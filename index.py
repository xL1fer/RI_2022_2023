"""
    index.py

    ====================================

    University of Aveiro
    Department of Electronics, Telecommunications and Informatics

    Information Retrieval (42596)
    Master's in Computer Engineering

    João Pedro dos Reis - 115513
    Luís Miguel Gomes Batista - 115279

    ====================================
    
    Information Retrieval Indexer System



    Base template created by: Tiago Almeida & Sérgio Matos
    Authors: 

    Indexer module

    Holds the code/logic addressing the Indexer class
    and the index managment.

"""

from utils import dynamically_init_class

import psutil                       # check available system memory
import os                           # manage folders
import sys                          # get size of objects
from time import time
import math
import json                         # save metadata in json format

def dynamically_init_indexer(**kwargs):
    """Dynamically initializes a Indexer object from this
    module.

    Parameters
    ----------
    kwargs : Dict[str, object]
        python dictionary that holds the variables and their values
        that are used as arguments during the class initialization.
        Note that the variable `class` must be here and that it will
        not be passed as an initialization argument since it is removed
        from this dict.
    
    Returns
    ----------
    object
        python instance
    """
    return dynamically_init_class(__name__, **kwargs)


class Indexer:
    """
    Top-level Indexer class
    
    This loosly defines a class over the concept of 
    an index.

    """
    def __init__(self, 
                 index_instance,
                 **kwargs):
        super().__init__()
        self._index = index_instance
    
    def get_index(self):
        return self._index
    
    def build_index(self, reader, tokenizer, index_output_folder):
        """
        Holds the logic for the indexing algorithm.
        
        This method should be implemented by more specific sub-classes
        
        Parameters
        ----------
        reader : Reader
            a reader object that knows how to read the collection
        tokenizer: Tokenizer
            a tokenizer object that knows how to convert text into
            tokens
        index_output_folder: str
            the folder where the resulting index or indexes should
            be stored, with some additional information.
            
        """
        raise NotImplementedError()
    

class SPIMIIndexer(Indexer):
    """
    The SPIMIIndexer represents an indexer that
    holds the logic to build an index according to the
    spimi algorithm.

    """
    def __init__(self, 
                 posting_threshold, 
                 memory_threshold,
                 rsv,
                 **kwargs):
        # lets suppose that the SPIMIIindex uses the inverted index, so
        # it initializes this type of index
        super().__init__(InvertedIndex(), **kwargs)
        print("init SPIMIIndexer|", f"{posting_threshold=}, {memory_threshold=}, {rsv=}")
        if kwargs:
            print(f"{self.__class__.__name__} also caught the following additional arguments {kwargs}")
            if ("smart_notation" in kwargs):
                self.smart_notation = kwargs["smart_notation"]

        self.rsv = rsv
        self.total_documents = 0
        self.docs_data = {}
        self.average_doc_length = 0     # avdl

        # statistics attributes
        self.indexing_time = 0.0
        self.merging_time = 0.0
        self.temp_ind = 0
        self.ind_size = 0.0
        self.voc_num = 0


    def save_metadata(self, minL, stopwords_path, stemmer_name, rsv, index_output_folder):
        """
        Method to save metadata used on index to be loaded later by the searcher
        """
        # make sure output folder exists
        if (not os.path.exists("metadata")):
            os.makedirs("metadata")

        print("\nSaving metadata file in path \"metadata/metadata.json\"... ")

        if rsv == "tfidf":
            metadata_dict = {"metadata": {"tokenizer": {"minL": minL, "stopwords_path": stopwords_path, "stemmer": stemmer_name}, "rsv": rsv, "smart_notation": self.smart_notation, "index_output_folder": index_output_folder} }
        else:
            metadata_dict = {"metadata": {"tokenizer": {"minL": minL, "stopwords_path": stopwords_path, "stemmer": stemmer_name}, "rsv": rsv, "index_output_folder": index_output_folder} }
        
            

        with open("metadata/metadata.json", "w") as metadata_file:
            json.dump(metadata_dict, metadata_file)
        

    def write_to_disk(self, index_output_folder, indexes_dict, terms_list, file_name, terms_data_file=None, terms_data_num=None):
        """
        Method that receives a dictionary and all its terms in an ordered list
        and writes the dictionary to disk in alphabetical order

        Parameters
        ----------
        index_output_folder
            output folder directory
        indexes_dict
            dictionary yo be written to file
        terms_list
            ordered terms list
        file_name
            output file name
        """
        # make sure output folder exists
        if (not os.path.exists(index_output_folder)):
            os.makedirs(index_output_folder)

        print("Writing indexes to file \"{}.txt\"... ".format(file_name), end="")

        with open("{}/{}.txt".format(index_output_folder, file_name), "w", encoding="utf-8") as output_file:
            for term in terms_list:
                # NOTE: what is the correct format to write to files?
                #output_file.write('{"%s": %s}\n' % (term, json.dumps(indexes_dict[term])))

                output_file.write(term)
                for doc_id, term_weight in indexes_dict[term].items():
                    output_file.write(';{}:{}:'.format(doc_id, term_weight))

                    output_file.write(self.term_positions[term + doc_id])

                output_file.write('\n')

                # in merging phase, also write a terms data file to memory
                if terms_data_file is not None:
                    # (term, idf, doc_index)
                    terms_data_file.write("{},{},{}\n".format(term, round(math.log(self.total_documents / len(indexes_dict[term]), 10), 2), terms_data_num))

        print("Done!")

        
    def build_index(self, reader, tokenizer, index_output_folder):
        """
        Function responsible for implementing the inverted index and
        merge the created blocks afterwards

        Parameters
        ----------
        reader
            reader object
        tokenizer
            tokenizer object
        index_output_folder
            output folder directory
        """
        self.available_memory = 2.0 * 1024 * 1024 * 1024                 # define an arbitrary available memory of 2GBytes
        self.memory_threshold = self.available_memory * 0.6              # use 60% of available memory as the memory threshold
        synthetic_memory = psutil.virtual_memory().available - self.memory_threshold    # our final synthetic memory

        print("< Current Memory Available: " + str(self.available_memory / 1073741824) + " GBytes >")
        print("< Memory Threshold: " + str(self.memory_threshold / 1073741824) + " GBytes >")

        print("\nIndexing some documents to \"{}/\" folder...".format(index_output_folder))
    
        block_counter = 0
        indexes_dict = {}
        token_stream = []

        index_start = time()    # start counter for indexing time

        self.term_positions = {}

        # read each line that is being returned by the reader
        for document in reader.read_json():
            # count total documents
            self.total_documents += 1

            # tokenize the document
            token_stream += tokenizer.tokenize(document["title"] + document["abstract"])

            # term positions list
            for i in range(len(token_stream)):
                if token_stream[i] + document["pmid"] in self.term_positions.keys():
                    self.term_positions[token_stream[i] + document["pmid"]] += ",{}".format(i)
                else:
                    self.term_positions[token_stream[i] + document["pmid"]] = str(i) 

            # TFIDF rsv
            if self.rsv == "tfidf":
                # get dictionary of weighted terms
                term_weight_dict = {term:(1 + math.log(token_stream.count(term), 10)) for term in token_stream}

                if self.smart_notation == "lnc.ltc" or self.smart_notation == "lnc.lnc":
                    # calculate length of the document
                    doc_length = math.sqrt(sum([value ** 2 for value in term_weight_dict.values()]))

                    # normalize weight
                    term_weight_dict = {k: round(v / doc_length, 2) for k, v in term_weight_dict.items()}

                elif self.smart_notation == "lnu.ltc":
                    # calculate length of the document
                    doc_length = len(term_weight_dict)

                    # normalize weight
                    term_weight_dict = {k: round(v / doc_length, 2) for k, v in term_weight_dict.items()}

            # BM25 rsv
            else:
                # get dictionary of weighted terms
                term_weight_dict = {term:token_stream.count(term) for term in token_stream}

                # total document terms
                total_terms = sum(term_weight_dict.values())

                # save document data containing document id and total terms
                self.docs_data[document["pmid"]] = total_terms

            # add "term: (docid, term_weight)" to dicionary - SPIMI inverted indexer
            for term, weight in term_weight_dict.items():
                self._index.add_term(term, document["pmid"], indexes_dict, weight)

            token_stream.clear()

            #print('Available: ', self.memory_threshold, synthetic_memory - psutil.virtual_memory().available)

            # check if we did not exceed our memory limit
            if (self.total_documents % 1000 == 0) or (synthetic_memory - psutil.virtual_memory().available < 0):
                continue

            # memory full, write to disk and reset indexes dictionary
            terms_list = list(indexes_dict.keys())
            terms_list.sort()

            self.write_to_disk(index_output_folder, indexes_dict, terms_list, block_counter)
            block_counter += 1

            indexes_dict.clear()
            self.term_positions.clear()

        # end of file, write to disk and reset indexes dictionary
        if len(indexes_dict) > 0:
            terms_list = list(indexes_dict.keys())
            terms_list.sort()

            self.write_to_disk(index_output_folder, indexes_dict, terms_list, block_counter)
            block_counter += 1

            indexes_dict.clear()
            self.term_positions.clear()

        self.indexing_time = (time() - index_start) # register total timestamp for indexing time

        # merging step #################

        merge_start = time()
        self.merge_blocks(index_output_folder)
        self.merging_time = (time() - merge_start)  # register total timestamp for merging time

        # register total index size on disk
        for file in os.scandir("{}/merged/".format(index_output_folder)):
            self.ind_size += os.path.getsize(file)

        # save aditional data file in case of BM25 rsv
        if self.rsv == "bm25":
            with open("{}/data/docs_data.txt".format(index_output_folder), "w", encoding="utf-8") as docs_data_file:
                avdl = sum(self.docs_data.values()) / self.total_documents
                for doc_id, dl in self.docs_data.items():
                    docs_data_file.write("{},{:.2f}\n".format(doc_id, dl / avdl))

            self.docs_data.clear()

    def get_smaller_term_index(self, list):
        """
        Auxiliar function to get the index of the alphabetical first term
        in a given list
        
        Parameters
        ----------
        list
            list to be searched
                
        Returns
        ----------
        min
            index of the first alphabetical term
        """
        min = None
        for i in range(len(list)):
            if min is None and list[i][0] is not None:
                min = i
            elif list[i][0] is not None and list[i][0] < list[min][0]:
                min = i

        return min

    
    def add_to_dictionary(self, line, dictionary):
        """
        Auxiliar function to add a list containing a term and the documents where
        it occurs to a dictionary
        
        Parameters
        ----------
        line
            list containing term and documents where it occurs
        dictionary
            dictionary where the parsed list will be added
        """
        for i in range(1, len(line)):
            doc_info = line[i].split(":")

            if line[0] not in dictionary.keys():
                dictionary[line[0]] = {doc_info[0]: float(doc_info[1])}
            else:
                dictionary[line[0]][doc_info[0]] = float(doc_info[1])

            # term positions list
            self.term_positions[line[0] + doc_info[0]] = doc_info[2]


    def get_next_line(self, file):
        """
        Function to get next line of given file and split by delimiter
        
        Parameters
        ----------
        file
            input file to get next line

        Returns
        ----------
        head_line
            next available line in the file, or None otherwise
        """
        # initialize empty head line
        head_line = [None]

        # try to fill head line with next element in the file
        for line in file:
            head_line = line.strip().split(";")
            break

        return head_line


    def merge_blocks(self, index_output_folder):
        """
        Method to merge a group of temporary index files
        into bigger sorted index
        
        Parameters
        ----------
        index_output_folder
            output folder directory
        """
        print("\nMerging some blocks to \"{}/merged/\" folder...".format(index_output_folder))

        dict_threshold = 20  # 20 MBytes

        # create data folder
        if (not os.path.exists("{}/data".format(index_output_folder))):
            os.makedirs("{}/data".format(index_output_folder))
            
        # create terms data file
        terms_data_file = open("{}/data/terms_data.txt".format(index_output_folder), "w", encoding="utf-8")

        index_files = [open("{}/{}".format(index_output_folder, file), encoding='utf8') for file in os.listdir(index_output_folder) if os.path.isfile("{}/{}".format(index_output_folder, file))]
        self.temp_ind = len(index_files)    # register number of temporary files

        merge_dict = {}
        head_terms = []
        terms_data_num = 0
        
        # fill head terms list with first term of each block
        for file in index_files:
            for line in file:
                head_terms.append(line.strip().split(";"))
                break

        # keep parsing the documents until eof
        while True:
            smaller_term_ind = self.get_smaller_term_index(head_terms)
            # break if all documents have been parsed
            if smaller_term_ind is None:
                break

            curr_line = head_terms[smaller_term_ind]

            # fill merge dictionary
            self.add_to_dictionary(curr_line, merge_dict)

            # fill current entry with next line in the file
            head_terms[smaller_term_ind] = self.get_next_line(index_files[smaller_term_ind])
            
            # write to disk in case we surpass the dictionary threshold
            if (sys.getsizeof(merge_dict) / 1048576 > dict_threshold):
                # add all the terms that are equal to the last in the dictionary to the dictionary
                last_term = list(merge_dict.keys())[-1]
                for i in range(len(head_terms)):
                    if head_terms[i][0] == last_term:
                        self.add_to_dictionary(head_terms[i], merge_dict)

                        # fill current entry with next line in the file
                        head_terms[i] = self.get_next_line(index_files[i])

                terms_list = list(merge_dict.keys())
                self.write_to_disk("{}/merged/".format(index_output_folder), merge_dict, terms_list, "{};{}_{}".format(terms_data_num, list(merge_dict.keys())[0], last_term), terms_data_file, terms_data_num)

                self.voc_num += len(merge_dict)         # add number of terms to vocabulary number
                merge_dict.clear()
                terms_data_num += 1                       # increase metadata file counter

        # write to disk in case we reach eof in all files
        if (len(merge_dict) > 0):
            terms_list = list(merge_dict.keys())
            self.write_to_disk("{}/merged/".format(index_output_folder), merge_dict, terms_list, "{};{}_{}".format(terms_data_num, list(merge_dict.keys())[0],list(merge_dict.keys())[-1]), terms_data_file, terms_data_num)

            self.voc_num += len(merge_dict)         # add number of terms to vocabulary number
            merge_dict.clear()

        # close all files after the merging step
        for file in index_files:
            file.close()
        terms_data_file.close()

        # delete temporary index files
        for file_name in os.listdir(index_output_folder) :
            if os.path.isfile("{}/{}".format(index_output_folder, file_name)):
                os.remove("{}/{}".format(index_output_folder, file_name))
        
class BaseIndex:
    """
    Top-level Index class
    
    This loosly defines a class over the concept of 
    an index.

    """
    def add_term(self, term, doc_id, *args, **kwargs):
        raise NotImplementedError()
    
    def print_statistics(self):
        raise NotImplementedError()

    @classmethod
    def load_from_disk(cls, path_to_folder:str, indexes_dict, term, doc_window_size):
        """
        Loads the index from disk, note that this
        the process may be complex, especially if your index
        cannot be fully loaded. Think of ways to coordinate
        this job and have a top-level abstraction that can
        represent the entire index even without being fully load
        in memory.
        
        Tip: The most important thing is to always know where your
        data is on disk and how to easily access it. Recall that the
        disk access are the slowest operation in a computation device, 
        so they should be minimized.
        
        Parameters
        ----------
        path_to_folder: str
            the folder where the index or indexes are stored.
        indexes_dict
            structure to save term and postings list
        term
            term to search for
        doc_window_size
            dictionary to hold documents' window size
        """
        with open(path_to_folder, "r", encoding="utf-8") as index_file:
            # read each line of the file
            for line in index_file:
                line = line.strip().split(";")
                
                # check if current line contains desired term
                if (line[0] == term):
                    # parse line
                    for i in range(1, len(line)):
                        doc_info = line[i].split(":")

                        if line[0] not in indexes_dict.keys():
                            indexes_dict[line[0]] = {doc_info[0]: float(doc_info[1])}
                        else:
                            indexes_dict[line[0]][doc_info[0]] = float(doc_info[1])

                        # window size calculation auxiliar structure
                        if doc_info[0] not in doc_window_size.keys():
                            doc_window_size[doc_info[0]] = {line[0]: doc_info[2].split(',')}
                        else:
                            doc_window_size[doc_info[0]][line[0]] = doc_info[2].split(',')

                    break


class InvertedIndex(BaseIndex):
    
    # make an efficient implementation of an inverted index
    def add_term(self, term, doc_id, *args, **kwargs):
        """
        Auxiliar function to do the inverted index step of the
        SPIMI algorithm
        
        Parameters
        ----------
        term
            document term
        doc_id
            term's document id
        *args
            array of extra possible parameters
        """
        # args[0] -> indexes dictionary
        # args[1] -> current term weight

        # term is not in dictionary
        if term not in args[0].keys():
            args[0][term] = {doc_id: args[1]}
        # if term in dictionary, try to find the same docid in the values
        # NOTE: this condition should never be met, since the same docid cannot reappear
        #elif doc_id in args[0][term]:
        #    args[0][term][doc_id] += args[1]
        # docid not found, add a new entry
        else:
            args[0][term][doc_id] = args[1]
        
    @classmethod
    def load_from_disk(cls, path_to_folder:str):
        """
        Loads the index from disk, note that this
        the process may be complex, especially if your index
        cannot be fully loaded. Think of ways to coordinate
        this job and have a top-level abstraction that can
        represent the entire index even without being fully load
        in memory.
        
        Tip: The most important thing is to always know where your
        data is on disk and how to easily access it. Recall that the
        disk access are the slowest operation in a computation device, 
        so they should be minimized.
        
        Parameters
        ----------
        path_to_folder: str
            the folder where the index or indexes are stored.
            
        """
        raise NotImplementedError()

    def print_statistics(self, indexing_time, merging_time, temp_ind, ind_size, voc_num, available_memory, memory_threshold):
        """
        Function to print statistics about the files
        
        Parameters
        ----------
        indexing_time
            total indexing time
        merging_time
            merging time (last SPIMI step)
        temp_ind
            number of temporary index segments written to disk (before merging)
        ind_size
            total index size on disk
        voc_num
            vocabulary size (number of terms)
        available_memory
            arbitrarily defined system memory
        memory_threshold
            memory threshold used based on a percentage of a given system memory (2GBytes by default)
        """
        #print("Print some stats about this index.. This should be implemented by the base classes")
        print(f"\n:: Statistics ::")
        print(f"> Total indexing time: {'%.3f' % indexing_time} seconds")
        print(f"> Total merging time: {'%.3f' % merging_time} seconds")
        print(f"> Number of temporary index files: {'%d' % temp_ind}")
        print(f"> Total index size: {'%.3f' % (ind_size / 1048576)} MBytes")
        print(f"> Vocabulary Size (number of terms): {'%d' % voc_num}")
        print(f"> Available system memory: {'%.3f' % (available_memory / 1073741824)} GBytes")
        print(f"> Memory threshold used: {'%.3f' % (memory_threshold / 1073741824)} GBytes")
