"""
    searcher.py

    ====================================

    University of Aveiro
    Department of Electronics, Telecommunications and Informatics

    Information Retrieval (42596)
    Master's in Computer Engineering

    João Pedro dos Reis - 115513
    Luís Miguel Gomes Batista - 115279

    ====================================
    
    Information Retrieval Indexer System



    Base template created by: Luís Batista & João Reis
    Authors: 

    Searcher module

    Holds the code/logic addressing the Searcher class
    and the search engine.

"""

from utils import dynamically_init_class

import math
import os
import operator
import json
import sys

def dynamically_init_searcher(**kwargs):
    """Dynamically initializes a Searcher object from this
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


class Searcher:
    """
    Top-level Searcher class
    
    This loosely defines a class over the concept of 
    a searcher.

    """
    
    def __init__(self,
                index_folder:str,
                metadata,
                **kwargs):
        super().__init__()

        self.index_folder = index_folder    # index files folder
        self.metadata = metadata            # metadata structure

        self.terms_data = {}                # terms data structure
        self.indexes_dict = {}              # holds term's postings list loaded to memory
        self.doc_scores = {}                # documents' score

        self.oldest_keys = []               # holds old indexes_dict terms by decreasing order (oldest term in first index and so on)

        self.doc_window_size = {}           # auxiliar structure to hold window size calculations


    def query_search(self):
        raise NotImplementedError()

    def load_metadata(self):
        """
        Auxiliar function to fill metadata structure with parameters used in indexer
        to help in the searcher phase
        
        Returns
        ----------
        True
            in case metadata was sucessfully loaded
        False
            otherwise

        """
        if (not os.path.exists("metadata/metadata.json")):
            print("Could not load \"metadata.json\" file.")
            return False

        with open("metadata/metadata.json", "r", encoding="utf-8") as metadata_file: 
            self.metadata = json.load(metadata_file)

        #print(list(metadata.keys())[5000], list(metadata.values())[5000])

        return True

    def load_terms_data(self):
        """
        Auxiliar function to fill terms data structure with pre-computed data
        to help in the searcher phase
        
        Returns
        ----------
        True
            in case terms data was sucessfully loaded
        False
            otherwise

        """
        if (not os.path.exists("{}/data/terms_data.txt".format(self.metadata["metadata"]["index_output_folder"]))):
            print("Could not load \"terms_data.txt\" file.")
            return False

        with open("{}/data/terms_data.txt".format(self.metadata["metadata"]["index_output_folder"]), "r", encoding="utf-8") as terms_data_file: 
            for line in terms_data_file:
                data = line.strip().split(",")
                self.terms_data[data[0]] = (float(data[1]), int(data[2]))

        #print(list(terms_data.keys())[5000], list(terms_data.values())[5000])

        return True

    def load_docs_data(self):
        """
        Auxiliar function to fill docs data structure with pre-computed data
        to help in the searcher phase
        
        Returns
        ----------
        True
            in case terms data was sucessfully loaded
        False
            otherwise

        """
        if (not os.path.exists("{}/data/docs_data.txt".format(self.metadata["metadata"]["index_output_folder"]))):
            print("Could not load \"docs_data.txt\" file.")
            return False

        with open("{}/data/docs_data.txt".format(self.metadata["metadata"]["index_output_folder"]), "r", encoding="utf-8") as docs_data_file: 
            for line in docs_data_file:
                data = line.strip().split(",")
                self.docs_data[data[0]] = float(data[1])

        return True

    def calculate_window_boost(self, min_window_size):
        """
        Auxiliar function to calculate window boost for the retrieved documents
        
        Parameters
        ----------
        min_window_size
            query's minimum window size
        """

        # calculate boost factor in case B value is given
        if (self.B != None and self.B.isnumeric()):
            
            # calculate window size for each document
            for docid, terms_positions in self.doc_window_size.items():
                terms_num = len(terms_positions.keys())

                # document must have have at least as many words as the minimum window size
                if terms_num >= min_window_size:
                    win_size = sys.maxsize

                    temp_array = [None] * terms_num
                    # initialize temp_array with first terms positions and remove them from their list
                    for i, positions in enumerate(terms_positions.values()):
                        temp_array[i] = int(positions.pop(0))

                    # continue to search for the best possible window size
                    while True:
                        # update window size
                        curr_win_size = abs(max(temp_array) - min(temp_array))
                        if (curr_win_size < win_size):
                            win_size = curr_win_size

                        ind = -1
                        min_value = sys.maxsize
                        
                        # fetch next min position
                        for i, positions in enumerate(terms_positions.values()):
                            if (len(positions) > 0 and int(positions[0]) < min_value):
                                min_value = int(positions[0])
                                ind = i

                        if (ind == -1):
                            break

                        list(terms_positions.values())[ind].pop(0)
                        #print(terms_positions.values())
                        temp_array[ind] = min_value

                    window_boost = int(self.B) / (1 + win_size)
                    if window_boost < 1:
                        window_boost = 1

                    #print("docid: {}, win: {}, boost: {}".format(docid, win_size, window_boost))
                    self.doc_scores[docid] *= window_boost

class TFIDFSearcher(Searcher):
    """
    TFIDFSearcher represents an searcher that
    holds the logic of a TFIDF search and ranking 
    engine for the pubmed files

    """
    def __init__(self,
                index_folder:str,
                metadata,
                **kwargs):
        super().__init__(index_folder, metadata, **kwargs)

        print("init TFIDFSearcher|", f"{index_folder=}")
        print("SMART notation: %s" % (self.metadata["metadata"]["smart_notation"]))
        if kwargs:
            print(f"{self.__class__.__name__} also caught the following additional arguments {kwargs}")
            self.B = kwargs['B']
            self.topk = kwargs['topk']

    def query_search(self, index, tokenizer, query):
        """
        Function to search and comput scores for a query inputed
        by the user
        
        Parameters
        ----------
        index
            index object
        tokenizer
            tokenizer object
        query
            user query

        """
        # SMART ddd.qqq schema -> document.query
        #
        # To calculate ltc, we need:
        #   l -> 1 + log (term_frequency)
        #   t -> idf -> log (number_of_documents / document_frequency) ; document_frequency is the number of documents that contain the term 
        #   c -> cosine normalization -> 1 / sqrt(w1^2 + w2^2 + ...)

        # postings list dictionary size threshold
        dict_threshold = 20  # 20 MBytes

        # tokenize query
        token_stream = tokenizer.tokenize(query)

        # NOTE: Minimum window size will be equal to tokenized query elements which have an idf
        # higher than 2.0; it is to note that the stop words filter already removes
        # some words which have small idf values (words very frequent in documents)
        min_window_size = 0

        # get dictionary of weighted terms
        query_terms_dict = {term:(1 + math.log(token_stream.count(term), 10)) for term in token_stream}

        # calculate length of the query
        query_length = math.sqrt(sum([value ** 2 for value in query_terms_dict.values()]))

        # normalize weight
        query_terms_dict = {k: round(v / query_length, 2) for k, v in query_terms_dict.items()}

        # get index files to an array
        index_files = [file for file in os.listdir(self.index_folder) if os.path.isfile("{}/{}".format(self.index_folder, file))]

        # itereate through query
        for term, weight in query_terms_dict.items():
            # calculate score in case term is present in documents
            if (term in self.terms_data.keys()):
                # count words that have a good idf value
                if (self.terms_data[term][0] > 2.0):
                    min_window_size += 1

                # fetch term postings list from disk in case term is not yet in our indexes dictionary
                if (term not in self.indexes_dict.keys()):

                    self.oldest_keys.append(term)

                    # memory managemet
                    if (sys.getsizeof(self.indexes_dict) / 1048576 > dict_threshold):
                        # remove oldest least used key from dictionary
                        self.indexes_dict.pop(self.oldest_keys[0])
                        # remove oldest least used key from list
                        self.oldest_keys.pop(0)
                        #self.indexes_dict.clear()

                    # NOTE: metadata[term][1] -> index of the file where the term is saved
                    index.load_from_disk("{}/{}".format(self.index_folder, index_files[self.terms_data[term][1]]), self.indexes_dict, term, self.doc_window_size)
                # update term in oldest used key list
                else:
                    self.oldest_keys.remove(term)
                    self.oldest_keys.append(term)

                # multiply query term's weight by idf in case it's "lnc.ltc" or lnu.ltc (it would be x1 in case of "lnc.lnc")
                if self.metadata["metadata"]["smart_notation"] == "lnc.ltc" or self.metadata["metadata"]["smart_notation"] == "lnu.ltc":
                    weight *= self.terms_data[term][0]

                # add score to dictionary
                for doc_id in self.indexes_dict[term]:
                    if doc_id not in self.doc_scores.keys():
                        self.doc_scores[doc_id] = round(self.indexes_dict[term][doc_id] * weight, 2)
                    else:
                        self.doc_scores[doc_id] += round(self.indexes_dict[term][doc_id] * weight, 2)

        # add window boost factor
        self.calculate_window_boost(min_window_size)

        # reverse sort scores dictionary by value
        self.doc_scores = dict(sorted(self.doc_scores.items(), key=operator.itemgetter(1),reverse=True))


class BM25Searcher(Searcher):
    """
    BM25Searcher represents an searcher that
    holds the logic of a BM25 search and ranking 
    engine for the pubmed files

    """
    def __init__(self,
                index_folder:str,
                metadata,
                k1,
                b,
                **kwargs):
        super().__init__(index_folder, metadata, **kwargs)

        self.docs_data = {}
        self.k1 = 0.0
        self.b = 0.0

        print("init BM25Searcher|", f"{index_folder=}")
        print("k1: %s, b: %s" % (k1, b))
        if kwargs:
            print(f"{self.__class__.__name__} also caught the following additional arguments {kwargs}")
            self.B = kwargs['B']
            self.topk = kwargs['topk']

    def query_search(self, index, tokenizer, query):
        """
        Function to search and comput scores for a query inputed
        by the user
        
        Parameters
        ----------
        index
            index object
        tokenizer
            tokenizer object
        query
            user query

        """
        # postings list dictionary size threshold
        dict_threshold = 20  # 20 MBytes

        # tokenize query
        token_stream = tokenizer.tokenize(query)

        # NOTE: Minimum window size will be equal to tokenized query elements which have an idf
        # higher than 2.0; it is to note that the stop words filter already removes
        # some words which have small idf values (words very frequent in documents)
        min_window_size = 0

        # get dictionary of weighted terms
        query_terms_dict = {term:token_stream.count(term) for term in token_stream}

        # get index files to an array
        index_files = [file for file in os.listdir(self.index_folder) if os.path.isfile("{}/{}".format(self.index_folder, file))]

        # itereate through query
        for term, weight in query_terms_dict.items():
            # calculate score in case term is present in documents
            if (term in self.terms_data.keys()):
                # count words that have a good idf value
                if (self.terms_data[term][0] > 2.0):
                    min_window_size += 1
                # fetch term postings list from disk in case term is not yet in our indexes dictionary
                if (term not in self.indexes_dict.keys()):

                    self.oldest_keys.append(term)

                    # memory managemet
                    if (sys.getsizeof(self.indexes_dict) / 1048576 > dict_threshold):
                        # remove oldest least used key from dictionary
                        self.indexes_dict.pop(self.oldest_keys[0])
                        # remove oldest least used key from list
                        self.oldest_keys.pop(0)
                        #self.indexes_dict.clear()

                    # NOTE: metadata[term][1] -> index of the file where the term is saved
                    index.load_from_disk("{}/{}".format(self.index_folder, index_files[self.terms_data[term][1]]), self.indexes_dict, term, self.doc_window_size)
                # update term in oldest used key list
                else:
                    self.oldest_keys.remove(term)
                    self.oldest_keys.append(term)

                idf = self.terms_data[term][0]

                # add score to dictionary
                for doc_id in self.indexes_dict[term]:
                    term_freq = self.indexes_dict[term][doc_id]

                    # get (dl / avdl) value from docs data file
                    # dl -> document length (how many terms the document have)
                    # avdl -> average document length
                    dl_avdl = self.docs_data[doc_id]

                    if doc_id not in self.doc_scores.keys():
                        self.doc_scores[doc_id] = idf * ((self.k1 + 1) * term_freq) / (self.k1 * ((1 - self.b) + self.b * dl_avdl) + term_freq)
                    else:
                        self.doc_scores[doc_id] += idf * ((self.k1 + 1) * term_freq) / (self.k1 * ((1 - self.b) + self.b * dl_avdl) + term_freq)

        # add window boost factor
        self.calculate_window_boost(min_window_size)

        # reverse sort scores dictionary by value
        self.doc_scores = dict(sorted(self.doc_scores.items(), key=operator.itemgetter(1),reverse=True))