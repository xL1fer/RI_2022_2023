"""
    reader.py

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

    Reader module

    Holds the code/logic addressing the Reader class
    and how to read text from a specific data format.

"""

from utils import dynamically_init_class

import json                         # parse json files
import gzip                         # decode gzip format files
from zipfile import ZipFile         # decompress zip files


def dynamically_init_reader(**kwargs):
    """Dynamically initializes a Reader object from this
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


class Reader:
    """
    Top-level Reader class
    
    This loosly defines a class over the concept of 
    a reader.
    
    Since there are multiple ways for implementing
    this class, we did not defined any specific method 
    in this started code.

    """
    def __init__(self, 
                 path_to_collection:str, 
                 **kwargs):
        super().__init__()
        self.path_to_collection = path_to_collection
        
    
class PubMedReader(Reader):
    """
    PubMedReader class
    
    This class is used to specifically read
    PubMed files and yield its inner documents

    """
    def __init__(self, 
                 path_to_collection:str,
                 **kwargs):
        super().__init__(path_to_collection, **kwargs)
        print("init PubMedReader|", f"{self.path_to_collection=}")
        if kwargs:
            print(f"{self.__class__.__name__} also caught the following additional arguments {kwargs}")
    
    def read_json(self):
        """
        Auxiliar function to open the gzip file and read each json line
                
        Yields
        ----------
        data
            each line of the files converted to dictionaries
            with keys and values
        """
        with gzip.open(self.path_to_collection, 'r') as json_file:
            # read each line of the file without loading the whole file to memory
            for json_line in json_file:
                # data is a dictionary containing each line of the file
                data = json.loads(json_line.decode('utf-8'))

                # yield each line at the end of the loop
                yield data

class QuestionsReader(Reader):
    """
    QuestionsReader class
    
    This class is used to retrieve a group of questions
    to be analysed and ranked by the searcher class

    """
    def __init__(self, 
                 path_to_questions:str,
                 **kwargs):
        super().__init__(path_to_questions, **kwargs)
        # I do not want to refactor Reader and here path_to_collection does not make any sense.
        # So consider using self.path_to_questions instead (but both variables point to the same thing, it just to not break old code)
        self.path_to_questions = self.path_to_collection
        print("init QuestionsReader|", f"{self.path_to_questions=}")
        if kwargs:
            print(f"{self.__class__.__name__} also caught the following additional arguments {kwargs}")

    def read_questions(self):
        """
        Auxiliar function to open the gzip file and read each json line
                
        Yields
        ----------
        data
            each line of the files converted to dictionaries
            with keys and values
        """
        #with open(self.path_to_questions, 'r') as questions_file:
        #    for question in questions_file:
                # yield each question
        #        yield question

        with ZipFile(self.path_to_questions, 'r') as zip:
            print()
            zip.printdir()
            print()

            for file_name in zip.namelist():
                file_data = (zip.read(file_name).decode('utf-8')).split('\n')

                for line in file_data:
                    line_data = line.split(',')

                    # we do this verification because there are some faulty lines in the json file, so we first need to make sure the line
                    # can be converted to json
                    if (len(line_data) > 1):
                        query_data = (json.loads(line)["query_text"], set([int(x) for x in json.loads(line)["documents_pmid"]]))

                        #print(query_data)
                        yield query_data

                #break # only read one question file for now 