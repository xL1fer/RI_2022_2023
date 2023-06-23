"""
    tokenizers.py

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

    Tokenizer module

    Holds the code/logic addressing the Tokenizer class
    and implemetns logic in how to process text into
    tokens.

"""

from utils import dynamically_init_class

from nltk.stem.porter import *                      # porter stemmer from nltk library
from nltk.stem.snowball import SnowballStemmer      # snowball stemmer
import re                                           # regex library to remove punctuation


def dynamically_init_tokenizer(**kwargs):
    """Dynamically initializes a Tokenizer object from this
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
    

class Tokenizer:
    """
    Top-level Tokenizer class
    
    This loosly defines a class over the concept of 
    an index.

    """
    def __init__(self, **kwargs):
        super().__init__()
    
    def tokenize(self, text):
        """
        Tokenizes a piece of text, this should be
        implemented by specific Tokenizer sub-classes.
        
        Parameters
        ----------
        text : str
            Sequence of text to be tokenized
            
        Returns
        ----------
        object
            An object that represent the output of the
            tokenization, yet to be defined by the students
        """
        raise NotImplementedError()

        
class PubMedTokenizer(Tokenizer):
    """
    An example of subclass that represents
    a special tokenizer responsible for the
    tokenization of articles from the PubMed.

    """
    def __init__(self, 
                 minL, 
                 stopwords_path, 
                 stemmer, 
                 *args, 
                 **kwargs):
        
        super().__init__(**kwargs)
        self.minL = minL
        self.regex_pattern = re.compile(r'[\s-]')
        self.stopwords_path = stopwords_path
        self.stopwords = self.init_stop_words()
        self.stemmer_name = stemmer
        if (self.stemmer_name == 'potterNLTK'):
            self.stemmer = PorterStemmer()
        elif (self.stemmer_name == 'showball'):
            self.stemmer = SnowballStemmer(language='english') 
        #self.stemmer = self.init_stemmer(stemmer)
        print("init PubMedTokenizer|", f"{minL=}, {stopwords_path=}, {stemmer=}")
        if kwargs:
            print(f"{self.__class__.__name__} also caught the following additional arguments {kwargs}")


    def init_stop_words(self):
        """
        Auxiliar function to initialize our stop words set
        (so that we do not initialize it more than once)
                
        Returns
        ----------
        stop_words
            stop words organized as a set data structure
        """
        # no stopwords file was specified
        if self.stopwords_path == None:
            return set()

        # read stop words file
        with open(self.stopwords_path) as f:
            stop_words = set(f.read().splitlines())

        return stop_words


    def init_stemmer(self, stemmer_name):
        """
        Auxiliar function to initialize a stemmer object
                
        Returns
        ----------
        stemmer
            initialized object, None otherwise
        """
        # no stemmer name was specified
        if stemmer_name is None:
            return None

        return PorterStemmer()

        
    def tokenize(self, text):
        """
        Function that receives a document (string) and converts each
        word into an admissible token

        Parameters
        ----------
        text
            string to be tokenized
                
        Returns
        ----------
        stemm_list
            list of processed tokens
        """
        # split phrase by spaces
        #token_stream = text.split()
        
        # NOTE: try to only split the terms by dash (-) in case
        #       the word matches the regex [a-zA-Z]+-[a-zA-Z]+

        # split phrase by spaces and dashes
        token_stream = self.regex_pattern.split(text)

        # remove ponctuation
        token_stream = [re.sub(r'[^\w]', '', word) for word in token_stream]

        # remove words that do not satisfie the minimum length required
        if self.minL is not None:
            token_stream = [word for word in token_stream if len(word) >= self.minL]

        # remove stop words
        token_stream = [word for word in token_stream if word.lower() not in self.stopwords]

        # stemm words
        if self.stemmer_name is not None:
            token_stream = [self.stemmer.stem(word) for word in token_stream]
        # lower case words in case no stemmer is specified
        else:
            token_stream = [word.lower() for word in token_stream]

        return token_stream
