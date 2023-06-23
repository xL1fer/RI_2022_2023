"""
    core.py

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

    Core functionality of the system, its the
    main system entry point and holds the top-level
    logic. It starts by spliting the execution path 
    based on the operation mode.

"""

from tokenizers import dynamically_init_tokenizer
from reader import dynamically_init_reader
from index import dynamically_init_indexer
from searcher import dynamically_init_searcher

import math
import os
import json
from time import time
from statistics import median

from index import BaseIndex

def add_more_options_to_indexer(indexer_parser, indexer_settings_parser, indexer_doc_parser):
    """Add more options to the main program argparser.
    This function receives three argparser as arguments,
    and each of them can be used to add parsing options
    for the indexer mode. Check the details below to
    understand the difference between them.

    Parameters
    ----------
    indexer_parser : ArgumentParser
        This is the base argparser used during the indexer
        mode.
    indexer_settings_parser : ArgumentParser
        Derives from the ìndexer_parser` and specifies a group setting
        for the indexer. This should be used if we aim to add options to 
        the indexer.
    indexer_doc_parser : ArgumentParser
        Derives from the ìndexer_parser` and specifies a group setting
        for the document processing classes. This should be used if we 
        aim to add options to tokenizer, reader or any other document
        processing class.

    """
    # Studends can implement if needed of additional argparse options
    pass

def engine_logic(args):
    """
    Entrypoint for the main engine logic. Here we split
    the current two modes of execution. The indexer mode 
    and the searcher mode. Read the Readme.md to better
    understand this methodology.
    
    Parameters
    ----------
    args : argparse.Namespace
        Output of the ArgumentParser::parse_args method
        it holds every parameter that was specified during the
        cli and its default values.
    
    """
    
    
    if args.mode == "indexer":
        
        indexer_logic(args.path_to_collection,
                      args.index_output_folder,
                      args.indexer,
                      args.reader,
                      args.tk,
                      args.tfidf)
        
    elif args.mode == "searcher":
        ## TO BE DONE
        # searcher_logic [This will be probably the name of the function still in revision]
        searcher_logic(args.path_to_questions,
                       args.index_folder,
                       args.reader,
                       args.tk,
                       args.bm25,
                       args.tfidf,
                       args.windowboost,
                       args.documents)
        
    else:
        # this should be ensured by the argparser
        raise RuntimeError("Enter the else condition on the main.py, which should never happen!")
    

def indexer_logic(path_to_collection, 
                  index_output_folder, 
                  indexer_args, 
                  reader_args, 
                  tk_args,
                  tfidf_args):
    """
    Entrypoint for the main indexer logic. Here we start by
    dynamically loading the main modules (reader, tokenizer,
    indexer). Then we build 
    
    Parameters
    ----------
    args : argparse.Namespace
        Output of the ArgumentParser::parse_args method
        it holds every parameter that was specified during the
        cli and its default values.
    
    """
    # Students can change if they want
    
    # init reader
    reader = dynamically_init_reader(path_to_collection=path_to_collection,
                                   **reader_args.get_kwargs())
    
    # init tokenizer
    tokenizer = dynamically_init_tokenizer(**tk_args.get_kwargs())

    # init indexer
    if indexer_args.get_kwargs()['rsv'] == "tfidf":
        indexer = dynamically_init_indexer(**indexer_args.get_kwargs(), **tfidf_args.get_kwargs())
    else:
        indexer = dynamically_init_indexer(**indexer_args.get_kwargs())

    if indexer.rsv != "tfidf" and indexer.rsv!= "bm25":
        print("RSV \"{}\" not supported.".format(indexer.rsv))
        return

    if (indexer.rsv == "tfidf" and indexer.smart_notation != "lnc.ltc" and indexer.smart_notation != "lnc.lnc" and indexer.smart_notation != "lnu.ltc"):
        print("Weighting notation \"{}\" not supported.".format(indexer.smart_notation))
        print("Supported notations: \"lnc.ltc\",\"lnc.lnc\",\"lnu.ltc\".")
        return

    if (os.path.exists("{}/data/terms_data.txt".format(index_output_folder))):
        print("\nWarning: previous index files found in index folder.\n")
    
    # execute the indexer logic
    indexer.build_index(reader, tokenizer, index_output_folder)

    # save metadata used during indexing phase
    indexer.save_metadata(tokenizer.minL, tokenizer.stopwords_path, tokenizer.stemmer_name, indexer.rsv, index_output_folder)
    
    # get the final index
    index = indexer.get_index()
    
    # print some statistics about the produced index
    index.print_statistics(indexer.indexing_time, indexer.merging_time, indexer.temp_ind, indexer.ind_size, indexer.voc_num, indexer.available_memory, indexer.memory_threshold)


def searcher_logic(path_to_questions,
                   index_folder,
                   reader_args,
                   tk_args,
                   bm25_args,
                   tfidf_args,
                   windowboost_args,
                   documents_args):
    """
    Entrypoint for the main indexer logic. Here we start by
    dynamically loading the main modules (reader, tokenizer,
    indexer). Then we build 

    Parameters
    ----------
    args : argparse.Namespace
        Output of the ArgumentParser::parse_args method
        it holds every parameter that was specified during the
        cli and its default values.

    """
    # start by loading metadata
    metadata = {}

    if (not os.path.exists("metadata/metadata.json")):
        print("Could not load \"metadata.json\" file.")
        return

    with open("metadata/metadata.json", "r", encoding="utf-8") as metadata_file: 
        metadata = json.load(metadata_file)

    # init searcher
    if metadata["metadata"]["rsv"] == "tfidf":
        # TFIDF
        searcher = dynamically_init_searcher(index_folder=index_folder,
                                            metadata=metadata,
                                            **tfidf_args.get_kwargs(),
                                            **windowboost_args.get_kwargs(),
                                            **documents_args.get_kwargs())
                                            #**searcher_args.get_kwargs())
    else:
        # BM25
        searcher = dynamically_init_searcher(index_folder=index_folder,
                                            metadata=metadata,
                                            **bm25_args.get_kwargs(),
                                            **windowboost_args.get_kwargs(),
                                            **documents_args.get_kwargs())
                                            #**searcher_args.get_kwargs())

    if not searcher.load_terms_data():                      # load terms data file
        return

    if (searcher.metadata["metadata"]["rsv"] == "bm25"):    # load docs data file in case of BM25 rsv
        if not searcher.load_docs_data():
            return

    # init reader
    reader = dynamically_init_reader(path_to_questions=path_to_questions,**reader_args.get_kwargs())

    # init tokenizer
    tokenizer = dynamically_init_tokenizer(minL=searcher.metadata["metadata"]["tokenizer"]["minL"], stopwords_path=searcher.metadata["metadata"]["tokenizer"]["stopwords_path"],
                                            stemmer=searcher.metadata["metadata"]["tokenizer"]["stemmer"], **tk_args.get_kwargs())

    # init index
    index = BaseIndex()

    #################################
    # questions loop       ##########
    #################################

    query_times = []

    for query_data in reader.read_questions():
        query_start = time()    # start counter for indexing time

        # query_data[0] -> question string
        # query_data[1] -> set with relevant documents' ids
        searcher.doc_scores.clear()
        searcher.doc_window_size.clear()

        print(query_data[0])
        searcher.query_search(index, tokenizer, query_data[0])

        if len(searcher.doc_scores) == 0:
            print("No matching documents found.")
            continue

        topk_scores = {k: searcher.doc_scores[k] for k in list(searcher.doc_scores)[0:documents_args.get_kwargs()["topk"]]}
        #for key, value in topk_scores.items():
            #print("%12s\t%10.2f" % (key, value))
        #print("Relevant documents", query_data[1], "\n")

        query_times.append(time() - query_start)

        # evaluation metrics
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        f_measure = 0.0
        precision = 0.0
        recall = 0.0
        average_precision_array = []
        average_precision = 0.0
        retrieved_doc_set = set(topk_scores.keys())

        for docid in topk_scores.keys():
            # check if retrieved document is relevant
            if (int(docid) in query_data[1]):
                tp += 1
                average_precision_array.append(tp / (tp + fp))
            else:
                fp += 1

        for docid in query_data[1]:
            # check if relevant document is retrieved
            if (str(docid) not in retrieved_doc_set):
                fn += 1

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        if (precision + recall) > 0.0:
            f_measure = 2 * precision * recall / (precision + recall)

        if len(average_precision_array) > 0:
            average_precision = sum(average_precision_array) / len(average_precision_array)

        print("Precision:", precision)
        print("Recall:", recall)
        print("F-measure:", f_measure)
        print("Average Precision:", average_precision)

        print("Query Time:", query_times[-1], "s")
        print("Average Query Time:", sum(query_times) / len(query_times))
        print("Median Query Time:", median(query_times))
        
        '''# Statistics writting to file
        import csv
        data = [round(precision,2),round(recall,2),round(f_measure,2),round(average_precision,2),round(query_times[-1],2),round(sum(query_times)/len(query_times),2),round(median(query_times),2)]
        with open('questions/tiny_top100_B2.csv','a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)'''

    #################################
    # searcher loop        ##########
    #################################

    """
    while True:
        searcher.doc_scores.clear()
        
        print("\n> Insert query (-exit):")
        query = input()
        if query == "-exit" or query == "-e":
            break

        searcher.query_search(index, tokenizer, query)

        if len(searcher.doc_scores) == 0:
            print("No matching documents found.")
            continue

        cur_page = 1
        max_page = math.ceil(len(searcher.doc_scores) / 10)

        # paginator loop
        while True:
            #print("Found {} results".format(len(doc_scores)))
            print("\nCurrent query: \"%s\"\n\n%12s\t%11s\n-------------------------------" % (query, "Document", "Score"))
            top10_scores = {k: searcher.doc_scores[k] for k in list(searcher.doc_scores)[(cur_page - 1) * 10:(cur_page) * 10]}
            for key, value in top10_scores.items():
                print("%12s\t%10.2f" % (key, value))

            print("\n< prev%8d/%d%14s\n%18s" % (cur_page, max_page, "next >", "(quit)"))

            user_input = input()
            if user_input == "prev" and cur_page > 1:
                cur_page -= 1
            elif user_input == "next" and cur_page < max_page:
                cur_page += 1
            elif user_input == "quit" or user_input == "q":
                break
    """

    # clear searcher attributes
    searcher.metadata.clear()
    searcher.terms_data.clear()
    searcher.indexes_dict.clear()
    searcher.doc_scores.clear()