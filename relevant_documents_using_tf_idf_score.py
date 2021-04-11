import pickle
import numpy as np
import math
import joblib
from collections import Counter
from question2.preprocessing import preProcessSentence
from question2.intersection_and_union import *

inverted_index = joblib.load('question2/inverted_index')


def compute_binary_variant(term_freq_matrix):

  for key in dictionary.keys():
    
    value_mapped_to_key = term_map[key]
    posting_list = dictionary[key][1]

    for document_number in posting_list:
      term_freq_matrix[value_mapped_to_key][document_number] = 1


def compute_raw_count_variant(term_freq_matrix):

  for key in dictionary.keys():

    value_mapped_to_key = term_map[key]
    posting_list = dictionary[key][1]

    for document_number in posting_list:
      prev_val = term_freq_matrix[value_mapped_to_key][document_number]
      term_freq_matrix[value_mapped_to_key][document_number] = 1 + prev_val


def compute_log_normalized_variant(term_freq_matrix):

  for key in dictionary.keys():

    value_mapped_to_key = term_map[key]
    posting_list = dictionary[key][1]

    for document_number in posting_list:
      prev_val = term_freq_matrix[value_mapped_to_key][document_number]
      term_freq_matrix[value_mapped_to_key][document_number] = 1 + prev_val

  for i in range(len(term_freq_matrix)):
    for j in range(len(term_freq_matrix[0])):
      term_freq_matrix[i][j] = math.log10(1 + term_freq_matrix[i][j])


def compute_term_freq_variant(term_freq_matrix):

  for key in dictionary.keys():

    value_mapped_to_key = term_map[key]
    posting_list = dictionary[key][1]

    for document_number in posting_list:
      prev_val = term_freq_matrix[value_mapped_to_key][document_number]
      term_freq_matrix[value_mapped_to_key][document_number] = 1 + prev_val

  sum_freq_of_all_terms_in_a_doc = np.sum(term_freq_matrix,axis=0) 

  for i in range(len(term_freq_matrix)):
    for j in range(len(term_freq_matrix[0])):
      term_freq_matrix[i][j] = (term_freq_matrix[i][j] / sum_freq_of_all_terms_in_a_doc[j])


def compute_double_normalization_variant(term_freq_matrix):

  for key in dictionary.keys():

    value_mapped_to_key = term_map[key]
    posting_list = dictionary[key][1]

    for document_number in posting_list:
      prev_val = term_freq_matrix[value_mapped_to_key][document_number]
      term_freq_matrix[value_mapped_to_key][document_number] = 1 + prev_val

  max_freq_in_a_doc = np.amax(term_freq_matrix,axis=0) 

  for i in range(len(term_freq_matrix)):
    for j in range(len(term_freq_matrix[0])):
      res = 0.5 * (term_freq_matrix[i][j] / max_freq_in_a_doc[j])
      term_freq_matrix[i][j] = 0.5 + res


def compute_term_freq_matrix(term_freq_matrix, variant):

  if (variant == "1"):
    compute_binary_variant(term_freq_matrix)
  elif (variant == "2"):
    compute_raw_count_variant(term_freq_matrix)
  elif (variant == "3"):
    compute_term_freq_variant(term_freq_matrix)
  elif (variant == "4"):
    compute_log_normalized_variant(term_freq_matrix)
  elif (variant == "5"):
    compute_double_normalization_variant(term_freq_matrix)



def get_query_vector(query_terms, norm_type="4"):
  query_vec = np.zeros((1,len(term_map.keys())))
  query_term_frequencies = Counter(query_terms)
  # print("query_term_frequencies",query_term_frequencies)

  for term in np.unique(query_terms):
    idf = math.log10( len(term_freq_matrix[0]) / (1 + dictionary[term][0][0]))
    value_mapped_to_key = term_map[term]
    if norm_type=="4":
      print("here")
      term_freq = math.log10(1 + query_term_frequencies[term])
    elif norm_type=="1":
      term_freq = 1
    elif norm_type=="2":
      term_freq = query_term_frequencies[term]
    elif norm_type=="3":
      term_freq = query_term_frequencies[term]/ sum(query_term_frequencies.values())
    elif norm_type=="5":
      term_freq = 0.5 + (0.5 *(query_term_frequencies[term]/(max(query_term_frequencies.values()))))



    query_vec[0][value_mapped_to_key] = term_freq * idf

  print("query_vec",query_vec,query_vec.shape,np.where((query_vec!=0)))
  return query_vec

def get_document_scores(term_freq_matrix,value_in_term_map):

    for doc_num in range(0, len(term_freq_matrix[0])):

        idf = math.log10( len(term_freq_matrix[0]) / (1 + dictionary[term][0][0]))

        tf = term_freq_matrix[value_in_term_map][doc_num]

        document_scores[doc_num] += (tf * idf)
    return document_scores

def get_document_scores_optimized(reduced_docs_list,value_in_term_map):
    for doc_num in reduced_docs_list:
          idf = math.log10( len(term_freq_matrix[0]) / (1 + dictionary[term][0][0]))

          tf = term_freq_matrix[value_in_term_map][doc_num-1]

          document_scores[doc_num-1] += (tf * idf)
    return document_scores

def cosine_similarity(query_vector, document_vector):
  numerator = np.dot(query_vector,document_vector.T)
  denominator = np.linalg.norm(query_vector,axis=1)* np.linalg.norm(document_vector,axis=1)
  cosine_similarity = numerator/denominator
  # print("cosine_similarity",cosine_similarity,cosine_similarity.shape)
  cosine_similarity = cosine_similarity.squeeze()
  return cosine_similarity

if __name__=="__main__":
  corpus = {} 

  # file to document number mapping
  document_map = {}

  # term to number mapping
  term_map = {}

  # key and corresponding posting list
  dictionary = {}

  with open('DocTerms_Spacy.pickle', 'rb') as handle:
    corpus = pickle.load(handle)


  # creating file to document mapping

  counter_doc = -1

  for key in corpus.keys():
    if (key not in document_map.keys()):
      counter_doc += 1
      document_map[key] = counter_doc

  # creating the postings

  for key in corpus.keys():
    
    list_of_words = corpus[key][0]

    for word in list_of_words:

      if (word in dictionary.keys()):
        l = dictionary[word]
        l[1].append(document_map.get(key))
      else :
        dictionary[word] = [[],[document_map.get(key)]]


  # creating term mapping

  counter_term = -1

  for key in dictionary.keys():
    if (key not in term_map.keys()):
      counter_term += 1
      term_map[key] = counter_term
      
  # print(term_map.keys())

  # creating term freq matrix

  term_freq_matrix = np.zeros((len(term_map.keys()), (len(document_map.keys()))))

  # print('Shape of term freq matrix -> ',term_freq_matrix.shape)

  # Filling term freq matrix

  freq_variant = str(input("Enter term frequency variant number"))

  compute_term_freq_matrix(term_freq_matrix, freq_variant)

  # print(term_freq_matrix)

  # removing duplicates

  for key in dictionary.keys():
    unique_postings = list(dict.fromkeys(dictionary[key][1]))
    dictionary[key][1] = unique_postings

  # inserting document count

  for key in dictionary.keys():
    doc_count = len(dictionary[key][1])
    dictionary[key][0].append(doc_count)
    

  number_of_queries = int(input("Enter number of queries"))

  for query_index in range(number_of_queries):
    query_terms = str(input("Enter the query terms separated by space"))
    optimization = str(input("Do you want to optimize retrieval by first doing index retrieval then ranking: YES or NO?"))

    query_terms = preProcessSentence(query_terms)
    query_terms = ' '.join(query_terms)
    operators = ["OR" for i in range(len(list(set(query_terms.split(" "))))-1)]
    print("The preprocessed version of query is: ",query_terms)

  # pre-processed text
 
  # print('Number of unique terms in dictionary -> ' + str(len(dictionary.keys())))
  # print()
  # print('Document count of term play -> ' + str(dictionary['play'][0][0]))
  # print()
  # print('Posting list of play -> ' + str(dictionary['play'][1]))
  # print()

  # print('Freq of play in document number 15 ->', term_freq_matrix[term_map['play']][15])

    document_scores = np.zeros((len(term_freq_matrix[0])))



    # query_terms = ['play', 'play', 'aluminum', 'tower', 'mansion']
    query_terms = query_terms.split(" ")
    query_vector = get_query_vector(query_terms,freq_variant)
    query_terms = list(set(query_terms))

    output =None
    # computing document scores
    if optimization.lower()=="yes":
            for index,operator in enumerate(operators):
                pos_lists = []
                NOT = False
                
                if "and" in operator.lower():
                    if index ==0:
                        pos_list_1 = inverted_index.get(query_terms[index])
                        if not pos_list_1:
                            print("Term not found: {}\nHence, operation has failed".
                                format(query_terms[index]))
                            skip = True
                            break
                        else:
                            pos_list_1 = pos_list_1[1]
                    else:
                        # to use the output from applying previous operation
                        pos_list_1  = output
                    pos_list_2 = inverted_index.get(query_terms[index+1])
                    # covers AND NOT
                    if "not" in operator.lower():
                        NOT = True
                        pos_list_2 = pos_list_2[1] if pos_list_2 else []
                    else:
                        if not pos_list_2:
                            print("Term not found: {}\nHence, operation has failed".
                                    format(query_terms[index]))
                            skip = True
                            break
                        else:
                            pos_list_2 = pos_list_2[1]
                    pos_list_1.sort()
                    pos_list_2.sort()
                    pos_lists.append(pos_list_1)
                    pos_lists.append(pos_list_2)
                    output,comparisons = AND_operator(pos_lists, NOT)
                    # print("Number of comparisons",comparisons_sum)
                    # print("The merged postings are", output)
                elif "or" in  operator.lower():
                    if index ==0:
                        pos_list_1 = inverted_index.get(query_terms[index])
                        if not pos_list_1:
                            pos_list_1 = []
                        else:
                            pos_list_1 = pos_list_1[1]
                    else:
                        pos_list_1  = output
                    pos_list_2 = inverted_index.get(query_terms[index+1])
                    if not pos_list_2:
                        pos_list_2 = []
                    else:
                        pos_list_2 = pos_list_2[1]
                    # covers or not
                    if "not" in operator.lower():
                        NOT = True
                        not_items = inverted_index.get(query_terms[index+1])
                        if not_items:
                            not_items = not_items[1]
                        else:
                            not_items = []
                        pos_list_2 = [x for x in all_docids if x not in not_items]
                    if not pos_list_2:
                        pos_list_2 = []
                    pos_list_1.sort()
                    pos_list_2.sort()
                    pos_lists.append(pos_list_1)
                    pos_lists.append(pos_list_2)
                    output,comparisons = OR_operator(pos_lists, NOT)
    query_terms = np.unique(query_terms)

    for term in query_terms:

      if (term not in term_map.keys()):
        continue

      no_of_comparisons=0
      value_in_term_map = term_map[term]
      if optimization.upper() == "NO":
          document_scores = get_document_scores(term_freq_matrix,value_in_term_map)
          no_of_comparisons = len(term_freq_matrix[0])
      else:               
          document_scores = get_document_scores_optimized(output,value_in_term_map)

    # print(document_scores)
          no_of_comparisons = len(output)
    doc_number_of_max_doc_score = np.argmax(document_scores)

    top_5_docs = document_scores.argsort()[-5:][::-1]

    document_term_matrix = term_freq_matrix.T
    joblib.dump(document_term_matrix,"document_term_matrix")
    # Finding name of document corresponding to document number
    document_index_map_keys = list(document_map.keys())
    document_index_map_values = list(document_map.values())
    for doc_number_top in top_5_docs:
        index = document_index_map_values.index(doc_number_top)
        print('The most relevant documents for the given query  and their tf-idf scores are -> ', document_index_map_keys[index], document_scores[doc_number_top])
    print("Total number of comparisons were", no_of_comparisons)

    # cosine simialrity based ranking Question 2 part 3

    document_cos_sim_scores = cosine_similarity(query_vector, document_term_matrix)

    top_5_docs_cos_sim = document_cos_sim_scores.argsort()[-5:][::-1]

    for doc_number_top in top_5_docs_cos_sim:
      index = document_index_map_values.index(doc_number_top)
      print('The most relevant documents for the given query  and their cosine similarity scores are -> ', document_index_map_keys[index], document_cos_sim_scores[doc_number_top])
