import pickle
import numpy as np
import math
import joblib
from collections import Counter
from question2.preprocessing import preProcessSentence


def get_query_vector(query_terms, norm_type="log_norm"):
  query_vec = np.zeros((1,len(term_map.keys())))
  query_term_frequencies = Counter(query_terms)
  # print("query_term_frequencies",query_term_frequencies)

  for term in np.unique(query_terms):
    idf = math.log10( len(term_freq_matrix[0]) / (1 + dictionary[term][0][0]))
    value_mapped_to_key = term_map[term]
    if norm_type=="log_norm":
      term_freq = math.log10(1 + query_term_frequencies[term])
    query_vec[0][value_mapped_to_key] = term_freq * idf

  print("query_vec",query_vec,query_vec.shape,np.where((query_vec!=0)))
  return query_vec

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

  with open('DocTerms.pickle', 'rb') as handle:
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

  for key in dictionary.keys():

    value_mapped_to_key = term_map[key]

    posting_list = dictionary[key][1]

    for document_number in posting_list:

      prev_val = term_freq_matrix[value_mapped_to_key][document_number]

      term_freq_matrix[value_mapped_to_key][document_number] = 1 + prev_val


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
    query_terms = preProcessSentence(query_terms)
    query_terms = ' '.join(query_terms)
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
    query_vector = get_query_vector(query_terms)
    query_terms = np.unique(query_terms)

    # computing document scores

    for term in query_terms:

      if (term not in term_map.keys()):
        continue

      value_in_term_map = term_map[term]

      for doc_num in range(0, len(term_freq_matrix[0])):

        idf = math.log10( len(term_freq_matrix[0]) / (1 + dictionary[term][0][0]))

        tf = math.log10(1 + term_freq_matrix[value_in_term_map][doc_num])

        document_scores[doc_num] += (tf * idf)


    # print(document_scores)

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


    # cosine simialrity based ranking Question 2 part 3

    document_cos_sim_scores = cosine_similarity(query_vector, document_term_matrix)

    top_5_docs_cos_sim = document_cos_sim_scores.argsort()[-5:][::-1]

    for doc_number_top in top_5_docs_cos_sim:
      index = document_index_map_values.index(doc_number_top)
      print('The most relevant documents for the given query  and their cosine similarity scores are -> ', document_index_map_keys[index], document_cos_sim_scores[doc_number_top])