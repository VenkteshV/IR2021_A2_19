import pickle
import numpy as np
import math

# pre-processed text
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
  

# print('Number of unique terms in dictionary -> ' + str(len(dictionary.keys())))
# print()
# print('Document count of term play -> ' + str(dictionary['play'][0][0]))
# print()
# print('Posting list of play -> ' + str(dictionary['play'][1]))
# print()

# print('Freq of play in document number 15 ->', term_freq_matrix[term_map['play']][15])

document_scores = np.zeros((len(term_freq_matrix[0])))

query_terms = ['play', 'play', 'aluminum', 'tower', 'mansion']

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


# Finding name of document corresponding to document number

for key, doc_number in document_map.items():
  if doc_number_of_max_doc_score == doc_number:
    print('The most relevant document for the given query is -> ', key)
    break
