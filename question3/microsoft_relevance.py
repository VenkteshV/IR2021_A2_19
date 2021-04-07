# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 17:17:25 2021

@author: Tharun
"""

#from preprocessing import preProcessSentence
import numpy as np
import matplotlib.pyplot as plt
import math

feature_list = []

with open('IR-assignment-2-data.txt', 'rb') as file:
    for line in file.readlines():
        #print(line.decode('utf8').split(" ")[1])
        feat = line.decode('utf8').split(" ")
        if feat[1] == "qid:4":
            feature_list.append(feat)

#print(len(feature_list))


sorted_feature_list = sorted(feature_list, reverse=True)

max_dcg_score = 0
i = 1
for feat in sorted_feature_list:
    if feat[0] == '0':
        break
    max_dcg_score += (pow(2, int(feat[0]))-1)/np.log2(1+i)
    i += 1

print("Part 1: Max DCG Score: ", max_dcg_score)

'''
with open('Max_dcg_one_example.txt', 'wb') as file:
    for feat in sorted_feature_list:
        feat = " ".join(feat)
        feat = feat.encode("utf8")
        file.write(feat)
'''      

count = [0 for _ in range(4)]

for feat in sorted_feature_list:
    count[int(feat[0])] += 1

print("Part 1: Combinations of files with max DCG: ", math.factorial(count[3])*
                                         math.factorial(count[2])*
                                         math.factorial(count[1]))

## NDCG at 50

ndcg_at_50_score = 0
i = 1
for feat in sorted_feature_list:
    if i == 51:
        break
    if feat[0] == '0':
        break
    ndcg_at_50_score += (pow(2, int(feat[0]))-1)/np.log2(1+i)
    i += 1
    
dcg_at_50_score = 0
i = 1
for feat in feature_list:
    if i == 51:
        break
    dcg_at_50_score += (pow(2, int(feat[0]))-1)/np.log2(1+i)
    i += 1
    
print("Normalized DCG at 50: ", dcg_at_50_score/ndcg_at_50_score)

##For whole dataset

ndcg_at_50_score = 0
i = 1
for feat in sorted_feature_list:
    ndcg_at_50_score += (pow(2, int(feat[0]))-1)/np.log2(1+i)
    i += 1
    
dcg_at_50_score = 0
i = 1
for feat in feature_list:
    dcg_at_50_score += (pow(2, int(feat[0]))-1)/np.log2(1+i)
    i += 1
    
print("Normalized DCG (for whole dataset): ", dcg_at_50_score/ndcg_at_50_score)

## Precision vs Recall graphs

sort_75_feature_list = sorted(feature_list, key=lambda x: x[76], reverse=True)

relevant_docs = sum(count[1:])

k = 0
prec = 0
prec_score = []
for feat in sort_75_feature_list:
    if feat[0] != '0':
        prec += 1
    k += 1
    prec_score.append(prec/k)
    
rec = 0
recall_score = []
for feat in sort_75_feature_list:
    if feat[0] != '0':
        rec += 1
    k += 1
    recall_score.append(rec/relevant_docs)
    
#x = [i for i in range(103)]
plt.plot(recall_score, prec_score)
plt.title("Precision vs Recall for documents ranked by tf-idf scores")
plt.xlabel("Recall @ K")
plt.ylabel("Precision @ K")
#plt.savefig("prec_recall@k.jpg")
plt.show()
