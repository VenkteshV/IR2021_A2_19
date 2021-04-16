# IR2021_A2_19

#Question1 

For Phrase queries run
  1) python PhraseQueries_withoutStopWords.py
  2) Enter number of queries:   1
  3) Enter the query with terms separated by space:  Telephones jangle and Typewriters


#Question2

For Jaccard similarity run 
  1) cd question2
  2) python jaccard_similarity_ranking.py
  3) Example run:  Enter number of queries 1
  4) Enter the query with terms separated by space universe was a small condensed hot spot
  5) Do you want to optimize retrieval by first doing index retrieval then ranking: YES or NO? YES
For part 2 and 3 of question 2 (tf-idf and cosine similairity run:
python relevant_documents_using_tf_idf_score.py

#Question 3:
Kindly download the dataset from -> https://drive.google.com/file/d/1okxas8RjrsGuKKSpEDRfxEq6aSa6CaDk/view

microsoft_relevance.py -> Kindly run the file once you place the dataset in the same folder. 
1) python microsoft_relevance.py

The following outputs can be observed:
	1) A random file order (text file saved in current folder) with descending relevance scores. Number of files possible in stdout
	2) Max DCG score for 50 and full dataset (qid:4)
	3) Precision-Recall curve as plot (fig saved as jpg file)