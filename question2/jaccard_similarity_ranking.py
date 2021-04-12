import joblib
from preprocessing import preProcessSentence
from intersection_and_union import *
document_token_set_mapping = joblib.load('documents_token_set')
def get_doc_token_set(document_name):
    return document_token_set_mapping[document_name]


# jaccard similarity
def jaccard_similarity(query_set,doc_set):
    intersection_tokens = set(query_set).intersection(set(doc_set))
    union_set = set(query_set).union(set(doc_set))
    # denomination can be also computed as len(query_set)+len(doc_set) - len(intersection_tokens)
    if len(union_set)!=0:
        jaccard_coeff = len(intersection_tokens)/len(union_set)
    else:
        return 0
    return jaccard_coeff


    
if __name__=="__main__":
    print("testing jaccard similarity")
    print("Example Jaccard:",jaccard_similarity(["a"],["a","d","a"]))

    
    inverted_index = joblib.load('inverted_index')
    document_index_map = joblib.load('document_index_map')
    document_index_map_keys = list(document_index_map.keys())
    document_index_map_values = list(document_index_map.values())


    #Makes a set of all document postings
    doc_ids = list(map(lambda x: x[1],inverted_index.values()))
    all_docids = set(functools.reduce(operator.iconcat, doc_ids, []))
    #all_docids = set(document_index_map_values)
    number_of_queries = int(input("Enter number of queries"))
    for query in range(number_of_queries):
        query = str(input("Enter the query with terms separated by space"))
        optimization = str(input("Do you want to optimize retrieval by first doing index retrieval then ranking: YES or NO?"))
        query = preProcessSentence(query)
        query=' '.join(query)
        print("query terms",query)
        query_terms = query.split(" ")
        # operation_sequence = str(input("Enter the operation sequence separated by comma"))
        # operation_sequence = 
        operators = ["OR" for i in range(len(query_terms)-1)]#operation_sequence.split(",")
        output = None
        comparisons_sum = 0
        skip = False
        query_tokens = list(set(query_terms))
        # if len(list(set(operators)))==1:
        #     query_terms.sort(key=lambda x: len(inverted_index.get(x)[1]))
        #     print("Sorted",query_terms)


        if optimization.upper() == "NO":
            no_of_comparisons=0
            relevant_docs = []
            for document_name in document_index_map_keys:
                token_set_from_doc = document_token_set_mapping[document_name]
                jaccard_sim = jaccard_similarity(query_tokens,token_set_from_doc)
                no_of_comparisons+=1
                relevant_docs.append((document_name,jaccard_sim))
            # print("relevant_docs",relevant_docs)
            relevant_docs.sort(key = lambda x: x[1], reverse=True)
            top_docs = relevant_docs[:5]



            print("Number of comparisons",no_of_comparisons)
            print("Number of documents retrieved", len(top_docs))
            print("The top relevant docs are",top_docs)

        # for each operator in the given list we iterate through them to perform the opertions and obtain final result
        else:
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
                    comparisons_sum+=comparisons
                    # print("Number of comparisons",comparisons_sum)
                    # print("The merged postings are", output)
                elif "or" in  operator.lower() and len(query_terms)>1:
                    print("here")
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
                    comparisons_sum+=comparisons
                else:
                    output = inverted_index.get(query_terms[0])[1]
            if len(operators)==0:
                    output = inverted_index.get(query_terms[0])[1]

            # skip scenario is true only if the terms are absent in dictionary
            no_of_comparisons = 0
            relevant_docs = []
            # print("token_set_from_doc",query_tokens,document_token_set_mapping['vgilante.txt'])
            for out in output:
                index = document_index_map_values.index(out)
                document_name = document_index_map_keys[index]
                token_set_from_doc = document_token_set_mapping[document_name]
                jaccard_sim = jaccard_similarity(query_tokens,token_set_from_doc)
                no_of_comparisons+=1
                relevant_docs.append((document_name,jaccard_sim))
            # print("relevant_docs",relevant_docs)
            relevant_docs.sort(key = lambda x: x[1], reverse=True)
            top_docs = relevant_docs[:5]



            print("Number of comparisons",no_of_comparisons)
            print("Number of documents retrieved", len(top_docs))
            print("The top relevant docs are",top_docs)
            # print("The retrieved documents are: [", end="")
            # for out in output:
            #     index = document_index_map_values.index(out)
            #     print(document_index_map_keys[index], end=", ")
            # print(']')