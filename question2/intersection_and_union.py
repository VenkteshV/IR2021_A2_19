import joblib
import functools
import operator
# from preprocessing import preProcessSentence

# handles AND and AND NOT operations
def intersection_op(pos_list_1,pos_list2, NOT):
    output = list()
    top_1 =0
    top_2 = 0
    comparisons = 0
    #AND NOT scenario retrieve all postings in first and not in second
    if NOT:
        while top_1<len(pos_list_1) and top_2 < len(pos_list2):
            comparisons+=1
            if pos_list_1[top_1] <pos_list2[top_2]:
                output.append(pos_list_1[top_1])
                top_1+=1
            elif pos_list_1[top_1] >pos_list2[top_2]:
                top_2+=1
            else:
                top_1+=1
                top_2+=1
        while top_1 < len(pos_list_1):
            output.append(pos_list_1[top_1])
            top_1+=1
    else:
        while top_1<len(pos_list_1) and top_2 < len(pos_list2):
            comparisons+=1
            if pos_list_1[top_1]==pos_list2[top_2]:
                output.append(pos_list_1[top_1])
                top_1+=1
                top_2+=1
            elif pos_list_1[top_1] <pos_list2[top_2]:
                top_1+=1
            else:
                top_2+=1
    return output,comparisons

# invokes the intersection method
def AND_operator(postings, NOT):
    comparison_accumulator =0
    initial_output = postings[0]
    for posting in postings[1:]:
        output, comparisons = intersection_op(initial_output,posting, NOT)
        comparison_accumulator+=comparisons
    return output, comparison_accumulator

# performs the OR operation also OR NOT is covered as in main method we first perform the NOT operation on the second operand
def Union_op(pos_list_1,pos_list_2, NOT):
    output = list()
    top_1 =0
    top_2 = 0
    comparisons = 0

    while top_1<len(pos_list_1) and top_2 < len(pos_list_2):
        comparisons+=1
        if pos_list_1[top_1]==pos_list_2[top_2]:
            output.append(pos_list_1[top_1])
            top_1+=1
            top_2+=1
        elif top_1<len(pos_list_1) and pos_list_1[top_1] <pos_list_2[top_2]:
            output.append(pos_list_1[top_1])
            top_1+=1
        else:
            output.append(pos_list_2[top_2])
            top_2+=1


    while top_1 < len(pos_list_1):
        output.append(pos_list_1[top_1])
        top_1+=1
    while top_2 < len(pos_list_2):
        output.append(pos_list_2[top_2])
        top_2+=1
    return output,comparisons    

# performs the OR operation
def OR_operator(postings, NOT):
    comparison_accumulator =0
    initial_output = postings[0]
    for posting in postings[1:]:
        output, comparisons = Union_op(initial_output,posting, NOT)
        comparison_accumulator+=comparisons
    return output, comparison_accumulator