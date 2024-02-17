import argparse
import csv
import math
import numpy as np
import sys
#import matplotlib
#import matplotlib.pyplot as plt


#     assume all attributes have values 0 and 1 only; further
#     assume that the left child corresponds to an attribute value
#     of 1, and the right child to a value of 0
class Node:
    def __init__(self, attribute=None, vote= None, left=None, right=None):
        self.attr = attribute
        self.left = left
        self.right = right
        self.vote = vote


class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth=max_depth

    def learn(self,data):
        self.root= self.maketree(data=data[1:],predictors=data[0])
       
    def maketree(self, data, predictors, depth=0):
        
        target=[row[-1] for row in data]
        if len(set(target)) == 1 or (self.max_depth is not None and depth >= self.max_depth):
            count={var: target.count(var) for var in set(target)}
            majorityelem= []
            for key,val in count.items():
                if val== max(count.values()):
                    majorityelem.append(key)
            #print("depth:", depth, "leafnode of vote: ", max(majorityelem))
            return Node(vote=max(majorityelem))

        best_feature = None 
        bestinformationgain = float("-inf")     

        #debug-start
        for var in predictors[:-1]:
            informationgain=  self.infogain(var, data, predictors.index(var))
            if informationgain > bestinformationgain:
                best_feature = var            
                bestinformationgain = informationgain
            # first column to break tie
            elif informationgain == bestinformationgain:
                col= min(predictors.index(var),predictors.index(best_feature))
                best_feature = predictors[col]            
                bestinformationgain = informationgain
            
        if bestinformationgain <= 0:
            count={var: target.count(var) for var in set(target)}
            majorityelem= []
            for key,val in count.items():
                if val== max(count.values()):
                    majorityelem.append(key)
            #print("depth:", depth, "leafnode of vote: ", max(majorityelem))
            return Node(vote=max(majorityelem))
        

        left_split= [row for row in data if row[predictors.index(best_feature)] == "1"]
        right_split= [row for row in data if row[predictors.index(best_feature)] == "0"]
        
        #debug-end
        
        
        left_child = self.maketree(data=left_split, predictors= predictors, depth=depth + 1)

        right_child = self.maketree(data=right_split, predictors= predictors, depth=depth + 1)
        

        return Node(attribute=best_feature, vote=None, left=left_child, right=right_child)
    
        
    def infogain(self, attr, data, attr_index):
        var_values = set(example[attr_index] for example in data)
        parent_entropy = self.getentropy([row[-1] for row in data])
        child_entropy = 0
        for a1 in var_values:
            subset = [row for row in data if row[attr_index] == a1]
            sub_entropy = self.getentropy([row[-1] for row in subset])
            child_entropy += ((len(subset) / len(data)) * sub_entropy)
        informationgain = parent_entropy - child_entropy
        return informationgain

    def getentropy(self, Y):
        total = len(Y)
        counts = {key: Y.count(key) for key in set(Y)}
        entropy = 0
        for subcount in counts.values():
            entropy += (subcount / total) * math.log((subcount / total),2)
        return -1*entropy
    
    def predict_obs(self, node, example, variables):
        if node.vote is not None: return node.vote
        attr_value = example[variables.index(node.attr)]
        if attr_value == "1": nxt = node.left 
        else: nxt=node.right
        return self.predict_obs(nxt, example,variables)

    def predict(self, datatopridct):
        return [self.predict_obs(self.root, line, datatopridct[0]) for line in datatopridct[1:]]
    

def error(trueY, predictions):
    err=0
    for i in range(len(trueY)):
        if trueY[i]!=predictions[i]: err+=1
    return err/len(trueY)


def pre_order_traversal(node, depth=0,left=False):
    
    if node is not None:
        if (node.left is None and node.right is None): 
            if left==True: print(" | "*depth, "leafnode", "=1", "with majority vote:",node.vote)
            else: print(" | "*depth,"leafnode","=0", "with majority vote:",node.vote)
        else: 
            if left==True: print(" | "*depth, "attr:",node.attr,"=1")  
            else: print(" | "*depth, "attr:",node.attr,"=0")  
        pre_order_traversal(node.left, depth+1,left=True)  
        pre_order_traversal(node.right, depth+1)  

if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the train_input path, you can use `args.train_input`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to training input .tsv file')
    parser.add_argument("test_input", type=str, help='path to the test input .tsv file')
    parser.add_argument("max_depth", type=int, 
                        help='maximum depth to which the tree should be built')
    parser.add_argument("train_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the training data should be written')
    parser.add_argument("test_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the test data should be written')
    parser.add_argument("metrics_out", type=str, 
                        help='path of the output .txt file to which metrics such as train and test error should be written')
    args = parser.parse_args()

    with open(args.train_input) as file2:     
        train_df= list(csv.reader(file2, delimiter="\t"))
    with open(args.test_input) as file2:     
        test_df= list(csv.reader(file2, delimiter="\t"))
    
    """trainingerrors=[]
    testingerrors=[]"""
    mytree = DecisionTree(max_depth=args.max_depth)
    mytree.learn(train_df)
    train_predictions = mytree.predict(train_df)
    test_predictions = mytree.predict(test_df)

    pre_order_traversal(mytree.root)

    train_error = error([row[-1] for row in train_df[1:]], train_predictions)
    test_error = error([row[-1] for row in test_df[1:]], test_predictions)





    """trainingerrors.append(train_error)
    testingerrors.append(test_error)
    """
    #print(train_error,test_error )

    """x= [i for i in range(len(train_df[0]))]

    plt.plot(x, trainingerrors, label='Training Error')
    plt.plot(x, testingerrors, label='Testing Error')
    plt.xlabel('max_depth')
    plt.ylabel('Error Rate')
    plt.title('Error Rate vs. max_depth')
    plt.legend()
    plt.show()"""

    

    predicted_train = open(args.train_out, 'w')

    for i in train_predictions:
        predicted_train.write(i)
        predicted_train.write("\n")

    predicted_train.close()

    predicted_test = open(args.test_out, 'w')

    for i in test_predictions:
        predicted_test.write(i)
        predicted_test.write("\n")
    predicted_test.close()

    metrics = open(args.metrics_out, 'w')

    metrics.write("error(train): ")
    metrics.write(str(train_error))
    metrics.write('\n')

    metrics.write("error(test): ")
    metrics.write(str(test_error))
    metrics.write('\n')

    metrics.close

    



        