# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
def safelog2(x):
    if x == 0:
        return(0)
    else:
        return(np.log2(x))


training = np.loadtxt("hw05_data_set_train.csv",delimiter = ",")
test = np.loadtxt("hw05_data_set_test.csv",delimiter = ",")
x_train = training[:,0]
y_train = training[:,1]
x_test = test[:,0]
y_test = test[:,1]
N_train = len(y_train)
N_test = len(y_test)


def regression_tree(P):
    node_indices = {}
    is_terminal = {}
    need_split = {}

    node_features = {}
    node_splits = {}
    node_averages = {}
    # put all training instances into the root node
    node_indices[1] = np.array(range(N_train))
    is_terminal[1] = False
    need_split[1] = True
    # learning algorithm
    while True:
    # find nodes that need splitting
        split_nodes = [key for key, value in need_split.items()
                   if value == True]
    # check whether we reach all terminal nodes
        if len(split_nodes) == 0:
            break
    # find best split positions for all nodes
        for split_node in split_nodes:
            data_indices = node_indices[split_node]
            need_split[split_node] = False
            if len(data_indices) <= P:
                is_terminal[split_node] = True
                node_averages[split_node] = np.mean(y_train[data_indices])
            
            else:
                is_terminal[split_node] = False

                best_score = 0.0
                best_split = 0.0
            
                unique_values = np.sort(np.unique(x_train[data_indices]))
                split_positions = (unique_values[1:len(unique_values)] + \
                                   unique_values[0:(len(unique_values) - 1)]) / 2
                split_scores = np.repeat(0.0, len(split_positions))
                for s in range(len(split_positions)):
                    left_indices = data_indices[x_train[data_indices] > \
                                   split_positions[s]]
                    right_indices = data_indices[x_train[data_indices] <= \
                                    split_positions[s]]
                
                
                
                    split_scores[s] = (np.sum(np.square(y_train[left_indices] - np.mean(y_train[left_indices]))) + \
                                   np.sum(np.square(y_train[right_indices] - np.mean(y_train[right_indices]))))/ len(data_indices)
                
                
                
                
                
                best_score = np.min(split_scores)
                best_split = split_positions[np.argmin(split_scores)]
            
                node_features[split_node] = best_score
                node_splits[split_node] = best_split
            
                # create left node using the selected split
                left_indices = data_indices[x_train[data_indices] > \
                           best_split]
                node_indices[2 * split_node] = left_indices
                is_terminal[2 * split_node] = False
                need_split[2 * split_node] = True
      
                # create right node using the selected split
                right_indices = data_indices[x_train[data_indices] <= \
                            best_split]
                node_indices[2 * split_node + 1] = right_indices
                is_terminal[2 * split_node + 1] = False
                need_split[2 * split_node + 1] = True
                
                
    y_predicted_train = np.repeat(0.0, N_train)
    for i in range(N_train):
        index = 1
        while True:
            if is_terminal[index] == True:
                y_predicted_train[i] = node_averages[index]
                break
            else:
                if x_train[i] > node_splits[index]:
                    index = index * 2
                else:
                    index = index * 2 + 1
                    
    y_predicted_test = np.repeat(0.0, N_test)
    for i in range(N_test):
        index = 1
        while True:
            if is_terminal[index] == True:
                y_predicted_test[i] = node_averages[index]
                break
            else:
                if x_test[i] > node_splits[index]:
                    index = index * 2
                else:
                    index = index * 2 + 1
    train_error = np.sqrt(np.sum(np.square(y_train - y_predicted_train))/len(y_train))
    test_error = np.sqrt(np.sum(np.square(y_test - y_predicted_test))/len(y_test))
    
    return node_indices,is_terminal,node_averages,node_splits,train_error,test_error


P = 30
node_indices,is_terminal,node_averages,node_splits,train_error,test_error = regression_tree(P)


# +
plt.figure(figsize = (10,6))

train_dic = {}
for key,value in node_indices.items():
    for v in value:
        if key in node_averages:
            if x_train[v] not in train_dic:
                train_dic[x_train[v]] = node_averages[key]

train_dic = {k: v for k, v in sorted(train_dic.items(), key=lambda item: item[0])}

plt.plot(x_train,y_train,"b.",markersize = 10,label= "training")
for b in range(1,len(train_dic)):
    plt.plot([list(train_dic.keys())[b-1], list(train_dic.keys())[b]], [list(train_dic.values())[b], list(train_dic.values())[b]] , "k-")
for b in range(len(train_dic) -1):
    plt.plot([list(train_dic.keys())[b], list(train_dic.keys())[b]], [list(train_dic.values())[b], list(train_dic.values())[b + 1]] , "k-")
plt.legend(frameon = True)
plt.show()
# -

plt.figure(figsize = (10,6))
plt.plot(x_test,y_test,"r.",markersize = 10,label= "test")
for b in range(1,len(train_dic)):
    plt.plot([list(train_dic.keys())[b-1], list(train_dic.keys())[b]], [list(train_dic.values())[b], list(train_dic.values())[b]] , "k-")
for b in range(len(train_dic) -1):
    plt.plot([list(train_dic.keys())[b], list(train_dic.keys())[b]], [list(train_dic.values())[b], list(train_dic.values())[b + 1]] , "k-")
plt.legend(frameon = True)
plt.show()

print("RMSE on training set is " + str(train_error) +" when P is " + str(P))
print("RMSE on training set is " + str(test_error) +" when P is " + str(P))

train_error = {}
test_error = {}

for i in range(10,51,5):
    _,_,_,_,train,test = regression_tree(i)

    train_error[i] = train
    test_error[i] = test


plt.figure(figsize = (12,8))
plt.plot(train_error.keys(),train_error.values(),".b-",label = "training")
plt.plot(test_error.keys(),test_error.values(),".r-",label = "test")
plt.legend(frameon= True)
plt.show()


