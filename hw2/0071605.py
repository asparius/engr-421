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

#necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def safelog(x):
    return(np.log(x + 1e-100))


images= np.loadtxt("hw02_data_set_images.csv",delimiter = ",") 
labels = np.loadtxt("hw02_data_set_labels.csv").astype(int)

# +
data = np.array(list(zip(images,labels))) # zipped together for further export

# extraction of data training and test sets from data
training_set = np.array([data[labels == c + 1][:25] for c in range(5)])
test_set = np.array([data[labels == c+ 1][25:] for c in range(5)])
training_set = np.concatenate(training_set)
test_set = np.concatenate(test_set)
# -

# preprocessing
x_training = np.concatenate(training_set[:,0].reshape(125,1),)
y_training = np.stack(training_set[:,1])
x_test = np.concatenate(test_set[:,0].reshape(70,1),)
y_test = np.stack(test_set[:,1])


class_priors = np.array([np.mean(y_training==c + 1) for c in range(5)]) # class frequency in training data set
pcd = np.array([np.mean(x_training[y_training==c +1]) for c in range(5)]) # class means for an image

print(pcd)

print(class_priors)


# +
# prediction or naive bayes classifier
# directly returns predicted class labels
def prediction_for_x(x,pcd,class_priors):
    
    score_for_1 = np.sum([x[d]*safelog(pcd[0][d]) + (1-x[d])*safelog(1-pcd[0][d]) for d in range(len(x))]) + np.log(class_priors[0])
    score_for_2 = np.sum([x[d]*safelog(pcd[1][d]) + (1-x[d])*safelog(1-pcd[1][d]) for d in range(len(x))]) + np.log(class_priors[1])
    score_for_3 = np.sum([x[d]*safelog(pcd[2][d]) + (1-x[d])*safelog(1-pcd[2][d]) for d in range(len(x))]) + np.log(class_priors[2])
    score_for_4 = np.sum([x[d]*safelog(pcd[3][d]) + (1-x[d])*safelog(1-pcd[3][d]) for d in range(len(x))]) + np.log(class_priors[3])
    score_for_5 = np.sum([x[d]*safelog(pcd[4][d]) + (1-x[d])*safelog(1-pcd[4][d]) for d in range(len(x))]) + np.log(class_priors[4])
    
    return np.argmax(np.array([score_for_1,score_for_2,score_for_3,score_for_4,score_for_5])) + 1
        
        
        
    
# -

# a classifier for any given data set 
def prediction_for_X(X,pcd,class_priors):
    prediction = []
    for i in X:
        prediction.append(prediction_for_x(i,pcd,class_priors))
    return np.array(prediction)


# prediction for training
predicted = prediction_for_X(x_training,pcd,class_priors)

# +

confusion_matrix = pd.crosstab(predicted,y_training,rownames = ["y_pred"],colnames = ["y_truth"])
print(confusion_matrix)
# -

# prediction for test set
predicted_test = prediction_for_X(x_test,pcd,class_priors)
confusion_matrix_test = pd.crosstab(predicted_test,y_test,rownames = ["y_pred"],colnames = ["y_truth"])
print(confusion_matrix_test)


