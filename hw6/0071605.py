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

import cvxopt as cvx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.spatial.distance as dt

X = np.genfromtxt("hw06_data_set_images.csv", delimiter = ",")
Y = np.genfromtxt("hw06_data_set_labels.csv",delimiter= ",")
x_train = X[:1000]
x_test = X[1000:]
y_train = Y[:1000]
y_test = Y[1000:]

y_test.shape

N_train = x_train.shape[0]
N = x_train.shape[1]
bin_width = 4
left_borders = np.arange(0, 256, bin_width)
right_borders = np.arange(0 + bin_width, 256 + bin_width, bin_width)
H_train = np.asarray([[np.sum((left_borders[b] <= x) & (x < right_borders[b])) / N for b in range(len(left_borders))] for x in x_train])
H_test = np.asarray([[np.sum((left_borders[b] <= x) & (x < right_borders[b])) / N for b in range(len(left_borders))] for x in x_test])

print(H_train[0:5,0:5])
print(H_test[0:5,0:5])


def hist_kernel(h1,h2):
    kernel = np.zeros((h1.shape[0],h1.shape[0]))
    for i in range(len(h1)):
        for j in range((len(h2))):
            kernel[i][j] = np.sum(np.minimum(h1[i],h2[j]))
    return kernel 


K_train = hist_kernel(H_train,H_train)
K_test = hist_kernel(H_test,H_train)

print(K_train[0:5,0:5])
print(K_test[0:5,0:5])


def svm(C,K_train,K_test,y_train,y_test):
    
   
    yyK = np.matmul(y_train[:,None], y_train[None,:]) * K_train

    # set learning parameters
    
    epsilon = 0.001

    P = cvx.matrix(yyK)
    q = cvx.matrix(-np.ones((N_train, 1)))
    G = cvx.matrix(np.vstack((-np.eye(N_train), np.eye(N_train))))
    h = cvx.matrix(np.vstack((np.zeros((N_train, 1)), C * np.ones((N_train, 1)))))
    A = cvx.matrix(1.0 * y_train[None,:])
    b = cvx.matrix(0.0)
                    
    # use cvxopt library to solve QP problems
    result = cvx.solvers.qp(P, q, G, h, A, b)
    alpha = np.reshape(result["x"], N_train)
    alpha[alpha < C * epsilon] = 0
    alpha[alpha > C * (1 - epsilon)] = C

    # find bias parameter
    support_indices, = np.where(alpha != 0)
    active_indices, = np.where(np.logical_and(alpha != 0, alpha < C))
    w0 = np.mean(y_train[active_indices] * (1 - np.matmul(yyK[np.ix_(active_indices, support_indices)], alpha[support_indices])))
    f_predicted_train = np.matmul(K_train, y_train[:,None] * alpha[:,None]) + w0
    y_predicted_train = 2 * (f_predicted_train > 0.0) - 1
    f_predicted_test = np.matmul(K_test, y_train[:,None] * alpha[:,None]) + w0
    y_predicted_test = 2 * (f_predicted_test > 0.0) - 1
    y_pred_train = np.concatenate(y_predicted_train)
    y_pred_test = np.concatenate(y_predicted_test)
    acc_train = (y_train == y_pred_train).sum()/len(y_train)
    acc_test = (y_test == y_pred_test).sum() / len(y_test)
    
    return y_predicted_train,y_predicted_test,acc_train,acc_test

y_predicted_train,y_predicted_test,_,_ = svm(10,K_train,K_test,y_train,y_test)


# +
confusion_matrix_train = pd.crosstab(np.reshape(y_predicted_train, N_train), y_train,
                                     rownames = ["y_predicted"], colnames = ["y_train"])

confusion_matrix_test = pd.crosstab(np.reshape(y_predicted_test, N_train), y_test,
                                     rownames = ["y_predicted"], colnames = ["y_test"])
print(confusion_matrix_train)
print(confusion_matrix_test)

# +
train_error = {}
test_error = {}
C = np.arange(-1,3.5,0.5)

for i in C:
    _,_,train,test = svm(10 ** i,K_train,K_test,y_train,y_test)

    train_error[i] = train
    test_error[i] = test

# +

plt.figure(figsize = (10,6))
plt.plot(train_error.keys(),train_error.values(),".b-",label = "training")
plt.plot(test_error.keys(),test_error.values(),".r-",label = "test")
plt.xlabel("Regularization Parameter(log(C))")
plt.ylabel("Accuracy")
plt.legend(frameon= True)
plt.show()

# -




