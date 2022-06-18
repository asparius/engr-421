# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg
import pandas as pd
import scipy.stats as st
import scipy.spatial.distance as dt

images = np.genfromtxt("hw07_data_set_images.csv",delimiter = ",")
labels = np.genfromtxt("hw07_data_set_labels.csv",delimiter = ",")

x_train = images[:2000]
x_test = images[2000:]
y_train = labels[:2000].astype(int)
y_test = labels[2000:].astype(int)
N = x_train.shape[0]
D = x_train.shape[1]
K = np.max(y_train).astype(int)

class_means = np.asarray([np.mean(x_train[y_train == c + 1],axis = 0,keepdims=True) for c in range(K)])

class_mean = np.mean(x_train,axis = 0,keepdims= True)

SW = np.sum(np.asarray([np.matmul((x_train[y_train == c + 1] - class_means[c]).T,x_train[y_train == c + 1] - class_means[c]) for c in range(K)]),axis = 0)

SW[0:4,0:4]

SB = np.sum(np.asarray([np.matmul((class_means[c] - class_mean).T,class_means[c] - class_mean) * len(x_train[y_train == c+1]) for c in range(K)]),axis = 0)

print(SB[0:4,0:4])

W = np.matmul(linalg.cho_solve(linalg.cho_factor(SW), np.eye(D)), SB)

values,vectors = linalg.eig(W)
values = np.real(values)
vectors = np.real(vectors)
print(values[0:9])

# +
figure, (ax1,ax2) = plt.subplots(1, 2,figsize = (16,9))

# calculate two-dimensional projections
Z = np.matmul(x_train - np.mean(x_train, axis = 0), vectors[:,[0, 1]])

# plot two-dimensional projections

point_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6"])
for c in range(K):
    ax1.plot(Z[y_train == c + 1, 0], Z[y_train == c + 1, 1], marker = "o", markersize = 4, linestyle = "none", color = point_colors[c])
ax1.legend(["t-shirt-top", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"],
           loc = "upper left", markerscale = 2)
ax1.set_xlim(-6,6)
ax1.set_ylim(-6,6)
ax1.set_xlabel("Component#1")
ax1.set_ylabel("COmponent#2")


Z = np.matmul(x_test - np.mean(x_test, axis = 0), vectors[:,[0, 1]])

# plot two-dimensional projections

point_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6"])
for c in range(K):
    ax2.plot(Z[y_test == c + 1, 0], Z[y_test == c + 1, 1], marker = "o", markersize = 4, linestyle = "none", color = point_colors[c])
ax2.legend(["t-shirt-top", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"],
           loc = "upper left", markerscale = 2)
ax2.set_ylim(-6,6)
ax2.set_xlim(-6,6)
ax2.set_xlabel("Component#1")
ax2.set_ylabel("COmponent#2")
plt.show()

# +

Z_train = np.matmul(x_train - np.mean(x_train, axis = 0), vectors[:,:9])
Z_test = np.matmul(x_test-np.mean(x_test,axis = 0),vectors[:,:9])

# +
predictions_train = st.mode(y_train[np.argsort(dt.cdist(Z_train,Z_train),axis = 0)][0:11,:],axis = 0)[0][0]
predictions_test = np.concatenate(st.mode(y_train[np.argsort(dt.cdist(Z_test,Z_train))][:,0:11],axis = 1)[0])



# +
confusion_matrix_train = pd.crosstab(predictions_train, y_train,
                                     rownames = ["y_predicted"], colnames = ["y_train"])



confusion_matrix_test = pd.crosstab(predictions_test, y_test,
                                     rownames = ["y_predicted"], colnames = ["y_test"])
# -

print(confusion_matrix_train)

print(confusion_matrix_test)




