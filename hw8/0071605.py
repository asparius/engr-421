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

import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as spa
from scipy.stats import multivariate_normal


X = np.genfromtxt("hw08_data_set.csv",delimiter=",")

means = np.genfromtxt("hw08_initial_centroids.csv",delimiter = ",")


real_means = [[5,5],[-5,5],[-5,-5],[5,-5],[5,0],[0,5],[-5,0],[0,-5],[0,0]]
real_cov = [[[0.8,-0.6],[-0.6,0.8]],[[0.8,0.6],[0.6,0.8]],[[0.8,-0.6],[-0.6,0.8]],[[0.8,0.6],[0.6,0.8]],[[0.2,0.0],[0.0,1.2]],[[1.2,0.0],[0.0,0.2]],[[0.2,0.0],[0.0,1.2]],[[1.2,0.0],[0.0,0.2]],[[1.2,0.0],[0.0,0.2]],[[1.6,0.0],[0.0,1.6]]]

K = means.shape[0]
N = X.shape[0]
X = X.reshape(N,2)

prior_labels = np.argmin(spa.distance_matrix(X,means),axis = 1)

priors = np.asarray([np.sum(prior_labels == c) /N for c in range(K)]).reshape(K,1)


cov = np.asarray([np.matmul((X[prior_labels == c] - means[c]).T,X[prior_labels == c] - means[c]) /len(X[prior_labels == c]) for c in range(K)])


# +
def m_step(X,memberships):
    priors = np.asarray([np.sum(memberships[:,c])/N for c in range(K)])
    means = np.vstack([np.matmul(X.T,memberships[:,c])/np.sum(memberships[:,c]) for c in range(K)])
    cov = np.asarray([np.matmul((X-means[c]).T,(X-means[c]) * (memberships[:,c].reshape(N,1))) /np.sum(memberships[:,c]) for c in range(K)])
    return priors,means,cov
    
    
# -

def e_step(X,means,cov,priors):
    memberships = np.zeros((N,K))
    for c in range(K):
        var = multivariate_normal(means[c],cov[c])
        memberships[:,c] = var.pdf(X) * priors[c]
        
    memberships = memberships / (np.sum(memberships,axis = 1).reshape(N,1))
    return memberships


iteration = 0
while True:

    memberships = e_step(X,means,cov,priors)
    priors,means,cov = m_step(X,memberships)
    if iteration > 100:
        break

    iteration +=1

print(means)

labels = np.argmax(memberships,axis = 1)

means

mean_order = [2,6,1,7,8,5,3,4,0]

# +
cluster_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#b15928",
                               "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6", "#ffff99"])
x1_interval = np.linspace(-8, +8, 2401)
x2_interval = np.linspace(-8, +8, 2401)
x1_grid, x2_grid = np.meshgrid(x1_interval, x2_interval)
X_grid = np.vstack((x1_grid.flatten(), x2_grid.flatten())).T
plt.figure(figsize=(12,8))
plt.xlim(-8,8)
plt.ylim(-8,8)

for i in range(K):
    D = multivariate_normal.pdf(X_grid,means[i,:],cov = cov[i,:,:])
    D = D.reshape((len(x1_interval), len(x2_interval)))
    D1 = multivariate_normal.pdf(X_grid,real_means[mean_order[i]],real_cov[mean_order[i]])
    D1 = D1.reshape((len(x1_interval), len(x2_interval)))
    
    
    
    

    plt.contour(x1_grid, x2_grid, D, levels = [0.05],
            colors = cluster_colors[i], linestyles = "dashed")
    plt.plot(X[labels == i][:,0],X[labels == i][:,1], color = cluster_colors[i],marker = ".",linestyle = "",markersize = 10)
    plt.contour(x1_grid, x2_grid, D1, levels = [0.05],
            colors = cluster_colors[i], linestyles = "solid")
    
# -




