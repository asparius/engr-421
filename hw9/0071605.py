#!/usr/bin/env python
# coding: utf-8

# In[173]:


import numpy as np
import scipy.spatial as sp
import matplotlib.pyplot as plt
import scipy.linalg as linalg


# In[174]:


x = np.genfromtxt("hw09_data_set.csv",delimiter = ",")


# In[175]:


N = x.shape[0]
threshold = 2


# In[176]:


B = (sp.distance_matrix(x,x) < threshold).astype(int)
for i in range(N):
    B[i][i] = 0


# In[177]:


conn1,conn2 = np.where(B == 1)
for i in range(len(conn1)):
    x_values = np.vstack([x[conn1[i],0],x[conn2[i],0]])
    y_values = np.vstack([x[conn1[i],1],x[conn2[i],1]])
    
    plt.plot(x_values,y_values,color = "gray")
plt.plot(x[:,0],x[:,1],".",color = "black")


# In[178]:


D = np.zeros((B.shape[0],B.shape[1]))
for i in range(N):
    D[i][i] = np.sum(B[i])
    


# In[179]:


D_inverse = linalg.cho_solve(linalg.cho_factor(D),np.eye(N))
Laplacian = np.eye(N) - np.matmul(np.matmul(np.sqrt(D_inverse),B),np.sqrt(D_inverse))


# In[180]:


Laplacian[0:5,0:5]


# In[181]:


R = 5
values,vectors = np.linalg.eig(Laplacian)
values = np.real(values)
vectors = np.real(vectors)

Z = np.vstack([vectors[:,np.argsort(values)[i]] for i in range(1,R + 1)]).T
centr_args =  [242, 528, 570, 590, 648, 667, 774, 891,955] 

centroids = np.vstack([Z[i] for i in centr_args])
K = centroids.shape[0]


# In[196]:


Z[0:5,0:5]


# In[182]:


def e_step(centroids,Z):
    memberships = np.zeros((N,K))
    cent_labels = np.argmin(sp.distance_matrix(Z,centroids),axis = 1)
    for i in range(len(cent_labels)):
        memberships[i][cent_labels[i]] = 1

    return memberships

def m_step(memberships):
    centroids = np.concatenate([np.mean(Z[np.argwhere(memberships==1)[:,1] == k],axis =0).reshape(1,5) for k in range(K)])
    return centroids
    
    


# In[183]:


clusters = np.argwhere(membership == 1)[:,1]
cluster_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#b15928",
                               "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6", "#ffff99"])


# In[195]:


for c in range(K):
            
            plt.plot(x[clusters == c, 0], x[clusters == c, 1], ".", markersize = 10,
                     color = cluster_colors[c])
            plt.plot(np.mean(x[clusters == c,0]),np.mean(x[clusters ==c,1 ]),"s",markersize = 12,markerfacecolor = cluster_colors[c],markeredgecolor = "black")
            


# In[ ]:




