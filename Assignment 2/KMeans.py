#%%
from copy import deepcopy
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
from sklearn import datasets
from scipy.spatial.distance import cdist
from sklearn.datasets.samples_generator import make_blobs
import time
#%%
iris = datasets.load_iris()

## change wich data is being run on
X, y  = make_blobs(n_samples=600, centers=5, random_state=0)
#X = iris.data[:,:2]
#%%

    

# Euclidean Distance Caculator
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

#%%

def KMeans(k):
    # Number of clusters
    
    # X coordinates of random centroids
    mean = np.mean(X, axis = 0)
    std = np.std(X, axis = 0)
    C = np.random.randn(k,2)*std + mean

    # To store the value of centroids when it updates
    C_old = np.zeros(C.shape)
    # Cluster Lables(0, 1, 2)
    clusters = np.zeros(len(X))
    # Error func. - Distance between new centroids and old centroids
    error = dist(C, C_old, None)
    # Loop will run till the error becomes zero

    #%%
    while error != 0:
        # Assigning each value to its closest cluster
        for i in range(len(X)):
            distances = dist(X[i], C)
            cluster = np.argmin(distances)
            clusters[i] = cluster
        # Storing the old centroid values
        C_old = deepcopy(C)
        # Finding the new centroids by taking the average value
        for i in range(k):
            points = [X[j] for j in range(len(X)) if clusters[j] == i]
            C[i] = np.mean(points, axis=0)

        
        error = dist(C, C_old, None)
    plotKMeans(C, k, clusters)
    return C

#%%
def plotKMeans(C, k, clusters):
    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    fig, ax = plt.subplots()
    for i in range(k):
            points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
            ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i%len(colors)])
    ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='#FFFFFF')
   
#%%

Ks = range(2, 7)
distortions = []
for i in Ks:
    km = KMeans(i)
    # Finds the distoritions of obtimal KMeans centers and append them to a list
    distortions.append(sum(np.min(cdist(X,km, 'euclidean'), axis=1)) / X.shape[0])
plt.show()
plt.plot(Ks, distortions, 'bx-')
plt.xlabel('Ks')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()
#%%