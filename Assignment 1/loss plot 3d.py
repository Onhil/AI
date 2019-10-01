#%%
import numpy as np
import matplotlib.pyplot as plt


data = np.loadtxt(open("ex1data2.txt", "r"), delimiter=",")
X = data[:, 0:2] #Reading two features from the input file i.e. the area in squre feet and the number of bedrooms
y = data[:, 2]   #labels in the form of price of houses
m = len(y)
#%%
def feature_normalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0, ddof=1)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma


#%%
X, mu, sigma = feature_normalize(X)
X = np.hstack((np.ones((m, 1)), X))


#%%
alpha = 0.01
num_iters = 50

beta = np.zeros(3) #(B_0, B_1, B_2)


#%%
def compute_cost_multi(X, y, beta):
    m = len(y)
    diff = X.dot(beta) - y
    J = 1.0 / (2 * m) * diff.T.dot(diff)
    return J


#%%
def gradient_descent_multi(X, y, beta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros(num_iters)

    for i in range(num_iters):
        beta -= alpha / m * ((X.dot(beta) - y).T.dot(X))
        J_history[i] = compute_cost_multi(X, y, beta)

    return beta, J_history


#%%
beta, J_history = gradient_descent_multi(X, y, beta, alpha, num_iters)
#%%
#Training is complete. The model can be tested with test data. 
def f(a, b):
       return a**2 + b**2
from matplotlib import cm
fig = plt.figure()
ax = fig.gca(projection='3d')
a, b = np.meshgrid(J_history, J_history)
c = f(a, b)
surf = ax.plot_surface(a,b,c, cmap=cm.Spectral)
plt.show()


#%%



