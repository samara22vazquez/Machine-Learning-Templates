# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
import os 
import scipy as sp 
import seaborn as sns 
from sklearn import datasets
from sklearn.neighbors.kde import KernelDensity


def f1(x, m, c): # y = m*x + c (ground truth function)
    return m*x + c 
xmin, xmax, npts = [-4, 10, 50]

# defining domain of the function as a vector of 50 real numbers between -4 and 10
X = np.linspace(xmin, xmax, npts) 

# Create data from ground truth function that is corrupted by additive Gaussian noise of mean 0 and std. dev. 4
y0 = f1(X, -3., 9.) + np.random.normal(0,scale=4, size=np.shape(X))  
plt.scatter(X, y0, marker='o', c='k')

def designmat1(Xmat):
    X = np.atleast_2d(Xmat).T
    col1 = np.ones(np.shape(X)[0]).reshape(np.shape(X)[0],1)
    X = np.concatenate((col1, X), axis=1) 
    return X


def gradsqloss(Amat, y, wt):
    n, p = Amat.shape
    return (-2/n)*Amat.T.dot((y-Amat.dot(wt)))

def gradientdescent(Amat, y, winit, rate, numiter):
    n, p = Amat.shape
    whistory = []
    meanrsshistory = [] 
    w = winit
    
    for i in range(numiter): 
        meanrss = np.square(y-Amat.dot(w)).mean()
        whistory.append(w)
        meanrsshistory.append(meanrss)
        grad = gradsqloss(Amat, y, w)
        w = w - rate*grad
    return w, np.asarray(whistory), np.asarray(meanrsshistory)

Xmat = designmat1(X)
n, p = Xmat.shape
w0 = np.random.randn(p)
num_iters = 70
rates = [.001,.005,.01,.02]
xinput = np.linspace(-4,10,100)
fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(12,4))
ax[0].scatter(X,y0,marker='o',color='k')

for i, r in enumerate(rates):
    wfin, whist, meanlosstrace = gradientdescent(Xmat, y0,  w0, r, num_iters)
    ax[1].plot(meanlosstrace,label=r)
    ax[0].plot(xinput,wfin[0]+wfin[1]*xinput, label=r)
ax[1].legend() 
ax[1].set_title("Gradient descent for different rates")
ax[1].set_xlabel("number of iterations")
ax[1].set_ylabel("mean of sum of squares of residual")

plt.show()