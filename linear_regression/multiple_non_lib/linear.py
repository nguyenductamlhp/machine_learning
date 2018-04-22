# -*- encoding: utf-8 -*-
import numpy as np
from pandas import read_csv
import pandas as pd
import os
from sklearn import preprocessing
from numpy import set_printoptions


def normalizeFeature(data_file):
    '''
    Read data from file "data_file" in format numpy array
    Nomalize data and return
    '''
    data = pd.read_csv(data_file)
    data = (data - data.mean()) / data.std()
    return data

def computeCost(X, y, theta):
    cost = np.power(((np.dot(X, theta.T))-y),2)
    return np.sum(cost) / (2 * len(X))

def computeGradient(X, y, theta):
    gradient = np.sum(X * (np.dot(X, theta.T) - y), axis=0)
    return gradient

def gradientDescent(X, y, theta, iters, alpha):
    cost = np.zeros(iters)
    for i in range(iters):
        theta = theta - (alpha/len(X)) * computeGradient(X, y, theta)
        cost[i] = computeCost(X, y, theta)
    return theta, cost

def predict():
    
def main():
    # Read data from file and nomalize
    my_data = normalizeFeature('ex1data2.txt')
    # Get area and number of room
    X = my_data.iloc[:, 0:2]
    ones = np.ones([X.shape[0], 1])
    print ">>>, ones:", ones
    X = np.concatenate((ones, X), axis=1)
    
    # Get price
    y = my_data.iloc[:, 2:3].values #.values converts it from pandas.core.frame.DataFrame to numpy.ndarray
    theta = np.zeros([1, 3])
    # Initial parameters
    alpha = 0.05
    iters = 10000
    # running the gd and cost function
    g, cost = gradientDescent(X, y, theta, iters, alpha)
    print(">>>g", g)
    
    cost = computeCost(X, y, g)
    print(">>> finalCost", cost)
 
if __name__ == "__main__":
    main()
