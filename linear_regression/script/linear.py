# -*- encoding: utf-8 -*-
import numpy as np
from pandas import read_csv
import pandas as pd
import os
from sklearn import preprocessing
from numpy import set_printoptions
import matplotlib.pyplot as plt

my_data = pd.read_csv('ex1data2.txt',names=["size","bedroom","price"])

def getData(file_path):
    """
    (string) -> numpy array
    Return numpy array ffrom file
    """
    dataFile = None
    if os.path.exists(file_path):
        dataFile = np.loadtxt(file_path, skiprows=0, delimiter=',')
    else:
        print "File not found!"
    return dataFile

def normalizeFeature(input_data):
    '''
    (numpy array) -> numpy array
    Return numpy array of data which is Normalized
    '''
    output = []
    sums = np.sum(input_data, axis=0)
    maxs = np.max(input_data, axis=0)
    mins = np.min(input_data, axis=0)
    for i in range(len(input_data[0])):
        re = []
        col = input_data[:, i]
        delta = maxs[i] - mins[i]
        avg = sums[i] / len(col)
        for item in col:
            re.append((1.0 * item - avg) / delta)
        output.append(re)
    return np.array(output)


# def predict():
#     return 0

def main():
    my_data = pd.read_csv('ex1data2.txt',names=["size","bedroom","price"])
    my_data = (my_data - my_data.mean())/my_data.std()
    print my_data.head()
    #setting the matrixes
    X = my_data.iloc[:,0:2]
    ones = np.ones([X.shape[0],1])
    X = np.concatenate((ones,X),axis=1)
    
    y = my_data.iloc[:,2:3].values #.values converts it from pandas.core.frame.DataFrame to numpy.ndarray
    theta = np.zeros([1,3])
    
    #set hyper parameters
    alpha = 0.01
    iters = 1000
    #computecost
    def computeCost(X,y,theta):
        tobesummed = np.power(((np.dot(X, theta.T))-y),2)
        return np.sum(tobesummed)/(2 * len(X))
    
    #gradient descent
    def gradientDescent(X,y,theta,iters,alpha):
        cost = np.zeros(iters)
        for i in range(iters):
            theta = theta - (alpha/len(X)) * np.sum(X * (np.dot(X, theta.T) - y), axis=0)
            cost[i] = computeCost(X, y, theta)
        
        return theta,cost
    #running the gd and cost function
    g,cost = gradientDescent(X,y,theta,iters,alpha)
    print(">>>g", g)
    
    finalCost = computeCost(X,y,g)
    print(">>> finalCost", finalCost)
 
if __name__ == "__main__":
    main()
