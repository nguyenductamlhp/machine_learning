# -*- encoding: utf-8 -*-
import numpy as np
from pandas import read_csv
import os
from sklearn import preprocessing
from numpy import set_printoptions

def normalizeFeature(feature):
    '''
    Normalize data
    '''
    input_data = np.loadtxt('ex1data2.txt',skiprows=0, delimiter=',')
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


print normalizeFeature(1)

# def computeCost():
#     return 0
    

# def computeGradient():
#     return 0

# def gradientDescent():
#     return 0

# def predict():
#     return 0