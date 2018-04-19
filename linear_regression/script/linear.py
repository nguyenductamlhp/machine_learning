# -*- encoding: utf-8 -*-
import numpy as np
from pandas import read_csv
import pandas as pd
import os
from sklearn import preprocessing
from numpy import set_printoptions
import matplotlib.pyplot as plt

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

def computeCost(theta_0, theta_1, theta_2, arr_x, arr_y):
    sum = 0
    for i in arr_x:
        temp = (theta_0 + theta_1 * arr_x[i][0] + theta_2 * arr_x[i][1] - arr_y[i]) ** 2
        sum = sum + temp
    return sum * 1.0 / (2 * len(arr_x))

def computeGradient():
    return 0



# areas = [1000, 2000, 4000]
# prices = [200000, 250000, 300000]
# theta_0 = 0
# testno = 200
# costs = [computeSingleCost(theta_0, theta_1, areas, prices) for theta_1 in np.arange(testno)]
# print ">>> costs", costs
# plt.plot(np.arange(testno), costs)



# def gradientDescent():
#     return 0

# def predict():
#     return 0

def main():
    print normalizeFeature(getData('ex1data2.txt'))
 
if __name__ == "__main__":
    main()
