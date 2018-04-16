# -*- encoding: utf-8 -*-
import numpy as np
from pandas import read_csv
import os
from sklearn import preprocessing
from numpy import set_printoptions

def normalizeFeature(feature):
    '''
    '''
    data = np.loadtxt('ex1data2.txt',skiprows=0, delimiter=',')
    print data
    
normalizeFeature(1)

# def computeCost():
#     return 0
    

# def computeGradient():
#     return 0

# def gradientDescent():
#     return 0

# def predict():
#     return 0