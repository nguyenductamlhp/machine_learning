# -*- encoding: utf-8 -*-
import numpy as np
from pandas import read_csv
import os
from sklearn import preprocessing
from numpy import set_printoptions

def normalizeFeature(feature):
    '''
    '''
    duongDan = os.getcwd() + '/ex1data2.txt'
    tenCot = ['area', 'no_room', 'price']
    duLieu = read_csv(duongDan, names=tenCot)
    maTran = duLieu.values
    X = maTran[:,0:4]
    y = maTran[:,4]
    dieuChinh = preprocessing.Normalizer().fit (X)# lớp Normalizer
    X_dieuChinh = dieuChinh.transform(X)
    set_printoptions(precision=3)
    print (X_dieuChinh[:5])
    return 0

normalizeFeature(1)

# def computeCost():
#     return 0
    

# def computeGradient():
#     return 0

# def gradientDescent():
#     return 0

# def predict():
#     return 0