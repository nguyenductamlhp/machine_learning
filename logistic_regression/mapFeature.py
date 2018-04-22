import numpy as np 
import csv

def mapFeature(X1, X2):
# MAPFEATURE Feature mapping function to polynomial features
#
#   MAPFEATURE(X1, X2) maps the two input features
#   to quadratic features used in the regularization exercise.
#
#   Returns a new feature array with more features, comprising of 
#   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
#
#   Inputs X1, X2 must be the same size
    degree = 6
    out = np.ones([len(X1), (degree+1)*(degree+2)/2])
    idx = 1
    for i in range(1, degree+1):
        for j in range(0, i+1):
            a1 = X1 ** (i-j)
            a2 = X2 ** j
            out[:, idx] = a1*a2
            idx += 1
    return out

if __name__ == "__main__":
    with open('ex2data2.txt', 'rb') as csvfile:
        data = np.loadtxt(csvfile, delimiter=",")
        new_data = mapFeature(data[:,0], data[:,1])