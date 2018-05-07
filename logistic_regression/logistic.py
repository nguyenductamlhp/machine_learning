# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from random import randrange


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

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    cross_validation_dataset = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset_copy)/n_folds)
    
    for i in range(0, n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        cross_validation_dataset.append(fold)
    return cross_validation_dataset

def sigmoid(X):
    return 1/(1 + np.exp(- X))
    
def pre_processing(matrix):
    range_ = 10
    b = np.apply_along_axis(lambda x: (x-np.mean(x))/range_, 0, matrix)
    return b

def cost_function(X, y, theta):
    h_theta = sigmoid(np.dot(X, theta))
    log_l = (-y)*np.log(h_theta) + (1 - y)*np.log(1 - h_theta)
    return log_l.mean()

def calculate_gradient(X, y, theta, index, X_count):
    dummy_theta = sigmoid(np.dot(X, theta))
    sum_ = 0.0
    for i in range(dummy_theta.shape[0]):
        sum_ = sum_ + (dummy_theta[i] - y[i]) * X[i][index]
    return sum_


def gradient_descent(training_set, alpha, max_iterations, plot_graph):
    iter_count = 0

    training_set = np.asarray(training_set)
    X = training_set.T[0:2].T
    y = training_set.T[2].T
    X_count = X.shape[1]

    theta = np.zeros(X_count)
    x_vals = []
    y_vals = []
    regularization_parameter = 1
    while(iter_count < max_iterations):
        iter_count += 1
        for i in range(X_count):
            prediction = calculate_gradient(X, y, theta, i, X_count)
            prev_theta = theta[i]
            if i != 0:
                prediction += (regularization_parameter/X_count)*prev_theta
            theta[i] = prev_theta - alpha * prediction
            
            if plot_graph:
                mean = cost_function(X, y, theta)
                x_vals.append(iter_count)
                y_vals.append(mean)
    
    return theta

def compute_efficiency(test_set, theta):
    test_set = np.asarray(test_set)
    X = test_set.T[0:2].T
    y = test_set.T[2].T
    X_count = X.shape[0]
    correct = 0
    
    for i in range(X_count):
        prediction = 0
        value  = np.dot(theta, X[i])
        if value >= 0.5:
            prediction = 1
        else:
            prediction = 0
        if prediction == y[i]:
            correct+=1
    return correct*100/X.shape[0]
    
    
def evaluate_algorithm(dataset, n_folds, alpha, max_iterations, plot_graph):
    folds = cross_validation_split(dataset, n_folds)
    results = []
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
        
        theta = gradient_descent(train_set, alpha, max_iterations, plot_graph)
        results.append(compute_efficiency(test_set, theta))
    return np.asarray(results)


if __name__ == "__main__":
    with open('ex2data2.txt', 'rb') as csvfile:
        data = np.loadtxt(csvfile, delimiter=",")
        print ">>> data", data
        new_data = mapFeature(data[:,0], data[:,1])
        print new_data
    f_len = len(new_data[0])
    X = new_data[:, :f_len - 1]
    print "........", X
    y = new_data[:, f_len -1]
    print ",,,,,,,,,,,y", y
    X = pre_processing(X)
    reshaped_y = y.reshape(y.shape[0], -1)
    processed_dataset = np.concatenate((X, reshaped_y), axis=1)
    results = evaluate_algorithm(processed_dataset.tolist(), n_folds=10,
                                 alpha=0.01, max_iterations=500, plot_graph=False)
    print("Mean : ",np.mean(results))