# -*- encoding: utf-8 -*-
import numpy as np
from pandas import read_csv
import pandas as pd
import os
from numpy import set_printoptions
import json
from pprint import pprint


def normalizeFeature(data_file):
    '''
    Read data from file "data_file" in format numpy array
    Nomalize data and return
    '''
    data = pd.read_csv(data_file)
    data = (data - data.mean()) / data.std()
    return data

def computeCost(X, y, theta):
    cost = np.power(((np.dot(X, theta.T))-y), 2)
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

def predict(size=0, bedroom=0):
    model = json.load(open('model.json'))
    theta_0 = model['Theta'][0]
    theta_1 = model['Theta'][1]
    theta_2 = model['Theta'][2]
    price = theta_0 + theta_1 * size+ theta_2 * bedroom
    return price

def main():

    # Read config
    conf = json.load(open('config.json'))
    alpha = conf['Alpha']
    iters = conf['NumIter']
    theta = np.array([conf['Theta']])
    dataset = conf['Dataset']

    # Read data from file and nomalize
    train_data = normalizeFeature(dataset)

    # Get area and number of room
    X = train_data.iloc[:, 0:2]
    # create vector x_zero as 0 which corressponing with theta_0 and append to vector X
    ones = np.ones([X.shape[0], 1])
    X = np.concatenate((ones, X), axis=1)

    # Get price
    y = train_data.iloc[:, 2:3].values

    # running the gradient decense and compute cost function
    theta_result, cost = gradientDescent(X, y, theta, iters, alpha)
    cost = computeCost(X, y, theta_result)

    # Save model to model.json
    model = {
        "Theta": theta_result[0].tolist(),
        "Cost": cost
    }
    with open('model.json', 'w') as fp:
        json.dump(model, fp)

    # predict for house with area as 1650, 3 room
    price = predict(1656, 3)
    result = {
        "Size": 1650,
        "Bedroom": 3,
        "Price": price,
    }
    with open('price.json', 'w') as fp:
        json.dump(result, fp)

if __name__ == "__main__":
    main()
