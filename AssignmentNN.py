#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 00:39:57 2020

@author: abhirakshit
"""

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import datasets

def sigmoid(z):
    return 1.0/(1 + math.exp(-1.0*z))

def sigmoid_vec(x):
    return np.array([sigmoid(z) for z in x])

def der_sigmoid(z):
    return sigmoid(z)*(1-sigmoid(z))

def der_sigmoid_vec(x):
    return np.array([der_sigmoid(z) for z in x])

def onehot(i, n):
    x = np.zeros(n)
    x[i] = 1
    return x

def cross_entropy(y):
	return -math.log(y, 2) 

def softmax(a):
    n = len(a)
    den = sum([math.exp(a[i]) for i in range(n)])
    return np.array([math.exp(a[i])/den for i in range(n)])


# class hiddenlayer:
#     def _init_(self, n_size, )


class hiddenlayer:
    def __init__(self, X, outdim):
        self.input      = X
        self.weights   = np.random.rand(outdim, self.input.shape[0]) 
        self.bias = np.random.rand(outdim)
        self.output = np.zeros(outdim)
    
    def forward(self, func):
        self.output = func(self.weights.dot(self.input) + self.bias)

class NeuralNet:
    def __init__(self, X, ydim, depth):
        self.input      = X[0]
        self.hiddenlayers = [hiddenlayer(X.shape[1], X.shape[1]) for i in range(depth)]
        self.outputlayer = hiddenlayer(X.shape[1], ydim)
        self.h_params = [X[0] for i in range(depth+2)]
        self.a_params = [sigmoid_vec(X[0]) for i in range(depth+2)]
        self.output = onehot(0, ydim)
        
    def feedforward(self, actfunc, outfunc):
        self.h_params[0] = self.input
        L = len(self.hiddenlayers)+1
        for k in range(1, L):
            ht = np.array([self.h_params[k-1]]).transpose()
            W = self.hiddenlayers[k-1].weights
            b = np.array([self.hiddenlayers[k-1].bias]).transpose()
            self.a_params[k] = b + W.dot(ht) 
            self.h_params[k] = actfunc(self.a_params[k])
            
        ht = np.array([self.h_params[L-1]]).transpose()
        W = self.outputlayer.weights
        b = np.array([self.outputlayer.bias]).transpose()
        self.a_params[L] = b + W.dot(ht)
        self.output = outfunc(self.a_params[L])
def hadamardprod(v1, v2):
    prod = np.array([v1[i]*v2[i] for i in range(len(v1))])
    return np.array([prod])

def forwardpass(X, layers, actfunc, outfunc):
    layers[0].input = X
    for i in range(len(layers)-1):
        layers[i].forward(actfunc)
        layers[i+1].input = layers[i].output
        
    layers[-1].forward(outfunc)

def backprop(layers, y):
    yhat = layers[-1].output
    L = len(layers)-1
    grad_a, grad_b, grad_w, grad_h, h, a = []
    grad_a[L] = np.array([-1*(y - yhat)])
    h[L] = np.array([layers[L].input])
    a[L] = np.array([layers[L].output])
    
    for k in reversed(range(L+1)):
        h[k-1] = np.array([layers[k-1].input])
        grad_w[k] =  grad_a[k]*h[k-1].transpose()
        grad_b[k] = grad_a[k]
        grad_h[k-1] = layers[k].weights.transpose()*grad_a[k]
        grad_a[k-1] = np.multiply(grad_h[k-1], np.array([der_sigmoid_vec(layers[k-1].output)]))
        
        
        
        
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y = iris.target
NN = NeuralNet(X, 3, 2)
NN.feedforward(sigmoid_vec, softmax)
L = len(NN.hiddenlayers)+1
y = np.array([1, 0, 0])
yhat = NN.output
grad_a, grad_w, grad_h, grad_b, W = ([0 for i in range(L+1)], )*5
W[1:L] = [NN.hiddenlayers[k].weights for k in range(len(NN.hiddenlayers))]
W[L] = NN.outputlayer.weights
grad_a[L] = np.matrix([-1*(y - yhat)]).transpose()
k = L
print(np.matrix([NN.h_params[k-1]]).shape)
# grad_w[k] =  grad_a[k] * np.matrix([NN.h_params[k-1]])
# print(grad_w[k].shape)
# # grad_b[k] = grad_a[k]
# # grad_h[k-1] = np.matmul(W[k].transpose(), grad_a[k])
# # grad_a[k-1] = np.multiply(grad_h[k-1], np.array([der_sigmoid_vec(layers[k-1].input)]))
# print(grad_a[k])