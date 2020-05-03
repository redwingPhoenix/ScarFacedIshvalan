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

class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1], self.input.shape[1]) 
        self.weights2   = np.random.rand(self.input.shape[1], self.input.shape[1])                 
        self.y          = y
        
        self.output     = np.zeros(self.y.shape)

    def feedforward(self, func, outf):
        self.layer1 = func(np.dot(self.input, self.weights1))
        self.output = outf(np.dot(self.layer1, self.weights2))
        
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

layer1 = hiddenlayer(X[0], 2)
layer1.weights = np.array([[1, 0], [0, 1]])
layer1.bias = -1*X[0]
# layer1.forward(sigmoid_vec)
print(layer1.output)

layer2 = hiddenlayer(layer1.output, 2)
layer2.weights = np.array([[1, 0], [0, 1]])
layer2.bias = np.array([2, 1])
# layer2.forward(softmax)

outlayer = hiddenlayer(layer2.output, 3)
# outlayer.weights = np.array([[1, 0], [0, 1]])
outlayer.bias = np.array([1, 1, 1])

layers = [layer1, layer2, outlayer]
forwardpass(X[0], layers, sigmoid_vec, softmax)
print(outlayer.output)

y = np.array([1, 0, 0])
yhat = layers[-1].output
L = len(layers)-1
grad_a, grad_b, grad_w, grad_h, h, a = ([i for i in np.zeros(L+1)], ) * 6
grad_a[L] = np.array([-1*(y - yhat)])
h[L] = np.array([layers[L].output])
a[L] = np.array([layers[L].input])

for k in reversed(range(L+1)):
    print(k)
    h[k-1] = np.array([layers[k-1].output])
    grad_w[k] =  grad_a[k].dot(h[k-1].transpose())
    grad_b[k] = grad_a[k]
    grad_h[k-1] = layers[k].weights.transpose().dot(grad_a[k])
    grad_a[k-1] = np.multiply(grad_h[k-1], np.array([der_sigmoid_vec(layers[k-1].input)]))
