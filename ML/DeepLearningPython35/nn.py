# -*- coding: utf-8 -*-

import random
import numpy as np

def sigmoid(z):
	return 1.0/(1.0+np.exp(-z))

def dsigmoid(z):
	return sigmoid(z)*(1-sigmoid(z))
def sigmoid_prime(z):
	return sigmoid(z)*(1-sigmoid(z))

def relu(z):
	# retrn np.max(0,z);
	return np.clip(z,0,np.inf)
def relu_prime(z):
	return (z>0).astype(int)
def drelu(z):
	return (z>0).astype(int)
