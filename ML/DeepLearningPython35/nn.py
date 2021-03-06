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
	# retrn np.max(0.0,z);
	return np.clip(z,0,np.inf)
def relu_prime(z):
	return (z>0).astype(int)
def drelu(z):
	return (z>0).astype(int)


class Network(object):

	def __init(self,sizes):
		self.num_layers=len(sizes)
		self.sizes=sizes
		self.biases = [np.random.randn(y,1) for y in sizes[1:]]
		self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]

	def feedforward(self,a):
		for b,w in zip(self.biases, self.weights):
			a=sigmoid(np.dot(w,a)+b)
		return a
