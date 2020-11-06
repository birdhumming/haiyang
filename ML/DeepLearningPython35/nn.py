# -*- coding: utf-8 -*-

import random
import numpy as np

def sigmoid(z):
	return 1.0/(1.0+np.exp(-z))

def relu(z):
	# retrn np.max(0,z);
	return np.clip(z,0,np.inf)

