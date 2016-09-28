from __future__ import print_function
from collections import OrderedDict

import os
import sys
import timeit

import scipy.io as sio
import numpy as np
import theano
import theano.tensor as T

import nnet as nn
import criteria	as er
import util

'''Model Definition/Construct'''

class DVNN(object):	
	"""
	The semi-supervised model Domain-Adversial Variational Autoencoder
	To deal with the semi-supervised model that source, target domain data will walk though same path. Use shared layer idea by copy the weight
	The domain label s will constuct inside this class
	For abbreviation: HL refer to hiddenlayer, GSL refer to Gaussian Sample Layer, CSL refer to Cat Sample Layer
	Encoder refer to Encoder NN, Decoder refer to Decoder NN	
	"""

	def __init__(self, rng, input_x, label_y, batch_size,
				 l1_struct,
				 in_dim, out_dim):
		
		######################################
		######################################
		######################################
		#Stochastic Neural Network: x -> z
		self.input_x = input_x
		self.label_y = label_y
		
		self.mu1 = nn.Stacked_NN_0L(
			rng=rng,
			input=input_x,
			struct = l1_struct,
			name='mu1'
		)		 

		self.sigma1 = nn.Stacked_NN_0L(
			rng=rng,
			input=input_x,
			struct = l1_struct,
			name='sigma1'
		)		  
		
		#Sample layer
		self.z1 = nn.GaussianSampleLayer(
			mu=self.mu1.output,
			log_sigma=self.sigma1.output,
			n_in = l1_struct.layer_dim[-1],
			batch_size = batch_size
		)
	   
		z1_dim = l1_struct.layer_dim[-1]
		self.l1_mu = self.mu1.output
		self.l1_log_sigma = self.sigma1.output
		self.l1_sigma = T.exp(self.l1_log_sigma)
		self.l1_z = self.z1.output
		
		self.l1_params = self.mu1.params + self.sigma1.params
		self.l1_outputs = [self.l1_mu, self.l1_log_sigma, self.l1_z]
		self.l1_outputs_name = ["l1_mu", "l1_log_sigma", "l1_z"]
		######################################
		######################################
		######################################
		#Prediction
		self.y = nn.SoftmaxLayer(
			rng=rng,
			input=self.l1_z,
			n_in=l1_struct.layer_dim[-1],
			n_out=out_dim,
			name='Predict'
		)		 
		
		
		PR_y_dim = out_dim
		self.PR_y = self.y.output
		self.y_params = self.y.params
		self.y_outputs = [self.PR_y]
		self.y_outputs_name = ["PR_y"]
		
		# This is test out put, get the test accuracy
		self.predictor = T.argmax(self.PR_y, axis=1, keepdims=False)
		
		# Cross entropy
		self.CE = T.nnet.categorical_crossentropy(self.PR_y, label_y);
				
		#Cost function
		self.cost = self.CE.mean()
						
		# the parameters of the model
		self.params = self.l1_params + self.y_params
		
		# all output of VAE
		self.outputs = self.l1_outputs + self.y_outputs
		self.outputs_name = self.l1_outputs_name + self.y_outputs_name


