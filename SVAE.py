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

class Supervised_VAE(object):	
	"""
	The semi-supervised model Domain-Adversial Variational Autoencoder
	To deal with the semi-supervised model that source, target domain data will walk though same path. Use shared layer idea by copy the weight
	The domain label s will constuct inside this class
	For abbreviation: HL refer to hiddenlayer, GSL refer to Gaussian Sample Layer, CSL refer to Cat Sample Layer
	Encoder refer to Encoder NN, Decoder refer to Decoder NN	
	"""

	def __init__(self, rng, input_x, label_y, batch_size,
				 phi_1_struct, theta_1_struct, theta_pi_struct):
		"""Initialize the parameters for the multilayer perceptron

		:type rng: numpy.random.RandomState
		:param rng: a random number generator used to initialize weights

		:type input_source: theano.tensor.TensorType
		:param input: symbolic variable that describes the "Source Domain" input of the architecture (one minibatch)
		
		:type input_target: theano.tensor.TensorType
		:param input: symbolic variable that describes the "Target Domain" input of the architecture (one minibatch)		

		:type xxx_struct: class NN_struct
		:param xxx_strucat: define the structure of each NN
		"""
		#------------------------------------------------------------------------
		#Encoder 1 Neural Network: present q_\phi(z_n | x_n, y_n)
		phi_xy_in = T.concatenate([input_x, label_y], axis=1)
		
		'''
		self.phi_1 = nn.Stacked_NN_0L(
			rng=rng,
			input=phi_xy_in,
			struct = phi_1_struct,
			name='Encoder1'
		)	 
		'''
		
		self.phi_mu = nn.Stacked_NN_0L(
			rng=rng,
			input=phi_xy_in,
			struct = phi_1_struct,
			name='Encoder1_mu'
		)		 

		self.phi_sigma = nn.Stacked_NN_0L(
			rng=rng,
			input=phi_xy_in,
			struct = phi_1_struct,
			name='Encoder1_sigma'
		)		  
		
		#Sample layer
		self.phi_z = nn.GaussianSampleLayer(
			mu=self.phi_mu,
			log_sigma=self.phi_sigma,
			n_in = phi_1_struct.layer_dim[-1],
			batch_size = batch_size
		)
	   
		EC_z_dim = phi_1_struct.layer_dim[-1]
		self.EC_mu = self.phi_mu.output
		self.EC_log_sigma = self.phi_sigma.output
		self.EC_sigma = T.exp(self.EC_log_sigma)
		self.EC_z = self.phi_z.output
		
		self.phi_1_params = self.phi_mu.params + self.phi_sigma.params
		self.phi_1_outputs = [self.EC_mu, self.EC_log_sigma, self.EC_z]
		self.phi_1_outputs_name = ["EC_mu", "EC_log_sigma", "EC_z"]		   
		
		#------------------------------------------------------------------------
		#Decoder 1 Neural Network: present p_\theta(z_n | x_n)
		'''
		theta_1 = nn.NNLayer(
			rng=rng,			
			input_source=input_x,
			struct = theta_1_struct,
			name='Decoder1'
		) 
		'''
		self.theta_mu = nn.Stacked_NN_0L(
			rng=rng,			
			input=input_x,
			struct = theta_1_struct,
			name='Decoder1_mu'
		)		 

		self.theta_sigma = nn.Stacked_NN_0L(
			rng=rng,			
			input=input_x,
			struct = theta_1_struct,
			name='Decoder1_sigma'
		)  
		
		self.DC_mu = self.theta_mu.output
		self.DC_log_sigma = self.theta_sigma.output
		self.DC_sigma = T.exp(self.DC_log_sigma)
		
		self.theta_1_params = self.theta_mu.params + self.theta_sigma.params
		self.theta_1_outputs = [self.DC_mu, self.DC_log_sigma]
		self.theta_1_outputs_name = ["DC_mu", "DC_log_sigma"]		 
		
		#------------------------------------------------------------------------
		#Predict 1 Neural Network: E_q_\phi(z_n | x_n, y_n) (p_\theta(y_n | x_n, z_n))
		self.theta_xz_in = T.concatenate([input_x, phi_z], axis=1)		
		self.theta_2 = nn.Stacked_NN_0L(
			rng=rng,			
			input=self.theta_xz_in,
			struct = theta_2_struct,
			name='Predict1'
		) 
		
		self.theta_pi = nn.Stacked_NN_0L(
			rng=rng,			
			input=self.theta_2.output,
			struct = theta_pi_struct,
			name='Predict1_pi'
		)		 
		
		self.y_hat = nn.CatSampleLayer(
			pi=self.theta_pi.output,
			n_in = theta_pi_struct.layer_dim[-1],
			batch_size = batch_size 
		)
		
		PR_y_hat_dim = theta_pi_struct.layer_dim[-1]
		self.PR_theta_2 = self.theta_2.output
		self.PR_pi = self.theta_pi.output
		self.PR_y_hat = self.y_hat.output
		
		self.theta_2_params = self.theta_2.params + self.theta_pi.params + self.y_hat.params
		self.theta_2_outputs = [self.PR_theta_2, self.PR_pi, self.PR_y_hat]
		self.theta_2_outputs_name = ["PR_theta_2", "PR_pi", "PR_y_hat"]	 
							
		#------------------------------------------------------------------------
		# Error Function Set				
		# KL(q(z|x,y)||p(z|x)) -----------
		self.KL = er.KLGaussianGaussian(self.EC_mu, self.EC_sigma, self.DC_mu, self.DC_sigma)
			 
		#threshold = 0.0000001				  
		# Cross entropy
		self.CE = T.nnet.categorical_crossentropy(self.PR_y_hat, label_y);
				
		#Cost function
		self.cost = self.KL - self.CE
						
		# the parameters of the model
		self.params = self.phi_1_params + self.theta_1_params + self.theta_2_params
		
		# all output of VAE
		self.outputs = self.phi_1_outputs + self.theta_1_outputs + self.theta_2_outputs
		self.outputs_name = self.phi_1_outputs_name + self.theta_1_outputs_name + self.theta_2_outputs_name

########################This version witout categorical sample###############################
class Supervised_VAE_v1(object):
	def __init__(self, rng, input_x, label_y, batch_size,
				 phi_1_struct, theta_1_struct, theta_2_struct, in_dim, out_dim):
		"""Initialize the parameters for the multilayer perceptron

		:type rng: numpy.random.RandomState
		:param rng: a random number generator used to initialize weights

		:type input_source: theano.tensor.TensorType
		:param input: symbolic variable that describes the "Source Domain" input of the architecture (one minibatch)
		
		:type input_target: theano.tensor.TensorType
		:param input: symbolic variable that describes the "Target Domain" input of the architecture (one minibatch)		

		:type xxx_struct: class NN_struct
		:param xxx_strucat: define the structure of each NN
		"""
		#------------------------------------------------------------------------
		#Encoder 1 Neural Network: present q_\phi(z_n | x_n, y_n)
		self.label_y = label_y
		phi_xy_in = T.concatenate([input_x, label_y], axis=1)
		
		'''
		self.phi_1 = nn.Stacked_NN_0L(
			rng=rng,
			input=phi_xy_in,
			struct = phi_1_struct,
			name='Encoder1'
		)	 
		'''
		
		self.phi_mu = nn.Stacked_NN_0L(
			rng=rng,
			input=phi_xy_in,
			struct = phi_1_struct,
			name='Encoder1_mu'
		)		 

		self.phi_sigma = nn.Stacked_NN_0L(
			rng=rng,
			input=phi_xy_in,
			struct = phi_1_struct,
			name='Encoder1_sigma'
		)  
		
		
		#Sample layer
		self.phi_z = nn.GaussianSampleLayer(
			mu=self.phi_mu.output,
			log_sigma=self.phi_sigma.output,
			n_in = phi_1_struct.layer_dim[-1],
			batch_size = batch_size
		)
	   
		EC_z_dim = phi_1_struct.layer_dim[-1]
		self.EC_mu = self.phi_mu.output
		self.EC_log_sigma = self.phi_sigma.output
		self.EC_sigma = T.exp(self.EC_log_sigma)
		self.EC_z = self.phi_z.output
		
		self.phi_1_params = self.phi_mu.params + self.phi_sigma.params
		self.phi_1_outputs = [self.EC_mu, self.EC_log_sigma, self.EC_z]
		self.phi_1_outputs_name = ["EC_mu", "EC_log_sigma", "EC_z"]
		
		#------------------------------------------------------------------------
		#Decoder 1 Neural Network: present p_\theta(z_n | x_n)
		'''
		theta_1 = nn.NNLayer(
			rng=rng,			
			input_source=input_x,
			struct = theta_1_struct,
			name='Decoder1'
		) 
		'''
		self.theta_mu = nn.Stacked_NN_0L(
			rng=rng,			
			input=input_x,
			struct = theta_1_struct,
			name='Decoder1_mu'
		)		 

		self.theta_sigma = nn.Stacked_NN_0L(
			rng=rng,			
			input=input_x,
			struct = theta_1_struct,
			name='Decoder1_sigma'
		)  
		
		self.DC_mu = self.theta_mu.output
		self.DC_log_sigma = self.theta_sigma.output
		self.DC_sigma = T.exp(self.DC_log_sigma)
		
		self.theta_1_params = self.theta_mu.params + self.theta_sigma.params
		self.theta_1_outputs = [self.DC_mu, self.DC_log_sigma]
		self.theta_1_outputs_name = ["DC_mu", "DC_log_sigma"]
		
		#------------------------------------------------------------------------
		#Predict 1 Neural Network: E_q_\phi(z_n | x_n, y_n) (p_\theta(y_n | x_n, z_n))
		self.theta_xz_in = T.concatenate([input_x, self.EC_z], axis=1)		
		self.theta_2 = nn.Stacked_NN_0L(
			rng=rng,			
			input=self.theta_xz_in,
			struct = theta_2_struct,
			name='Predict1'
		) 
		
		self.predict = nn.SoftmaxLayer(
			rng=rng,			
			input=self.theta_2.output,
			n_in=theta_2_struct.layer_dim[-1],
			n_out=out_dim,
			name='Predict2'
		)		 
		
		
		PR_y_hat_dim = out_dim
		self.PR_theta_2 = self.theta_2.output
		self.PR_y_pred = self.predict.output
		
		# This is test out put, get the test accuracy
		self.predictor = T.argmax(self.PR_y_pred, axis=1, keepdims=False)
		
		self.theta_2_params = self.theta_2.params + self.predict.params
		self.theta_2_outputs = [self.PR_theta_2, self.PR_y_pred]
		self.theta_2_outputs_name = ["PR_theta_2", "PR_y_pred"]
							
		#------------------------------------------------------------------------
		# Error Function Set				
		# KL(q(z|x,y)||p(z|x)) -----------
		self.KL = er.KLGaussianGaussian(self.EC_mu, self.EC_sigma, self.DC_mu, self.DC_sigma)
			 
		threshold = 0.0000001
		# Cross entropy
		self.CE = T.nnet.categorical_crossentropy(self.PR_y_pred, label_y);
				
		#Cost function
		self.cost = -self.KL.mean() + self.CE.mean()
						
		# the parameters of the model
		self.params = self.phi_1_params + self.theta_1_params + self.theta_2_params
		
		# all output of VAE
		self.outputs = self.phi_1_outputs + self.theta_1_outputs + self.theta_2_outputs
		self.outputs_name = self.phi_1_outputs_name + self.theta_1_outputs_name + self.theta_2_outputs_name
		

		
		
########################This version witout categorical sample###############################
class Supervised_VAE_v2(object):
	def __init__(self, rng, input_x, label_y, batch_size,
				 phi_1_struct, theta_1_struct, theta_2_struct, in_dim, out_dim):
		"""Initialize the parameters for the multilayer perceptron

		:type rng: numpy.random.RandomState
		:param rng: a random number generator used to initialize weights

		:type input_source: theano.tensor.TensorType
		:param input: symbolic variable that describes the "Source Domain" input of the architecture (one minibatch)
		
		:type input_target: theano.tensor.TensorType
		:param input: symbolic variable that describes the "Target Domain" input of the architecture (one minibatch)		

		:type xxx_struct: class NN_struct
		:param xxx_strucat: define the structure of each NN
		"""
		#------------------------------------------------------------------------
		#Encoder 1 Neural Network: present q_\phi(z_n | x_n, y_n)
		self.label_y = label_y
		phi_xy_in = T.concatenate([input_x, label_y], axis=1)
		
		'''
		self.phi_1 = nn.Stacked_NN_0L(
			rng=rng,
			input=phi_xy_in,
			struct = phi_1_struct,
			name='Encoder1'
		)	 
		'''
		
		self.phi_mu = nn.Stacked_NN_0L(
			rng=rng,
			input=phi_xy_in,
			struct = phi_1_struct,
			name='Encoder1_mu'
		)		 

		self.phi_sigma = nn.Stacked_NN_0L(
			rng=rng,
			input=phi_xy_in,
			struct = phi_1_struct,
			name='Encoder1_sigma'
		)  
		
		
		#Sample layer
		self.phi_z = nn.GaussianSampleLayer(
			mu=self.phi_mu.output,
			log_sigma=self.phi_sigma.output,
			n_in = phi_1_struct.layer_dim[-1],
			batch_size = batch_size
		)
	   
		EC_z_dim = phi_1_struct.layer_dim[-1]
		self.EC_mu = self.phi_mu.output
		self.EC_log_sigma = self.phi_sigma.output
		self.EC_sigma = T.exp(self.EC_log_sigma)
		self.EC_z = self.phi_z.output
		
		self.phi_1_params = self.phi_mu.params + self.phi_sigma.params
		self.phi_1_outputs = [self.EC_mu, self.EC_log_sigma, self.EC_z]
		self.phi_1_outputs_name = ["EC_mu", "EC_log_sigma", "EC_z"]
		
		#------------------------------------------------------------------------
		#Decoder 1 Neural Network: present p_\theta(z_n | x_n)
		'''
		theta_1 = nn.NNLayer(
			rng=rng,			
			input_source=input_x,
			struct = theta_1_struct,
			name='Decoder1'
		) 
		'''
		self.theta_mu = nn.Stacked_NN_0L(
			rng=rng,			
			input=input_x,
			struct = theta_1_struct,
			name='Decoder1_mu'
		)		 

		self.theta_sigma = nn.Stacked_NN_0L(
			rng=rng,			
			input=input_x,
			struct = theta_1_struct,
			name='Decoder1_sigma'
		)  
		
		#Sample layer
		self.theta_z = nn.GaussianSampleLayer(
			mu=self.theta_mu.output,
			log_sigma=self.theta_sigma.output,
			n_in = theta_1_struct.layer_dim[-1],
			batch_size = batch_size
		)
	   
		DC_z_dim = phi_1_struct.layer_dim[-1]
		self.DC_mu = self.theta_mu.output
		self.DC_log_sigma = self.theta_sigma.output
		self.DC_sigma = T.exp(self.DC_log_sigma)
		self.DC_z = self.theta_z.output
		
		self.theta_1_params = self.theta_mu.params + self.theta_sigma.params
		self.theta_1_outputs = [self.DC_mu, self.DC_log_sigma, self.DC_z]
		self.theta_1_outputs_name = ["DC_mu", "DC_log_sigma", "DC_z"]
		
		#------------------------------------------------------------------------
		#Predict 1 Neural Network: E_q_\phi(z_n | x_n, y_n) (p_\theta(y_n | x_n, z_n))
		self.theta_xz_in = T.concatenate([input_x, self.DC_z], axis=1)		
		self.theta_2 = nn.Stacked_NN_0L(
			rng=rng,			
			input=self.theta_xz_in,
			struct = theta_2_struct,
			name='Predict1'
		) 
		
		self.predict = nn.SoftmaxLayer(
			rng=rng,			
			input=self.theta_2.output,
			n_in=theta_2_struct.layer_dim[-1],
			n_out=out_dim,
			name='Predict2'
		)		 
		
		
		PR_y_hat_dim = out_dim
		self.PR_theta_2 = self.theta_2.output
		self.PR_y_pred = self.predict.output
		
		# This is test out put, get the test accuracy
		self.predictor = T.argmax(self.PR_y_pred, axis=1, keepdims=False)
		
		self.theta_2_params = self.theta_2.params + self.predict.params
		self.theta_2_outputs = [self.PR_theta_2, self.PR_y_pred]
		self.theta_2_outputs_name = ["PR_theta_2", "PR_y_pred"]
							
		#------------------------------------------------------------------------
		# Error Function Set				
		# KL(q(z|x,y)||p(z|x)) -----------
		self.KL = er.KLGaussianGaussian(self.EC_mu, self.EC_sigma, self.DC_mu, self.DC_sigma)
			 
		threshold = 0.0000001
		# Cross entropy
		self.CE = T.nnet.categorical_crossentropy(self.PR_y_pred, label_y);
				
		#Cost function
		self.cost = -self.KL.mean() + self.CE.mean()
						
		# the parameters of the model
		self.params = self.phi_1_params + self.theta_1_params + self.theta_2_params
		
		# all output of VAE
		self.outputs = self.phi_1_outputs + self.theta_1_outputs + self.theta_2_outputs
		self.outputs_name = self.phi_1_outputs_name + self.theta_1_outputs_name + self.theta_2_outputs_name
		

		
########################This version witout categorical sample###############################
class Supervised_VAE_v3(object):
	def __init__(self, rng, input_x, label_y, batch_size,
				 phi_1_struct, theta_1_struct, theta_2_struct, in_dim, out_dim,
				 model=None):
		"""Initialize the parameters for the multilayer perceptron

		:type rng: numpy.random.RandomState
		:param rng: a random number generator used to initialize weights

		:type input_source: theano.tensor.TensorType
		:param input: symbolic variable that describes the "Source Domain" input of the architecture (one minibatch)
		
		:type input_target: theano.tensor.TensorType
		:param input: symbolic variable that describes the "Target Domain" input of the architecture (one minibatch)		

		:type xxx_struct: class NN_struct
		:param xxx_strucat: define the structure of each NN
		"""
		#------------------------------------------------------------------------
		#Encoder 1 Neural Network: present q_\phi(z_n | x_n, y_n)
		self.label_y = label_y
		phi_xy_in = T.concatenate([input_x, label_y], axis=1)
		
		'''
		self.phi_1 = nn.Stacked_NN_0L(
			rng=rng,
			input=phi_xy_in,
			struct = phi_1_struct,
			name='Encoder1'
		)	 
		'''
		
		phi_1_struct.activation[0] = None
		self.phi_mu = nn.Stacked_NN_0L(
			rng=rng,
			input=phi_xy_in,
			struct = phi_1_struct,
			name='Encoder1_mu'
		)
		phi_1_struct.activation[0] = T.nnet.softplus
		self.phi_sigma = nn.Stacked_NN_0L(
			rng=rng,
			input=phi_xy_in,
			struct = phi_1_struct,
			name='Encoder1_sigma'
		)  
		
		
		#Sample layer
		self.phi_z = nn.GaussianSampleLayer(
			mu=self.phi_mu.output,
			log_sigma=self.phi_sigma.output,
			n_in = phi_1_struct.layer_dim[-1],
			batch_size = batch_size
		)
	   
		EC_z_dim = phi_1_struct.layer_dim[-1]
		self.EC_mu = self.phi_mu.output
		self.EC_log_sigma = self.phi_sigma.output
		self.EC_sigma = T.exp(self.EC_log_sigma)
		self.EC_z = self.phi_z.output
		
		self.phi_1_params = self.phi_mu.params + self.phi_sigma.params
		self.phi_1_outputs = [self.EC_mu, self.EC_log_sigma, self.EC_z]
		self.phi_1_outputs_name = ["EC_mu", "EC_log_sigma", "EC_z"]
		
		#------------------------------------------------------------------------
		#Decoder 1 Neural Network: present p_\theta(z_n | x_n)
		'''
		theta_1 = nn.NNLayer(
			rng=rng,			
			input_source=input_x,
			struct = theta_1_struct,
			name='Decoder1'
		) 
		'''
		theta_1_struct.activation[0] = None
		self.theta_mu = nn.Stacked_NN_0L(
			rng=rng,
			input=input_x,
			struct = theta_1_struct,
			name='Decoder1_mu'
		)
		if model is not None:
			self.theta_mu.OL.W.set_value(model.phi_mu.OL.W.get_value())
			self.theta_mu.OL.b.set_value(model.phi_mu.OL.b.get_value())
		theta_1_struct.activation[0] = T.nnet.softplus
		self.theta_sigma = nn.Stacked_NN_0L(
			rng=rng,
			input=input_x,
			struct = theta_1_struct,
			name='Decoder1_sigma'
		)
		if model is not None:		
			self.theta_sigma.OL.W.set_value(model.phi_sigma.OL.W.get_value())
			self.theta_sigma.OL.b.set_value(model.phi_sigma.OL.b.get_value())
		#Sample layer
		self.theta_z = nn.GaussianSampleLayer(
			mu=self.theta_mu.output,
			log_sigma=self.theta_sigma.output,
			n_in = theta_1_struct.layer_dim[-1],
			batch_size = batch_size
		)
	   
		DC_z_dim = phi_1_struct.layer_dim[-1]
		self.DC_mu = self.theta_mu.output
		self.DC_log_sigma = self.theta_sigma.output
		self.DC_sigma = T.exp(self.DC_log_sigma)
		self.DC_z = self.theta_z.output
		
		self.theta_1_params = self.theta_mu.params + self.theta_sigma.params
		self.theta_1_outputs = [self.DC_mu, self.DC_log_sigma, self.DC_z]
		self.theta_1_outputs_name = ["DC_mu", "DC_log_sigma", "DC_z"]
		
		#------------------------------------------------------------------------
		#Predict 1 Neural Network: E_q_\phi(z_n | x_n, y_n) (p_\theta(y_n | x_n, z_n))
		self.theta_xz_in = T.concatenate([input_x, self.EC_z], axis=1)
		theta_2_struct.activation[0] = T.nnet.sigmoid
		self.theta_2 = nn.Stacked_NN_0L(
			rng=rng,			
			input=self.theta_xz_in,
			struct = theta_2_struct,
			name='Predict1'
		) 
		
		self.predict = nn.SoftmaxLayer(
			rng=rng,			
			input=self.theta_2.output,
			n_in=theta_2_struct.layer_dim[-1],
			n_out=out_dim,
			name='Predict2'
		)		 
		
		
		PR_y_hat_dim = out_dim
		self.PR_theta_2 = self.theta_2.output
		self.PR_y_pred = self.predict.output
		
		
		# This predictor is used to test training output, get the training accuracy
		self.predictor = T.argmax(self.PR_y_pred, axis=1, keepdims=False)
		
		self.theta_2_params = self.theta_2.params + self.predict.params
		self.theta_2_outputs = [self.PR_theta_2, self.PR_y_pred]
		self.theta_2_outputs_name = ["PR_theta_2", "PR_y_pred"]
							
		#------------------------------------------------------------------------
		# Error Function Set				
		# KL(q(z|x,y)||p(z|x)) -----------
		self.KL = er.KLGaussianGaussian(self.EC_mu, self.EC_sigma, self.DC_mu, self.DC_sigma)
		# Cross entropy
		self.CE = T.nnet.categorical_crossentropy(self.PR_y_pred, label_y);
				
		#Cost function
		self.cost = -self.KL.mean() + self.CE.mean()
						
		# the parameters of the model
		self.params = self.phi_1_params + self.theta_1_params + self.theta_2_params
		
		# all output of VAE
		self.outputs = self.phi_1_outputs + self.theta_1_outputs + self.theta_2_outputs
		self.outputs_name = self.phi_1_outputs_name + self.theta_1_outputs_name + self.theta_2_outputs_name
		
		#------------------------------------------------------------------------
		#Test predict Neural Network
		self.test_xz = T.concatenate([input_x, self.DC_z], axis=1)		
		self.test_nn = nn.Stacked_NN_0L(
			rng=rng,			
			input=self.test_xz,
			struct = theta_2_struct,
			name='Predict1_test',
			#W=self.theta_2.OL.W,
			#b=self.theta_2.OL.b
		)
		self.test_nn.OL.W.set_value(self.theta_2.OL.W.get_value())
		self.test_nn.OL.b.set_value(self.theta_2.OL.b.get_value())
		
		self.test_predict = nn.SoftmaxLayer(
			rng=rng,			
			input=self.test_nn.output,
			n_in=theta_2_struct.layer_dim[-1],
			n_out=out_dim,
			name='Predict2_test',
			#W=self.predict.W,
			#b=self.predict.b
		)
		self.test_predict.W.set_value(self.predict.W.get_value())
		self.test_predict.b.set_value(self.predict.b.get_value())
		
		self.PR_test = self.test_predict.output
		# This predictor is to test training output, get the training accuracy
		self.predictor_test = T.argmax(self.PR_test, axis=1, keepdims=False)
		
class Supervised_VAE_v3_CE(object):
	def __init__(self, rng, input_x, label_y, batch_size,
				 phi_1_struct, theta_2_struct, in_dim, out_dim):
		#------------------------------------------------------------------------
		#Encoder 1 Neural Network: present q_\phi(z_n | x_n, y_n)
		self.label_y = label_y
		phi_xy_in = T.concatenate([input_x, label_y], axis=1)
		
		'''
		self.phi_1 = nn.Stacked_NN_0L(
			rng=rng,
			input=phi_xy_in,
			struct = phi_1_struct,
			name='Encoder1'
		)	 
		'''
		
		phi_1_struct.activation[0] = T.nnet.sigmoid
		phi_1_struct.activation[1] = None
		self.phi_mu = nn.Stacked_NN_1L(
			rng=rng,
			input=phi_xy_in,
			struct = phi_1_struct,
			name='Encoder1_mu'
		)
		
		phi_1_struct.activation[0] = T.nnet.sigmoid
		phi_1_struct.activation[1] = None
		self.phi_sigma = nn.Stacked_NN_1L(
			rng=rng,
			input=phi_xy_in,
			struct = phi_1_struct,
			name='Encoder1_sigma'
		)  
		
		
		#Sample layer
		self.phi_z = nn.GaussianSampleLayer(
			mu=self.phi_mu.output,
			log_sigma=self.phi_sigma.output,
			n_in = phi_1_struct.layer_dim[-1],
			batch_size = batch_size
		)
	   
		EC_z_dim = phi_1_struct.layer_dim[-1]
		self.EC_mu = self.phi_mu.output
		self.EC_log_sigma = self.phi_sigma.output
		self.EC_z = self.phi_z.output
		
		self.phi_1_params = self.phi_mu.params + self.phi_sigma.params
		self.phi_1_outputs = [self.EC_mu, self.EC_log_sigma, self.EC_z]
		self.phi_1_outputs_name = ["EC_mu", "EC_log_sigma", "EC_z"]
		
		
		#------------------------------------------------------------------------
		#Predict 1 Neural Network: E_q_\phi(z_n | x_n, y_n) (p_\theta(y_n | x_n, z_n))
		self.theta_xz_in = T.concatenate([input_x, self.EC_z], axis=1)
		theta_2_struct.activation[0] = T.nnet.sigmoid
		theta_2_struct.activation[1] = T.nnet.sigmoid
		self.theta_2 = nn.Stacked_NN_1L(
			rng=rng,
			input=self.theta_xz_in,
			struct = theta_2_struct,
			name='Predict1'
		) 
		
		self.predict = nn.SoftmaxLayer(
			rng=rng,
			input=self.theta_2.output,
			n_in=theta_2_struct.layer_dim[-1],
			n_out=out_dim,
			name='Predict2'
		)		 
		
		
		PR_y_hat_dim = out_dim
		self.PR_theta_2 = self.theta_2.output
		self.PR_y_pred = self.predict.output
		
		
		# This predictor is to test training output, get the training accuracy
		self.predictor = T.argmax(self.PR_y_pred, axis=1, keepdims=False)
		
		self.theta_2_params = self.theta_2.params + self.predict.params
		self.theta_2_outputs = [self.PR_theta_2, self.PR_y_pred]
		self.theta_2_outputs_name = ["PR_theta_2", "PR_y_pred"]
		#------------------------------------------------------------------------
		# Error Function Set				
		# Cross entropy
		self.CE = T.nnet.categorical_crossentropy(self.PR_y_pred, label_y);
				
		#Cost function
		self.cost = self.CE.mean()
						
		# the parameters of the model
		self.params = self.phi_1_params + self.theta_2_params
		
		# all output of VAE
		self.outputs = self.phi_1_outputs + self.theta_2_outputs
		self.outputs_name = self.phi_1_outputs_name + self.theta_2_outputs_name

class Supervised_VAE_v3_KL(object):
	def __init__(self, rng, input_x, label_y, batch_size,
				 phi_1_struct, theta_1_struct, theta_2_struct, in_dim, out_dim,
				 model=None):
		#------------------------------------------------------------------------
		#Encoder 1 Neural Network: present q_\phi(z_n | x_n, y_n)
		self.label_y = label_y
		phi_xy_in = T.concatenate([input_x, label_y], axis=1)
		
		phi_1_struct.activation[0] = T.nnet.sigmoid
		phi_1_struct.activation[1] = None
		self.phi_mu = nn.Stacked_NN_1L(
			rng=rng,
			input=phi_xy_in,
			struct = phi_1_struct,
			name='Encoder1_mu',
			#W=model.phi_mu.OL.W,
			#b=model.phi_mu.OL.b
		)
		self.phi_mu.HL_1.W.set_value(model.phi_mu.HL_1.W.get_value())
		self.phi_mu.HL_1.b.set_value(model.phi_mu.HL_1.b.get_value())
		self.phi_mu.OL.W.set_value(model.phi_mu.OL.W.get_value())
		self.phi_mu.OL.b.set_value(model.phi_mu.OL.b.get_value())
		
		phi_1_struct.activation[0] = T.nnet.sigmoid
		phi_1_struct.activation[1] = None
		self.phi_sigma = nn.Stacked_NN_1L(
			rng=rng,
			input=phi_xy_in,
			struct = phi_1_struct,
			name='Encoder1_sigma',
			#W=model.phi_sigma.OL.W,
			#b=model.phi_sigma.OL.b
		)
		self.phi_sigma.HL_1.W.set_value(model.phi_sigma.HL_1.W.get_value())
		self.phi_sigma.HL_1.b.set_value(model.phi_sigma.HL_1.b.get_value())
		self.phi_sigma.OL.W.set_value(model.phi_sigma.OL.W.get_value())
		self.phi_sigma.OL.b.set_value(model.phi_sigma.OL.b.get_value())
		
		#Sample layer
		self.phi_z = nn.GaussianSampleLayer(
			mu=self.phi_mu.output,
			log_sigma=self.phi_sigma.output,
			n_in = phi_1_struct.layer_dim[-1],
			batch_size = batch_size
		)
	   
		EC_z_dim = phi_1_struct.layer_dim[-1]
		self.EC_mu = self.phi_mu.output
		self.EC_log_sigma = self.phi_sigma.output
		self.EC_z = self.phi_z.output
		
		self.phi_1_params = self.phi_mu.params + self.phi_sigma.params
		self.phi_1_outputs = [self.EC_mu, self.EC_log_sigma, self.EC_z]
		self.phi_1_outputs_name = ["EC_mu", "EC_log_sigma", "EC_z"]
		
		#------------------------------------------------------------------------
		#Decoder 1 Neural Network: present p_\theta(z_n | x_n)
		theta_1_struct.activation[0] = T.nnet.sigmoid
		theta_1_struct.activation[1] = None
		self.theta_mu = nn.Stacked_NN_1L(
			rng=rng,
			input=input_x,
			struct = theta_1_struct,
			name='Decoder1_mu'
		)
		theta_1_struct.activation[0] = T.nnet.sigmoid
		theta_1_struct.activation[1] = None
		self.theta_sigma = nn.Stacked_NN_1L(
			rng=rng,
			input=input_x,
			struct = theta_1_struct,
			name='Decoder1_sigma'
		)  
		
		#Sample layer
		self.theta_z = nn.GaussianSampleLayer(
			mu=self.theta_mu.output,
			log_sigma=self.theta_sigma.output,
			n_in = theta_1_struct.layer_dim[-1],
			batch_size = batch_size
		)
	   
		DC_z_dim = phi_1_struct.layer_dim[-1]
		self.DC_mu = self.theta_mu.output
		self.DC_log_sigma = self.theta_sigma.output
		self.DC_z = self.theta_z.output
		
		self.theta_1_params = self.theta_mu.params + self.theta_sigma.params
		self.theta_1_outputs = [self.DC_mu, self.DC_log_sigma, self.DC_z]
		self.theta_1_outputs_name = ["DC_mu", "DC_log_sigma", "DC_z"]
		
		
							
		#------------------------------------------------------------------------
		# Error Function Set				
		# KL(q(z|x,y)||p(z|x)) -----------
		self.KL = er.KLGaussianGaussian(self.EC_mu, self.EC_log_sigma, self.DC_mu, self.DC_log_sigma)
		#self.KL = er.KLGaussianGaussian(self.DC_mu, self.DC_log_sigma, self.EC_mu, self.EC_log_sigma)
		## Cross entropy
		#self.CE = T.nnet.categorical_crossentropy(self.PR_y_pred, label_y);
				
		#Cost function
		self.cost = -self.KL.mean()
						
		# the parameters of the model
		self.params = self.theta_1_params
		
		# all output of VAE
		self.outputs = self.theta_1_outputs
		self.outputs_name = self.theta_1_outputs_name
		
		#------------------------------------------------------------------------
		#Predict 1 Neural Network: E_q_\phi(z_n | x_n, y_n) (p_\theta(y_n | x_n, z_n))
		self.theta_xz_in = T.concatenate([input_x, self.EC_z], axis=1)
		theta_2_struct.activation[0] = T.nnet.sigmoid
		theta_2_struct.activation[1] = T.nnet.sigmoid
		self.theta_2 = nn.Stacked_NN_1L(
			rng=rng,
			input=self.theta_xz_in,
			struct = theta_2_struct,
			name='Predict1',
			#W=model.theta_2.OL.W,
			#b=model.theta_2.OL.b
		)
		self.theta_2.HL_1.W.set_value(model.theta_2.HL_1.W.get_value())
		self.theta_2.HL_1.b.set_value(model.theta_2.HL_1.b.get_value())
		self.theta_2.OL.W.set_value(model.theta_2.OL.W.get_value())
		self.theta_2.OL.b.set_value(model.theta_2.OL.b.get_value())
		
		self.predict = nn.SoftmaxLayer(
			rng=rng,
			input=self.theta_2.output,
			n_in=theta_2_struct.layer_dim[-1],
			n_out=out_dim,
			name='Predict2',
			#W=model.predict.W,
			#b=model.predict.b
		)
		self.predict.W.set_value(model.predict.W.get_value())
		self.predict.b.set_value(model.predict.b.get_value())
		
		self.PR_y_pred = self.predict.output
		
		
		# This predictor is to test training output, get the training accuracy
		self.predictor = T.argmax(self.PR_y_pred, axis=1, keepdims=False)
		
		#------------------------------------------------------------------------
		#Test predict Neural Network
		self.test_xz = T.concatenate([input_x, self.DC_z], axis=1)		
		self.test_nn = nn.Stacked_NN_1L(
			rng=rng,			
			input=self.test_xz,
			struct = theta_2_struct,
			name='Predict1_test',
			#W=model.theta_2.OL.W,
			#b=model.theta_2.OL.b
		)
		self.test_nn.HL_1.W.set_value(model.theta_2.HL_1.W.get_value())
		self.test_nn.HL_1.b.set_value(model.theta_2.HL_1.b.get_value())
		self.test_nn.OL.W.set_value(model.theta_2.OL.W.get_value())
		self.test_nn.OL.b.set_value(model.theta_2.OL.b.get_value())
		
		self.test_predict = nn.SoftmaxLayer(
			rng=rng,			
			input=self.test_nn.output,
			n_in=theta_2_struct.layer_dim[-1],
			n_out=out_dim,
			name='Predict2_test',
			#W=model.predict.W,
			#b=model.predict.b
		)
		self.test_predict.W.set_value(model.predict.W.get_value())
		self.test_predict.b.set_value(model.predict.b.get_value())
		
		self.PR_test = self.test_predict.output
		# This predictor is to test training output, get the training accuracy
		self.predictor_test = T.argmax(self.PR_test, axis=1, keepdims=False)

class Supervised_VAE_v3_KL_1(object):
	def __init__(self, rng, input_x, label_y, batch_size,
				 phi_1_struct, theta_1_struct, theta_2_struct, in_dim, out_dim,
				 model=None):
		#------------------------------------------------------------------------
		#Encoder 1 Neural Network: present q_\phi(z_n | x_n, y_n)
		self.label_y = label_y
		phi_xy_in = T.concatenate([input_x, label_y], axis=1)
		
		phi_1_struct.activation[0] = T.nnet.sigmoid
		phi_1_struct.activation[1] = None
		self.phi_mu = nn.Stacked_NN_1L(
			rng=rng,
			input=phi_xy_in,
			struct = phi_1_struct,
			name='Encoder1_mu',
			#W=model.phi_mu.OL.W,
			#b=model.phi_mu.OL.b
		)
		self.phi_mu.HL_1.W.set_value(model.phi_mu.HL_1.W.get_value())
		self.phi_mu.HL_1.b.set_value(model.phi_mu.HL_1.b.get_value())
		self.phi_mu.OL.W.set_value(model.phi_mu.OL.W.get_value())
		self.phi_mu.OL.b.set_value(model.phi_mu.OL.b.get_value())
		
		phi_1_struct.activation[0] = T.nnet.sigmoid
		phi_1_struct.activation[1] = None
		self.phi_sigma = nn.Stacked_NN_1L(
			rng=rng,
			input=phi_xy_in,
			struct = phi_1_struct,
			name='Encoder1_sigma',
			#W=model.phi_sigma.OL.W,
			#b=model.phi_sigma.OL.b
		)
		self.phi_sigma.HL_1.W.set_value(model.phi_sigma.HL_1.W.get_value())
		self.phi_sigma.HL_1.b.set_value(model.phi_sigma.HL_1.b.get_value())
		self.phi_sigma.OL.W.set_value(model.phi_sigma.OL.W.get_value())
		self.phi_sigma.OL.b.set_value(model.phi_sigma.OL.b.get_value())
		
		#Sample layer
		self.phi_z = nn.GaussianSampleLayer(
			mu=self.phi_mu.output,
			log_sigma=self.phi_sigma.output,
			n_in = phi_1_struct.layer_dim[-1],
			batch_size = batch_size
		)
	   
		EC_z_dim = phi_1_struct.layer_dim[-1]
		self.EC_mu = self.phi_mu.output
		self.EC_log_sigma = self.phi_sigma.output
		self.EC_z = self.phi_z.output
		
		self.phi_1_params = self.phi_mu.params + self.phi_sigma.params
		self.phi_1_outputs = [self.EC_mu, self.EC_log_sigma, self.EC_z]
		self.phi_1_outputs_name = ["EC_mu", "EC_log_sigma", "EC_z"]
		
		#------------------------------------------------------------------------
		#Decoder 1 Neural Network: present p_\theta(z_n | x_n)
		theta_1_struct.activation[0] = T.nnet.sigmoid
		theta_1_struct.activation[1] = None
		self.theta_mu = nn.Stacked_NN_1L(
			rng=rng,
			input=input_x,
			struct = theta_1_struct,
			name='Decoder1_mu'
		)
		theta_1_struct.activation[0] = T.nnet.sigmoid
		theta_1_struct.activation[1] = None
		self.theta_sigma = nn.Stacked_NN_1L(
			rng=rng,
			input=input_x,
			struct = theta_1_struct,
			name='Decoder1_sigma'
		)  
		
		#Sample layer
		self.theta_z = nn.GaussianSampleLayer(
			mu=self.theta_mu.output,
			log_sigma=self.theta_sigma.output,
			n_in = theta_1_struct.layer_dim[-1],
			batch_size = batch_size
		)
	   
		DC_z_dim = phi_1_struct.layer_dim[-1]
		self.DC_mu = self.theta_mu.output
		self.DC_log_sigma = self.theta_sigma.output
		self.DC_z = self.theta_z.output
		
		self.theta_1_params = self.theta_mu.params + self.theta_sigma.params
		self.theta_1_outputs = [self.DC_mu, self.DC_log_sigma, self.DC_z]
		self.theta_1_outputs_name = ["DC_mu", "DC_log_sigma", "DC_z"]
		
		
		#------------------------------------------------------------------------
		#Predict 1 Neural Network: E_q_\phi(z_n | x_n, y_n) (p_\theta(y_n | x_n, z_n))
		self.theta_xz_in = T.concatenate([input_x, self.EC_z], axis=1)
		theta_2_struct.activation[0] = T.nnet.sigmoid
		theta_2_struct.activation[1] = T.nnet.sigmoid
		self.theta_2 = nn.Stacked_NN_1L(
			rng=rng,
			input=self.theta_xz_in,
			struct = theta_2_struct,
			name='Predict1',
			#W=model.theta_2.OL.W,
			#b=model.theta_2.OL.b
		)
		self.theta_2.HL_1.W.set_value(model.theta_2.HL_1.W.get_value())
		self.theta_2.HL_1.b.set_value(model.theta_2.HL_1.b.get_value())
		self.theta_2.OL.W.set_value(model.theta_2.OL.W.get_value())
		self.theta_2.OL.b.set_value(model.theta_2.OL.b.get_value())
		
		self.predict = nn.SoftmaxLayer(
			rng=rng,
			input=self.theta_2.output,
			n_in=theta_2_struct.layer_dim[-1],
			n_out=out_dim,
			name='Predict2',
			#W=model.predict.W,
			#b=model.predict.b
		)
		self.predict.W.set_value(model.predict.W.get_value())
		self.predict.b.set_value(model.predict.b.get_value())
		
		self.PR_y_pred = self.predict.output
		
		
		# This predictor is to test training output, get the training accuracy
		self.predictor = T.argmax(self.PR_y_pred, axis=1, keepdims=False)
		
		#------------------------------------------------------------------------
		#Test predict Neural Network
		self.test_xz = T.concatenate([input_x, self.DC_z], axis=1)
		self.test_nn = nn.Stacked_NN_1L(
			rng=rng,			
			input=self.test_xz,
			struct = theta_2_struct,
			name='Predict1_test',
			#W=model.theta_2.OL.W,
			#b=model.theta_2.OL.b
		)
		self.test_nn.HL_1.W.set_value(model.theta_2.HL_1.W.get_value())
		self.test_nn.HL_1.b.set_value(model.theta_2.HL_1.b.get_value())
		self.test_nn.OL.W.set_value(model.theta_2.OL.W.get_value())
		self.test_nn.OL.b.set_value(model.theta_2.OL.b.get_value())
		
		self.test_predict = nn.SoftmaxLayer(
			rng=rng,			
			input=self.test_nn.output,
			n_in=theta_2_struct.layer_dim[-1],
			n_out=out_dim,
			name='Predict2_test',
			#W=model.predict.W,
			#b=model.predict.b
		)
		self.test_predict.W.set_value(model.predict.W.get_value())
		self.test_predict.b.set_value(model.predict.b.get_value())
		
		self.PR_test = self.test_predict.output
		# This predictor is to test training output, get the training accuracy
		self.predictor_test = T.argmax(self.PR_test, axis=1, keepdims=False)
		
		
		self.test_params = self.test_nn.params + self.test_predict.params
		
		#------------------------------------------------------------------------
		# Error Function Set				
		# KL(q(z|x,y)||p(z|x)) -----------
		self.KL = er.KLGaussianGaussian(self.EC_mu, self.EC_log_sigma, self.DC_mu, self.DC_log_sigma)
		#self.KL = er.KLGaussianGaussian(self.DC_mu, self.DC_log_sigma, self.EC_mu, self.EC_log_sigma)
		# Cross entropy
		self.CE = T.nnet.categorical_crossentropy(self.PR_test, label_y);
				
		#Cost function
		self.cost = -self.KL.mean() + self.CE.mean()
						
		# the parameters of the model
		self.params = self.theta_1_params + self.test_params
		
		# all output of VAE
		self.outputs = self.theta_1_outputs
		self.outputs_name = self.theta_1_outputs_name
		
class Supervised_VAE_v3_KL_CE(object):
	def __init__(self, rng, input_x, label_y, batch_size,
				 phi_1_struct, theta_1_struct, theta_2_struct, in_dim, out_dim,
				 model=None):
		#------------------------------------------------------------------------
		#Encoder 1 Neural Network: present q_\phi(z_n | x_n, y_n)
		self.label_y = label_y
		phi_xy_in = T.concatenate([input_x, label_y], axis=1)
		
		phi_1_struct.activation[0] = T.nnet.sigmoid
		phi_1_struct.activation[1] = None
		self.phi_mu = nn.Stacked_NN_1L(
			rng=rng,
			input=phi_xy_in,
			struct = phi_1_struct,
			name='Encoder1_mu',
			#W=model.phi_mu.OL.W,
			#b=model.phi_mu.OL.b
		)
		self.phi_mu.HL_1.W.set_value(model.phi_mu.HL_1.W.get_value())
		self.phi_mu.HL_1.b.set_value(model.phi_mu.HL_1.b.get_value())
		self.phi_mu.OL.W.set_value(model.phi_mu.OL.W.get_value())
		self.phi_mu.OL.b.set_value(model.phi_mu.OL.b.get_value())
		
		phi_1_struct.activation[0] = T.nnet.sigmoid
		phi_1_struct.activation[1] = None
		self.phi_sigma = nn.Stacked_NN_1L(
			rng=rng,
			input=phi_xy_in,
			struct = phi_1_struct,
			name='Encoder1_sigma',
			#W=model.phi_sigma.OL.W,
			#b=model.phi_sigma.OL.b
		)
		self.phi_sigma.HL_1.W.set_value(model.phi_sigma.HL_1.W.get_value())
		self.phi_sigma.HL_1.b.set_value(model.phi_sigma.HL_1.b.get_value())
		self.phi_sigma.OL.W.set_value(model.phi_sigma.OL.W.get_value())
		self.phi_sigma.OL.b.set_value(model.phi_sigma.OL.b.get_value())
		#Sample layer
		self.phi_z = nn.GaussianSampleLayer(
			mu=self.phi_mu.output,
			log_sigma=self.phi_sigma.output,
			n_in = phi_1_struct.layer_dim[-1],
			batch_size = batch_size
		)
	   
		EC_z_dim = phi_1_struct.layer_dim[-1]
		self.EC_mu = self.phi_mu.output
		self.EC_log_sigma = self.phi_sigma.output
		self.EC_z = self.phi_z.output
		
		self.phi_1_params = self.phi_mu.params + self.phi_sigma.params
		self.phi_1_outputs = [self.EC_mu, self.EC_log_sigma, self.EC_z]
		self.phi_1_outputs_name = ["EC_mu", "EC_log_sigma", "EC_z"]
		
		#------------------------------------------------------------------------
		#Decoder 1 Neural Network: present p_\theta(z_n | x_n)
		theta_1_struct.activation[0] = T.nnet.sigmoid
		theta_1_struct.activation[1] = None
		self.theta_mu = nn.Stacked_NN_1L(
			rng=rng,
			input=input_x,
			struct = theta_1_struct,
			name='Decoder1_mu'
		)
		theta_1_struct.activation[0] = T.nnet.sigmoid
		theta_1_struct.activation[1] = None
		self.theta_sigma = nn.Stacked_NN_1L(
			rng=rng,
			input=input_x,
			struct = theta_1_struct,
			name='Decoder1_sigma'
		)  
		
		#Sample layer
		self.theta_z = nn.GaussianSampleLayer(
			mu=self.theta_mu.output,
			log_sigma=self.theta_sigma.output,
			n_in = theta_1_struct.layer_dim[-1],
			batch_size = batch_size
		)
	   
		DC_z_dim = phi_1_struct.layer_dim[-1]
		self.DC_mu = self.theta_mu.output
		self.DC_log_sigma = self.theta_sigma.output
		self.DC_z = self.theta_z.output
		
		self.theta_1_params = self.theta_mu.params + self.theta_sigma.params
		self.theta_1_outputs = [self.DC_mu, self.DC_log_sigma, self.DC_z]
		self.theta_1_outputs_name = ["DC_mu", "DC_log_sigma", "DC_z"]
		
		#------------------------------------------------------------------------
		#Predict 1 Neural Network: E_q_\phi(z_n | x_n, y_n) (p_\theta(y_n | x_n, z_n))
		self.theta_xz_in = T.concatenate([input_x, self.EC_z], axis=1)
		theta_2_struct.activation[0] = T.nnet.sigmoid
		theta_2_struct.activation[1] = T.nnet.sigmoid
		self.theta_2 = nn.Stacked_NN_1L(
			rng=rng,
			input=self.theta_xz_in,
			struct = theta_2_struct,
			name='Predict1',
			#W=model.theta_2.OL.W,
			#b=model.theta_2.OL.b
		)
		self.theta_2.HL_1.W.set_value(model.theta_2.HL_1.W.get_value())
		self.theta_2.HL_1.b.set_value(model.theta_2.HL_1.b.get_value())
		self.theta_2.OL.W.set_value(model.theta_2.OL.W.get_value())
		self.theta_2.OL.b.set_value(model.theta_2.OL.b.get_value())
		
		self.predict = nn.SoftmaxLayer(
			rng=rng,
			input=self.theta_2.output,
			n_in=theta_2_struct.layer_dim[-1],
			n_out=out_dim,
			name='Predict2',
			#W=model.predict.W,
			#b=model.predict.b
		)
		self.predict.W.set_value(model.predict.W.get_value())
		self.predict.b.set_value(model.predict.b.get_value())
		
		self.PR_y_pred = self.predict.output
		
		
		# This predictor is used to test training output, get the training accuracy
		self.predictor = T.argmax(self.PR_y_pred, axis=1, keepdims=False)
		
		self.theta_2_params = self.theta_2.params + self.predict.params
		self.theta_2_outputs = [self.PR_theta_2, self.PR_y_pred]
		self.theta_2_outputs_name = ["PR_theta_2", "PR_y_pred"]
							
		#------------------------------------------------------------------------
		# Error Function Set				
		# KL(q(z|x,y)||p(z|x)) -----------
		self.KL = er.KLGaussianGaussian(self.EC_mu, self.EC_log_sigma, self.DC_mu, self.DC_log_sigma)
		# Cross entropy
		self.CE = T.nnet.categorical_crossentropy(self.PR_y_pred, label_y);
				
		#Cost function
		self.cost = -self.KL.mean() + self.CE.mean()
						
		# the parameters of the model
		self.params = self.theta_1_params + self.phi_1_params + self.theta_2_params
		
		# all output of VAE
		self.outputs = self.theta_1_outputs + self.phi_1_outputs + self.theta_2_outputs
		self.outputs_name = self.theta_1_outputs_name + self.phi_1_outputs_name + self.theta_2_outputs_name
		
		
		
		#------------------------------------------------------------------------
		#Test predict Neural Network
		self.test_xz = T.concatenate([input_x, self.DC_z], axis=1)		
		self.test_nn = nn.Stacked_NN_0L(
			rng=rng,			
			input=self.test_xz,
			struct = theta_2_struct,
			name='Predict1_test',
			#W=model.theta_2.OL.W,
			#b=model.theta_2.OL.b
		)
		self.test_nn.OL.W.set_value(model.theta_2.OL.W.get_value())
		self.test_nn.OL.b.set_value(model.theta_2.OL.b.get_value())
		
		self.test_predict = nn.SoftmaxLayer(
			rng=rng,			
			input=self.test_nn.output,
			n_in=theta_2_struct.layer_dim[-1],
			n_out=out_dim,
			name='Predict2_test',
			#W=model.predict.W,
			#b=model.predict.b
		)
		self.test_predict.W.set_value(model.predict.W.get_value())
		self.test_predict.b.set_value(model.predict.b.get_value())
		
		self.PR_test = self.test_predict.output
		# This predictor is to test training output, get the training accuracy
		self.predictor_test = T.argmax(self.PR_test, axis=1, keepdims=False)
		
		
class Supervised_VAE_v3_CE_2L(object):
	def __init__(self, rng, input_x, label_y, batch_size,
				 phi_1_struct, theta_2_struct, in_dim, out_dim):
		#------------------------------------------------------------------------
		#Encoder 1 Neural Network: present q_\phi(z_n | x_n, y_n)
		self.label_y = label_y
		phi_xy_in = T.concatenate([input_x, label_y], axis=1)
		
		phi_1_struct.activation[0] = T.nnet.sigmoid
		phi_1_struct.activation[1] = T.nnet.sigmoid
		phi_1_struct.activation[2] = None
		self.phi_mu = nn.Stacked_NN_2L(
			rng=rng,
			input=phi_xy_in,
			struct = phi_1_struct,
			name='Encoder1_mu'
		)
		phi_1_struct.activation[0] = T.nnet.sigmoid
		phi_1_struct.activation[1] = T.nnet.sigmoid
		phi_1_struct.activation[2] = None
		self.phi_sigma = nn.Stacked_NN_2L(
			rng=rng,
			input=phi_xy_in,
			struct = phi_1_struct,
			name='Encoder1_sigma'
		)  
		
		
		#Sample layer
		self.phi_z = nn.GaussianSampleLayer(
			mu=self.phi_mu.output,
			log_sigma=self.phi_sigma.output,
			n_in = phi_1_struct.layer_dim[-1],
			batch_size = batch_size
		)
	   
		EC_z_dim = phi_1_struct.layer_dim[-1]
		self.EC_mu = self.phi_mu.output
		self.EC_log_sigma = self.phi_sigma.output
		self.EC_z = self.phi_z.output
		
		self.phi_1_params = self.phi_mu.params + self.phi_sigma.params
		self.phi_1_outputs = [self.EC_mu, self.EC_log_sigma, self.EC_z]
		self.phi_1_outputs_name = ["EC_mu", "EC_log_sigma", "EC_z"]
		
		
		#------------------------------------------------------------------------
		#Predict 1 Neural Network: E_q_\phi(z_n | x_n, y_n) (p_\theta(y_n | x_n, z_n))
		self.theta_xz_in = T.concatenate([input_x, self.EC_z], axis=1)
		theta_2_struct.activation[0] = T.nnet.sigmoid
		theta_2_struct.activation[1] = T.nnet.sigmoid
		theta_2_struct.activation[2] = T.nnet.sigmoid
		self.theta_2 = nn.Stacked_NN_2L(
			rng=rng,
			input=self.theta_xz_in,
			struct = theta_2_struct,
			name='Predict1'
		) 
		
		self.predict = nn.SoftmaxLayer(
			rng=rng,
			input=self.theta_2.output,
			n_in=theta_2_struct.layer_dim[-1],
			n_out=out_dim,
			name='Predict2'
		)		 
		
		
		PR_y_hat_dim = out_dim
		self.PR_theta_2 = self.theta_2.output
		self.PR_y_pred = self.predict.output
		
		
		# This predictor is to test training output, get the training accuracy
		self.predictor = T.argmax(self.PR_y_pred, axis=1, keepdims=False)
		
		self.theta_2_params = self.theta_2.params + self.predict.params
		self.theta_2_outputs = [self.PR_theta_2, self.PR_y_pred]
		self.theta_2_outputs_name = ["PR_theta_2", "PR_y_pred"]
		#------------------------------------------------------------------------
		# Error Function Set				
		# Cross entropy
		self.CE = T.nnet.categorical_crossentropy(self.PR_y_pred, label_y);
				
		#Cost function
		self.cost = self.CE.mean()
						
		# the parameters of the model
		self.params = self.phi_1_params + self.theta_2_params
		
		# all output of VAE
		self.outputs = self.phi_1_outputs + self.theta_2_outputs
		self.outputs_name = self.phi_1_outputs_name + self.theta_2_outputs_name

class Supervised_VAE_v3_KL_2L(object):
	def __init__(self, rng, input_x, label_y, batch_size,
				 phi_1_struct, theta_1_struct, theta_2_struct, in_dim, out_dim,
				 model=None):
		#------------------------------------------------------------------------
		#Encoder 1 Neural Network: present q_\phi(z_n | x_n, y_n)
		self.label_y = label_y
		phi_xy_in = T.concatenate([input_x, label_y], axis=1)
		
		phi_1_struct.activation[0] = T.nnet.sigmoid
		phi_1_struct.activation[1] = T.nnet.sigmoid
		phi_1_struct.activation[2] = None
		self.phi_mu = nn.Stacked_NN_2L(
			rng=rng,
			input=phi_xy_in,
			struct = phi_1_struct,
			name='Encoder1_mu',
		)
		self.phi_mu.HL_1.W.set_value(model.phi_mu.HL_1.W.get_value())
		self.phi_mu.HL_1.b.set_value(model.phi_mu.HL_1.b.get_value())
		self.phi_mu.HL_2.W.set_value(model.phi_mu.HL_2.W.get_value())
		self.phi_mu.HL_2.b.set_value(model.phi_mu.HL_2.b.get_value())
		self.phi_mu.OL.W.set_value(model.phi_mu.OL.W.get_value())
		self.phi_mu.OL.b.set_value(model.phi_mu.OL.b.get_value())
		
		phi_1_struct.activation[0] = T.nnet.sigmoid
		phi_1_struct.activation[1] = T.nnet.sigmoid
		phi_1_struct.activation[2] = None
		self.phi_sigma = nn.Stacked_NN_2L(
			rng=rng,
			input=phi_xy_in,
			struct = phi_1_struct,
			name='Encoder1_sigma',
		)
		self.phi_sigma.HL_1.W.set_value(model.phi_sigma.HL_1.W.get_value())
		self.phi_sigma.HL_1.b.set_value(model.phi_sigma.HL_1.b.get_value())
		self.phi_sigma.HL_2.W.set_value(model.phi_sigma.HL_2.W.get_value())
		self.phi_sigma.HL_2.b.set_value(model.phi_sigma.HL_2.b.get_value())
		self.phi_sigma.OL.W.set_value(model.phi_sigma.OL.W.get_value())
		self.phi_sigma.OL.b.set_value(model.phi_sigma.OL.b.get_value())
		
		#Sample layer
		self.phi_z = nn.GaussianSampleLayer(
			mu=self.phi_mu.output,
			log_sigma=self.phi_sigma.output,
			n_in = phi_1_struct.layer_dim[-1],
			batch_size = batch_size
		)
	   
		EC_z_dim = phi_1_struct.layer_dim[-1]
		self.EC_mu = self.phi_mu.output
		self.EC_log_sigma = self.phi_sigma.output
		self.EC_z = self.phi_z.output
		
		self.phi_1_params = self.phi_mu.params + self.phi_sigma.params
		self.phi_1_outputs = [self.EC_mu, self.EC_log_sigma, self.EC_z]
		self.phi_1_outputs_name = ["EC_mu", "EC_log_sigma", "EC_z"]
		
		#------------------------------------------------------------------------
		#Decoder 1 Neural Network: present p_\theta(z_n | x_n)
		theta_1_struct.activation[0] = T.nnet.sigmoid
		theta_1_struct.activation[1] = T.nnet.sigmoid
		theta_1_struct.activation[2] = None
		self.theta_mu = nn.Stacked_NN_2L(
			rng=rng,
			input=input_x,
			struct = theta_1_struct,
			name='Decoder1_mu'
		)
		theta_1_struct.activation[0] = T.nnet.sigmoid
		theta_1_struct.activation[1] = T.nnet.sigmoid
		theta_1_struct.activation[2] = None
		self.theta_sigma = nn.Stacked_NN_2L(
			rng=rng,
			input=input_x,
			struct = theta_1_struct,
			name='Decoder1_sigma'
		)  
		
		#Sample layer
		self.theta_z = nn.GaussianSampleLayer(
			mu=self.theta_mu.output,
			log_sigma=self.theta_sigma.output,
			n_in = theta_1_struct.layer_dim[-1],
			batch_size = batch_size
		)
	   
		DC_z_dim = phi_1_struct.layer_dim[-1]
		self.DC_mu = self.theta_mu.output
		self.DC_log_sigma = self.theta_sigma.output
		self.DC_z = self.theta_z.output
		
		self.theta_1_params = self.theta_mu.params + self.theta_sigma.params
		self.theta_1_outputs = [self.DC_mu, self.DC_log_sigma, self.DC_z]
		self.theta_1_outputs_name = ["DC_mu", "DC_log_sigma", "DC_z"]
		
		
							
		#------------------------------------------------------------------------
		# Error Function Set				
		# KL(q(z|x,y)||p(z|x)) -----------
		self.KL = er.KLGaussianGaussian(self.EC_mu, self.EC_log_sigma, self.DC_mu, self.DC_log_sigma)
		## Cross entropy
		#self.CE = T.nnet.categorical_crossentropy(self.PR_y_pred, label_y);
				
		#Cost function
		self.cost = -self.KL.mean()
						
		# the parameters of the model
		self.params = self.theta_1_params
		
		# all output of VAE
		self.outputs = self.theta_1_outputs
		self.outputs_name = self.theta_1_outputs_name
		
		#------------------------------------------------------------------------
		#Predict 1 Neural Network: E_q_\phi(z_n | x_n, y_n) (p_\theta(y_n | x_n, z_n))
		self.theta_xz_in = T.concatenate([input_x, self.EC_z], axis=1)
		theta_2_struct.activation[0] = T.nnet.sigmoid
		theta_2_struct.activation[1] = T.nnet.sigmoid
		theta_2_struct.activation[2] = T.nnet.sigmoid
		self.theta_2 = nn.Stacked_NN_2L(
			rng=rng,
			input=self.theta_xz_in,
			struct = theta_2_struct,
			name='Predict1',
		)
		self.theta_2.HL_1.W.set_value(model.theta_2.HL_1.W.get_value())
		self.theta_2.HL_1.b.set_value(model.theta_2.HL_1.b.get_value())
		self.theta_2.HL_2.W.set_value(model.theta_2.HL_2.W.get_value())
		self.theta_2.HL_2.b.set_value(model.theta_2.HL_2.b.get_value())
		self.theta_2.OL.W.set_value(model.theta_2.OL.W.get_value())
		self.theta_2.OL.b.set_value(model.theta_2.OL.b.get_value())
		
		self.predict = nn.SoftmaxLayer(
			rng=rng,
			input=self.theta_2.output,
			n_in=theta_2_struct.layer_dim[-1],
			n_out=out_dim,
			name='Predict2',
		)
		self.predict.W.set_value(model.predict.W.get_value())
		self.predict.b.set_value(model.predict.b.get_value())
		
		self.PR_y_pred = self.predict.output
		
		
		# This predictor is to test training output, get the training accuracy
		self.predictor = T.argmax(self.PR_y_pred, axis=1, keepdims=False)
		
		#------------------------------------------------------------------------
		#Test predict Neural Network
		self.test_xz = T.concatenate([input_x, self.DC_z], axis=1)
		self.test_nn = nn.Stacked_NN_2L(
			rng=rng,			
			input=self.test_xz,
			struct = theta_2_struct,
			name='Predict1_test',
		)
		self.test_nn.HL_1.W.set_value(model.theta_2.HL_1.W.get_value())
		self.test_nn.HL_1.b.set_value(model.theta_2.HL_1.b.get_value())
		self.test_nn.HL_2.W.set_value(model.theta_2.HL_2.W.get_value())
		self.test_nn.HL_2.b.set_value(model.theta_2.HL_2.b.get_value())
		self.test_nn.OL.W.set_value(model.theta_2.OL.W.get_value())
		self.test_nn.OL.b.set_value(model.theta_2.OL.b.get_value())
		
		self.test_predict = nn.SoftmaxLayer(
			rng=rng,			
			input=self.test_nn.output,
			n_in=theta_2_struct.layer_dim[-1],
			n_out=out_dim,
			name='Predict2_test',
		)
		self.test_predict.W.set_value(model.predict.W.get_value())
		self.test_predict.b.set_value(model.predict.b.get_value())
		
		self.PR_test = self.test_predict.output
		# This predictor is to test training output, get the training accuracy
		self.predictor_test = T.argmax(self.PR_test, axis=1, keepdims=False)
		
		

		
		
########################This version witout categorical sample###############################
class Supervised_VAE_v4(object):
	def __init__(self, rng, input_x, label_y, batch_size,
				 x_struct, phi_1_struct, theta_1_struct, theta_2_struct, in_dim, out_dim):
		"""Initialize the parameters for the multilayer perceptron

		:type rng: numpy.random.RandomState
		:param rng: a random number generator used to initialize weights

		:type input_source: theano.tensor.TensorType
		:param input: symbolic variable that describes the "Source Domain" input of the architecture (one minibatch)
		
		:type input_target: theano.tensor.TensorType
		:param input: symbolic variable that describes the "Target Domain" input of the architecture (one minibatch)		

		:type xxx_struct: class NN_struct
		:param xxx_strucat: define the structure of each NN
		"""
		#------------------------------------------------------------------------
		#Encoder 1 Neural Network: present q_\phi(z_n | x_n, y_n)
		self.label_y = label_y
		self.input_x = input_x
		
		
		
		self.NN_x = nn.Stacked_NN_0L(
			rng=rng,
			input=input_x,
			struct = x_struct,
			name='NN_x'
		)		 
		
		x_dim = x_struct.layer_dim[-1]
		self.x_hat = self.NN_x.output
		
		self.x_hat_params = self.NN_x.params
		self.x_hat_outputs = self.NN_x.output
		self.x_hat_outputs_name = ["x_hat"]
		
		phi_xy_in = T.concatenate([self.x_hat_outputs, label_y], axis=1)
		
		'''
		self.phi_1 = nn.Stacked_NN_0L(
			rng=rng,
			input=phi_xy_in,
			struct = phi_1_struct,
			name='Encoder1'
		)	 
		'''
		
		self.phi_mu = nn.Stacked_NN_0L(
			rng=rng,
			input=phi_xy_in,
			struct = phi_1_struct,
			name='Encoder1_mu'
		)		 

		self.phi_sigma = nn.Stacked_NN_0L(
			rng=rng,
			input=phi_xy_in,
			struct = phi_1_struct,
			name='Encoder1_sigma'
		)  
		
		
		#Sample layer
		self.phi_z = nn.GaussianSampleLayer(
			mu=self.phi_mu.output,
			log_sigma=self.phi_sigma.output,
			n_in = phi_1_struct.layer_dim[-1],
			batch_size = batch_size
		)
	   
		EC_z_dim = phi_1_struct.layer_dim[-1]
		self.EC_mu = self.phi_mu.output
		self.EC_log_sigma = self.phi_sigma.output
		self.EC_sigma = T.exp(self.EC_log_sigma)
		self.EC_z = self.phi_z.output
		
		self.phi_1_params = self.phi_mu.params + self.phi_sigma.params
		self.phi_1_outputs = [self.EC_mu, self.EC_log_sigma, self.EC_z]
		self.phi_1_outputs_name = ["EC_mu", "EC_log_sigma", "EC_z"]
		
		#------------------------------------------------------------------------
		#Decoder 1 Neural Network: present p_\theta(z_n | x_n)
		'''
		theta_1 = nn.NNLayer(
			rng=rng,			
			input_source=input_x,
			struct = theta_1_struct,
			name='Decoder1'
		) 
		'''
		self.theta_mu = nn.Stacked_NN_0L(
			rng=rng,			
			input=self.x_hat_outputs,
			struct = theta_1_struct,
			name='Decoder1_mu'
		)		 

		self.theta_sigma = nn.Stacked_NN_0L(
			rng=rng,			
			input=self.x_hat_outputs,
			struct = theta_1_struct,
			name='Decoder1_sigma'
		)  
		
		#Sample layer
		self.theta_z = nn.GaussianSampleLayer(
			mu=self.theta_mu.output,
			log_sigma=self.theta_sigma.output,
			n_in = theta_1_struct.layer_dim[-1],
			batch_size = batch_size
		)
	   
		DC_z_dim = phi_1_struct.layer_dim[-1]
		self.DC_mu = self.theta_mu.output
		self.DC_log_sigma = self.theta_sigma.output
		self.DC_sigma = T.exp(self.DC_log_sigma)
		self.DC_z = self.theta_z.output
		
		self.theta_1_params = self.theta_mu.params + self.theta_sigma.params
		self.theta_1_outputs = [self.DC_mu, self.DC_log_sigma, self.DC_z]
		self.theta_1_outputs_name = ["DC_mu", "DC_log_sigma", "DC_z"]
		
		#------------------------------------------------------------------------
		#Predict 1 Neural Network: E_q_\phi(z_n | x_n, y_n) (p_\theta(y_n | x_n, z_n))
		self.theta_xz_in = T.concatenate([self.x_hat_outputs, self.EC_z], axis=1)		
		self.theta_2 = nn.Stacked_NN_0L(
			rng=rng,			
			input=self.theta_xz_in,
			struct = theta_2_struct,
			name='Predict1'
		) 
		
		self.predict = nn.SoftmaxLayer(
			rng=rng,			
			input=self.theta_2.output,
			n_in=theta_2_struct.layer_dim[-1],
			n_out=out_dim,
			name='Predict2'
		)		 
		
		
		PR_y_hat_dim = out_dim
		self.PR_theta_2 = self.theta_2.output
		self.PR_y_pred = self.predict.output
		
		# This is test out put, get the test accuracy
		self.predictor = T.argmax(self.PR_y_pred, axis=1, keepdims=False)
		
		self.theta_2_params = self.theta_2.params + self.predict.params
		self.theta_2_outputs = [self.PR_theta_2, self.PR_y_pred]
		self.theta_2_outputs_name = ["PR_theta_2", "PR_y_pred"]
							
		#------------------------------------------------------------------------
		# Error Function Set				
		# KL(q(z|x,y)||p(z|x)) -----------
		self.KL = er.KLGaussianGaussian(self.EC_mu, self.EC_sigma, self.DC_mu, self.DC_sigma)
			 
		threshold = 0.0000001
		# Cross entropy
		self.CE = T.nnet.categorical_crossentropy(self.PR_y_pred, label_y);
				
		#Cost function
		self.cost = -self.KL.mean() + self.CE.mean()
						
		# the parameters of the model
		self.params = self.x_hat_params + self.phi_1_params + self.theta_1_params + self.theta_2_params
		
		# all output of VAE
		self.outputs = self.x_hat_outputs + self.phi_1_outputs + self.theta_1_outputs + self.theta_2_outputs
		self.outputs_name = self.x_hat_outputs_name + self.phi_1_outputs_name + self.theta_1_outputs_name + self.theta_2_outputs_name

		
########################This version witout categorical sample###############################
class Supervised_VAE_test(object):
	def __init__(self, rng, input_x, label_y, batch_size, \
				theta_2_struct, hidden_z, in_dim, out_dim, param):
		self.label_y = label_y
		self.hidden_z = hidden_z
		#------------------------------------------------------------------------
		#Predict Neural Network: E_q_\phi(z_n | x_n, y_n) (p_\theta(y_n | x_n, z_n))
		self.xz_in = T.concatenate([input_x, hidden_z], axis=1)
		self.theta_2 = nn.Stacked_NN_0L(
			rng=rng,			
			input=self.xz_in,
			struct = theta_2_struct,
			name='Predict1',
			W=param[8],
			b=param[9]
		) 
		
		self.predict = nn.SoftmaxLayer(
			rng=rng,			
			input=self.theta_2.output,
			n_in=theta_2_struct.layer_dim[-1],
			n_out=out_dim,
			name='Predict2',
			W=param[10],
			b=param[11]
		)		 
		
		
		PR_y_dim = out_dim
		self.PR_theta_2 = self.theta_2.output
		self.PR_predict = self.predict.output
		
		
		# This is test out put, get the test accuracy
		self.predictor = T.argmax(self.PR_predict, axis=1, keepdims=False)
		
		self.predict_params = self.theta_2.params + self.predict.params 
		self.predict_outputs = [self.PR_theta_2, self.PR_predict]
		self.predict_outputs_name = ["PR_theta_2", "PR_predict"]
							
		#------------------------------------------------------------------------
		
		# all output
		self.outputs = self.PR_theta_2 + self.predict_outputs
		self.outputs_name = self.predict_outputs_name

		
		
########################This version witout categorical sample###############################
class Supervised_VAE_test_v2(object):
	def __init__(self, rng, input_x, label_y, batch_size, \
				theta_1_struct, theta_2_struct, \
				in_dim, out_dim, param):
		
		#------------------------------------------------------------------------
		#Decoder 1 Neural Network: present p_\theta(z_n | x_n)
		self.label_y = label_y
		
		#print(input_x.eval())
		
		self.theta_mu = nn.Stacked_NN_0L(
			rng=rng,			
			input=input_x,
			struct = theta_1_struct,
			name='Decoder1_mu',
			W=param[4],
			b=param[5]
		)		 

		self.theta_sigma = nn.Stacked_NN_0L(
			rng=rng,			
			input=input_x,
			struct = theta_1_struct,
			name='Decoder1_sigma',
			W=param[6],
			b=param[7]
		)  
		
		#Sample layer
		self.theta_z = nn.GaussianSampleLayer(
			mu=self.theta_mu.output,
			log_sigma=self.theta_sigma.output,
			n_in = theta_1_struct.layer_dim[-1],
			batch_size = batch_size
		)
			
		DC_z_dim = theta_1_struct.layer_dim[-1]
		self.DC_mu = self.theta_mu.output
		self.DC_log_sigma = self.theta_sigma.output
		self.DC_sigma = T.exp(self.DC_log_sigma)
		self.DC_z = self.theta_z.output
		
		self.theta_1_params = self.theta_mu.params + self.theta_sigma.params
		self.theta_1_outputs = [self.DC_mu, self.DC_log_sigma, self.DC_z]
		self.theta_1_outputs_name = ["DC_mu", "DC_log_sigma", "DC_z"]

		#------------------------------------------------------------------------
		#Predict Neural Network: E_q_\phi(z_n | x_n, y_n) (p_\theta(y_n | x_n, z_n))
		self.xz_in = T.concatenate([input_x, self.DC_z], axis=1)
		self.theta_2 = nn.Stacked_NN_0L(
			rng=rng,			
			input=self.xz_in,
			struct = theta_2_struct,
			name='Predict1',
			W=param[8],
			b=param[9]
		) 
		
		self.predict = nn.SoftmaxLayer(
			rng=rng,			
			input=self.theta_2.output,
			n_in=theta_2_struct.layer_dim[-1],
			n_out=out_dim,
			name='Predict2',
			W=param[10],
			b=param[11]
		)		 
		
		
		PR_y_dim = out_dim
		self.PR_theta_2 = self.theta_2.output
		self.PR_predict = self.predict.output
		
		
		# This is test out put, get the test accuracy
		self.predictor = T.argmax(self.PR_predict, axis=1, keepdims=False)
		
		self.predict_params = self.theta_2.params + self.predict.params 
		self.predict_outputs = [self.PR_theta_2, self.PR_predict]
		self.predict_outputs_name = ["PR_theta_2", "PR_predict"]
							
		#------------------------------------------------------------------------
		
		# all output
		self.outputs = self.theta_1_outputs + self.predict_outputs
		self.outputs_name = self.theta_1_outputs_name + self.predict_outputs_name

		
		
########################This version witout categorical sample###############################
class Supervised_VAE_test_v1(object):
	def __init__(self, rng, input_x, label_y, batch_size, \
				phi_1_struct, theta_2_struct, \
				in_dim, out_dim, param):
		
		#------------------------------------------------------------------------
		#Encoder 1 Neural Network: present q_\phi(z_n | x_n, y_n)
		self.label_y = label_y
		phi_xy_in = T.concatenate([input_x, label_y], axis=1)
		
		
		self.phi_mu = nn.Stacked_NN_0L(
			rng=rng,
			input=phi_xy_in,
			struct = phi_1_struct,
			name='Encoder1_mu',
			W=param[0],
			b=param[1]
		)		 

		self.phi_sigma = nn.Stacked_NN_0L(
			rng=rng,
			input=phi_xy_in,
			struct = phi_1_struct,
			name='Encoder1_sigma',
			W=param[2],
			b=param[3]
		)  
		
		
		#Sample layer
		self.phi_z = nn.GaussianSampleLayer(
			mu=self.phi_mu.output,
			log_sigma=self.phi_sigma.output,
			n_in = phi_1_struct.layer_dim[-1],
			batch_size = batch_size
		)
	   
		EC_z_dim = phi_1_struct.layer_dim[-1]
		self.EC_mu = self.phi_mu.output
		self.EC_log_sigma = self.phi_sigma.output
		self.EC_sigma = T.exp(self.EC_log_sigma)
		self.EC_z = self.phi_z.output
		
		self.phi_1_params = self.phi_mu.params + self.phi_sigma.params
		self.phi_1_outputs = [self.EC_mu, self.EC_log_sigma, self.EC_z]
		self.phi_1_outputs_name = ["EC_mu", "EC_log_sigma", "EC_z"]
		
		#------------------------------------------------------------------------
		#Predict 1 Neural Network: E_q_\phi(z_n | x_n, y_n) (p_\theta(y_n | x_n, z_n))
		self.theta_xz_in = T.concatenate([input_x, self.EC_z], axis=1)		
		self.theta_2 = nn.Stacked_NN_0L(
			rng=rng,			
			input=self.theta_xz_in,
			struct = theta_2_struct,
			name='Predict1',
			W=param[8],
			b=param[9]
		) 
		
		self.predict = nn.SoftmaxLayer(
			rng=rng,			
			input=self.theta_2.output,
			n_in=theta_2_struct.layer_dim[-1],
			n_out=out_dim,
			name='Predict2',
			W=param[10],
			b=param[11]
		)		 
		
		
		PR_y_hat_dim = out_dim
		self.PR_theta_2 = self.theta_2.output
		self.PR_y_pred = self.predict.output
		
		# This is test out put, get the test accuracy
		self.predictor = T.argmax(self.PR_y_pred, axis=1, keepdims=False)
		
		self.theta_2_params = self.theta_2.params + self.predict.params
		self.theta_2_outputs = [self.PR_theta_2, self.PR_y_pred]
		self.theta_2_outputs_name = ["PR_theta_2", "PR_y_pred"]
							
		
		# the parameters of the model
		self.params = self.phi_1_params + self.theta_2_params
		
		# all output of VAE
		self.outputs = self.phi_1_outputs + self.theta_2_outputs
		self.outputs_name = self.phi_1_outputs_name + self.theta_2_outputs_name
		
		
		
		
class Supervised_VAE_test_v3(object):
	def __init__(self, rng, input_x, label_y, batch_size, \
				phi_1_struct, theta_1_struct, theta_2_struct, \
				in_dim, out_dim, param):
		#------------------------------------------------------------------------
		#Encoder 1 Neural Network: present q_\phi(z_n | x_n, y_n)
		self.label_y = label_y
		phi_xy_in = T.concatenate([input_x, label_y], axis=1)
		
		
		self.phi_mu = nn.Stacked_NN_0L(
			rng=rng,
			input=phi_xy_in,
			struct = phi_1_struct,
			name='Encoder1_mu',
			W=param[2],
			b=param[3]
		)		 

		self.phi_sigma = nn.Stacked_NN_0L(
			rng=rng,
			input=phi_xy_in,
			struct = phi_1_struct,
			name='Encoder1_sigma',
			W=param[4],
			b=param[5]
		)  
		
		
		#Sample layer
		self.phi_z = nn.GaussianSampleLayer(
			mu=self.phi_mu.output,
			log_sigma=self.phi_sigma.output,
			n_in = phi_1_struct.layer_dim[-1],
			batch_size = batch_size
		)
	   
		EC_z_dim = phi_1_struct.layer_dim[-1]
		self.EC_mu = self.phi_mu.output
		self.EC_log_sigma = self.phi_sigma.output
		self.EC_sigma = T.exp(self.EC_log_sigma)
		self.EC_z = self.phi_z.output
		
		self.phi_1_params = self.phi_mu.params + self.phi_sigma.params
		self.phi_1_outputs = [self.EC_mu, self.EC_log_sigma, self.EC_z]
		self.phi_1_outputs_name = ["EC_mu", "EC_log_sigma", "EC_z"]
		
		#------------------------------------------------------------------------
		#Decoder 1 Neural Network: present p_\theta(z_n | x_n)
		self.label_y = label_y
		
		#print(input_x.eval())
		
		self.theta_mu = nn.Stacked_NN_0L(
			rng=rng,			
			input=input_x,
			struct = theta_1_struct,
			name='Decoder1_mu',
			W=param[6],
			b=param[7]
		)		 

		self.theta_sigma = nn.Stacked_NN_0L(
			rng=rng,			
			input=input_x,
			struct = theta_1_struct,
			name='Decoder1_sigma',
			W=param[8],
			b=param[9]
		)  
		
		#Sample layer
		self.theta_z = nn.GaussianSampleLayer(
			mu=self.theta_mu.output,
			log_sigma=self.theta_sigma.output,
			n_in = theta_1_struct.layer_dim[-1],
			batch_size = batch_size
		)
			
		DC_z_dim = theta_1_struct.layer_dim[-1]
		self.DC_mu = self.theta_mu.output
		self.DC_log_sigma = self.theta_sigma.output
		self.DC_sigma = T.exp(self.DC_log_sigma)
		self.DC_z = self.theta_z.output
		
		self.theta_1_params = self.theta_mu.params + self.theta_sigma.params
		self.theta_1_outputs = [self.DC_mu, self.DC_log_sigma, self.DC_z]
		self.theta_1_outputs_name = ["DC_mu", "DC_log_sigma", "DC_z"]

		#------------------------------------------------------------------------
		#Predict Neural Network: E_q_\phi(z_n | x_n, y_n) (p_\theta(y_n | x_n, z_n))
		self.xz_in = T.concatenate([input_x, self.DC_z], axis=1)
		self.theta_2 = nn.Stacked_NN_0L(
			rng=rng,			
			input=self.xz_in,
			struct = theta_2_struct,
			name='Predict1',
			W=param[10],
			b=param[11]
		) 
		
		self.predict = nn.SoftmaxLayer(
			rng=rng,			
			input=self.theta_2.output,
			n_in=theta_2_struct.layer_dim[-1],
			n_out=out_dim,
			name='Predict2',
			W=param[12],
			b=param[13]
		)		 
		
		
		PR_y_dim = out_dim
		self.PR_theta_2 = self.theta_2.output
		self.PR_predict = self.predict.output
		
		
		# This is test out put, get the test accuracy
		self.predictor = T.argmax(self.PR_predict, axis=1, keepdims=False)
		
		self.predict_params =  self.theta_2.params + self.predict.params 
		self.predict_outputs = [self.PR_theta_2, self.PR_predict]
		self.predict_outputs_name = [ "PR_theta_2", "PR_predict"]
							
		#------------------------------------------------------------------------
		
		# all output
		self.outputs = self.phi_1_outputs + self.theta_1_outputs + self.predict_outputs
		self.outputs_name = self.phi_1_outputs_name + self.theta_1_outputs_name + self.predict_outputs_name
'''		
########################This version witout categorical sample###############################
class Supervised_VAE_test(object):
	def __init__(self, rng, input_x, label_y, batch_size, \
				phi_1_struct, theta_1_struct, theta_2_struct, \
				in_dim, out_dim, obj):
		
		#------------------------------------------------------------------------
		#Encoder 1 Neural Network: present q_\phi(z_n | x_n, y_n)
		self.label_y = label_y
		phi_xy_in = T.concatenate([input_x, label_y], axis=1)
		
		
		self.phi_mu = nn.Stacked_NN_0L(
			rng=rng,
			input=phi_xy_in,
			struct = phi_1_struct,
			name='Encoder1_mu',
			W=param[0],
			b=param[1]
		)		 

		self.phi_sigma = nn.Stacked_NN_0L(
			rng=rng,
			input=phi_xy_in,
			struct = phi_1_struct,
			name='Encoder1_sigma',
			W=param[2],
			b=param[3]
		)  
		
		
		#Sample layer
		self.phi_z = nn.GaussianSampleLayer(
			mu=self.phi_mu.output,
			log_sigma=self.phi_sigma.output,
			n_in = phi_1_struct.layer_dim[-1],
			batch_size = batch_size
		)
	   
		EC_z_dim = phi_1_struct.layer_dim[-1]
		self.EC_mu = self.phi_mu.output
		self.EC_log_sigma = self.phi_sigma.output
		self.EC_sigma = T.exp(self.EC_log_sigma)
		self.EC_z = self.phi_z.output
		
		self.phi_1_params = self.phi_mu.params + self.phi_sigma.params
		self.phi_1_outputs = [self.EC_mu, self.EC_log_sigma, self.EC_z]
		self.phi_1_outputs_name = ["EC_mu", "EC_log_sigma", "EC_z"]
		
		#------------------------------------------------------------------------
		#Predict 1 Neural Network: E_q_\phi(z_n | x_n, y_n) (p_\theta(y_n | x_n, z_n))
		self.theta_xz_in = T.concatenate([input_x, self.EC_z], axis=1)		
		self.theta_2 = nn.Stacked_NN_0L(
			rng=rng,			
			input=self.theta_xz_in,
			struct = theta_2_struct,
			name='Predict1',
			W=param[8],
			b=param[9]
		) 
		
		self.predict = nn.SoftmaxLayer(
			rng=rng,			
			input=self.theta_2.output,
			n_in=theta_2_struct.layer_dim[-1],
			n_out=out_dim,
			name='Predict2',
			W=param[10],
			b=param[11]
		)		 
		
		
		PR_y_hat_dim = out_dim
		self.PR_theta_2 = self.theta_2.output
		self.PR_y_pred = self.predict.output
		
		# This is test out put, get the test accuracy
		self.predictor = T.argmax(self.PR_y_pred, axis=1, keepdims=False)
		
		self.theta_2_params = self.theta_2.params + self.predict.params
		self.theta_2_outputs = [self.PR_theta_2, self.PR_y_pred]
		self.theta_2_outputs_name = ["PR_theta_2", "PR_y_pred"]
							
		
		# the parameters of the model
		self.params = self.phi_1_params + self.theta_2_params
		
		# all output of VAE
		self.outputs = self.phi_1_outputs + self.theta_2_outputs
		self.outputs_name = self.phi_1_outputs_name + self.theta_2_outputs_name
		'''