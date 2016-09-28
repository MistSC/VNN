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
import criteria as er
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
                 phi_1_struct, theta_1_struct,
                 in_dim, out_dim):
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
        #Encoder 1 Neural Network: present q_\phi(z_n | x_n )
        self.input_x = input_x
        self.label_y = label_y

        self.phi_mu = nn.Stacked_NN_0L(
            rng=rng,
            input=self.input_x,
            struct = phi_1_struct,
            name='Encoder1_mu'
        )        

        self.phi_sigma = nn.Stacked_NN_0L(
            rng=rng,
            input=self.input_x,
            struct = phi_1_struct,
            name='Encoder1_sigma'
        )         
        
        #Sample layer
        #sample z from q_\phi(z_n | x_n)
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
        #Decoder 1 Neural Network: present p_\theta(y_n | z_n)
        self.theta = nn.Stacked_NN_0L(
            rng=rng,            
            input=self.EC_z,
            struct = theta_1_struct,
            name='Decoder1'
        )        
        self.y_hat = nn.SoftmaxLayer(
            rng=rng,
            input=self.theta.output,
            n_in=theta_1_struct.layer_dim[-1],
            n_out=out_dim,
            name='Predict'
        )        
        
        PR_y_dim = out_dim
        self.PR_y_hat = self.y_hat.output
        
        self.theta_1_params = self.theta.params + self.y_hat.params
        self.theta_1_outputs = [self.PR_y_hat]
        self.theta_1_outputs_name = ["PR_y_hat"]         
        
        
        # This is test output, get the test accuracy
        self.predictor = T.argmax(self.PR_y_hat, axis=1, keepdims=False)
        #------------------------------------------------------------------------
        # Error Function Set                
        # KL(q(z|x)||p(z)): p(z)~N(0,I) prior----------
        self.KL = er.KLGaussianStdGaussian(self.EC_mu, self.EC_sigma)

        # Cross entropy
        self.CE = T.nnet.categorical_crossentropy(self.PR_y_hat, label_y);
                
        #Cost function
        self.cost = -self.KL.mean() + self.CE.mean()
                        
        # the parameters of the model
        self.params = self.phi_1_params + self.theta_1_params
        
        # all output of VAE
        self.outputs = self.phi_1_outputs + self.theta_1_outputs
        self.outputs_name = self.phi_1_outputs_name + self.theta_1_outputs_name
        
        
        
class Unsupervised_VAE(object): 
    """
    The semi-supervised model Domain-Adversial Variational Autoencoder
    To deal with the semi-supervised model that source, target domain data will walk though same path. Use shared layer idea by copy the weight
    The domain label s will constuct inside this class
    For abbreviation: HL refer to hiddenlayer, GSL refer to Gaussian Sample Layer, CSL refer to Cat Sample Layer
    Encoder refer to Encoder NN, Decoder refer to Decoder NN    
    """

    def __init__(self, rng, input_x, label_y, batch_size,
                 phi_1_struct, theta_1_struct,
                 in_dim, out_dim):
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
        #Encoder 1 Neural Network: present q_\phi(z_n | x_n )
        self.input_x = input_x
        self.label_y = label_y

        self.phi_mu = nn.Stacked_NN_0L(
            rng=rng,
            input=self.input_x,
            struct = phi_1_struct,
            name='Encoder1_mu'
        )        

        self.phi_sigma = nn.Stacked_NN_0L(
            rng=rng,
            input=self.input_x,
            struct = phi_1_struct,
            name='Encoder1_sigma'
        )         
        
        #Sample layer
        #sample z from q_\phi(z_n | x_n)
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
        #Decoder 1 Neural Network: present p_\theta(x_n | z_n)
        self.theta = nn.Stacked_NN_0L(
            rng=rng,            
            input=self.EC_z,
            struct = theta_1_struct,
            name='Decoder1'
        )        
        self.x_hat = nn.NNLayer(
            rng=rng,
            input=self.theta.output,
            n_in=theta_1_struct.layer_dim[-1],
            n_out=out_dim,
            name='Predict'
        )        
        
        PR_x_dim = out_dim
        self.PR_x_hat = self.x_hat.output
        
        self.theta_1_params = self.theta.params + self.x_hat.params
        self.theta_1_outputs = [self.PR_x_hat]
        self.theta_1_outputs_name = ["PR_x_hat"]         
        
        
        # This is test output, get the test accuracy
        #self.predictor = T.argmax(self.PR_y_hat, axis=1, keepdims=False)
        self.predictor = self.PR_x_hat
        #------------------------------------------------------------------------
        # Error Function Set                
        # KL(q(z|x)||p(z)): p(z)~N(0,I) prior----------
        self.KL = er.KLGaussianStdGaussian(self.EC_mu, self.EC_sigma)

        # Cross entropy
        self.CE = T.nnet.binary_crossentropy(self.PR_x_hat, input_x);
                
        #Cost function
        self.cost = - self.KL.mean() + self.CE.mean()
        #self.cost = - self.KL.sum(axis=1) + self.CE.sum(axis=1)

        # the parameters of the model
        self.params = self.phi_1_params + self.theta_1_params
        
        # all output of VAE
        self.outputs = self.phi_1_outputs + self.theta_1_outputs
        self.outputs_name = self.phi_1_outputs_name + self.theta_1_outputs_name

class Unsupervised_VAE_xy(object): 
    """
    The semi-supervised model Domain-Adversial Variational Autoencoder
    To deal with the semi-supervised model that source, target domain data will walk though same path. Use shared layer idea by copy the weight
    The domain label s will constuct inside this class
    For abbreviation: HL refer to hiddenlayer, GSL refer to Gaussian Sample Layer, CSL refer to Cat Sample Layer
    Encoder refer to Encoder NN, Decoder refer to Decoder NN    
    """

    def __init__(self, rng, input_x, label_y, batch_size,
                 phi_1_struct, theta_1_struct,
                 in_dim, out_dim):
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
        #Encoder 1 Neural Network: present q_\phi(z_n | x_n, y_n )
        self.input_x = input_x
        self.label_y = label_y
        self.input = T.concatenate([input_x, label_y], axis=1)

        self.phi_mu = nn.Stacked_NN_0L(
            rng=rng,
            input=self.input,
            struct = phi_1_struct,
            name='Encoder1_mu'
        )        

        self.phi_sigma = nn.Stacked_NN_0L(
            rng=rng,
            input=self.input,
            struct = phi_1_struct,
            name='Encoder1_sigma'
        )         
        
        #Sample layer
        #sample z from q_\phi(z_n | x_n, y_n)
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
        #Decoder 1 Neural Network: present p_\theta(x_n, y_n | z_n)
        self.theta = nn.Stacked_NN_0L(
            rng=rng,            
            input=self.EC_z,
            struct = theta_1_struct,
            name='Decoder1'
        )        
        self.x_hat = nn.NNLayer(
            rng=rng,
            input=self.theta.output,
            n_in=theta_1_struct.layer_dim[-1],
            n_out=out_dim,
            name='Predict'
        )        
        
        PR_x_dim = out_dim
        self.PR_x_hat = self.x_hat.output
        
        self.theta_1_params = self.theta.params + self.x_hat.params
        self.theta_1_outputs = [self.PR_x_hat]
        self.theta_1_outputs_name = ["PR_x_hat"]         
        
        
        # This is test output, get the test accuracy
        #self.predictor = T.argmax(self.PR_y_hat, axis=1, keepdims=False)
        self.predictor = self.PR_x_hat
        #------------------------------------------------------------------------
        # Error Function Set                
        # KL(q(z|x)||p(z)): p(z)~N(0,I) prior----------
        self.KL = er.KLGaussianStdGaussian(self.EC_mu, self.EC_sigma)

        # Cross entropy
        self.CE = T.nnet.binary_crossentropy(self.PR_x_hat, input_x);
                
        #Cost function
        self.cost = - self.KL.mean() + self.CE.mean()
        #self.cost = - self.KL.sum(axis=1) + self.CE.sum(axis=1)

        # the parameters of the model
        self.params = self.phi_1_params + self.theta_1_params
        
        # all output of VAE
        self.outputs = self.phi_1_outputs + self.theta_1_outputs
        self.outputs_name = self.phi_1_outputs_name + self.theta_1_outputs_name