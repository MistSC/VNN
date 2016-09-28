#%matplotlib inline
from __future__ import print_function
import matplotlib.pyplot as plt
from collections import OrderedDict
from six.moves import cPickle

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
import SVAE

def KLKL(mu1,log_sigma1,mu2,log_sigma2):
	KL = np.sum(0.5 * (2 * log_sigma2 - 2 * log_sigma1 + (np.exp(log_sigma1)**2 + (mu1 - mu2)**2) / np.exp(log_sigma2)**2 - 1), axis=1)

def get_acc(pred, true):
	ll = pred - true
	ll = np.array(ll)
	acc = 1 - (np.nonzero(ll)[0].shape[0])/float(ll.shape[0])
	return acc
	
def train_svae_mnist():
	'''Load Data'''
	train_file = 'train_1ok.npy'
	valid_file = 'valid_1ok.npy'
	test_file = 'test_1ok.npy'
	
	train=np.load(train_file)
	valid=np.load(valid_file)
	test=np.load(test_file)
	
	#train_list=np.load('train.npy')[1]
	
	
	train_feat, train_label = util.shared_dataset(train)
	valid_feat, valid_label = util.shared_dataset(valid)
	test_feat, test_label = util.shared_dataset(test)
	
  
	'''Coefficient Initial'''		 
	batch_size = 50
	epsilon_std = 0.01
	n_epochs = 500
	learning_rate = 0.0001
	
	n_train_batches = train_feat.get_value(borrow=True).shape[0] // batch_size
	n_valid_batches = valid_feat.get_value(borrow=True).shape[0] // batch_size
	n_test_batches = test_feat.get_value(borrow=True).shape[0] // batch_size
	print('number of minibatch at one epoch: train	%i, validation %i, test %i' %
		(n_train_batches, n_valid_batches, n_test_batches))
	
	z_dim = 5 #dimension of latent variable 
	x_dim = train_feat.get_value(borrow=True).shape[1]
	#x_dim = 100
	y_dim = train_label.get_value(borrow=True).shape[1]
	activation = None
	
	print(train_feat.get_value(borrow=True).shape[0])
	print(train_label.get_value(borrow=True).shape[0])
	print(train_feat.get_value(borrow=True).shape[1])
	print(train_label.get_value(borrow=True).shape[1])
	'''
	x_struct=nn.NN_struct()
	x_struct.layer_dim = [x_dim, x_dim]
	x_struct.activation = [activation]		
	'''
	phi_1_struct=nn.NN_struct()
	phi_1_struct.layer_dim = [x_dim+y_dim, z_dim]
	phi_1_struct.activation = [activation]
	
	theta_1_struct=nn.NN_struct()
	theta_1_struct.layer_dim = [x_dim, z_dim]
	theta_1_struct.activation = [activation]
	
	theta_2_struct=nn.NN_struct()
	theta_2_struct.layer_dim = [z_dim+x_dim, y_dim]
	theta_2_struct.activation = [activation]		
	
	
	
	#theta_pi_struct=nn.NN_struct()
	#theta_pi_struct.layer_dim = [y_dim, y_dim]
	#theta_pi_struct.activation = [activation] 
	
	######################
	# BUILD ACTUAL MODEL #
	######################
	print('... building the model')
	
	
	# allocate symbolic variables for the data
	#index_source = T.lscalar()	 # index to a [mini]batch
	#index_target = T.lscalar()	 # index to a [mini]batch
	index = T.lscalar()	 # index to a [mini]batch
	x = T.matrix('x')  # the data is presented as rasterized images
	y = T.matrix('y')  # the labels are presented as signal vector	   
	
	rng = np.random.RandomState(1234)
		
	# construct the DAVAE class
   
	classifier = SVAE.Supervised_VAE_v3(
		rng=rng,
		input_x = x,
		label_y = y,
		batch_size = batch_size,
		phi_1_struct = phi_1_struct,
		theta_1_struct = theta_1_struct,
		theta_2_struct = theta_2_struct,
		in_dim = x_dim,
		out_dim = y_dim,
		)
	'''
	classifier_test = SVAE.Supervised_VAE_test_v3(
		rng=rng,
		input_x = x,
		label_y = y,
		batch_size = batch_size,
		phi_1_struct = phi_1_struct,
		theta_1_struct = theta_1_struct,
		theta_2_struct = theta_2_struct,
		in_dim = x_dim,
		out_dim = y_dim,
		param = classifier.params,
		)
		
	'''
	
	cost = (classifier.cost)
		
	gparams = [T.grad(cost, param) for param in classifier.params]
				   
	updates = [
		(param, param - learning_rate * gparam)
		for param, gparam in zip(classifier.params, gparams)
	]
	
	
	
	validate_model = theano.function(
		inputs=[index],
		outputs=classifier.cost,
		givens={
			x: valid_feat[index * batch_size : (index + 1) * batch_size, :],
			y: valid_label[index * batch_size : (index + 1) * batch_size, :]
		}		 
	)				 
	
	
	train_model = theano.function(
		inputs=[index],
		outputs=[classifier.cost, classifier.KL, classifier.CE, classifier.predictor, classifier.label_y, \
				classifier.EC_z, classifier.DC_z, classifier.theta_2.params[0], classifier.theta_2.params[1], \
				classifier.predict.params[0], classifier.predict.params[1]],
		updates=updates,
		givens={
			x: train_feat[index * batch_size : (index + 1) * batch_size, :],
			y: train_label[index * batch_size : (index + 1) * batch_size, :]
		}		
	)	
	
	'''
	#z = T.matrix('z')
	classifier_test = SVAE.Supervised_VAE_test(
		rng=rng,
		input_x = x,
		label_y = y,
		batch_size = batch_size,
		theta_2_struct = theta_2_struct,
		hidden_z = classifier.EC_z,
		in_dim = x_dim,
		out_dim = y_dim,
		param = classifier.params,
		)
	
	test_model = theano.function(
		inputs=[index],
		outputs=[classifier_test.predictor, classifier_test.label_y, classifier_test.hidden_z], #, classifier_test.EC_z, classifier_test.DC_z],
		givens={
			x: test_feat[index * batch_size : (index + 1) * batch_size, :],
			y: test_label[index * batch_size : (index + 1) * batch_size, :],
		}		 
	)
	'''
	
	###############
	# TRAIN MODEL #
	###############
	'''
	Define :
		xx_loss : Cost function value
		xx_score : Classification accuracy rate
	'''		   
	
	print('... training')
	
	# early-stopping parameters
	patience = 10000  # look as this many examples regardless
	patience_increase = 2  # wait this much longer when a new best is
						   # found
	improvement_threshold = 0.995  # a relative improvement of this much is
								   # considered significant
	validation_frequency = min(n_train_batches, patience // 2)
								  # go through this many
								  # minibatche before checking the network
								  # on the validation set; in this case we
								  # check every epoch
	
	#validation_frequency = n_train_batches
	
	best_iter = 0
	best_train_loss = np.inf
	best_validation_loss = np.inf  
	test_loss = np.inf
	train_score = 0.
	validation_score = 0.
	test_score = 0.	   
	start_time = timeit.default_timer()

	epoch = 0
	done_looping = False

	kl_store=[]
	ce_store=[]
	while (epoch < n_epochs) and (not done_looping):
		epoch = epoch + 1
		train_acc=[]
		
		for minibatch_index in range(n_train_batches):

			[minibatch_avg_cost, KL_loss, CE_loss, pred, lab, EC_z, DC_z, \
			theta_2_params_w, theta_2_params_b ,preidct_params_w, preidct_params_b] \
			= train_model(minibatch_index)
			#print(minibatch_index)
						
			# iteration number
			iter = (epoch - 1) * n_train_batches + minibatch_index
			
			train_acc.append(get_acc(pred,np.nonzero(lab)[1]))
			
			
			if (iter + 1) % validation_frequency == 0:
				# compute loss on validation set
				validation_losses = [validate_model(i) for i in range(n_valid_batches)]	
				this_validation_loss = np.mean(validation_losses)
				
				#print(classifier.params)
				#print(test_EC_z[-1])
				#print(test_DC_z[-1])
				print('KL loss: %f, CE loss: %f' % (np.mean(KL_loss),np.mean(CE_loss)))
				kl_store.append(np.mean(KL_loss))
				ce_store.append(np.mean(CE_loss))
				print('epoch training accuracy: %f' \
					% (np.mean(np.array(train_acc))))
				print(
					'epoch %i, minibatch %i/%i, validation loss %f %%' %
					(
						epoch,
						minibatch_index + 1,
						n_train_batches,
						this_validation_loss
					)
				)

				# if we got the best validation score until now
				if this_validation_loss < best_validation_loss:
					#improve patience if loss improvement is good enough
					if (
						this_validation_loss < best_validation_loss *
						improvement_threshold
					):
						patience = max(patience, iter * patience_increase)

					best_validation_loss = this_validation_loss   
					best_iter = iter

					# get training accuracy
					print('best training accuracy: %f' % (np.mean(np.array(train_acc))))
					# test it on the test set
					#test_losses = [test_model(i) for i in range(n_test_batches)]
					#test_score = np.mean(test_losses)

					print(('	 epoch %i, minibatch %i/%i, best train accuracy: %f') % \
						  (epoch, minibatch_index + 1, n_train_batches, \
						   np.mean(np.array(train_acc))))

			if patience <= iter:
				done_looping = True
				break
		'''		
		#shared_z = theano.shared(EC_z, "shared_z")
		classifier_test = SVAE.Supervised_VAE_test_v3(
			rng=rng,
			input_x = x,
			label_y = y,
			batch_size = batch_size,
			phi_1_struct = phi_1_struct, 
			theta_1_struct = theta_1_struct, 
			theta_2_struct = theta_2_struct,
			in_dim = x_dim,
			out_dim = y_dim,
			param = classifier.params,
			)
	
		test_model = theano.function(
			inputs=[index],
			outputs=[classifier_test.predictor, classifier_test.label_y, classifier_test.EC_z], #, classifier_test.EC_z, classifier_test.DC_z],
			givens={
				x: test_feat[index * batch_size : (index + 1) * batch_size, :],
				y: test_label[index * batch_size : (index + 1) * batch_size, :],
			}		 
		)	
		test_acc=[]
		for i in range(n_test_batches):
			[test_pred, test_lab, test_EC_z] = test_model(i)
			test_acc.append(get_acc(test_pred,np.nonzero(test_lab)[1]))
		'''		
		#print('~~~~~~~~~~~~~~~test z:~~~~~~~~~~~~~~~~')
		#print(test_EC_z[-1])
		print('===============train z================')		
		#print(EC_z[-2])
		print(EC_z[-1])
		print(DC_z[-1])
		#print('test accuracy: %f' % np.mean(np.array(test_acc)))
	
	np.save('kl.npy',np.array(kl_store))
	np.save('ce.npy',np.array(ce_store))
	#f = open('svae_train_obj.save', 'wb')
	#cPickle.dump(classifier, f, protocol=cPickle.HIGHEST_PROTOCOL)
	#f.close()

	'''
	####################
	# BUILD TEST MODEL #
	####################
	print('... building test model')
	
	
	# allocate symbolic variables for the data
	#index_source = T.lscalar()	 # index to a [mini]batch
	#index_target = T.lscalar()	 # index to a [mini]batch
	index = T.lscalar()	 # index to a [mini]batch
	x = T.matrix('x')  # the data is presented as rasterized images
	y = T.matrix('y')  # the labels are presented as signal vector
	
	rng = np.random.RandomState(1234)
		
	# construct the DAVAE class
   
	classifier = SVAE.Supervised_VAE_v1(
		rng=rng,
		input_x = x,
		label_y = y,
		batch_size = batch_size,
		phi_1_struct = phi_1_struct,
		theta_1_struct = theta_1_struct,
		theta_2_struct = theta_2_struct,
		in_dim = x_dim,
		out_dim = y_dim
		)
		   

	
	cost = (classifier.cost)
		
	gparams = [T.grad(cost, param) for param in classifier.params]
				   
	updates = [
		(param, param - learning_rate * gparam)
		for param, gparam in zip(classifier.params, gparams)
	]
	
	
	test_model = theano.function(
		inputs=[index],
		outputs=classifier.predictor,
		givens={
			x: test_feat[index * batch_size : (index + 1) * batch_size, :],
			y: test_label[index * batch_size : (index + 1) * batch_size, :]
		}		 
	)
	
	validate_model = theano.function(
		inputs=[index],
		outputs=classifier.cost,
		givens={
			x: valid_feat[index * batch_size : (index + 1) * batch_size, :],
			y: valid_label[index * batch_size : (index + 1) * batch_size, :]
		}		 
	)				 
	
	
	train_model = theano.function(
		inputs=[index],
		outputs=[classifier.cost, classifier.KL, classifier.CE, classifier.predictor, classifier.label_y],
		updates=updates,
		givens={
			x: train_feat[index * batch_size : (index + 1) * batch_size, :],
			y: train_label[index * batch_size : (index + 1) * batch_size, :]
		}		
	)					
	'''
def test_svae_mnist():
	'''Load Data'''
	test_file = 'test_1ok.npy'
	
	test=np.load(test_file)
	f = open('svae_train_1000.save', 'rb')
	trained_params = cPickle.load(f)
	f.close()
	test_feat, test_label = util.shared_dataset(test)
	
  
	'''Coefficient Initial'''		 
	batch_size = 50
	
	
	n_test_batches = test_feat.get_value(borrow=True).shape[0] // batch_size
	print('number of minibatch at one epoch:  test %i' % ( n_test_batches))
	
	z_dim = 10 #dimension of latent variable 
	x_dim = test_feat.get_value(borrow=True).shape[1]
	y_dim = test_label.get_value(borrow=True).shape[1]
	activation = None
	
	print(test_feat.get_value(borrow=True).shape[0])
	print(test_label.get_value(borrow=True).shape[0])
	print(test_feat.get_value(borrow=True).shape[1])
	print(test_label.get_value(borrow=True).shape[1])
	
	phi_1_struct=nn.NN_struct()
	phi_1_struct.layer_dim = [x_dim+y_dim, z_dim]
	phi_1_struct.activation = [activation]
	
	theta_1_struct=nn.NN_struct()
	theta_1_struct.layer_dim = [x_dim, z_dim]
	theta_1_struct.activation = [activation]
	
	theta_2_struct=nn.NN_struct()
	theta_2_struct.layer_dim = [x_dim+z_dim, y_dim]
	theta_2_struct.activation = [activation]		
	
	
	######################
	# BUILD ACTUAL MODEL #
	######################
	print('... building the test model')
	
	
	# allocate symbolic variables for the data
	#index_source = T.lscalar()	 # index to a [mini]batch
	#index_target = T.lscalar()	 # index to a [mini]batch
	index = T.lscalar()	 # index to a [mini]batch
	x = T.matrix('x')  # the data is presented as rasterized images
	y = T.matrix('y')  # the labels are presented as signal vector	   
	
	rng = np.random.RandomState(1234)
		
	# construct the DAVAE class
   
	classifier = SVAE.Supervised_VAE_test_v2(
		rng=rng,
		input_x = x,
		label_y = y,
		batch_size = batch_size,
		phi_1_struct = phi_1_struct,
		theta_2_struct = theta_2_struct,
		in_dim = x_dim,
		out_dim = y_dim,
		param = trained_params
		)
	
	test_model = theano.function(
		inputs=[index],
		outputs=[classifier.predictor, classifier.label_y],
		givens={
			x: test_feat[index * batch_size : (index + 1) * batch_size, :],
			y: test_label[index * batch_size : (index + 1) * batch_size, :]
		}		 
	)

	train_acc=[]
	for minibatch_index in range(n_test_batches):
		[pred, lab]= test_model(minibatch_index)
		print(pred)
		print(np.nonzero(lab)[1])
		# iteration number
		iter = minibatch_index
		train_acc.append(get_acc(pred,np.nonzero(lab)[1]))
	
	print('test accuracy: %f' % (np.mean(train_acc)))
			
		
	
	
	
	'''Model Construct'''
if __name__ == '__main__':	  
	train_svae_mnist()
	#test_svae_mnist()
	
	
