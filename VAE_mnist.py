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
import criteria as er
import util
import VAE


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
    batch_size = 100
    n_epochs = 5
    learning_rate = 0.01
    
    n_train_batches = train_feat.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_feat.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_feat.get_value(borrow=True).shape[0] // batch_size
    print('number of minibatch at one epoch: train  %i, validation %i, test %i' %
        (n_train_batches, n_valid_batches, n_test_batches))
    
    z_dim = 100 #dimension of latent variable 
    x_dim = train_feat.get_value(borrow=True).shape[1]
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
    phi_1_struct.layer_dim = [x_dim, z_dim]
    phi_1_struct.activation = [activation]
    
    theta_1_struct=nn.NN_struct()
    theta_1_struct.layer_dim = [z_dim, y_dim]
    theta_1_struct.activation = [activation]
    
    
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')
    
    
    # allocate symbolic variables for the data
    #index_source = T.lscalar()  # index to a [mini]batch
    #index_target = T.lscalar()  # index to a [mini]batch
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.matrix('y')  # the labels are presented as signal vector     
    
    rng = np.random.RandomState(1234)
        
    # construct the DAVAE class
   
    classifier = VAE.Supervised_VAE(
        rng=rng,
        input_x = x,
        label_y = y,
        batch_size = batch_size,
        phi_1_struct = phi_1_struct,
        theta_1_struct = theta_1_struct,
        in_dim = x_dim,
        out_dim = y_dim
        )
    
    
    cost = (classifier.cost)
        
    gparams = [T.grad(cost, param) for param in classifier.params]
                   
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]
    
    print('... prepare training model')
    train_model = theano.function(
        inputs=[index],
        outputs=[classifier.cost, classifier.predictor, classifier.label_y],
        updates=updates,
        givens={
            x: train_feat[index * batch_size : (index + 1) * batch_size, :],
            y: train_label[index * batch_size : (index + 1) * batch_size, :]
        }       
    )   
    
    
    print('... prepare validate model')
    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.cost,
        givens={
            x: valid_feat[index * batch_size : (index + 1) * batch_size, :],
            y: valid_label[index * batch_size : (index + 1) * batch_size, :]
        }        
    )                
    
    
    print('... prepare test model')
    test_model = theano.function(
        inputs=[index],
        outputs=[classifier.predictor, classifier.label_y],
        givens={
            x: test_feat[index * batch_size : (index + 1) * batch_size, :],
            y: test_label[index * batch_size : (index + 1) * batch_size, :],
        }        
    )
    
    
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


    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        train_acc=[]
        
        
        for minibatch_index in range(n_train_batches):

            [minibatch_avg_cost, pred, lab] = train_model(minibatch_index)
            #print(minibatch_index)
                        
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index
            
            train_acc.append(get_acc(pred,np.nonzero(lab)[1]))
            
            
            if (iter + 1) % validation_frequency == 0:
                # compute loss on validation set
                validation_losses = [validate_model(i) for i in range(n_valid_batches)] 
                this_validation_loss = np.mean(validation_losses)
                
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
                    
                    #[test_pred, test_lab] = [test_model(i) for i in range(n_test_batches)] 
                    #this_test_loss = np.mean(test_losses)
                    #test_acc.append(get_acc(test_pred,np.nonzero(lab)[1]))
                    test_acc=[]
                    for i in range(n_test_batches):
                        [test_pred, test_lab]=test_model(i)
                        test_acc.append(get_acc(test_pred,np.nonzero(test_lab)[1]))
                        #print(test_pred)
                        #print(test_lab)
                    print(np.mean(np.array(test_acc)))
                    
                    best_validation_loss = this_validation_loss   
                    best_iter = iter

                    # get training accuracy
                    print('best training accuracy: %f' % (np.mean(np.array(train_acc))))
                    # test it on the test set
                    #test_losses = [test_model(i) for i in range(n_test_batches)]
                    #test_score = np.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, best train accuracy: %f') % \
                          (epoch, minibatch_index + 1, n_train_batches, \
                           np.mean(np.array(train_acc))))

            if patience <= iter:
                done_looping = True
                break

                
def train_uvae_mnist():
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
    batch_size = 100
    n_epochs = 100
    learning_rate = 0.007
    
    n_train_batches = train_feat.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_feat.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_feat.get_value(borrow=True).shape[0] // batch_size
    print('number of minibatch at one epoch: train  %i, validation %i, test %i' %
        (n_train_batches, n_valid_batches, n_test_batches))
    
    z_dim = 5 #dimension of latent variable 
    x_dim = train_feat.get_value(borrow=True).shape[1]
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
    phi_1_struct.layer_dim = [x_dim, z_dim]
    phi_1_struct.activation = [activation]
    
    theta_1_struct=nn.NN_struct()
    theta_1_struct.layer_dim = [z_dim, x_dim]
    theta_1_struct.activation = [activation]
    
    
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')
    
    
    # allocate symbolic variables for the data
    #index_source = T.lscalar()  # index to a [mini]batch
    #index_target = T.lscalar()  # index to a [mini]batch
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    #y = T.matrix('y')  # the labels are presented as signal vector    
    
    rng = np.random.RandomState(1234)
        
    # construct the DAVAE class
   
    classifier = VAE.Unsupervised_VAE(
        rng=rng,
        input_x = x,
        label_y = x,
        batch_size = batch_size,
        phi_1_struct = phi_1_struct,
        theta_1_struct = theta_1_struct,
        in_dim = x_dim,
        out_dim = x_dim
        )
    
    
    cost = (classifier.cost)
        
    gparams = [T.grad(cost, param) for param in classifier.params]
                   
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]
    
    print('... prepare training model')
    train_model = theano.function(
        inputs=[index],
        outputs=[classifier.cost, classifier.predictor, classifier.label_y],
        updates=updates,
        givens={
            x: train_feat[index * batch_size : (index + 1) * batch_size, :],
            #y: train_label[index * batch_size : (index + 1) * batch_size, :]
        }       
    )   
    
    
    print('... prepare validate model')
    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.cost,
        givens={
            x: valid_feat[index * batch_size : (index + 1) * batch_size, :],
            #y: valid_label[index * batch_size : (index + 1) * batch_size, :]
        }        
    )                
    
    
    print('... prepare test model')
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.predictor,
        givens={
            x: test_feat[index * 1 : (index + 1) * 1, :],
            #y: test_label[index * batch_size : (index + 1) * batch_size, :],
        }        
    )
    
    
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


    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        train_acc=[]
        
        
        for minibatch_index in range(n_train_batches):

            [minibatch_avg_cost, pred, lab] = train_model(minibatch_index)
            #print(minibatch_index)
                        
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index
            
            #train_acc.append(get_acc(pred,np.nonzero(lab)[1]))
            
            
            if (iter + 1) % validation_frequency == 0:
                # compute loss on validation set
                validation_losses = [validate_model(i) for i in range(n_valid_batches)] 
                this_validation_loss = np.mean(validation_losses)
                
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
                    
                    #[test_pred, test_lab] = [test_model(i) for i in range(n_test_batches)] 
                    #this_test_loss = np.mean(test_losses)
                    #test_acc.append(get_acc(test_pred,np.nonzero(lab)[1]))
                    
                    
                    best_validation_loss = this_validation_loss   
                    best_iter = iter

                    # get training accuracy
                    print('best training accuracy: %f' % (np.mean(np.array(train_acc))))
                    # test it on the test set
                    #test_losses = [test_model(i) for i in range(n_test_batches)]
                    #test_score = np.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, best train accuracy: %f') % \
                          (epoch, minibatch_index + 1, n_train_batches, \
                           np.mean(np.array(train_acc))))

            if patience <= iter:
                done_looping = True
                break

    test_pred=[]
    for i in range(10):
        test_pred[i]=test_model(i)
        #test_acc.append(get_acc(test_pred,np.nonzero(test_lab)[1]))
        #print(test_pred)
        #print(test_lab)

    '''
    print('saving final model')
    f = open('model.save', 'wb')
    cPickle.dump(classifier, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
    '''

    '''
    test_acc=[]
    for i in range(n_test_batches):
        [test_pred, test_lab]=test_model(i)
        test_acc.append(get_acc(test_pred,np.nonzero(test_lab)[1]))
        #print(test_pred)
        #print(test_lab)
    print(np.mean(np.array(test_acc)))
    '''


    '''Model Construct'''
if __name__ == '__main__':    
    #train_svae_mnist()
    train_uvae_mnist()
    
    
    
    
    
    
    
