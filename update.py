from __future__ import print_function
from collections import OrderedDict

import os
import sys
import timeit

import scipy.io as sio
import numpy as np
import theano
import theano.tensor as T

def adam(loss, all_params, learning_rate=0.0002, beta1=0.1, beta2=0.001,
         epsilon=1e-8, gamma=1-1e-7):
    """
    ADAM update rules
    Default values are taken from [Kingma2014]
    References:
    [Kingma2014] Kingma, Diederik, and Jimmy Ba.
    "Adam: A Method for Stochastic Optimization."
    arXiv preprint arXiv:1412.6980 (2014).
    :parameters:
        - loss : Theano expression
            specifying loss
        - all_params : list of theano.tensors
            Gradients are calculated w.r.t. tensors in all_parameters
        - learning_Rate : float
        - beta1 : float
            Exponentioal decay rate on 1. moment of gradients
        - beta2 : float
            Exponentioal decay rate on 2. moment of gradients
        - epsilon : float
            For numerical stability
        - gamma: float
            Decay on first moment running average coefficient
        - Returns: list of update rules
    """

    updates = []
    all_grads = theano.grad(loss, all_params)

    i = theano.shared(np.float32(1))  # HOW to init scalar shared?
    i_t = i + 1.
    fix1 = 1. - (1. - beta1)**i_t
    fix2 = 1. - (1. - beta2)**i_t
    beta1_t = 1-(1-beta1)*gamma**(i_t-1)   # ADDED
    learning_rate_t = learning_rate * (T.sqrt(fix2) / fix1)

    for param_i, g in zip(all_params, all_grads):
        m = theano.shared(
            np.zeros(param_i.get_value().shape, dtype=theano.config.floatX))
        v = theano.shared(
            np.zeros(param_i.get_value().shape, dtype=theano.config.floatX))

        m_t = (beta1_t * g) + ((1. - beta1_t) * m) # CHANGED from b_t TO use beta1_t
        v_t = (beta2 * g**2) + ((1. - beta2) * v)
        g_t = m_t / (T.sqrt(v_t) + epsilon)
        param_i_t = param_i - (learning_rate_t * g_t)

        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((param_i, param_i_t) )
    updates.append((i, i_t))
    return updates                