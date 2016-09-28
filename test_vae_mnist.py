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

with open('model.save', 'rb') as f:
    model = pickle.load(f)



