import math
from itertools import izip
import numpy as np
import theano
import theano.tensor as T



def create_shared_variable(data_variable):
    shared_var = theano.shared(np.zeros((1,) * data_variable.ndim,
                                               dtype=data_variable.dtype)) 
    shared_var.name = "%s_shared"%data_variable
    return shared_var

def create_shared_variables(inputs):
    return { var:create_shared_variable(var)
                for var in inputs }

def build_trainer(inputs,updates,outputs=None,batch_size=256,mapping=None):
    """
    Creates a shared variables and a function to load chunk into shared variables and train
    """
    if mapping is None:
        mapping = create_shared_variables(inputs)

    idx = T.iscalar('idx')
    train = theano.function(
            inputs  = [idx],
            outputs = outputs,
            updates = updates,
            givens  = { var:shared_var[idx*batch_size:(idx+1)*batch_size]
                            for var,shared_var in mapping.iteritems() },
        )
    def chunk_train(chunk):
        batch_count = int(math.ceil(chunk[0].shape[0]/float(batch_size)))
        for in_var,data in izip(inputs,chunk):
            mapping[in_var].set_value(data)
        for i in xrange(batch_count):
            if outputs is None:
                train(i)
            else:
                print train(i)

    return chunk_train
