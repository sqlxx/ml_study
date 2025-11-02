import numpy as np
import theano 

class FullyConnectedLayer(object):

    def __init__(self, n_in, n_out, activation_fn=sigmoid, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout

        self.w = theano.shared(
            np.asarray(np.random.normal(loc=0.0, scale=np.sqrt(1.0/n_out), size=(n_in, n_out)), dtype=theano.config.floatX)
        )