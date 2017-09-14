from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as nr
import numpy.linalg as nl
import tensorflow as tf

class wavefunction():

    def __init__(self,sess=tf.get_default_session(),num_sites,num_hidden=None, num_layers=1, act_fun="sigmoid"):
        # Process inputs
        if num_hidden is None:
            num_hidden = num_sites

        # Store vars
        self.sess       = sess
        self.num_sites  = num_sites
        self.num_hidden = num_hidden
        self.act_fun    = act_fun


        # Create funcs to initialise weights/biases
        tf_weight_gen   = lambda x: tf.Variable(tf.random_normal(x, dtype=conf.DTYPE))
        tf_bias_gen     = lambda x: tf.Variable(tf.random_normal(x, dtype=conf.DTYPE))




