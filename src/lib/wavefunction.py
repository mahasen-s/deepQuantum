from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as nr
import numpy.linalg as nl
import tensorflow as tf

class wavefunction():

    def __init__(self,sess=tf.get_default_session(),num_sites,num_hidden=None, num_layers=1, act_fun="sigmoid", var_dtype=tf.float64):
        # Process inputs
        if num_hidden is None:
            num_hidden = num_sites

        # Store vars
        self.sess       = sess
        self.num_sites  = num_sites
        self.num_hidden = num_hidden
        self.act_fun    = act_fun

        # Create funcs to initialise weights/biases
        tf_weight_gen   = lambda s,x: tf.get_variable(s,x, dtype=conf.DTYPE, initializer=tf.random_normal_initializer)
        tf_bias_gen     = lambda s,x: tf.get_variable(s,x, dtype=conf.DTYPE, initializer=tf.random_normal_initializer)

        # Create list of weights, biases
        self.biases     = []
        self.weights    = []

        with tf.variable_scope("wavefunction"):
            for ind in range(num_layers):
                str_biases  = "bias_layer_%d" %(ind)
                str_weights = "weights_layer_%d" %(ind)

                self.biases.append(tf_bias_gen(str_biases,self.hidden_num))

                if ind==0:
                    self.weights.append(tf_weight_gen(str_weights,[self.hidden_num,self.input_num]))
                else:
                    self.weights.append(tf_weight_gen(str_weights,[self.hidden_num,self.hidden_num]))

    def buildOp(self,input_states):
        # generate einsum string
        input_dims  = len(input_states.get_shape());
        alphabet    = 'abcdefghklmnopqrstuvwxyz';
        einsum_string   = 'ij,%sj->%si' %(alphabet[0:input_dims-1],alphabet[0:input_dims-1])


        # need an operator which will compute the wavefunction on the innermost dimension
        for inf in range(self.num_layers):
            if ind==0:
                curr_net    = self.act_fun(tf.add(self.biases[ind],tf.einsum(einsum_string,self.weights[ind],input_states)))
            else:
                curr_net    = self.act_fun(tf.add(self.biases[ind],tf.einsum(einsum_string,self.weights[ind],curr_net)))

        # shoudl there be an activation function on this?
        psi = tf.add(self.biases[-1],tf.einsum(einsum_string[0:-1],self.weights[-1],curr_net))

        return psi



