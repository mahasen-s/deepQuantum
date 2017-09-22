from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as nr
import numpy.linalg as nl
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

class wavefunction():

    def __init__(self,num_sites=1,num_hidden=None, num_layers=1, num_output=2, act_fun="sigmoid", var_dtype=tf.float64):
        # Process inputs
        if num_hidden is None:
            num_hidden = num_sites

        # Store vars
        self.num_sites  = num_sites
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.num_output = num_output
        self.dtype      = var_dtype

        # Set activation func
        if act_fun == 'sigmoid':
            self.act_fun = tf.nn.sigmoid
        if act_fun == 'ReLU':
            self.act_fun = tf.nn.relu
        if act_fun == 'tanh':
            self.act_fun = tf.nn.tanh

        # Create funcs to initialise weights/biases
        tf_weight_gen   = lambda s,x: tf.get_variable(s,x, dtype=self.dtype, initializer=tf.random_normal_initializer)
        tf_bias_gen     = lambda s,x: tf.get_variable(s,x, dtype=self.dtype, initializer=tf.random_normal_initializer)

        # Create list of weights, biases
        self.biases     = []
        self.weights    = []

        with tf.variable_scope("wavefunction"):
            for ind in range(num_layers+1):
                str_biases  = "bias_layer_%d" %(ind)
                str_weights = "weights_layer_%d" %(ind)

                if ind==0:
                    # input -> 1st hidden layer
                    self.biases.append(tf_bias_gen(str_biases,self.num_hidden))
                    self.weights.append(tf_weight_gen(str_weights,[self.num_hidden,self.num_sites]))
                elif ind==num_layers:
                    # last hidden layer -> output
                    self.biases.append(tf_bias_gen(str_biases,self.num_output))
                    self.weights.append(tf_weight_gen(str_weights,[self.num_output,self.num_hidden]))
                else:
                    # hidden layer -> hidden layer
                    self.biases.append(tf_bias_gen(str_biases,self.num_hidden))
                    self.weights.append(tf_weight_gen(str_weights,[self.num_hidden,self.num_hidden]))

    def buildOp(self,input_states):
        # generate einsum string
        input_dims  = len(input_states.get_shape());
        alphabet    = 'abcdefghklmnopqrstuvwxyz';
        einsum_string   = 'ij,%sj->%si' %(alphabet[0:input_dims-1],alphabet[0:input_dims-1])

        # check
        print('\nBiases shapes:')
        for x in self.biases:
            print('%s:\t' %(x.name) + repr(x.shape))
        print('\nWeights shapes:')
        for x in self.weights:
            print('%s:\t' %(x.name) + repr(x.shape))

        # need an operator which will compute the wavefunction on the innermost dimension
        print('\nInput shape at each layer:')
        print('Initial:\t' + repr(input_states.shape))
        for ind in range(self.num_layers):
            if ind==0:
                curr_net    = self.act_fun(tf.add(self.biases[ind],tf.einsum(einsum_string,self.weights[ind],input_states)))
#                curr_net    = self.act_fun(tf.add(self.biases[ind],tf.einsum('ij,j->i',self.weights[ind],input_states)))
            else:
                curr_net    = self.act_fun(tf.add(self.biases[ind],tf.einsum(einsum_string,self.weights[ind],curr_net)))
#                curr_net    = self.act_fun(tf.add(self.biases[ind],tf.einsum('ij,j->i',self.weights[ind],curr_net)))
            print('Layer %d:\t' %(ind) + repr(curr_net.shape))

        # shoudl there be an activation function on this?
        psi = tf.add(self.biases[-1],tf.einsum(einsum_string,self.weights[-1],curr_net))
        print('Output:\t\t'+ repr(psi.shape))
#        psi = tf.add(self.biases[-1],tf.einsum('ij->',self.weights[-1],curr_net))

        return psi


def test_wavefunction():
    sess    = tf.Session()
    wf  = wavefunction(num_sites=5,num_hidden=10,num_layers=3)

    sess.run(tf.global_variables_initializer())

    input_1D    = tf.constant(np.ones([5]))
    input_2D    = tf.constant(np.ones([4,5]))
    input_3D    = tf.constant(np.ones([3,4,5]))

    op_1D       = wf.buildOp(input_1D)
    op_2D       = wf.buildOp(input_2D)
    op_3D       = wf.buildOp(input_3D)

    assert sess.run(op_1D).shape==(2,)
    assert sess.run(op_2D).shape==(4,2)
    assert sess.run(op_3D).shape==(3,4,2)

#test_wavefunction()
