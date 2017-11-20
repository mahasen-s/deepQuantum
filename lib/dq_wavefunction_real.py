# Class for evaluating a wavefunction
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as nr
import numpy.linalg as nl
import tensorflow as tf

from . import config_real as conf
from . import dq_utils as utils
class wavefunction():

    '''
    Defines the a deep neural net compression of a wavefunction.

    Args:
        sess (tensorflow session): The session to run all tf commands through.

    Keyword:
        nn_type (String): The learner type. 'mf_single' single layer net. 'mf_deep' deep nerual net. Default: 'deep'.
        input_num (int): The number of inputs for the net. Default 1.
    
    
        hidden_num (int): The number of hidden units. If none set to input_num. Default None.
        layers_num (int): The number of layer in the deep neural net. Not used for shallow.
        activation_function (string): The activation function used for wavefunction. Not used for shallow

    '''
    def __init__(self,sess,nn_type='deep',input_num=1, hidden_num=None, layers_num=None, activation_function='sigmoid'):

        # Generates weights for network which are used by a factory function to produce the network representation of a wavefunction
        tf_weight_gen = lambda x: tf.Variable(tf.random_normal(x, dtype=conf.DTYPE))
        tf_bias_gen = lambda x: tf.Variable(tf.random_normal(x, dtype=conf.DTYPE))

        self.sess = sess
        self.input_num = input_num
        self.output_num = 2
        if hidden_num is None:
            self.hidden_num = self.input_num*2
        else:
            self.hidden_num = hidden_num
        if layers_num is None:
            layers_num = self.input_num # why?
        else:
            self.layers_num = layers_num
        self.nn_type = nn_type
        self.activation_function = activation_function

        self.input_state  = tf.placeholder(conf.DTYPE,[self.input_num,1]); # single state for MC
        self.weights = []
        self.biases = []

        if nn_type == 'mf':
            self.biases.append( tf_bias_gen([self.hidden_num,1]) )
            self.weights.append( tf_weight_gen([self.output_num, self.input_num]) )
            self.weights.append( tf_weight_gen([self.hidden_num, self.input_num]) )

            self.layers_num = 1
            self.activation_function = 'custom'

        elif nn_type == 'deep':

            if activation_function == 'sigmoid': # doesn't support complex128
                act_func = tf.nn.sigmoid
                if conf.DTYPE == tf.complex128:
                    act_func = utils.tf_sigmoid_complex128
            elif activation_function == 'ReLU': # doesn't support complex64, complex128
                act_func = tf.nn.relu
                if conf.DTYPE == tf.complex64 or conf.DTYPE == tf.complex128:
                    # Use softmax function; does it even make sense for complex?
                    act_func = utils.tf_softmax_complex
            elif activation_function == 'tanh': # doesn't support complex128
                act_func = tf.nn.tanh
                if conf.DTYPE == tf.complex128:
                    act_func = utils.tf_tanh_complex128
            else:
                print('layer named incorrectly')
                raise ValueError

            self.act_func   = act_func;
            for ind in range(layers_num):

                self.biases.append(tf_bias_gen([self.hidden_num,1]))

                if ind == 0:
                    self.weights.append( tf_weight_gen( [ self.hidden_num, self.input_num ] ) )
                else:
                    self.weights.append( tf_weight_gen( [ self.hidden_num, self.hidden_num] ) )

            self.weights.append(tf_weight_gen([self.output_num,self.hidden_num]))
            self.biases.append(tf_bias_gen([self.output_num,]))
        else:
            print('Unknown nn_type' + repr(nn_type))
            raise ValueError

        # Build op for self.input_states
        self.wavefunction_tf_op = self.buildOp(self.input_state);

        self.input_state_overlap = tf.placeholder(conf.DTYPE,[self.input_num,pow(2,self.input_num)]) # SHOULD ONLY BE CREATED IF OVERLAP IS TO BE COMPUTED
        self.wavefunction_overlap_op = self.buildOp(self.input_state_overlap)

    def buildOp(self,input_states):
        '''
        Returns a tensorflow operator which computes the wavefunction for input_states based on the 
        network topology and network weights defined during initialisation of the class.
        NEED TO CHECK THAT WEIGHTS ARE UPDATED!
        '''
        # expand dimensions of input state
        if len(input_states.get_shape()) == 2:
            input_states = tf.expand_dims(input_states,0);
        elif len(input_states.get_shape()) != 3:
            print('Incompatible dimensions of' + repr(input_states) + 'for wavefunction')
            raise ValueError

        if self.nn_type == 'shallow':
            psi_exp     = tf.exp(tf.einsum('ij,ajk->ak',self.weights[0],input_states)); # a x M array
            psi_F_arg   = tf.add(self.biases[0],tf.einsum('ij,ajk->aik',self.weights[1],input_states));
            psi_F       = tf.add(tf.exp(psi_F_arg),tf.exp(tf.negative(psi_F_arg)));
            if conf.DTYPE == tf.complex64 or conf.DTYPE == tf.complex128:
                psi_F_prod  = utils.tf_reduce_prod_cmplx(psi_F, 1);
            else:
                psi_F_prod  = tf.reduce_prod(psi_F, 1);

            psi = tf.multiply( psi_exp, psi_F_prod );

        elif self.nn_type =='deep':
            for ind in range(self.layers_num):
                if ind == 0:
                    curr_net  = self.act_func( tf.add( self.biases[ind], tf.einsum('ij,ajk->aik', self.weights[ind], input_states) ) );
                    # print('\nInd = %d' % ind)
                    # print(input_states.shape)
                    # print(self.biases[ind].shape)
                    # print(self.weights[ind].shape)
                    # print(curr_net.shape)
                else:
                    curr_net  = self.act_func( tf.add( self.biases[ind], tf.einsum('ij,ajk->aik', self.weights[ind], curr_net) ) );
                    # print('\nInd = %d' % ind)
                    # print(self.biases[ind].shape)
                    # print(self.weights[ind].shape)
                    # print(curr_net.shape)

            
            # print('\nInd = End')
            # print(self.biases[-1].shape)
            # print(self.weights[-1].shape)
            psi = tf.add(tf.einsum('ij,ajk->aki',self.weights[-1],curr_net),self.biases[-1]);

        return psi



    '''
    Returns a list of complex values correspoding to the list of configurations provided.

    Args:
        states (np.array): List of configurations, each configuration must be the same size as the input_num for the wavefunction.
    Returns:
        list: List of complex numbers corresponding to the configurations provided.
    '''

    def eval(self, states):
        psi = self.sess.run(self.wavefunction_tf_op, feed_dict={self.input_state: states});
        if conf.DTYPE == tf.float32:
            nptype = np.complex64
        elif conf.DTYPE == tf.float64:
            nptype = np.complex128

        # Convert to [N] complex array
        psi = np.add.reduce(np.multiply([nptype(1.),nptype(1.j)],psi),axis=-1)
        return psi

    def evalOverlap(self,states):
        # Psi is a [N,2] array 
        psi = self.sess.run(self.wavefunction_overlap_op, feed_dict={self.input_state_overlap: states})

        if conf.DTYPE == tf.float32:
            nptype = np.complex64
        elif conf.DTYPE == tf.float64:
            nptype = np.complex128

        # Convert to [N] complex array
        psi = np.add.reduce(np.multiply([nptype(1.),nptype(1.j)],psi),axis=-1)

        return psi


