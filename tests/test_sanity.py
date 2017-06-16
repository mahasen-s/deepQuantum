# Example of run file
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from timeit import default_timer as timer

import lib.config as conf

from lib.dq_sampling import metropolisSample
from lib.dq_wavefunction import wavefunction
from lib.dq_computeOverlap import computeOverlap

from models.TFI.TFI_sampling import sampling_tfi as sampling_fun
from models.TFI.TFI_hamiltonian import hamiltonian_tfi as hamiltonian

class wf_test():
    def __init__(self,sess,wf_fun,input_num=2,output_num=1):
        self.sess = sess
        self.input_num = input_num
        self.output_num = output_num
        self.wf_fun = wf_fun # function which is actually evaluated to return psi

        self.input_state = tf.placeholder(conf.DTYPE,[self.input_num,5])
        self.wavefunction_tf_op = self.buildOp(self.input_state)


    def buildOp(self,input_states):
        # input_states.get_shape() should have 2-3 elements. If 2, 0th index is the
        # site index, 1st is sample index. If 3, 0th is aux index, 1st is site
        # index, and 2nd is sample index
        if len(input_states.get_shape())==2:
            input_states = tf.expand_dims(input_states,0)
        elif len(input_states.get_shape()) != 3:
            print('Incompatible dimensions of '+repr(input_states)+' for wavefunction')
            raise ValueError

        psi     = self.wf_fun(self,input_states)
        return psi

    def eval(self,states):
        psi = self.sess.run(self.wavefunction_tf_op, feed_dict={self.input_state: states})
        return psi

def wf_GHZ_fun(caller,input_states):
    N   = tf.constant(caller.input_num,dtype=conf.DTYPE)
    div = tf.constant(1/np.sqrt(2),dtype=conf.DTYPE)
    psi = tf.cast(tf.equal(tf.abs(tf.einsum('ijk->ik',input_states)),N),conf.DTYPE)
    return psi

def wf_W_fun(caller,input_states):
    N   = tf.constant(caller.input_num-2,dtype=conf.DTYPE)
    div = tf.constant(1/np.sqrt(caller.input_num),dtype=conf.DTYPE)
    psi = tf.cast(tf.equal(tf.abs(tf.einsum('ijk->ik',input_states)),N),conf.DTYPE)
    return psi


N       = 3;
sess    = tf.Session();

# Create states
wf_GHZ  = wf_test(sess,wf_GHZ_fun,input_num=N)
wf_W    = wf_test(sess,wf_W_fun,input_num=N)

# Initialise variables
sess.run(tf.global_variables_initializer());

# Define input states
test    = np.array([[1,1,1],[-1,-1,-1],[1,-1,-1],[-1,1,-1],[-1,-1,1]])
test    = np.transpose(test)

M       = test.shape[1];

# GHZ
H_GHZ   = hamiltonian(wf_GHZ,M,h_drive=0,h_inter=1,h_detune=0)
H_W     = hamiltonian(wf_W,M,h_drive=0,h_inter=1,h_detune=0)
E_GHZ   = np.zeros(M)
E_W     = np.zeros(M)

#for i in range(0,M-1):
[sigx,sigz] = H_GHZ.getAuxVars(test)
E_GHZ    = sess.run(H_GHZ.E_vals, feed_dict={H_GHZ.input_states:test, H_GHZ.sigx: sigx, H_GHZ.sigz:sigz})
E_W      = sess.run(H_W.E_vals, feed_dict={H_W.input_states:test, H_W.sigx: sigx, H_W.sigz:sigz})

print("Wavefunction tests:")
print(wf_GHZ.eval(test))
print(wf_W.eval(test))

print("Hamiltonian tests:" )
# Print
print(E_GHZ)
print(E_W)

print("Sigz, sigx")
print(sigz)
print(sigx)

print("New test")
print(sess.run(H_GHZ.input_states, feed_dict={H_GHZ.input_states:test}))

print("Test multiply")
print(sess.run(tf.multiply(wf_GHZ.buildOp(H_GHZ.input_states),sigz), feed_dict={H_GHZ.input_states:test}))
