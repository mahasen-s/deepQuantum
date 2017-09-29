from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as nr
import numpy.linalg as nl
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import utils as utils

class hamiltonian_tfi():
    # Produces graphs which compute the local energy and summary statistics for a given input and 
    # wavefunction
    def __init__(self,h_drive=1,h_inter=1,h_detune=0,dtype=tf.float64):
        self.h_drive    = h_drive
        self.h_inter    = h_inter
        self.h_detune   = h_detune
        self.alphabet   = 'abcdefghklmnopqrstuvwxyz';

    def build_energy_local(self,input_states=None,wavefunction=None):
        if input_states is None:
            raise ValueError('input_states must be provided')
        if wavefunction is None:
            raise ValueError('wavefunction must be provided')
        if input_dims>24:
            raise ValueError('Number of dimensions of input_states too large (>24)')

        # Calculate sigz, sigx
        num_sites       = input_states.get_shape().as_list()[-1]
        input_dims      = len(input_states.get_shape())

        sigz            = tf.reduce_sum(tf.multiply(input_states,utils.tf_roll(input_states,1,axis=-1)),axis=-1,keep_dims=True)

        einsum_string   = '%sk,ak->%sak' %(self.alphabet[0:input_dims-1],self.alphabet[0:input_dims-1])
        flip_mat        = tf.subtract(tf.ones(num_sites,dtype=dtype),tf.scalar_mul(2.0,tf.eye(num_sites,dtype=dtype)))
        sigx            = tf.einsum(einsum_string,input_states,flip_mat)

        # Get energy from sigz, sigx
        psi             = wavefunction.build_wf(input_states)
        E_sigz          = tf.scalar_mul(self.h_inter,tf.multiply(psi,sigz)) # will fail because dimensions of psi, sigz are wrong
        E_sigx          = tf.scalar_mul(self.h_drive,tf.reduce_sum(wavefunction.build_wf(sigx),axis=-1))

        # Output energy


def test_hamiltonian_TFI():
    import wavefunction as wavefunction
    import numpy.random as nr
    wf = wavefunction.wavefunction(num_sites=5,num_hidden=7,num_layers=3)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    input_1D    = tf.constant(2*nr.randint(0,high=2,size=[5]).astype(float)-1.0)
    input_2D    = tf.constant(2*nr.randint(0,high=2,size=[4,5]).astype(float)-1.0)
    input_3D    = tf.constant(2*nr.randint(0,high=2,size=[3,4,5]).astype(float)-1.0)

    op_1D       = wf.build_wf(input_1D)
    op_2D       = wf.build_wf(input_2D)
    op_3D       = wf.build_wf(input_3D)

    import pdb; pdb.set_trace()

def test_hamiltonian_TFI_energy():
    return 1
