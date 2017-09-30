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
        self.dtype      = dtype
        self.alphabet   = 'abcdefghklmnopqrstuvwxyz';

    def build_energy_local(self,input_states=None,wavefunction=None):
        if input_states is None:
            raise ValueError('input_states must be provided')
        if wavefunction is None:
            raise ValueError('wavefunction must be provided')
        input_dims      = len(input_states.get_shape())
        if input_dims>24:
            raise ValueError('Number of dimensions of input_states too large (>24)')

        # Calculate sigz, sigx
        num_sites       = input_states.get_shape().as_list()[-1]

        sigz            = tf.reduce_sum(tf.multiply(input_states,utils.tf_roll(input_states,1,axis=-1)),axis=-1,keep_dims=True)

        einsum_string   = '%si,ji->%sij' %(self.alphabet[0:input_dims-1],self.alphabet[0:input_dims-1])
        flip_mat        = tf.subtract(tf.ones(num_sites,dtype=self.dtype),tf.scalar_mul(2.0,tf.eye(num_sites,dtype=self.dtype)))
        sigx            = tf.einsum(einsum_string,input_states,flip_mat)

        # Get energy from sigz, sigx
        psi             = wavefunction.build_wf(input_states)
        E_sigz          = tf.scalar_mul(self.h_inter,tf.multiply(psi,sigz)) # will fail because dimensions of psi, sigz are wrong
        E_sigx          = tf.scalar_mul(self.h_drive,tf.reduce_sum(wavefunction.build_wf(sigx),axis=-2))

        # Output energy
        E_proj_unnorm   = tf.add(E_sigz,E_sigx);
        E_local         = tf.divide(E_proj_unnorm,psi)

        return E_local

def test_hamiltonian_TFI():
    # check that output is dimensionally correct
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

    H           = hamiltonian_tfi()
    H_1D        = H.build_energy_local(input_states=input_1D,wavefunction=wf)
    H_2D        = H.build_energy_local(input_states=input_2D,wavefunction=wf)
    H_3D        = H.build_energy_local(input_states=input_3D,wavefunction=wf)

    assert sess.run(H_1D).shape==(2,)
    assert sess.run(H_2D).shape==(4,2)
    assert sess.run(H_3D).shape==(3,4,2)



def ntest_hamiltonian_TFI_energy():
    import wavefunction as wavefunction
    import numpy.random as nr

    # Make wavefunction
    wf = wavefunction.wavefunction(num_sites=5,num_hidden=7,num_layers=3)
    sess = tf.Session()

    # Gen all states
    [states_sites,states_hilbert]   = utils.genAllStates(5)

    # Load exact energy, hDrive=1, hInter=1, hDetune=0
    exactFile = "../exact/results/exact_TFI_hDrive=1p0_hInter=1p0_hDetune=0p0.mat"

    # Make wavefunction

    # Make Hamiltonian
    H   = hamiltonian_tfi()
    E_local     = H.build_energy_local(states_sites)

    return 1
