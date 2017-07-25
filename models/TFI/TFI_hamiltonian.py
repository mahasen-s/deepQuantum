# Class for metropolis sampling
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as nr
import numpy.linalg as nl
import tensorflow as tf

from lib import config as conf
from lib import dq_utils as utils

class hamiltonian_tfi():
    def __init__(self,wf,M=1,h_drive=1,h_inter=1,h_detune=0):
        self.M          = M;
        self.N          = wf.input_num;
        self.h_drive    = h_drive
        self.h_inter    = h_inter
        self.h_detune   = h_detune
        self.wf         = wf;   # shallow copy?

        self.input_states   = tf.placeholder(conf.DTYPE,shape=[self.N,M]);

        self.sigx           = tf.placeholder(conf.DTYPE,shape=[self.N,self.N,M])
        self.sigz           = tf.placeholder(conf.DTYPE,shape=[M])

        self.build_E_graphs()

    def build_E_graphs(self):
        # Add sig_z terms
        self.psi    = self.wf.buildOp(self.input_states);
        E_sig_z     = tf.scalar_mul(self.h_inter,tf.multiply(self.psi,self.sigz));
        E_sig_x     = tf.scalar_mul(self.h_drive,tf.einsum('ij->j',self.wf.buildOp(self.sigx)))

        if utils.tf_is_cmplx(conf.DTYPE)==True:
            absFun  = utils.tf_cmplx_abs_to_cmplx
        else:
            absFun = tf.abs

        self.E_proj_unnorm  = tf.add(E_sig_z,E_sig_x)
        self.E_locs         = tf.divide(self.E_proj_unnorm,self.psi)
        self.E_locs_mean_re = tf.reduce_mean(tf.real(self.E_locs))
        self.E_locs_var_re = tf.reduce_mean(tf.square(tf.real(self.E_locs) - tf.reduce_mean(tf.real(self.E_locs), keep_dims=True)))
        self.E_locs_std_re = tf.sqrt(self.E_locs_var_re)


    def getAuxVars(self,S):
        if conf.DTYPE==tf.float64 or conf.DTYPE==tf.complex128:
            nptype  = np.single
        else:
            nptype = np.double
        # Get N
        N       = S.shape[0];

        # Get number of spin-aligned pairs in each sample
        np_sigz  = np.add.reduce(np.multiply(S,np.roll(S,1,0)),0);

        # Get NxNxM array of single-site flipped states
        np_sigx  = np.einsum('ij,ai->aij',S,(np.ones(N) - 2*np.eye(N)))

        return nptype(np_sigx),nptype(np_sigz);
