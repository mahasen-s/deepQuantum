# Class for metropolis sampling
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as nr
import numpy.linalg as nl
import tensorflow as tf

from lib import config_real as conf
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
        # need to make this agree with [x,x,2] sized psi arrays
        self.psi    = self.wf.buildOp(self.input_states);
        E_sig_z     = tf.scalar_mul(self.h_inter,tf.multiply(self.psi, tf.expand_dims(self.sigz,-1))); # expand sigz, then broadcast multiply on innermost dim
        E_sig_x     = tf.scalar_mul(self.h_drive,tf.einsum('ijk->jk',self.wf.buildOp(self.sigx))) # 'ij->j' becomes 'ijk->jk' where the innermost dim is preserved

        if utils.tf_is_cmplx(conf.DTYPE)==True:
            absFun  = utils.tf_cmplx_abs_to_cmplx
        else:
            absFun = tf.abs

        self.E_proj_unnorm  = tf.add(E_sig_z,E_sig_x)


        # Constructing local energies
        # self.E_locs         = tf.divide(self.E_proj_unnorm,self.psi)
        denom = tf.reduce_sum(tf.square(self.psi),axis=-1,keep_dims=True)
        conj_mask = tf.constant([1,-1],dtype=conf.DTYPE)
        psi_conj = tf.multiply(self.psi,conj_mask)

        # real, imag part of numerator of self.E_locs before multiplying by E_proj_unnorm
        num_real = tf.reduce_sum(tf.multiply(self.E_proj_unnorm[:,:,:1],psi_conj),axis=-1,keep_dims=True)
        num_imag = tf.reduce_sum(tf.multiply(self.E_proj_unnorm[:,:,1:2],tf.reverse(psi_conj,axis=[-1])),axis=-1,keep_dims=True)

        # construct E_locs
        E_locs_real = tf.divide(num_real,denom)
        E_locs_imag = tf.divide(num_imag,denom)
        self.E_locs = tf.concat([E_locs_real,E_locs_imag],axis=-1)

        print(self.E_proj_unnorm.shape)
        print(denom.shape)
        print(psi_conj.shape)
        print(num_real.shape)
        print(num_imag.shape)
        print(E_locs_real.shape)
        print(E_locs_imag.shape)
        print(self.E_locs.shape)

        self.E_locs_mean_re = tf.reduce_mean(tf.squeeze(E_locs_real))
        self.E_locs_mean_im = tf.reduce_mean(tf.squeeze(E_locs_imag))
        self.E_locs_var_re = tf.reduce_mean(tf.square(tf.squeeze(E_locs_real) - tf.reduce_mean(tf.squeeze(E_locs_real), keep_dims=True)))
        self.E_locs_var_im = tf.reduce_mean(tf.square(tf.squeeze(E_locs_imag) - tf.reduce_mean(tf.squeeze(E_locs_imag), keep_dims=True)))
        self.E_locs_std_re = tf.sqrt(self.E_locs_var_re)
        self.E_locs_std_im = tf.sqrt(self.E_locs_var_im)

        # self.E_locs_mean_re = tf.reduce_mean(tf.real(self.E_locs))
        # self.E_locs_mean_im = tf.reduce_mean(tf.imag(self.E_locs))
        # self.E_locs_var_re = tf.reduce_mean(tf.square(tf.real(self.E_locs) - tf.reduce_mean(tf.real(self.E_locs), keep_dims=True)))
        # self.E_locs_var_im = tf.reduce_mean(tf.square(tf.imag(self.E_locs) - tf.reduce_mean(tf.imag(self.E_locs), keep_dims=True)))
        # self.E_locs_std_re = tf.sqrt(self.E_locs_var_re)
        # self.E_locs_std_im = tf.sqrt(self.E_locs_var_im)


    def getAuxVars(self,S):
        if conf.DTYPE==tf.float64 or conf.DTYPE==tf.complex128:
            nptype  = np.double
        else:
            nptype = np.single
        # Get N
        N       = S.shape[0];

        # Get number of spin-aligned pairs in each sample
        np_sigz  = np.add.reduce(np.multiply(S,np.roll(S,1,0)),0);

        # Get NxNxM array of single-site flipped states
        np_sigx  = np.einsum('ij,ai->aij',S,(np.ones(N) - 2*np.eye(N)))

        return nptype(np_sigx),nptype(np_sigz);
