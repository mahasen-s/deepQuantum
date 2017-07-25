# Example of run file
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as nr
import tensorflow as tf
import matplotlib.pyplot as plt
from timeit import default_timer as timer

import lib.config as conf

from lib.dq_sampling import metropolisSample
from lib.dq_wavefunction import wavefunction
from lib.dq_computeOverlap import computeOverlap

from lib.dq_utils import genAllStates

from models.TFI.TFI_sampling import sampling_tfi as sampling_fun
from models.TFI.TFI_hamiltonian import hamiltonian_tfi as hamiltonian

from exact.pythonLoadHDF5 import loadMatlabHDF5 as pyMatLoad

from functools import partial, wraps

from lib.dq_utils import Bunch
import pickle

tf.logging.set_verbosity(tf.logging.ERROR)

from models.TFI.TFI_sampling_singleSite import markovChainGenerator as mcg
from models.TFI.TFI_sampling_singleSite import sampler_TFI as sampler

import sys

def gen_data(fileOut='pickledData'):
    
    N = 6
    P = 16
    L = 2
    M = 4
    
    full_steps = 500
    mc_steps = 0
    
    burnIn = 16
    thinning = 4
    
    learn_rate_full=0.05
    learn_rate_mc=0.005
    h_drive=1
    h_inter=0.5
    h_detune=0
    mcg_useFinal=False
    
    # Calculate variational energy using the full set of states and compare with a MC sample

    # EXACT VARS
    exactFile   = "/exact/results/exact_TFI_hDrive=%2.1f_hInter=%2.1f_hDetune=%2.1f" % (h_drive,h_inter,h_detune)
    exactFile   = '.' + exactFile.replace('.','p') + '.mat'
    try:
        exactVals   = pyMatLoad(exactFile,'data')
    except:
        raise ValueError('File not found: %s' % exactFile)
    wf_exact    = exactVals[N-2]
    
    E_0 = wf_exact[2]
    
    # FULL STATE SPACE
    states,_= genAllStates(N)
    print("States:")
    print(states)
    M_full  = states.shape[1]
    
    sess    = tf.Session()

    # construct wavefunction

    wf_full = wavefunction(sess,input_num=N,hidden_num=P,nn_type='deep',layers_num=L)

    # construct Hamiltonian
    H_full       = hamiltonian(wf_full,M=M_full,h_drive=h_drive,h_inter=h_inter,h_detune=h_detune)
    H_samp       = hamiltonian(wf_full,M=M,h_drive=h_drive,h_inter=h_inter,h_detune=h_detune)

    # IO vars
    H_list = []
    O_list = []

    # Get aux vars
    [sigx_full,sigz_full] = H_full.getAuxVars(states)

    # Build quantity to minimise
    denom_full  = tf.reduce_sum(tf.pow(tf.abs(H_full.psi),2.0))
    num_full    = tf.einsum('ij,ij->',tf.conj(H_full.psi),H_full.E_proj_unnorm)
    H_avg_full  = tf.real(tf.divide(num_full,tf.complex(denom_full,np.float64(0.0))))
    
    fullTrainStep   = tf.train.AdamOptimizer(learn_rate_full).minimize(H_avg_full);
    
    H_avg_mc = H_samp.E_locs_mean_re 
    #H_avg_mc = H_samp.E_locs_var_re
    #H_avg_mc = H_samp.E_locs_mean_re + 3*H_samp.E_locs_std_re
    
    mcTrainStep = tf.train.AdamOptimizer(learn_rate_mc).minimize(H_avg_mc)
    
    # Initialise vars
    sess.run(tf.global_variables_initializer())

    # Create sampler
    def probFun(x):
        return np.power(np.abs(np.squeeze(wf_full.eval(x))),2.0)
    
    samp    = sampler(N=N,probFun=probFun)
    mcg1    = mcg(samp,burnIn=burnIn,thinning=thinning)
    sample  = mcg1.getSample_MH(M)

    # Feeding wrappers
    def feed(H,f,S):
        [sigx_t,sigz_t] = H.getAuxVars(S)
        return sess.run(f, feed_dict={H.input_states: S, H.sigx: sigx_t, H.sigz: sigz_t})

    startTime   = timer()
    print("Start full")
    for i in range(full_steps):
        
        print("\nStep %d" % i)
        feed(H_full,fullTrainStep,states)

        H_avg_now   = feed(H_full,H_avg_full,states)
        H_list.append(H_avg_now)
        print('H_avg_full err\t=%4.3e' % (H_avg_now-E_0))
        
        overlap     = computeOverlap(wf_full,wf_exact)
        O_list.append(overlap)
        print('Overlap\t=%4.3e' % overlap)
    
    print("Start mc")
    for j in range(mc_steps):
        
        print("\nStep %d" % j)
        sample = mcg1.getSample_MH(M,useFinal=mcg_useFinal)
        feed(H_samp,mcTrainStep,sample)
        
        H_mc_now   = feed(H_samp,H_avg_mc,sample)
        H_list.append(H_mc_now)
        print('H_avg_full err\t=%4.3e' % (H_mc_now-E_0))
        
        overlap     = computeOverlap(wf_full,wf_exact)
        O_list.append(overlap)
        print('Overlap\t=%4.3e' % overlap)

    endTime     = timer()
    print('Time elapsed\t=%d seconds\n' % (endTime-startTime))

    
    H_array = np.array(H_list)
    O_array = np.array(O_list)
    
    # Save data
    data    = Bunch()
    data.H_array    = H_array
    data.O_array    = O_array
    data.E_0 = E_0

    fileObj     = open(fileOut,'wb')
    pickle.dump(data,fileObj)
    fileObj.close()

    sess.close()
    
def make_plots(fileOut,plotLog=False):
    fileObj     = open(fileOut,'rb')
    data        = pickle.load(fileObj)
    fileObj.close()

    plt.figure(1)
    plt.plot(np.real(data.H_array - data.E_0))
    plt.title('Energy Diff')
    
    plt.figure(2)
    plt.plot(np.real(data.O_array))
    plt.title('Overlap')
    
    plt.show()

def main(argv):
    gen_data(fileOut='pickledData')
    make_plots(fileOut='pickledData')
    
if __name__ == '__main__':
    main(sys.argv[1:])
    