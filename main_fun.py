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

def run_net(N=2,alpha=2,learn_rate=0.1,optim='gradient_descent',M=100,mcs=1000,h_drive=1,h_inter=0.5,h_detune=0,wf_exact=[],fileOut=[],verbose=False):
    P           = alpha*N;

    # Instantiate TF session
    sess        = tf.Session();

    # Instantiate model: wavefunction, hamiltonian, sampling
    wf          = wavefunction(sess,input_num=N,hidden_num=P,nn_type='deep',layers_num=2);
    sampling    = sampling_fun();
    samp        = metropolisSample(N,sampling);
    H           = hamiltonian(wf,M=M,h_drive=h_drive,h_inter=h_inter,h_detune=h_detune);

    # IO variables
    E_var_list  = np.zeros(mcs,dtype=np.float);
    Overlap_list= np.zeros(mcs,dtype=np.float);

    # Exact file
    computeOverlapFlag = False
    if len(wf_exact):
        if wf_exact[0]==N and len(wf_exact[1])==pow(2,N):
            computeOverlapFlag = True

    # Build training
    if optim == 'gradient_descent':
        trainStep   = tf.train.GradientDescentOptimizer(learn_rate).minimize(H.E_var);
    elif optim == 'adagrad':
        trainStep   = tf.train.AdagradOptimizer(learn_rate).minimize(H.E_var);
    elif optim == 'adam':
        trainStep   = tf.train.AdamOptimizer(learn_rate).minimize(H.E_var);


    # Initialize variables
    sess.run(tf.global_variables_initializer());

    if verbose:
        print('Biases')
        print(sess.run(wf.biases))
        print('Weights')
        print(sess.run(wf.weights))

    # Run
    startTime   = timer();
    for i in range(1,mcs):
        # Print values of network parameters
    #    print(sess.run(wf.a));
    #    print(sess.run(wf.b));

        # Get new sample
        if verbose:
            print('Begin sample')
        sample          = samp.getSample(M,wf.eval);
        [sigx,sigz]     = H.getAuxVars(sample);

        # Run 1 training step. sample should be implicitly cast to an appropriate type
        if verbose:
            print('Begin step')
        sess.run(trainStep, feed_dict={H.input_states: sample, H.sigx: sigx, H.sigz: sigz});
        E_var_current   = sess.run(H.E_var, feed_dict={H.input_states: sample, H.sigx: sigx, H.sigz: sigz});
       # print("MC iter\t= %d\nE_var\t= %4.3f\n\n" % (i,E_var_current) );
       # Why is E_var three elements?
        print('E_var\t=%4.3e' % E_var_current)
        E_var_list[i-1] = E_var_current;

        # Compute overlap
        if computeOverlapFlag == True:
            overlap = computeOverlap(wf,wf_exact)
            print('Overlap\t=%4.3e' % overlap)
            Overlap_list[i-1] = overlap

        if verbose:
            print('Biases')
            print(sess.run(wf.biases))
            print('Weights')
            print(sess.run(wf.weights))
            print('End step')

    # Close session, free resources
    sess.close();
    endTime     = timer();
    print('Time elapsed\t=%d seconds\n' % (endTime-startTime))

    ## SAVE RESULTS TO FILE
    np.savetxt(fileOut, (E_var_list,Overlap_list))
    np.savez(fileOut, E_var_list=E_var_list, Overlap_list=Overlap_list)

    ## PLOT RESULTS
    #plt.semilogy(np.arange(mcs),E_var_list)
    #plt.xlabel('MC step')
    #plt.ylabel('Groundstate energy')
    #plt.grid(True)
    #plt.show()
