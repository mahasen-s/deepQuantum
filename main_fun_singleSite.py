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

from models.TFI.TFI_sampling_singleSite import markovChainGenerator as mcg
from models.TFI.TFI_sampling_singleSite import sampler_TFI as sampler
from models.TFI.TFI_hamiltonian import hamiltonian_tfi as hamiltonian

def run_net(N=2,alpha=4,learn_rate=0.1,optim='gradient_descent',M=100,mcs=1000,h_drive=1,h_inter=0.5,h_detune=0,wf_exact=[],fileOut=[],verbose=False,nn_type='deep',layers_num=2):
    P           = alpha*N;

    # Instantiate TF session
    sess        = tf.Session();

    # Instantiate model: wavefunction, hamiltonian, sampling
#    wf          = wavefunction(sess,input_num=N,hidden_num=P,nn_type='deep',layers_num=4);
    wf          = wavefunction(sess,input_num=N,hidden_num=P,nn_type=nn_type,layers_num=layers_num);
    def costFun(x):
        psi = np.power(np.abs(wf.eval(x)),2.0)
        return psi
    samp        = sampler(N=N,probFun=costFun)
    mcgFun      = mcg(samp,burnIn=10*N,thinning=2*N)
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
    minVar  = tf.real(H.E_var)

    if optim == 'gradient_descent':
        trainStep   = tf.train.GradientDescentOptimizer(learn_rate).minimize(minVar);
    elif optim == 'adagrad':
        trainStep   = tf.train.AdagradOptimizer(learn_rate).minimize(minVar);
    elif optim == 'adam':
        trainStep   = tf.train.AdamOptimizer(learn_rate).minimize(minVar);


    # Initialize variables
    sess.run(tf.global_variables_initializer());

    if verbose:
        print('Biases')
        print(sess.run(wf.biases))
        print('Weights')
        print(sess.run(wf.weights))

    # Run
    startTime   = timer();
    for i in range(mcs):
        # Print values of network parameters
    #    print(sess.run(wf.a));
    #    print(sess.run(wf.b));

        # Get new sample
        if verbose:
            print('Begin sample')
        sample          = mcgFun.getSample_MH(M,useFinal=False)
        [sigx,sigz]     = H.getAuxVars(sample);

        # Run 1 training step. sample should be implicitly cast to an appropriate type
        if verbose:
            print('Begin step')
        sess.run(trainStep, feed_dict={H.input_states: sample, H.sigx: sigx, H.sigz: sigz});
        E_var_current   = sess.run(H.E_var, feed_dict={H.input_states: sample, H.sigx: sigx, H.sigz: sigz})
        E_vals_current  = sess.run(H.E_vals, feed_dict={H.input_states: sample, H.sigx: sigx, H.sigz: sigz})
       # print("MC iter\t= %d\nE_var\t= %4.3f\n\n" % (i,E_var_current) );

        print('E_var\t\t\t= %4.3e' % E_var_current)
        print('Mean E_local real\t= %4.3e' % np.mean(np.real(E_vals_current)))
        print('Mean E_local imag\t= %4.3e' % np.mean(np.imag(E_vals_current)))
        print('Std. E_local real\t= %4.3e' % np.std(np.real(E_vals_current)))
        print('Std. E_local imag\t= %4.3e' % np.std(np.imag(E_vals_current)))

        E_var_list[i] = E_var_current;

        # Compute overlap
        if computeOverlapFlag == True:
            overlap = computeOverlap(wf,wf_exact)
            print('Overlap\t\t\t= %4.3e\n' % overlap)
            Overlap_list[i] = overlap

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
    np.savez(fileOut, E_var_list=E_var_list, Overlap_list=Overlap_list,mcs=mcs)

    ## PLOT RESULTS
#    plt.plot(np.arange(mcs),E_var_list)
#    plt.xlabel('MC step')
#    plt.ylabel('Groundstate energy')
#    plt.grid(True)
#    plt.show()
