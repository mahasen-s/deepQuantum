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

class wf_test():
    def __init__(self,sess,wf_fun,input_num=2,output_num=1,nStates=1):
        self.sess = sess
        self.input_num = input_num
        self.output_num = output_num
        self.wf_fun = wf_fun # function which is actually evaluated to return psi

        self.input_state = tf.placeholder(conf.DTYPE,[self.input_num,nStates])
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
    N   = tf.constant(-caller.input_num+2,dtype=conf.DTYPE)
    div = tf.constant(1/np.sqrt(caller.input_num),dtype=conf.DTYPE)
    psi = tf.cast(tf.equal(tf.einsum('ijk->ik',input_states),N),conf.DTYPE)
    return psi

def wf_absSum_fun(caller,input_states):
    psi = tf.cast(tf.abs(tf.einsum('ijk->ik',input_states)),conf.DTYPE)
    return psi

def wf_rand_fun_maker(N):
    probs   = nr.rand(2**N).reshape([2**N,1])
    states,_= genAllStates(N)
    states  = np.transpose(states)

    def wf_rand_fun(state):
        S = state.reshape(1,N)

        for i in range(2**N):
            if (states[i,:]==S).all()==True:
                # match
                break

        return probs[i]

    return probs, states, wf_rand_fun

def test_wf():
    N       = 3;
    sess    = tf.Session();

    # Create states
    wf_GHZ  = wf_test(sess,wf_GHZ_fun,input_num=N,nStates=5)
    wf_W    = wf_test(sess,wf_W_fun,input_num=N,nStates=5)

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

    print("States")
    print(test)

    print("Wavefunction tests:")
    print(wf_GHZ.eval(test))
    print(wf_W.eval(test))

    print("Hamiltonian tests:" )
    # Print
    print(E_GHZ)
    print(E_W)

    print("Sigz")
    print(sigz)
    print("Sigx")
    print(sigx)

    print("New test")
    print(sess.run(H_GHZ.input_states, feed_dict={H_GHZ.input_states:test}))

    print("Test multiply")
    print(sess.run(tf.multiply(wf_GHZ.buildOp(H_GHZ.input_states),sigz), feed_dict={H_GHZ.input_states:test}))
    print(sess.run(tf.multiply(wf_W.buildOp(H_W.input_states),sigz), feed_dict={H_W.input_states:test}))

def test_mcg():
    N = 5
    sess = tf.Session()

    # Create states
    wf_W    = wf_test(sess,wf_W_fun,input_num=N)

    # initialise variables
    sess.run(tf.global_variables_initializer())

    # Import 
    from models.TFI.TFI_sampling_singleSite import markovChainGenerator as mcg
    from models.TFI.TFI_sampling_singleSite import sampler_TFI as sampler

    M = 20;
    samp    = sampler(N=N,probFun=wf_W.eval)
    mcg1    = mcg(samp,burnIn=10,thinning=1)

    # Get sample, unseeded
    S       = mcg1.getSample_MH(M)
    print('\nMarkov chain')
    print(repr(np.transpose(S)))

def test_mcg_nonUniformDist():
    # Produce random distribution, then use mcg to produce sample which has the same statistics
    N  = 5
    M = 200;

    from models.TFI.TFI_sampling_singleSite import markovChainGenerator as mcg
    from models.TFI.TFI_sampling_singleSite import sampler_TFI as sampler

    # Create list of states, probs, psi
    probs, states, wf_rand_fun = wf_rand_fun_maker(N)
    probs   = np.divide(probs,np.sum(probs))
    samp    = sampler(N=N,probFun=wf_rand_fun)
    mcg1    = mcg(samp,burnIn=100,thinning=10)

    # Calculate analytics
    # mean
    mean_analytic   = np.sum(np.multiply(states,probs),axis=0)
    std_analytic    = np.sqrt(np.sum(np.multiply(np.power(np.subtract(states,mean_analytic),2.0),probs),axis=0))

    print('Mean and corrected StdDev at each site are')
    print(mean_analytic)
    print(std_analytic)

    # Get sample
    S       = mcg1.getSample_MH(M)
    S       = np.transpose(S)

    # Calcualte quantities
    mean_sample     = np.divide(np.sum(S,axis=0),M)
    std_sample      = np.sqrt(np.divide(np.sum(np.power(np.subtract(S,mean_sample),2.0),axis=0),M-1))


    print('\nMean and corrected StdDev at each site from sample')
    print(mean_sample)
    print(std_sample)

    # Compare difference between analytic and sample means relative to analytic standard deviation
    d_mean  = np.abs(np.subtract(mean_analytic,mean_sample))
    mean_std= np.divide(std_analytic,np.sqrt(M))
    d_norm  = np.divide(d_mean,mean_std)
    print('Distance of sample mean from analytic mean, normalised by std.Dev of mean')
    print(d_norm)
    print('Average distance')
    print(np.mean(d_norm))

def test_mcg_nonUniformDist_Plot():
    # Same as before, but with plotting!
    N = 5

    from models.TFI.TFI_sampling_singleSite import markovChainGenerator as mcg
    from models.TFI.TFI_sampling_singleSite import sampler_TFI as sampler
    import matplotlib.pyplot as plt

    # Create list of states, probs, psi
    probs, states, wf_rand_fun = wf_rand_fun_maker(N)
    probs   = np.divide(probs,np.sum(probs))
    samp    = sampler(N=N,probFun=wf_rand_fun)
    mcg1    = mcg(samp,burnIn=100,thinning=10)

    # Calculate analytics
    mean_analytic   = np.sum(np.multiply(states,probs),axis=0)
    std_analytic    = np.sqrt(np.sum(np.multiply(np.power(np.subtract(states,mean_analytic),2.0),probs),axis=0))

    print('Mean and corrected StdDev at each site are')
    print(mean_analytic)
    print(std_analytic)

    Ml = np.linspace(1000,10000,10)
    M = np.repeat(Ml,10)
    means   = np.zeros([len(M),N])
    stds    = np.zeros([len(M),N])

    for i in range(len(M)):
        print('Processing m=%d\n' % M[i])
        m   = M[i]
        # Get sample
        S       = mcg1.getSample_MH(m.astype(int))
        S       = np.transpose(S)

        # Calcualte quantities
        mean_sample     = np.divide(np.sum(S,axis=0),m)
        std_sample      = np.sqrt(np.divide(np.sum(np.power(np.subtract(S,mean_sample),2.0),axis=0),m-1))

        means[i,:]      = mean_sample
        stds[i,:]       = std_sample

    d_mean  = np.abs(np.subtract(means,mean_analytic))
    d_mean_avg = np.divide(np.sum(d_mean,axis=1),N)
    d_norms = np.zeros([len(M),N])

    for i in range(len(M)):
        mean_std = np.divide(std_analytic,np.sqrt(M[i]))
        d_norms[i,:] = np.divide(d_mean[i,:],mean_std)

    d_norms_avg = np.divide(np.sum(d_norms,axis=1),N)
    np.savez('plotNonUniDist',means=means,stds=stds,probs=probs,mean_analytic=mean_analytic,std_analytic=std_analytic,d_mean=d_mean,d_mean_avg=d_mean_avg,d_norms=d_norms,d_norms_avg=d_norms_avg)

    f, axarr = plt.subplots(2, sharex=True)
    d_mean_avg_plot = np.divide(np.sum(np.reshape(d_mean_avg,[len(Ml),10]),axis=1),10)
    d_norms_avg_plot = np.divide(np.sum(np.reshape(d_norms_avg,[len(Ml),10]),axis=1),10)
    axarr[0].plot(Ml,d_mean_avg_plot,marker='o')
    axarr[0].set_ylabel('Abs. Avg. Dist.')
    axarr[0].grid(True)
    axarr[1].plot(Ml,d_norms_avg_plot,marker='o')
    axarr[1].set_ylabel('Normed Avg. Dist.')
    axarr[1].set_xlabel('Samples')
    axarr[1].grid(True)
    plt.show()

def test_fullStateSpace_as_sample(N=4,alpha=4,mcs=250,learn_rate=0.05,optim='gradient_descent',h_drive=1,h_inter=0,h_detune=0):
    # Calculate variational energy using the full set of states

    states,_= genAllStates(N)
    print("States:")
    print(states)

    M       = states.shape[1]

    # construct wavefunction
    P       = alpha*N
    sess    = tf.Session()
    wf      = wavefunction(sess,input_num=N,hidden_num=P,nn_type='deep',layers_num=2)

    # construct Hamiltonian
    H       = hamiltonian(wf,M=M,h_drive=1,h_inter=0.5,h_detune=0)

    # IO vars
    H_avg_list  = np.zeros(mcs,dtype=np.float)
    OverlapList = np.zeros(mcs,dtype=np.float)

    # Exact 
    exactFile   = "/exact/results/exact_TFI_hDrive=%2.1f_hInter=%2.1f_hDetune=%2.1f" % (h_drive,h_inter,h_detune)
    exactFile   = '.' + exactFile.replace('.','p') + '.mat'
    try:
        exactVals   = pyMatLoad(exactFile,'data')
    except:
        raise ValueError('File not found: %s' % exactFile)


    wf_exact    = exactVals[N-2]

    # Get aux vars
    [sigx,sigz] = H.getAuxVars(states)

    # Build quantity to minimise
    #H_avg       = tf.divide(tf.reduce_sum(tf.multiply(H.psi,H.E_vals)), tf.multiply(tf.conj(H.psi),H.psi))
    denom       = tf.reduce_sum(tf.pow(tf.abs(H.psi),2.0))
    num         = tf.einsum('ij,ij->',H.psi,H.E_vals)
    H_avg       = tf.divide(num,tf.complex(denom,np.float64(0.0)))

    H_avg       = tf.real(H_avg)

    if optim == 'gradient_descent':
        trainStep   = tf.train.GradientDescentOptimizer(learn_rate).minimize(H_avg);
    elif optim == 'adagrad':
        trainStep   = tf.train.AdagradOptimizer(learn_rate).minimize(H_avg);
    elif optim == 'adam':
        trainStep   = tf.train.AdamOptimizer(learn_rate).minimize(H_avg);

    sess.run(tf.global_variables_initializer())

    # Feeding wrapper
    def feed(f):
        return sess.run(f, feed_dict={H.input_states: states, H.sigx: sigx, H.sigz: sigz})

    # Print values
    print("\n<state|Psi> for each state in space")
    psi_vals    = feed(H.psi)
    print(psi_vals)

    print("\nLocal energy for each state in space")
    local_Evals = feed(H.E_vals)
    print(local_Evals)

    print('\nInitial H_avg')
    print(feed(H_avg))

    startTime   = timer()

    for i in range(mcs):
        print("Epoch %d" % i)
        psi_vals    = feed(trainStep)

        # Compute H_avg
        H_avg_now   = feed(H_avg)
        print('H_avg\t=%4.3e' % H_avg_now)
        H_avg_list[i]   = H_avg_now

        # Compute overlap
        overlap     = computeOverlap(wf,wf_exact)
        print('Overlap\t=%4.3e\n' % overlap)
        OverlapList[i]  = overlap

    endTime     = timer()
    print('Time elapses\t=%d seconds\n' % (endTime-startTime))
    np.savez('test_fullStateSpace_as_sample_data',H_avg_list=H_avg_list,OverlapList=OverlapList,mcsList=range(mcs))

    print("<state|Psi> for each state in space, end of sim")
    psi_vals    = feed(H.psi)
    print(psi_vals)

    print("Local energy for each state in space, end of sim")
    local_Evals = feed(H.E_vals)
    print(local_Evals)

    print("Error = %4.3e\n" % (feed(H_avg)-wf_exact[2]))
    sess.close()

    return np.arange(mcs), H_avg_list, OverlapList, wf_exact[2]

def test_fullStateSpace_as_sample_plot():
    # Plot H_avg as function of epoch

    mcs_list, H_avg_list, OverlapList,_ = test_fullStateSpace_as_sample(N=4,alpha=4,mcs=250,learn_rate=0.05,optim='gradient_descent',h_drive=1,h_inter=0.5,h_detune=0)
    plt.figure(1)
    plt.subplot(211)
    plt.plot(mcs_list,H_avg_list,marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Abs(H_avg-E_0)')
    plt.title('hDrive=1, hInter=0.5')
    plt.grid(True)

    plt.subplot(212)
    plt.plot(mcs_list,OverlapList,marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Overlap')
    plt.grid(True)

    plt.show()

def test_fullStateSpace_as_sample_plot_multi():
    # Plot absolute difference between deepQuantum and analytic
    mcs     = 4000;
    optim = 'adam'
    lrate=0.0025
    mcs_list, H1, _, E1 = test_fullStateSpace_as_sample(N=6,alpha=6,mcs=mcs,learn_rate=lrate,optim=optim,h_drive=1,h_inter=0.5,h_detune=0)
    mcs_list2, H2, _ , E2 = test_fullStateSpace_as_sample(N=8,alpha=6,mcs=mcs,learn_rate=lrate,optim=optim,h_drive=1,h_inter=0.5,h_detune=0)
    _, H3, _ , E3 = test_fullStateSpace_as_sample(N=6,alpha=6,mcs=mcs,learn_rate=lrate,optim=optim,h_drive=0.5,h_inter=1,h_detune=0)
    _, H4, _ , E4= test_fullStateSpace_as_sample(N=8,alpha=6,mcs=mcs,learn_rate=lrate,optim=optim,h_drive=0.5,h_inter=1,h_detune=0)

    np.savez('data_FSS_as_sample_multi',mcs_list=mcs_list,mcs_list2=mcs_list2,H1=H1,H2=H2,H3=H3,H4=H4,E1=E1,E2=E2,E3=E3,E4=E4)
    def wrapFun(x):
        return np.abs(x)

    def plotFun(*args,**kwargs):
        return plt.semilogy(*args,**kwargs)

    plt.figure(1)
    plt.subplot(221)
    plotFun(mcs_list,wrapFun(H1-E1),marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Abs(H_avg-E_0)')
    plt.title('N=6, hDrive=1, hInter=0.5, learn_rate=0.0025')
    plt.grid(True)

    plt.subplot(222)
    plotFun(mcs_list,wrapFun(H3-E3),marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Abs(H_avg-E_0)')
    plt.title('N=6, hDrive=0.5, hInter=1, learn_rate=0.0025')
    plt.grid(True)

    plt.subplot(223)
    plotFun(mcs_list2,wrapFun(H2-E2),marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Abs(H_avg-E_0)')
    plt.title('N=8, hDrive=1, hInter=0.5, learn_rate=0.0025')
    plt.grid(True)

    plt.subplot(224)
    plotFun(mcs_list2,wrapFun(H4-E4),marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Abs(H_avg-E_0)')
    plt.title('N=8, hDrive=0.5, hInter=1, learn_rate=0.0025')
    plt.grid(True)

    plt.show()

def test_fullStateSpace_and_MC_as_sample(N=4,alpha=4,mcs=250,M_samp=10,learn_rate=0.05,optim='gradient_descent',h_drive=1,h_inter=0,h_detune=0,mcg_useFinal=False,fileOut='pickledData'):
    # Calculate variational energy using the full set of states and compare with a MC sample
    from models.TFI.TFI_sampling_singleSite import markovChainGenerator as mcg
    from models.TFI.TFI_sampling_singleSite import sampler_TFI as sampler

    # EXACT VARS
    exactFile   = "/exact/results/exact_TFI_hDrive=%2.1f_hInter=%2.1f_hDetune=%2.1f" % (h_drive,h_inter,h_detune)
    exactFile   = '.' + exactFile.replace('.','p') + '.mat'
    try:
        exactVals   = pyMatLoad(exactFile,'data')
    except:
        raise ValueError('File not found: %s' % exactFile)
    wf_exact    = exactVals[N-2]

    # FULL STATE SPACE
    states,_= genAllStates(N)
    print("States:")
    print(states)
    M_full  = states.shape[1]

    # construct wavefunction
    P       = alpha*N
    sess    = tf.Session()
    wf_full = wavefunction(sess,input_num=N,hidden_num=P,nn_type='deep',layers_num=2)

    # construct Hamiltonian
    H_full       = hamiltonian(wf_full,M=M_full,h_drive=1,h_inter=0.5,h_detune=0)
    H_samp       = hamiltonian(wf_full,M=M_samp,h_drive=1,h_inter=0.5,h_detune=0)

    # IO vars
    H_list_full = np.zeros(mcs,dtype=np.float)
    OL_list_full= np.zeros(mcs,dtype=np.float)
    var_list_full=np.zeros(mcs,dtype=np.float)
    E_vals_list_full    = np.zeros([mcs,M_full],dtype=complex)

    # Get aux vars
    [sigx_full,sigz_full] = H_full.getAuxVars(states)

    # Build quantity to minimise
    denom_full  = tf.reduce_sum(tf.pow(tf.abs(H_full.psi),2.0))
    num_full    = tf.einsum('ij,ij->',tf.conj(H_full.psi),H_full.E_proj_unnorm)
    H_avg_full  = tf.real(tf.divide(num_full,tf.complex(denom_full,np.float64(0.0))))

    denom_samp  = tf.reduce_sum(tf.pow(tf.abs(H_samp.psi),2.0))
    num_samp    = tf.einsum('ij,ij->',tf.conj(H_samp.psi),H_samp.E_proj_unnorm)
    H_avg_samp  = tf.real(tf.divide(num_samp,tf.complex(denom_samp,np.float64(0.0))))

    if optim == 'gradient_descent':
        trainStep   = tf.train.GradientDescentOptimizer(learn_rate).minimize(H_avg_full);
    elif optim == 'adagrad':
        trainStep   = tf.train.AdagradOptimizer(learn_rate).minimize(H_avg_full);
    elif optim == 'adam':
        trainStep   = tf.train.AdamOptimizer(learn_rate).minimize(H_avg_full);

    # SAMPLE SPACE
    H_list_samp = np.zeros(mcs,dtype=np.float)
    OL_list_samp= np.zeros(mcs,dtype=np.float)
    var_list_samp= np.zeros(mcs,dtype=np.float)
    E_vals_list_samp    = np.zeros([mcs,M_samp],dtype=complex)

    # Initialise vars
    sess.run(tf.global_variables_initializer())

    # Create sampler
    def probFun(x):
        return np.power(np.abs(np.squeeze(wf_full.eval(x))),2.0)
    samp    = sampler(N=N,probFun=probFun)
    mcg1    = mcg(samp,burnIn=5*N,thinning=2*N)
    sample  = mcg1.getSample_MH(M_samp)

    # Feeding wrappers
    def feed(H,f,S):
        [sigx_t,sigz_t] = H.getAuxVars(S)
        return sess.run(f, feed_dict={H.input_states: S, H.sigx: sigx_t, H.sigz: sigz_t})

    # Print values
    print("\n<state|Psi> for each state in space")
    psi_vals    = feed(H_full,H_full.psi,states)
    print(psi_vals)

    print("\nLocal energy for each state in space")
    local_Evals = feed(H_full,H_full.E_locs,states)
    print(local_Evals)

    print('\nInitial H_avg over all states')
    print(feed(H_full,H_avg_full,states))

    print('\nInitial H_avg over all sample')
    print(feed(H_samp,H_avg_samp,sample))

    startTime   = timer()

    for i in range(mcs):
        print("\nEpoch %d" % i)
        psi_vals    = feed(H_full,trainStep,states)

        # Get new sample
        sample = mcg1.getSample_MH(M_samp,useFinal=mcg_useFinal)

        # Compute H_avg
        H_avg_now   = feed(H_full,H_avg_full,states)
        print('H_avg_full err\t=%4.3e' % (H_avg_now-wf_exact[2]))
        H_list_full[i]   = H_avg_now

        H_avg_now   = feed(H_samp,H_avg_samp,sample)
        print('H_avg_samp err\t=%4.3e' % (H_avg_now-wf_exact[2]))
        H_list_samp[i]   = H_avg_now

        # Compute overlap
        overlap     = computeOverlap(wf_full,wf_exact)
        print('Overlap\t=%4.3e' % overlap)
        OL_list_full[i]  = overlap

        # Compute local energies and store
        E_vals_list_full[i:i+1,:]   = feed(H_full,H_full.E_locs,states)
        E_vals_list_samp[i:i+1,:]   = feed(H_samp,H_samp.E_locs,sample)

    endTime     = timer()
    print('Time elapsed\t=%d seconds\n' % (endTime-startTime))


    # Save data
    data    = Bunch()
    data.H_list_full    = H_list_full
    data.H_list_samp    = H_list_samp
    data.OL_list_full   = OL_list_full
    data.E_vals_list_full   = E_vals_list_full
    data.E_vals_list_samp   = E_vals_list_samp
    data.N              = N
    data.alpha          = alpha
    data.mcs            = mcs
    data.h_drive        = h_drive
    data.h_inter        = h_inter
    data.h_detune       = h_detune
    data.M              = M_samp
    data.mcs_list       = np.arange(mcs)
    data.E_0            = wf_exact[2]
    data.learn_rate     = learn_rate

    fileObj     = open(fileOut,'wb')
    pickle.dump(data,fileObj)
    fileObj.close()

    print("<state|Psi> for each state in space, end of sim")
    psi_vals    = feed(H_full,H_full.psi,states)
    print(psi_vals)

    print("Local energy for each state in space, end of sim")
    local_Evals = feed(H_full,H_full.E_locs,states)
    print(local_Evals)

    print("Full Error \t= %4.3e\n" % (feed(H_full,H_avg_full,states)-wf_exact[2]))
    print("Samp Error \t= %4.3e\n" % (feed(H_samp,H_avg_samp,sample)-wf_exact[2]))
    sess.close()

def test_fullStateSpace_and_MC_as_sample_run():
    N           = 4
    alpha       = 4
    mcs         = 200
    learn_rate  = 0.05
    h_drive     = 1
    h_inter     = 0.5
    h_detune    = 0
    fileOut     = 'test_data1'

    test_fullStateSpace_and_MC_as_sample(   N=N,alpha=alpha,mcs=mcs,learn_rate=learn_rate,\
                                            optim='adam',\
                                            h_drive=h_drive,h_inter=h_inter,h_detune=h_detune,\
                                            mcg_useFinal=True, \
                                            fileOut     = fileOut)


def test_fullStateSpace_and_MC_as_sample_plot(fileOut,plotLog=False):
    fileObj     = open(fileOut,'rb')
    data        = pickle.load(fileObj)
    fileObj.close()

    def plotFun(x,y,label):
        if plotLog == False:
            handle  = plt.plot(x,y,marker='o',label=label)
        else:
            handle  = plt.semilogy(x,np.abs(y),marker='o',label=label)
        return handle

    plt.figure(1)
    # Plot <H>_var-E_0
    plt.subplot(231)
    p1,  = plotFun(data.mcs_list,data.H_list_samp-data.E_0,'Sample')
    p2,  = plotFun(data.mcs_list,data.H_list_full-data.E_0,'Full space')
    plt.xlabel('Step')
    plt.ylabel('H_avg-E_0')
    plt.title('N=%d, hDrive=%2.1f, hInter=%2.1f, learn_rate=%2.2e' % (data.N,data.h_drive,data.h_inter,data.learn_rate))
    plt.grid(True)
    plt.legend(handles=[p1,p2])

    # Plot distance from exact
    plt.subplot(234)
    plotFun(data.mcs_list,data.OL_list_full-1,'Overlap')
    plt.xlabel('Step')
    plt.ylabel('Overlap-1')
    plt.title('Distance from perfect overlap of NN and Exact wf')
    plt.grid(True)

    # Plot mean of real part of local energies at each step
    plt.subplot(232)
    p2, = plotFun(data.mcs_list,np.mean(np.real(data.E_vals_list_samp),axis=1),'Sample')
    plt.xlabel('Step')
    plt.ylabel('mean(re(E_loc))')
    plt.title('Mean of real part of E_loc')
    plt.grid(True)

    # Plot mean of imag part of local energies at each step
    plt.subplot(233)
    p1, = plotFun(data.mcs_list,np.mean(np.imag(data.E_vals_list_samp),axis=1),'Sample')
    p2, = plotFun(data.mcs_list,np.mean(np.imag(data.E_vals_list_full),axis=1),'Full space')
    plt.xlabel('Step')
    plt.ylabel('mean(im(E_loc))')
    plt.title('Mean of imag part of E_loc')
    plt.grid(True)

    # Plot unbiased var of real part of local energies at each step
    plt.subplot(235)
    p1, = plotFun(data.mcs_list,np.var(np.real(data.E_vals_list_full),axis=1,ddof=1),'Full space')
    plt.xlabel('Step')
    plt.ylabel('var(re(E_loc))')
    plt.title('Var in real part of E_loc')
    plt.grid(True)

    # Plot unbiased var of imag part of local energies at each step
    plt.subplot(236)
    p1, = plotFun(data.mcs_list,np.var(np.imag(data.E_vals_list_full),axis=1,ddof=1),'Full space')
    plt.xlabel('Epoch')
    plt.ylabel('var(im(E_loc))')
    plt.title('Var in imag part of E_loc')
    plt.grid(True)

    plt.show()

#test_fullStateSpace_and_MC_as_sample_run()
test_fullStateSpace_and_MC_as_sample_plot('test_data1',plotLog=True)
