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

def test_fullStateSpace_as_sample():
    # Calculate variational energy using the full set of states
    N       = 4
    alpha   = 4
    mcs     = 250
    optim   = 'gradient_descent'
    learn_rate  = 0.05

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

    sess.run(tf.global_variables_initializer())

    # IO vars
    H_avg_list  = np.zeros(mcs,dtype=np.float)
    OverlapList = np.zeros(mcs,dtype=np.float)

    # Exact 
    #exactVals   = pyMatLoad('./exact/results/exact_TFI_hDrive=1p0_hInter=0p5_hDetune=0p0.mat','data')
    exactVals   = pyMatLoad('test.mat','data')
    wf_exact    = exactVals[N-2]

    # Print psi, local energy
    [sigx,sigz] = H.getAuxVars(states)
    print("\n<state|Psi> for each state in space")
    psi_vals    = sess.run(H.psi, feed_dict={H.input_states: states, H.sigx: sigx, H.sigz: sigz})
    print(psi_vals)

    print("\nLocal energy for each state in space")
    local_Evals = sess.run(H.E_vals, feed_dict={H.input_states: states, H.sigx: sigx, H.sigz: sigz})
    print(local_Evals)

    # Build quantity to minimise
    #H_avg       = tf.divide(tf.reduce_sum(tf.multiply(H.psi,H.E_vals)), tf.multiply(tf.conj(H.psi),H.psi))
    denom       = tf.reduce_sum(tf.pow(tf.abs(H.psi),2.0))
    num         = tf.einsum('ij,ij->',H.psi,H.E_vals)
    H_avg       = tf.divide(num,tf.complex(denom,np.float64(0.0)))

    print('\nInitial H_avg')
    print(sess.run(H_avg, feed_dict={H.input_states: states, H.sigx: sigx, H.sigz: sigz}))

    H_avg       = tf.real(H_avg)

    if optim == 'gradient_descent':
        trainStep   = tf.train.GradientDescentOptimizer(learn_rate).minimize(H_avg);
    elif optim == 'adagrad':
        trainStep   = tf.train.AdagradOptimizer(learn_rate).minimize(H_avg);
    elif optim == 'adam':
        trainStep   = tf.train.AdamOptimizer(learn_rate).minimize(H_avg);

    startTime   = timer()
    for i in range(mcs):
        psi_vals    = sess.run(trainStep, feed_dict={H.input_states: states, H.sigx: sigx, H.sigz: sigz})

        # Compute H_avg
        H_avg_now   = sess.run(H_avg, feed_dict={H.input_states: states, H.sigx: sigx, H.sigz: sigz})
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
    psi_vals    = sess.run(H.psi, feed_dict={H.input_states: states, H.sigx: sigx, H.sigz: sigz})
    print(psi_vals)

    print("Local energy for each state in space, end of sim")
    local_Evals = sess.run(H.E_vals, feed_dict={H.input_states: states, H.sigx: sigx, H.sigz: sigz})
    print(local_Evals)

    print("Error = %4.3e\n" % (sess.run(H_avg, feed_dict={H.input_states: states, H.sigx: sigx, H.sigz: sigz})-wf_exact[2]))

    sess.close()

    plt.figure(1)
    plt.subplot(211)
    plt.plot(np.arange(mcs),H_avg_list,marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('H_avg')
    plt.title('hDrive=1, hInter=0.5')
    plt.grid(True)

    plt.subplot(212)
    plt.plot(np.arange(mcs),OverlapList,marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Overlap')
    plt.grid(True)

    plt.show()

test_fullStateSpace_as_sample()
