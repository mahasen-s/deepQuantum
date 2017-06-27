# Class 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as nr

class markovChainGenerator():
    # Generates a Markov Chain of states
    # In future this should generate a list of states, which should allow easier naive paralleisation at the cost of vectorisation on the sample index
    # This would also make this code easier to generalise, although maybe not? Apparently we should have many machines searching the phase space concurrently using the same graph?
    #
    # gen_initState might produce an infinite loop. If it works, probOld is guaranteed to be positive and finite, and the only probNew that 
    # gen_propState does not protect against probNew being infinite

    # Needs the following parameters

    #   sampler             : the wavefunction object which has an eval method that accepts single configurations S. Should be a shallow copy
    #   burnIn              : number of samples to be thrown away initially (e.g. 1e3)
    #   thinning            : number of samples to be thrown away between samples of the Markov chain (e.g. 10)

    # sampler needs to have the following attributes
    #   N                   : the number of spins
    #   probDist(S)         : evaluates the probability distribution associated with Psi(S)
    #   propDist(A,B)       : evaluates the conditional proposal distribution p(A|B) i.e. probability of proposing A given B
    # sampler needs to have the following methods
    #   initGen()           : generates a candidate initial state
    # 	propGen(S) 			: generates a candidate state based on a given state S

    # DESIGN ELEMENTS
    # Need to be able to start the chain from an element with non-zero probability
    # Need to be able to propose a start-point
    # New chains should be able to be initialised from the final state of the previous chain

    def __init__(self,sampler,burnIn=None,thinning=None):
        # Should be some sort of checking here to make sure that sampler is as prescribed
        self.sampler = sampler

        if burnIn is None:
            # use some default value
            burnIn = 10*N
        self.burnIn = burnIn

        if thinning is None:
            # use some default value
            thinning = 2*N;
        self.thinning= thinning

        # Create a variable to hold the final state of the previous chain
        self.finalState = None

    def get_initState(self):
        # generates a random initial state. Initial state will be accepted if it has non-zero probability
        prob    = 0;
        while self.isLegalProb(prob)==False:
            # iterate until 0<prob<inf
            state   = self.sampler.initGen()
            prob    = self.sampler.probFun(state)
        self.probOld    = prob
        return state

    def get_propState(self,S):
        # generates a new proposed state given a state S
        accepted = False

        # Check if probability of old state has been calculated
        self.probOld = self.sampler.probFun(S)

        while not accepted:
            Snew            = self.sampler.propGen(S)
            probNew         = self.sampler.probFun(Snew)
            #prAccept 		= min(1,(probNew/self.probOld)*(self.sampler.propFun(S,Snew)/self.sampler.propFun(Snew,S)))
            prAccept        = min(1,(probNew/self.probOld))

            if self.probOld == 0:
                raise ValueError('probOld is zero, everything is broken')
            elif np.isfinite(probNew)== False:
                raise ValueError('probNew is not finite, everything is broken')

            if np.random.rand() < prAccept:
                # step is successful
                accepted = True
                self.probOld= probNew

        return Snew


    def getSample_MH(self,M,initState=None,useFinal=False):
        # generates a Markov Chain of length M using Metropolis-Hastings

        # Get initial state
        if useFinal == True and self.finalState != None:
            # Use end of previous chain
            self.initState = self.finalState
        else:
            # Was an initial state supplied?
            if initState is None:
                self.initState = self.get_initState()
            else:
                # Check if state is valid
                self.probOld    = self.sampler.probFun(initState)
                if self.isLegalProb(self.probOld)==False:
                    raise ValueError('Supplied state has pathological probability')
                else:
                    self.initState = initState

        # Print init
        print('Init state:')
        print(self.initState)
        print(self.sampler.probFun(self.initState))

        # Create array to store states
        samples = np.zeros([self.sampler.N,M])

        # Do burn-in
        currState = self.initState
        for i in range(self.burnIn):
            currState = self.get_propState(currState)

        # Get samples
        for i in range(M):
            for j in range(self.thinning):
                currState       = self.get_propState(currState)
            samples[:,i:i+1]    = currState

        # Store final state
        self.finalState     = currState
        return samples

    def isLegalProb(self,x):
        return np.isfinite(x)==True and x>0

class sampler_TFI:
    def __init__(self,N=None,probFun=None):
        # Input checking
        if N is None:
            raise ValueError('N must be specified')
        else:
            self.N = N

        if probFun is None:
            raise ValueError('probFun must be supplied and must be a unary function on state space which returns the relative probability of a given state')
        else:
            self.probFun = probFun

    def initGen(self):
        # Generates candidate initial state
        return 2*(nr.randint(2,size=[self.N,1]))-1

    def propGen(self,S):
        # Generates candidate state given previous state S
        Snew	= S

        # Flip a random site
        Snew[np.random.randint(self.N)] *= -1

        return Snew
