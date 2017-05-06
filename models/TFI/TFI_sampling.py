# Class 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as nr

class sampling_tfi():
    # Must define 4 factory functions whose outputs are defined by:
    #   stateGen()          : generates random state during Metropolis-Hastings
    #   stateGen_init()     : generates initial random state for Metropolis-Hastings
    #   probDist(S,psiFun)  : evaluates the probability distribution associated with Psi(S)
    #   propDists(A,B)      : evaluates the conditional proposal distribution p(A|B) i.e. probability of proposing A given B
    def __init__(self):
        self.make_stateGen_init = self.make_stateGen;

    def make_stateGen(self,N):
        def stateGen():
            sample  = nr.randint(2, size=[N,1]);
            sample  = 2*sample -1;              # Use {-1,1}
            return sample
        return stateGen

    def make_probDist(self,N):
        def probDist(S,psiFun):
            psi         = psiFun(S);
            psi         = pow(abs(psi),2);  # encountered overflow for some pathological S
            return psi
        return probDist

    def make_propDist(self,N):
        def propDist(A,B):
            return 1
        return propDist

    def getAuxVars(self,S):
        # Get N
        N       = S.shape[0];

        # Get number of spin-aligned pairs in each sample
        sigz    = np.add.reduce(np.abs(S + np.roll(S,1,0)),0)/2;

        # Get NxNxM array of single-site flipped states
        sigx    = np.einsum('ij,ai->aij',S,(np.ones(N) - 2*np.eye(N)))

        return sigx,sigz;
