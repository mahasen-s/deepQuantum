# Class for metropolis sampling
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as nr
import numpy.linalg as nl

from . import config as conf

class metropolisSample():
    # Produces object which can be used to generate metropolis samplings from provided functions
    # which generate states S, Pr(S), and proposal probability Q(S,S') 

    def __init__(self,N,sampler):
        # N is the number of elements in each sample
        # M is the number of samples, i.e. length of walk
        # sampler is an object which has the following methods:
            # stateGen is some function which is able to produce random state configuration during the random walk
            # stateGen_init is some function which produces a random state config to start the random walk. E.g. it may be that this is the same as stateGen or that the initial proposed states should be conditioned on some other distribution
            # probDist is some function s.t. probDist(S) is the probability distribution associated with the configuration S
            # propDist is some function s.t. propDist(S,S') is the conditional proposal distribution, i.e. probability config S will be proposed given S'. It needs methods for conditional proposals and initial proposals
        self.N              = N;

        # Construct generators, probability fns from factory functions 
        self.stateGen       = sampler.make_stateGen(N);
        self.stateGen_init  = sampler.make_stateGen_init(N);
        self.probDist       = sampler.make_probDist(N);
        self.propDist       = sampler.make_propDist(N);

    def getSample(self,M,psiFun):
        probOld             = np.inf;
        probNew             = np.inf;

        while not np.isfinite(probOld):
            oldState            = self.stateGen_init();
            probOld             = self.probDist(oldState,psiFun);

        states              = np.zeros([self.N,M]);
        states[:,0:1]       = oldState;

        for i in range(1,M):
            while not np.isfinite(probNew):
                newState            = self.stateGen_init();
                probNew             = self.probDist(newState,psiFun);

            # Should this be scaled with Euclidean distance between spin configs?
            # Should we use sequential walking for new spin configs?
            probAcc             = probNew*self.propDist(oldState, newState)/ (probOld*self.propDist(newState, oldState));
            #print('Prob Old: %4.3e\nProb New: %4.3e\nProb Acc:%4.3e\n' %  (probOld,probNew,probAcc))

            if probAcc > 1 or probAcc > nr.rand():
                states[:,i:i+1]     = newState;
                probOld             = probNew;
                oldState            = newState;
            else:
                states[:,i:i+1]     = oldState;

        return states
