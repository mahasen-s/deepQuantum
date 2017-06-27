from models.TFI.TFI_sampling_singleSite import markovChainGenerator as mcg
from models.TFI.TFI_sampling_singleSite import sampler_TFI as sampler

import numpy as np
import numpy.random as nr

print('TEST WITH LAMBDA FUNCTION')
N           = 5;
M           = 10;
probFun     = lambda x: np.float(np.sum(x)==N)
samp        = sampler(N=N,probFun=probFun)

initState   = np.array(-np.ones([N,1]))
mcg1        = mcg(samp,burnIn=1,thinning=1)
S           = mcg1.getSample_MH(M)
print('\nMarkov chain')
print(repr(S))

