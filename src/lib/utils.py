import tensorflow as tf
import numpy as np

def tf_roll(x,n,axis=0):
    # Equivalent to numpy's roll
    return tf.gather(x,np.roll(np.arange(x.get_shape().as_list()[axis]),n),axis=axis)

def genAllStates(N,dtype=float):
    import numpy as np

    binarise = lambda x: np.array(list(format(x,'0'+str(N)+'b'))).astype(dtype)

    # Generate states as 0,1 site-indexed
    states  = np.zeros([pow(2,N),N],dtype=dtype);
    for i in range(pow(2,N)):
        states[i,:]     = binarise(i)

    # Generate states as element of Hilbert space
    up      = np.array([0,1])
    down    = np.array([1,0])
    states_hilbert   = np.zeros([pow(2,N),pow(2,N)],dtype=dtype)
    for i in range(pow(2,N)):
        newState = 1
        for j in range(N):
            if states[i,j] == 1:
                newState    = np.kron(newState,up)
            else:
                newState    = np.kron(newState,down)
        states_hilbert[i,:] = newState

    # Generate states as [-1,1] site indexed
    states_sites = 2*states-1
    return states_sites, states_hilbert

class Bunch(dict):
    def __init__(self, *args, **kwds):
        super(Bunch, self).__init__(*args, **kwds)
        self.__dict__ = self

def loadMatlabHDF5(fname,field):
    import h5py
    import numpy as np

    with h5py.File(fname) as f:
        N   = f[field][0].shape[0];
        data = [[np.array(f[element[ind]][0]) for element in f[field]] for ind in np.arange(N)]

    return data
