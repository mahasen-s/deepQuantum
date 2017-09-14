def loadMatlabHDF5(fname,field):
    import h5py
    import numpy as np

    with h5py.File(fname) as f:
        N   = f[field][0].shape[0];
        data = [[f[element[ind]][0] for element in f[field]] for ind in np.arange(N)]

    return data
