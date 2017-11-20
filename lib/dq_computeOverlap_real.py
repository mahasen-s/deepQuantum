from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from . import config_real as conf
from . import dq_utils  as utils

def computeOverlap(wf,wf_exact):
    # Check
    if wf_exact[0]!=wf.input_num:
        print('Wavefunction input_num inconsistent with wf_exact num')
        raise ValueError

    # Generate states
    [states_sites,states_hilbert]    = utils.genAllStates(wf.input_num)

    # Get neural-wf complex nums
    wf_neural_vals  = np.squeeze(wf.evalOverlap(states_sites))

    # Get exact-wf complex nums
    # wf_exact_vals  = np.einsum('ij,i->j',states_hilbert,wf_exact[1])
    wf_exact_vals = wf_exact[1]
    wf_exact_vals = wf_exact_vals.astype(wf_neural_vals.dtype)

    # Compute the overlap
    overlap     = np.power(np.abs(np.dot(np.conj(wf_neural_vals),wf_exact_vals)),2)
    norm        = np.multiply(np.dot(np.conj(wf_exact_vals),wf_exact_vals),np.dot(np.conj(wf_neural_vals),wf_neural_vals))

    return np.divide(overlap,norm)
