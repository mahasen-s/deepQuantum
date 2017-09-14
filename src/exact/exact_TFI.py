import numpy as np
from scipy import sparse
from scipy.sparse import linalg
import mkl
mkl.set_num_threads(6)

def exact_TFI(N,h):
    # sparse function
    sp_fun  = sparse.csr_matrix

    # Pauli matrices
    sx  = sp_fun(np.array([[0,1],[1, 0]]));
    sz  = sp_fun(np.array([[1,0],[0,-1]]));
    eye = sparse.identity(2);
    zer = sp_fun(np.array([[0,0],[0,0]]));

    # Blank H
    H   = sp_fun(1);
    for i in range(0,N):
        H = sparse.kron(H,zer)

    # Build H
    for i in range(0,N):
        # sig x
        sig_x   = sp_fun(1)
        sig_z   = sp_fun(1)
        for j in range(0,N):
            if i==j:
                sig_x   = sparse.kron(sig_x,sx)
                sig_z   = sparse.kron(sig_z,sz)
            else:
                sig_x   = sparse.kron(sig_x,eye)
                sig_z   = sparse.kron(sig_z,eye)

        H   += h*sig_x

        for k in range(0,i):
            sig_z2  = sp_fun(1)
            for j in range(0,N):
                if j==k:
                    sig_z2 = sparse.kron(sig_z2,sz)
                else:
                    sig_z2 = sparse.kron(sig_z2,eye)
            sig_z2  = sp_fun.multiply(sig_z,sig_z2)

            H += sig_z2

    # Solve H
    evals, evecs    = linalg.eigsh(H,1,which='SA')
    return evals, evecs
