### PYTHON
def genAllStates(N):
    import numpy as np
    from .dq_utils import getBitWidth

    bitWidth = getBitWidth();
    binarise = lambda x: np.array(list(format(x,'0'+str(N)+'b'))).astype(bitWidth)

    # Generate states as 0,1 site-indexed
    states  = np.zeros([N,pow(2,N)],dtype=bitWidth);
    for i in range(pow(2,N)):
        states[:,i]     = binarise(i)

    # Generate states as element of Hilbert space
    up      = np.array([0,1])
    down    = np.array([1,0])
    states_hilbert   = np.zeros([pow(2,N),pow(2,N)],dtype=bitWidth)
    for i in range(pow(2,N)):
        newState = 1
        for j in range(N):
            if states[j,i] == 1:
                newState    = np.kron(newState,up)
            else:
                newState    = np.kron(newState,down)
        states_hilbert[:,i] = newState

    # Generate states as [-1,1] site indexed
    states_sites = 2*states-1
    return states_sites, states_hilbert

def getBitWidth():
    # Return bit width of data type
    from . import config_real as conf
    import tensorflow as tf
    if conf.DTYPE == tf.complex64 or conf.DTYPE == tf.float32:
        bitWidth    = 'float32'
    else:
        bitWidth    = 'float64'

    return bitWidth


def loadMatlabHDF5(fname,field):
    import h5py
    import numpy as np

    with h5py.File(fname) as f:
        N   = f[field][0].shape[0];
        data = [[np.array(f[element[ind]][0]) for element in f[field]] for ind in np.arange(N)]

    return data

### TENSORFLOW

def tf_is_cmplx(x):
    # Checks if TF dtype is complex
    import tensorflow as tf
    if x == tf.complex64 or x==tf.complex128:
        return True
    else:
        return False

def tf_cmplx_abs_to_cmplx(x):
    # Function which returns the absolute value of a complex number as a complex data type
    # Is implicitly casting pythonic 0 to TF zero slower than explicitly using TF zero?
    import tensorflow as tf
    from .dq_utils import tf_cmplx_abs
    if x.dtype==tf.complex64:
        zero_dtype = tf.float32
    elif x.dtype==tf.complex128:
        zero_dtype = tf.float64
    else:
        print('Cannot use tf_complex_abs given variable')
        raise ValueError

    return tf.complex(tf_cmplx_abs(x),tf.zeros_like(x,dtype=zero_dtype))

def tf_cmplx_abs(x):
    import tensorflow as tf
    return tf.add(tf.square(tf.real(x)),tf.square(tf.imag(x)))

def tf_get_shape(x):
    s = x.get_shape();
    return [s[i].value for i in range(0,len(s))];

def tf_reduce_prod_cmplx(x,dim, keepdims=False):
    # Produces tensorflow operation which reduces along dimension dim 
    # For consistency with numpy, dim is indexed left to right
    # keep_dims controls whether or not singleton dimensions are removed
    from .dq_utils import tf_get_shape
    import tensorflow as tf

    x_red_shape     = tf_get_shape(x);
    x_red_dim_len   = x_red_shape[dim];
    x_red_shape[dim]= 1;
    num_dims        = len(x_red_shape);

    output  = tf.ones(x_red_shape,dtype=x.dtype);

    for i in range(0,x_red_dim_len):
        slice_ind   = [0]*num_dims;
        slice_ind[dim] = i;
        output = tf.multiply(output,tf.slice(x,slice_ind,x_red_shape));

    if keepdims == False:
        output      = tf.squeeze(output);

    return output;

def tf_to_complex64(real,imag):
    import tensorflow as tf
    return tf.complex(tf.to_float(real),tf.to_float(imag));

def tf_to_complex128(real,imag):
    import tensorflow as tf
    return tf.complex(tf.to_double(real),tf.to_double(imag));

def tf_sigmoid_complex128(x):
    import tensorflow as tf
    from .dq_utils import tf_to_complex128
    return tf.divide(tf_to_complex128(1,0),tf.add(tf_to_complex128(1,0),tf.exp(tf.negative(x))));

def tf_softmax_complex(x,dtype):
    import tensorflow as tf
    from .dq_utils import tf_to_complex128, tf_to_complex64
    if dtype == tf.complex64:
        dtype_fn = tf_to_complex64;
    else:
        dtype_fn = tf_to_complex128;
    return tf.log(tf.add(dtype_fn(1,0)),tf.exp(x))

def tf_tanh_complex128(x):
    import tensorflow as tf
    from .dq_utils import to_complex128
    num = tf.add(tf_to_complex128(1,0),tf.exp(tf.multiply(to_complex128(2,0),x)));
    den = tf.subtract(to_complex128(1,0),tf.exp(tf.negative(tf.multiply(tf_to_complex128(2,0),x))));
    return tf.divide(num,den);

class Bunch(dict):
    def __init__(self, *args, **kwds):
        super(Bunch, self).__init__(*args, **kwds)
        self.__dict__ = self
