import tensorflow as tf
import numpy as np

def tf_roll(x,n,axis=0):
    # Equivalent to numpy's roll
    return tf.gather(x,np.roll(np.arange(x.get_shape().as_list()[axis]),n),axis=axis)
