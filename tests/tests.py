import unittest
import tensorflow as tf

class wavefunction_test():
    def __init__(self,sess,input_num=N):
        self.sess = sess
        self.input_num = input_numi
        self.input_state = tf.placeholder(conf.DTYPE,[self.input_num,1]);
        self.wavefunction_tf_op = self.buildOp(self.input_state)

    def buildOp(self,input_states):
        psi = self.wfFun()
        return psi

    def eval

    def evalOverlap

class testHamiltonian_TFI(unittest.TestCase):
    self.N

    def wavefunction_test(state):
        N = 2;
        M = 1;
        h_drive = 1;
        h_inter = 1;
        h_detune = 0;

        sess = tf.Session();



    def test_hamiltonian_GHZstate:

    def test_hamiltonian_Wstate:

    def test_hamiltonian_AllDown:

    def test_hamiltonian_OneUp:

if __name__= '__main__':
    unittest.main()
