# deepQuantum
TensorFlow implementation of the algorithm from this [paper](https://arxiv.org/abs/1606.02318)
Finds groundstate solutions for arbitrary spin Hamiltonians.

# Current state
* Metropolis sampling accurately samples arbitrary distribution 
* Minimising H_avg (expectation of H) and feeding the whole configuration space works--Overlap asymptotically approaches 1, variation in Re(E_loc) approaches 0
* Minimising E_var for a MC sample doesn't seem to work
* Time taken even for simple problems is *very* large, slightly ominous
  * Inefficient implemntation? Profile using `cPython` and Tensorflow the `timeline` module
  * Metropolis takes a long time, TensorFlow rewrite might be useful
   

# To-Do

## Neural Net Wavefunction

### Refactor Hamiltonian
* Make hamiltonian template that accepts a function which defines the operations used to compute the variational energy

### Overlap 
* Overlap between DQ wavefunction and exact wavefunction is almost zero; problem either in Hamiltonian or overlap computation **Working**
  * Check overlap computation **Done**

### Function support
* Workaround lack of `complex` support with simpler operations **Done**
* Build CPU/GPU kernels for functions not supporting complex data types

## Metropolis sampler
* Implement single-site update **Done**
  * Flip one spin site randomly, decide whether to accept **Done**
  * Repeat for some number of steps (until the state is statistically uncorrelated with the last step) **Done**
  * Accept new state as new member of Metropolis sample **Done**
  * Implemented Metropolis sampler in TF

* Rewrite as RNN in tensorflow
  * Rewrite tests for RNN Metropolis

## Diagnostics
* Set up `tensorBoard`
* Write wrappers for `timeline`

## Exact solver
* Generalise to solve (simple) arbitrary spin Hamiltonians
* Write solver for simple 2D Hamiltonians

## MPS Solver
* Use ALPS

# Roadmap
1. Reproduce 1D TFI, AFH groundstate solutions from paper
2. Repeat 1D TFI, AFH groundstate solutions using deep nets
3. Reproduce 2D TFI, AFH groundstate solutions from paper
4. Repeat 2D TFI, AFH groundstate solutions using deep nets
5. Explore performance of DQ in non-integrable systems
