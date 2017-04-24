# deepQuantum
TensorFlow implementation of the algorithm from this [paper](https://arxiv.org/abs/1606.02318)
Finds groundstate solutions for arbitrary spin Hamiltonians.

# To-Do


## Tensorflow

### Refactor Hamiltonian
* Make hamiltonian template that accepts a function which defines the operations used to compute the variational energy

### Overlap 
* Overlap between DQ wavefunction and exact wavefunction is almost zero; problem either in Hamiltonian or overlap computation
  * Check overlap computation
  * Use dummy Hamiltonians

### Function support
* Workaround lack of `complex` support with simpler operations **Done**
* Build CPU/GPU kernels for functions not supporting complex data types

### Metropolis sample
* Implement single-site update
  * Flip one spin site randomly, decide whether to accept
  * Repeat for some number of steps (until the state is statistically uncorrelated with the last step)
  * Accept new state as new member of Metropolis sample

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
