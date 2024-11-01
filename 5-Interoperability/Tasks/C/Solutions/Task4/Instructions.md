# Interoperability: Task 4

This task solves the Poisson equation using Fast Fourier Transforms which run on the GPU, as provided by the GPU-accelerated cuFFT library.

Special care must be taken to run on the same CUDA Streams using `acc_get_cuda_stream()` and `cufftSetStream()`.

# Tasks

Check for the TODOs throughout the code in `poisson.c`.

* `solveKSpace()`: Parallelize the two loops which go through `Nx` and `Ny`
* `normalise()`: Parallelize the normalization loop
* `solveRSpace()`: Move data to the device, switch context to proper stream, pass correct addresses to `cufftExecZ2Z`s.

Compile with `make`, run with `make run`.
