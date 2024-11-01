# Interoperability: Task 2

In this task, data is copied to the GPU device by means of OpenACC. Computation on the GPU is done via CUDA. CUDA can be called directly from Fortran as CUDA Fortran. No extra file (like in the C version of this task) is needed.

## Tasks

* subroutine `cudaVecAdd` accepts pointers to data which is already on the GPU. Tell OpenACC to use the device pointers for the arrays when `cudaVecAdd` is called.
* call the subroutine `cudaVecAdd()` as a CUDA kernel with the `<<<A,B>>>` syntax, using as *A* and *B* the number of blocks and threads, respectively.
* `cudaVecAdd()`: Implement the core of the kernel, the addition. Think of the kernel as a implicitly called for loop.
