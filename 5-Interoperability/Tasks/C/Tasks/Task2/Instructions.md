# Interoperability: Task 2

In this task, data is copied to the GPU device by means of OpenACC. Computation on the GPU is done
via CUDA. Both OpenACC and CUDA codes should go into the main source code file, vecAddRed.c

## Program Idea

* `vecAddRed.c`: The main part of the program sets up the data and initializes it. `cudaVecAdd()` : Calculates `c = a + b` in parallel..


# Tasks

* `vecAddRed.c`: `cudaVecAdd()` accepts pointers to data which is already on the GPU. Tell OpenACC to use the device pointers for the arrays.
    - Call the kernel (Use the `<<<` syntax)
    - Implement vector addition
* Compile with `make`, run with `make run`
