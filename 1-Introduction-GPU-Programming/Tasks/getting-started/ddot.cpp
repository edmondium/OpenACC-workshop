#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mkl_cblas.h>
#include <mkl_vsl.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "errcheck.inc"

int main(int argc, char** argv){
    double* A;
    double* B;
    double* C;
    double* A_dev;
    double* B_dev;
    double res = 0.0;
    double res_cublas = 0.0;
    
    double alpha = 1.0;
    double beta = 0.0;
    int i;        
    struct timeval t1,t2, t3, t4, t5;
    
    const int SEED = 1;
    const int METHOD = 0;
    const int BRNG = VSL_BRNG_MCG31;
    VSLStreamStatePtr stream;
    int errcode;
    
    cublasStatus_t status;
    cublasHandle_t handle;
    
    double a=0.0, b= 1.0; // Uniform distribution between 0 and 1
    
    errcode = vslNewStream(&stream, BRNG, SEED);
    
    int width = 100;
    if (argc > 1){
        width = atoi(argv[1]);
    }
    A = (double*) malloc(width * sizeof(double));
    B = (double*) malloc(width * sizeof(double));
    /* Generate width random numbers between 0 and 1 to fill vectors A and B. */
    errcode = vdRngUniform(METHOD, stream, width, A, a, b);
    CheckVslError(errcode);
    errcode = vdRngUniform(METHOD, stream, width, B, a, b);
    CheckVslError(errcode);
    
    gettimeofday(&t1, NULL);

    res = cblas_ddot(width, A, 1, B, 1);
    gettimeofday(&t2, NULL);

    /* Now prepare the call to CUBLAS */

    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! CUBLAS initialization error\n");
        return EXIT_FAILURE;
    }

    /* Allocate memory for A, B, and C */
    if (cudaMalloc((void**)&A_dev, width * sizeof(double)) != cudaSuccess){
        fprintf(stderr, "!!!! device memory alocation error (allocate A)\n");
        return EXIT_FAILURE;
    }
    if (cudaMalloc((void**)&B_dev, width * sizeof(double)) != cudaSuccess){
        fprintf(stderr, "!!!! device memory alocation error (allocate B)\n");
        return EXIT_FAILURE;
    }
    gettimeofday(&t3, NULL);
    
    /* Copy data to device using CUBLAS routines */
    status = cublasSetVector(width, sizeof(double), A, 1, A_dev, 1);
    if (status != CUBLAS_STATUS_SUCCESS){
        fprintf(stderr, "!!!! device access error (write A)\n");
        return EXIT_FAILURE;
    }
    status = cublasSetVector(width, sizeof(double), B, 1, B_dev, 1);
    if (status != CUBLAS_STATUS_SUCCESS){
        fprintf(stderr, "!!!! device access error (write B)\n");
        return EXIT_FAILURE;
    }
    gettimeofday(&t5, NULL);
    
    /* Perform calculation */
    status = cublasDdot(handle, width, A_dev, 1, B_dev, 1, &res_cublas);
    if (status != CUBLAS_STATUS_SUCCESS){
        fprintf(stderr, "!!!! kernel execution error.\n");
        return EXIT_FAILURE;
    }
    
    gettimeofday(&t4, NULL);
    printf("Call to DDot took %lf s\n",(double) (t2.tv_sec
-t1.tv_sec)+1e-6*(t2.tv_usec-t1.tv_usec));
    printf("Call to cublasDdot took %lf s (%lf s for transfer)\n",(double) (t4.tv_sec - t3.tv_sec) +
1e-6 * (t4.tv_usec -
        t3.tv_usec),(double) (t5.tv_sec - t3.tv_sec) + 1e-6 * (t5.tv_usec -
        t3.tv_usec));
    printf("Result: %lf (MKL) %lf (CUBLAS)\n", res, res_cublas);
    free(A);
    free(B);
    
    cudaFree(A_dev);
    cudaFree(B_dev);
    
    status = cublasDestroy(handle);
    if (status != CUBLAS_STATUS_SUCCESS){
        fprintf(stderr, "!!!! shutdown error\n");
        return EXIT_FAILURE;
    }
    
    return 0;
}
