#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cublas_v2.h>

int main( int argc, char* argv[] ) {
    // Size of vectors
    int n = 1000000;
 
    // Input vectors
    double * restrict const a = (double*) malloc(n*sizeof(double));
    double * restrict const b = (double*) malloc(n*sizeof(double));
 
    // Initialize content of input vectors, vector a[i] = sin(i)^2, vector b[i] = cos(i)^2
    for(int i = 0; i < n; i++) {
        a[i] = sin(i)*sin(i);
        b[i] = cos(i)*cos(i);
    }
 
    float sum = 0.0;

    // Initialize cuBLAS
    cublasHandle_t handle;

    cublasStatus_t stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }
    // Copy data to GPU
    #pragma acc data copyin(a[0:n],b[0:n])
    {
        double alpha = 1;
        #pragma acc host_data use_device(a, b)
        cublasDaxpy(handle, n, &alpha, a, 1., b, 1.);  // Calculates b = alpha * a + b (with a and b having stride 1), output stored in b

        #pragma acc parallel loop reduction(+:sum)
        for(int i = 0; i < n; i++) {
            sum += b[i];
        }

    }
 
    sum = sum/n;
    printf("Result: %f\n", sum);
    const double diff = fabs( sum - 1. );
    if (diff > 1E-16) {
        printf("Result differs from 1 by %f!\nSomething went wrong.\n", diff);
    }
    // Release memory
    free(a);
    free(b);

    // Cleanup cuBLAS
    cublasDestroy(handle);
 
    return 0;
}

// Notes ///////////////////////////////////////////////////////////////////////

/* cublasDaxpy *******************************************************************
 * 
 * The signature of cublasDaxpy is
 * cublasDaxpy(cublasHandle_t handle, int n, const double *alpha, const double *x, int incx, double *y, int incy),
 * which calculates y = alpha * x + y with stride incx and incy for x and y (both should be 1).
 * The vector length is n.
 * 
 * See also: http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-axpy
 */
