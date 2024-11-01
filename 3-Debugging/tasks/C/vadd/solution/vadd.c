#include <stdio.h>
#include <stdlib.h>
#include <math.h>
 
int main( int argc, char* argv[] ) {
    // Size of vectors
    int n = 1000000;
 
    // Input vectors
    double * restrict const a = (double*) malloc(n*sizeof(double));
    double * restrict const b = (double*) malloc(n*sizeof(double));
    // Output vector
    double * restrict const c = (double*) malloc(n*sizeof(double));
 
    // Initialize content of input vectors, vector a[i] = sin(i)^2, vector b[i] = cos(i)^2
    for(int i = 0; i < n; i++) {
        a[i] = sin(i)*sin(i);
        b[i] = cos(i)*cos(i);
    }
 
    // Sum a and b component wise, save result into c, reduce
    float sum = 0.0;
    #pragma acc data copyin(a[0:n],b[0:n]) create(c[0:n])
    {
        #pragma acc parallel loop reduction(+:sum)
        for(int i = 0; i < n; i++) {
            c[i] = a[i] + b[i];
            sum += c[i];
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
    free(c);
 
    return 0;
}
