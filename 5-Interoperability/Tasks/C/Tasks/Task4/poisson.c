#include "utils.h"   // Helpers: getTime, CUFFTC

#include <openacc.h>           // OpenACC runtime functions
#include <stdio.h>             // File I/O
#include <cufft.h>             // DFT library

// Problem parameters //////////////////////////////////////////////////////////
#define cmplx cufftDoubleComplex
#define real  cufftDoubleReal

#define Nx 512
#define Ny Nx
#define Size (Nx*Ny)
#define eps 1e-9
#define pi 3.14159265359

// Initial charge distribution /////////////////////////////////////////////////
void mkCharge(cmplx* restrict rho) {
  real dx = 1./Nx, dy = 1./Ny;                             // Grid spacing
  for(int ix = 0; ix < Nx; ++ix) {                         // (x,y) in [0, 1)^2
    for(int iy = 0; iy < Ny; ++iy) {
      rho[iy + Ny*ix] = (cmplx) { cos(4*pi*ix*dx)*sin(2*pi*iy*dy), 0. };
    }
  }
}

// Fourier space solution /////////////////////////////////////////////////////
void solveKSpace(cmplx*  restrict rhoHat) {
    // TODO: Parallelize these loops
    // Hints:
    //   - The compiler needs to know that rhoHat is already on the device
    //   - There are two loops to be parallelized
    for(int ix = 0; ix < Nx; ++ix) {
      for(int iy = 0; iy < Ny; ++iy) {
        const real kx = (ix - (ix < Nx/2 ? 0 : Nx))*2*pi;   // Index to frequency
        const real ky = (iy - (iy < Ny/2 ? 0 : Ny))*2*pi;
        const real ik2 = 1./(kx*kx + ky*ky + eps);          // NB. Fuzzy factor to avoid div by 0
        rhoHat[iy + Ny*ix].x *= ik2;                       // NB. No operator overloads allowed
        rhoHat[iy + Ny*ix].y *= ik2;
      }
    }
    rhoHat[0].x = 0; rhoHat[0].y = 0.;                     // Force average value to zero
}

// Normalise result ////////////////////////////////////////////////////////////
void normalise(cmplx* restrict phi) {
  real s = 1./Size;                                        // ifft(fft(x)) = N*x
  // TODO: Parallelize this loop
  // Hint: phi is already on the GPU
  for(int idx = 0; idx < Size; ++idx) {
    phi[idx].x *= s; phi[idx].y *= s;
  }
}

// Real space solution /////////////////////////////////////////////////////////
void solveRSpace(cmplx* restrict rho, cmplx* restrict phi) {
  // TODO: Move rho and phi to the device and back
  {
    cufftHandle xform;
    CUFFTC(cufftPlan2d(&xform, Nx, Ny, CUFFT_Z2Z));       // Build cuFFT plan [Note:CufftPlan]
    cudaStream_t str = (cudaStream_t) 0;                  // TODO: Obtain OpenACC stream, see [Note:Stream]
    CUFFTC(cufftSetStream(xform, str));                   // Move cuFFT to OpenACC stream
    // TODO: Tell OpenACC to pass the correct addresses of rho and phi 
    CUFFTC(cufftExecZ2Z(xform, rho, phi, CUFFT_FORWARD)); // FFT
    solveKSpace(phi);                                     // div grad ~ -k^2
    // TODO: Tell OpenACC to pass the correct address of phi
    CUFFTC(cufftExecZ2Z(xform, phi, phi, CUFFT_INVERSE)); // iFFT inplace
    normalise(phi);                                       // Scale result
    cufftDestroy(xform);                                  // Destroy plan
  }
}

// Check result ////////////////////////////////////////////////////////////////
void dump(const char* nme, const cmplx* arr) {
  FILE* of = fopen(nme, "w");
  for(int ix = 0; ix < Nx; ++ix) {
    for(int iy = 0; iy < Ny; ++iy) {
      fprintf(of, "%15.10lf ", arr[ix*Ny + iy].x);
    }
    fprintf(of, "\n");
  }
  fclose(of);
}

void report(const cmplx* restrict rho, const cmplx* restrict phi, double t0, double t1) {
  // Write out fields for plotting
  dump("phi.txt", phi);
  dump("rho.txt", rho);
  // Compute mean square error
  real err = 0.;
  for(int idx = 0; idx < Nx*Ny; ++idx) {
    const real tmp = rho[idx].x - 20*pi*pi*phi[idx].x;      // Compare to analytical solution
    if(tmp*tmp > eps) 
      printf("%d %lf %lf\n", idx, rho[idx].x, phi[idx].x*20*pi*pi);
    err += tmp*tmp;
  }
  if(err < eps) {                                           // Report Time and mean error
    printf("#########################################################\n"
     "## Accumulated square error:               %7.3e   ##\n"
     "## Time to compute result:                 %9.6lfs  ##\n"
     "#########################################################\n", err, t1 - t0);
  } else {                                                  // Error out of bounds
    printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
     "!! ERROR            Result incorect              ERROR !!\n"
     "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
  }
}

// Main routine ////////////////////////////////////////////////////////////////
int main() {
  cmplx* restrict rho = (cmplx*) malloc(Size*sizeof(cmplx)); // Allocate charge density
  cmplx* restrict phi = (cmplx*) malloc(Size*sizeof(cmplx)); // Allocate potential

  mkCharge(rho);                                             // Setup density.
  real t0 = getTime();
  solveRSpace(rho, phi);                                     // Compute potential from charge density.
  real t1 = getTime();
  report(rho, phi, t0, t1);                                  // Print timings and errors, dump fields.

  free(rho);
  free(phi);
}

// Notes ///////////////////////////////////////////////////////////////////////

/* CufftPlan *******************************************************************
 *
 * A CUFFT/FFTW plan is an opaque handle specifying one type of FFT operation.
 * It encapsulates the data types, array ranks and extents.
 * This allows for specialised (read fast) executions by synthesing the algorithm
 * from smaller building blocks and is especially useful for multiple executions.
 *
 * In this case we have a 2d plan operating on double precision complex
 * numbers. The arrays are of size Nx x Ny.
 */

/* Stream **********************************************************************
 *
 * OpenACC, at least /this/ implementation, runs per default on some stream which
 * is not the default CUDA stream (0). CUFFT however runs on the the default
 * stream. So we need to
 *    a. obtain the OpenACC default stream by calling a OpenACC API method (not directive!) and
 *    b. force CUFFT on this stream via cufftSetStream
 * Using
 *    #pragma acc wait
 * does not help as it ignores CUDA stream 0!
 */

