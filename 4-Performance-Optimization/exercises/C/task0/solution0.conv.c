/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

// dimension of input and output image, for N^2 "pixels"
#define N 2048*2

void run_convolution_kernel_and_time(float A[N][N], float B[N][N], int sw, const int num_repetitions)
{
  // Create stencil coefficients: 
  int stencil_dim = 2*sw + 1;
  float stencil[stencil_dim][stencil_dim];
  for (int x=0; x < stencil_dim; ++x) {
    for (int y=0; y < stencil_dim; ++y) {
      stencil[x][y] = 0.1*x + 0.2*y; // arbitrary data - could be replaced with filter coefficients
    }
  }
  double duration;
  #pragma acc data copyin (A[0:N][0:N], stencil[0:stencil_dim][0:stencil_dim]) copyout (B[0:N][0:N]) 
  {
    double start = omp_get_wtime();
    for (int rep = 0; rep < num_repetitions; ++rep) {
      ///////////////////////////
      // Modifications only below
      ///////////////////////////
      // TODO: Parallelize the outer loop by adding the correct #pragma below the TODO
      #pragma acc parallel loop
      for (int x = sw; x < N - sw; ++x) {
        for (int y = sw; y < N - sw; ++y) {
          B[x][y] = 0;
          for (int sx = -sw; sx <= sw; ++sx) {
            for (int sy = -sw; sy <= sw; ++sy) {
              const float val = stencil[sw + sx][sw + sy] * A[x + sx][y + sy];
              B[x][y] += val;
            }
          }
        }
      }
      ///////////////////////////
      // Modifications only above
      ///////////////////////////
    }
    duration = 1000.*(omp_get_wtime() - start)/num_repetitions;
  }
  printf("Runtime %f ms\n", duration);
}

enum REF_MODE {
  WRITE_REF,
  READ_REF
};

int main(int argc, char** argv)
{
  const int num_repetitions = 15;
  // Parse arguments
  int retval = 0;
  int sw = 3;
  enum REF_MODE mode = READ_REF; 
  if (argc == 1) {
    printf("Usage: %s [sw] [recreate]\nsw - width of stencil\nrecreate - pass 'yes' to recreate ref data\n", argv[0]);
    return 1;
  }
  if (argc > 1) {
    sw = atoi(argv[1]);
  }
  if (argc > 2) {
    mode = WRITE_REF;
    printf("Recreating reference data...\n");
  }
  printf("Using stencil width = %d\n", sw);

  // Allocate input and output matrices
  const size_t n_bytes = sizeof(float[N][N]);
  float (*A)[N] = malloc(n_bytes);
  float (*B)[N] = malloc(n_bytes);
  if (!A || !B) {
    fprintf(stderr, "Allocation failure (OOM?), need %ld bytes per array\n", n_bytes);
    retval = 1;
    goto cleanup;
  }
  // Create arbitrary, but fixed, input data
  for (int x = 0; x < N; ++x) {
    for (int y = 0; y < N; ++y) {
      A[x][y] = N*x +y;
    }
  }
  // Run the actual computation and measure (average over multiple runs)
  run_convolution_kernel_and_time(A, B, sw, num_repetitions);
  // Compare results and cleanup
  char fname[255];
  sprintf(fname, "ref_sw%d_sz%d.bin", sw, N);
  if (mode == WRITE_REF) {
    FILE *f = fopen(fname, "wb");
    if (!f) {
      fprintf(stderr, "Could not open ref file for writing: %s\n", fname);
      retval = 1;
      goto cleanup;
    }
    fwrite(B, sizeof(float), N*N, f);
    fclose(f);
  }
  else {
    FILE *f = fopen(fname, "rb");
    if (!f) {
      fprintf(stderr, "Could not open ref file for reading: %s\n", fname);
      retval = 1;
      goto cleanup;
    }
    fread(A, sizeof(float), N*N, f);
    fclose(f);
    char error[255];
    int num_errors = 0;
    for (int x = 0; x < N; ++x) {
      for (int y = 0; y < N; ++y) {
        // relative tolerance of 10^-6, float accuracy
        if (fabs(B[x][y] - A[x][y]) > 1e-6f*(fabs(B[x][y])+fabs(A[x][y]))) {
          if (num_errors++ == 0) {
            sprintf(error, "at %d, %d: %f vs %f (ref)", x, y, B[x][y], A[x][y]);
            fprintf(stderr, "Found error: %s\n", error);
            retval = 1;
          }
        }
      }
    }
    if (num_errors) fprintf(stderr, "Total %d errors\n", num_errors);
  }
cleanup:
  free(A);
  free(B);
  return retval;
}
