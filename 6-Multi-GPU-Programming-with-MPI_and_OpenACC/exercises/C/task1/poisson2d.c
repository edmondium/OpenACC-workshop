/* Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
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

#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>

#include "common.h"

int main(int argc, char** argv)
{
    int ny = 8192;
    int nx = 8192;
    int iter_max = 1000;
    const real tol = 1.0e-5;

    int rank = 0;
    int size = 1;

    //Initialize MPI and determine rank and size
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
        
    //TODO: handle device affinity

    real* restrict const A    = (real*) malloc(nx*ny*sizeof(real));
    real* restrict const Aref = (real*) malloc(nx*ny*sizeof(real));
    real* restrict const Anew = (real*) malloc(nx*ny*sizeof(real));
    real* restrict const rhs  = (real*) malloc(nx*ny*sizeof(real));
    
    // set rhs
    for (int iy = 1; iy < ny-1; iy++)
    {
        for( int ix = 1; ix < nx-1; ix++ )
        {
            const real x = -1.0 + (2.0*ix/(nx-1));
            const real y = -1.0 + (2.0*iy/(ny-1));
            rhs[iy*nx+ix] = expr(-10.0*(x*x + y*y));
        }
    }

    #pragma acc enter data create(A[0:nx*ny],Aref[0:nx*ny],Anew[0:nx*ny],rhs[0:nx*ny])

    int ix_start = 1;
    int ix_end   = (nx - 1);

    //TODO: set first and last row to be processed by this rank.
    int iy_start = 1;
    int iy_end   = (ny - 1);
    
    //OpenACC Warm-up and initializing arrays
    #pragma acc parallel loop present(A,Aref,Anew)
    for( int iy = 0; iy < ny; iy++)
    {
        for( int ix = 0; ix < nx; ix++ )
        {
            Aref[iy*nx+ix] = 0.0;
            Anew[iy*nx+ix] = 0.0;
            A[iy*nx+ix] = 0.0;
        }
    }
    //Input data is assumed to be on the host
    #pragma acc update self(Aref[0:nx*ny],Anew[0:nx*ny],A[0:nx*ny])

    if ( rank == 0) printf("Jacobi relaxation Calculation: %d x %d mesh\n", ny, nx);

    if ( rank == 0) printf("Calculate reference solution and time serial execution.\n");
    double start = MPI_Wtime();
    poisson2d_serial( rank, iter_max, tol, Aref, Anew, nx, ny, rhs );
    double runtime_serial = MPI_Wtime() - start;
    
    //MPI Warm-up to establish CUDA IPC connections
    for (int i=0; i<2; ++i)
    {
        int top    = (rank == 0) ? (size-1) : rank-1;
        int bottom = (rank == (size-1)) ? 0 : rank+1;
        #pragma acc host_data use_device( A )
        {
            //1. Sent row iy_start (first modified row) to top receive lower boundary (iy_end) from bottom
            MPI_Sendrecv( A+iy_start*nx+ix_start, (ix_end-ix_start), MPI_REAL_TYPE, top   , 0,
                          A+iy_end*nx+ix_start,   (ix_end-ix_start), MPI_REAL_TYPE, bottom, 0,
                          MPI_COMM_WORLD, MPI_STATUS_IGNORE );

            //2. Sent row (iy_end-1) (last modified row) to bottom receive upper boundary (iy_start-1) from top
            MPI_Sendrecv( A+(iy_end-1)*nx+ix_start,   (ix_end-ix_start), MPI_REAL_TYPE, bottom, 0,
                          A+(iy_start-1)*nx+ix_start, (ix_end-ix_start), MPI_REAL_TYPE, top   , 0,
                          MPI_COMM_WORLD, MPI_STATUS_IGNORE );
        }
    }

    //Wait for all processes to ensure correct timing of the parallel version
    MPI_Barrier( MPI_COMM_WORLD );
    if ( rank == 0) printf("Parallel execution.\n");
    double mpi_time = 0.0;
    start = MPI_Wtime();
    int iter  = 0;
    real error = 1.0;
    
    #pragma acc update device(A[(iy_start-1)*nx:((iy_end-iy_start)+2)*nx],rhs[iy_start*nx:(iy_end-iy_start)*nx])
    while ( error > tol && iter < iter_max )
    {
        error = 0.0;

        #pragma acc parallel loop present(A,Anew,rhs)
        for (int iy = iy_start; iy < iy_end; iy++)
        {
            for( int ix = ix_start; ix < ix_end; ix++ )
            {
                Anew[iy*nx+ix] = -0.25 * (rhs[iy*nx+ix] - ( A[iy*nx+ix+1] + A[iy*nx+ix-1]
                                                       + A[(iy-1)*nx+ix] + A[(iy+1)*nx+ix] ));
                error = fmaxr( error, fabsr(Anew[iy*nx+ix]-A[iy*nx+ix]));
            }
        }
        
        real globalerror = 0.0;
        MPI_Allreduce( &error, &globalerror, 1, MPI_REAL_TYPE, MPI_MAX, MPI_COMM_WORLD );
        error = globalerror;
        
        #pragma acc parallel loop present(A,Anew)
        for (int iy = iy_start; iy < iy_end; iy++)
        {
            for( int ix = ix_start; ix < ix_end; ix++ )
            {
                A[iy*nx+ix] = Anew[iy*nx+ix];
            }
        }

        //Periodic boundary conditions
        //TODO: Handle top/bottom periodic boundary conditions and halo exchange with MPI
        #pragma acc parallel loop present(A)
        for( int ix = 1; ix < ny-1; ix++ )
        {
                A[0     *nx+ix] = A[(ny-2)*nx+ix];
                A[(ny-1)*nx+ix] = A[1     *nx+ix];
        }
        int top    = (rank == 0) ? (size-1) : rank-1;
        int bottom = (rank == (size-1)) ? 0 : rank+1;
        //TODO: Pass device ptr of A to MPI using host_data use_device
        {
            double start_mpi = MPI_Wtime();
            //TODO: 1. Sent row iy_start (first modified row) to top receive lower boundary (iy_end) from bottom
            //MPI_Sendrecv(const void *sendbuf, int sendcount, MPI_REAL_TYPE, int dest, 0,
            //                   void *recvbuf, int recvcount, MPI_REAL_TYPE, int source, 0,
            //                   MPI_COMM_WORLD, MPI_STATUS_IGNORE );

            //TODO: 2. Sent row (iy_end-1) (last modified row) to bottom receive upper boundary (iy_start-1) from top
            //MPI_Sendrecv(const void *sendbuf, int sendcount, MPI_REAL_TYPE, int dest, 0,
            //                   void *recvbuf, int recvcount, MPI_REAL_TYPE, int source, 0,
            //                   MPI_COMM_WORLD, MPI_STATUS_IGNORE );
            mpi_time += MPI_Wtime() - start_mpi;
        }

        #pragma acc parallel loop present(A)
        for (int iy = iy_start; iy < iy_end; iy++)
        {
                A[iy*nx+0]      = A[iy*nx+(nx-2)];
                A[iy*nx+(nx-1)] = A[iy*nx+1];
        }
        
        if(rank == 0 && (iter % 100) == 0) printf("%5d, %0.6f\n", iter, error);
        
        iter++;
    }
    #pragma acc update self(A[(iy_start-1)*nx:((iy_end-iy_start)+2)*nx])
    MPI_Barrier( MPI_COMM_WORLD );
    double runtime = MPI_Wtime() - start;

    int errors = 0;
    if (check_results( rank, ix_start, ix_end, iy_start, iy_end, tol, A, Aref, nx ))
    {
        if ( rank == 0 )
        {
            printf( "Num GPUs: %d.\n", size );
            printf( "%dx%d: 1 GPU: %8.4f s, %d GPUs: %8.4f s, speedup: %8.2f, efficiency: %8.2f%\n", ny,nx, runtime_serial, size, runtime, runtime_serial/runtime, runtime_serial/(size*runtime)*100 );
            printf( "MPI time: %8.4f s, inter GPU BW: %8.2f GiB/s\n", mpi_time, (iter*4*(ix_end-ix_start)*sizeof(real))/(1024*1024*1024*mpi_time) );
        }
    }
    else
    {
        errors = -1;
    }

    #pragma acc exit data delete(A,Aref,Anew,rhs)
    MPI_Finalize();
    
    free(rhs);
    free(Anew);
    free(Aref);
    free(A);
    return errors;
}
