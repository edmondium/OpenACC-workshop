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
#include <stdio.h>

#include "common.h"

void poisson2d_serial( int rank, int iter_max, real tol, real* restrict const Aref, real* restrict const Anew, int nx, int ny, const real* restrict const rhs )
{
    int iter  = 0;
    real error = 1.0;
    #pragma acc data present(Aref,Anew,rhs)
    {
    #pragma acc update device(Aref[0:nx*ny],rhs[0:nx*ny])
    while ( error > tol && iter < iter_max )
    {
        error = 0.0;

        #pragma acc parallel loop
        for( int iy = 1; iy < ny-1; iy++)
        {
            for( int ix = 1; ix < ny-1; ix++ )
            {
                Anew[iy*nx+ix] = -0.25 * (rhs[iy*nx+ix] - ( Aref[iy    *nx+(ix+1)] + Aref[iy    *nx+ix-1]
                                                        + Aref[(iy-1)*nx+ix]     + Aref[(iy+1)*nx+ix] ));
                error = fmaxr( error, fabsr(Anew[iy*nx+ix]-Aref[iy*nx+ix]));
            }
        }

        #pragma acc parallel loop
        for( int iy = 1; iy < ny-1; iy++)
        {
            for( int ix = 1; ix < ny-1; ix++ )
            {
                Aref[iy*nx+ix] = Anew[iy*nx+ix];
            }
        }

        //Periodic boundary conditions
        #pragma acc parallel loop
        for( int ix = 1; ix < ny-1; ix++ )
        {
                Aref[0     *nx+ix] = Aref[(ny-2)*nx+ix];
                Aref[(ny-1)*nx+ix] = Aref[1     *nx+ix];
        }
        #pragma acc parallel loop
        for( int iy = 1; iy < ny-1; iy++ )
        {
                Aref[iy*nx+0]      = Aref[iy*nx+(nx-2)];
                Aref[iy*nx+(nx-1)] = Aref[iy*nx+1];
        }

        if(rank == 0 && (iter % 100) == 0) printf("%5d, %0.6f\n", iter, error);

        iter++;
    }
    #pragma acc update self(Aref[0:nx*ny])
    }
}

int check_results( int rank, int ix_start, int ix_end,  int iy_start, int iy_end, real tol, const real* restrict const A, const real* restrict const Aref, int nx )
{
    int result_correct = 1;
    for( int iy = iy_start; iy < iy_end && (result_correct == 1); iy++)
    {
        for( int ix = ix_start; ix < ix_end && (result_correct == 1); ix++ )
        {
            if ( fabs ( Aref[iy*nx+ix] - A[iy*nx+ix] ) >= tol )
            {
                fprintf(stderr,"[MPI%d] ERROR: A[%d][%d] = %f does not match %f (reference)\n", rank, iy,ix, A[iy*nx+ix], Aref[iy*nx+ix]);
                result_correct = 0;
            }
        }
    }
#ifdef MPI_VERSION
    int global_result_correct = 0;
    MPI_Allreduce( &result_correct, &global_result_correct, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD );
    result_correct = global_result_correct;
#endif //MPI_VERSION
    return result_correct;
}
