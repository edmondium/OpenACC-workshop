! Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
!
! Redistribution and use in source and binary forms, with or without
! modification, are permitted provided that the following conditions
! are met:
!  * Redistributions of source code must retain the above copyright
!    notice, this list of conditions and the following disclaimer.
!  * Redistributions in binary form must reproduce the above copyright
!    notice, this list of conditions and the following disclaimer in the
!    documentation and/or other materials provided with the distribution.
!  * Neither the name of NVIDIA CORPORATION nor the names of its
!    contributors may be used to endorse or promote products derived
!    from this software without specific prior written permission.
!
! THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
! EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
! IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
! PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
! CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
! EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
! PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
! PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
! OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
! (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
! OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

PROGRAM poisson2d
#if _OPENACC
    USE openacc
#endif
    USE mpi
    IMPLICIT NONE
    INTEGER, PARAMETER :: nx = 8192
    INTEGER, PARAMETER :: ny = 8192
    INTEGER, PARAMETER :: iter_max = 1000
    REAL, PARAMETER :: tol = 1.0E-5
    INTEGER :: i,ix, iy, ix_start, ix_end, iy_start, iy_end, iter, mpi_rank, mpi_size, ngpus, local_rank, num_devs, local_comm, ierror
    INTEGER(KIND=acc_device_kind) :: device_type
    INTEGER :: chunk_size, right, left
    REAL :: x,y, error, globalerror
    REAL*8 :: runtime_serial, runtime, start, finish, mpi_time, mpi_start_time
    LOGICAL, EXTERNAL :: check_results
    LOGICAL :: errors
    REAL, DIMENSION(:,:), ALLOCATABLE :: a, a_ref, a_new, rhs
    
    mpi_rank = 0
    mpi_size = 1
    
    !Initialize MPI and determine rank and size
    CALL MPI_Init(ierror)
    CALL MPI_Comm_rank(MPI_COMM_WORLD,mpi_rank,ierror)
    CALL MPI_Comm_size(MPI_COMM_WORLD,mpi_size,ierror)
    
    CALL MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,mpi_rank,MPI_INFO_NULL,local_comm,ierror)
    
    CALL MPI_Comm_rank(local_comm,local_rank,ierror)
    
    CALL MPI_Comm_free(local_comm,ierror)
    
    #if _OPENACC
    device_type = acc_get_device_type()
    IF ( acc_device_nvidia == device_type ) THEN
        num_devs = acc_get_num_devices( acc_device_nvidia )
        call acc_set_device_num( MOD(local_rank,num_devs), acc_device_nvidia )
    END IF
    call acc_init( device_type )
    #endif
    
    ALLOCATE( a(nx,ny) )
    ALLOCATE( a_ref(nx,ny) )
    ALLOCATE( a_new(nx,ny) )
    ALLOCATE( rhs(nx,ny) )
    
    a = 0.0
    a_ref = 0.0
    
    DO iy = 2, ny-1
        DO ix = 2, nx-1
            x = -1.0 + (2.0*ix/(nx-1.0))
            y = -1.0 + (2.0*iy/(ny-1.0))
            rhs(ix,iy) = EXP(-10.0*(x*x+y*y))
        END DO
    END DO
    
    !$acc enter data create(a,a_ref,a_new,rhs)
    
    ix_start = 2
    ix_end   = nx-1
    
    !set first and last row to be processed by this rank.
    !Ensure correctness if ny%size != 0
    chunk_size = CEILING( (1.0*ny)/mpi_size )
    iy_start = mpi_rank * chunk_size
    iy_end = iy_start + chunk_size - 1
    
    !Do not process boundaries
    iy_start = MAX( iy_start, 2 )
    iy_end = MIN( iy_end, ny-1 )
    
    !OpenACC Warm-up
    !$acc parallel loop present(a,a_ref)
    DO iy = 1, ny
        DO ix = 1, nx
            a(ix,iy) = 0.0
            a_ref(ix,iy) = 0.0
        END DO
    END DO
    !$acc end parallel loop

    IF ( mpi_rank == 0 ) THEN
        WRITE(*,"('Jacobi relaxation Calculation: ',I4,' x ',I4,' mesh')") nx,ny
        WRITE(*,*) 'Calculate reference solution and time serial execution.'
    END IF
    start = MPI_WTIME()
    CALL poisson2d_serial( nx, ny, iter_max, mpi_rank, tol, a_ref, a_new, rhs )
    finish = MPI_WTIME()
    runtime_serial = finish-start
    
    !MPI Warm-up to establish CUDA IPC connections
    DO i = 1,2
        left = mpi_rank-1
        IF ( mpi_rank == 0 ) THEN
            left = mpi_size-1
        END IF
        right = mpi_rank+1
        IF ( mpi_rank == mpi_size-1 ) THEN
            right = 0
        END IF
        !$acc host_data use_device( a )
            !1. Sent column iy_start (first modified column) to left receive right boundary (iy_end+1) from right
            CALL MPI_Sendrecv( a(ix_start,iy_start), (ix_end-ix_start)+1, MPI_REAL_TYPE, left   , 0, &
                              a(ix_start,iy_end+1), (ix_end-ix_start)+1, MPI_REAL_TYPE, right, 0, &
                              MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierror )

            !2. Sent column iy_end (last modified column) to right receive left boundary (iy_start-1) from left
            CALL MPI_Sendrecv( a(ix_start,iy_end), (ix_end-ix_start)+1, MPI_REAL_TYPE, right, 0, &
                               a(ix_start,(iy_start-1)), (ix_end-ix_start)+1, MPI_REAL_TYPE, left   , 0, &
                               MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierror )
        !$acc end host_data
    END DO
    
    !Wait for all processes to ensure correct timing of the parallel version
    CALL MPI_Barrier( MPI_COMM_WORLD, ierror )
    
    IF ( mpi_rank == 0 ) THEN
        WRITE(*,*) 'Parallel execution.'
    END IF 
    
    start = MPI_WTIME()
    iter = 1
    error = 1.0
    !$acc update device(a(1:nx,iy_start:iy_end),rhs(1:nx,iy_start:iy_end))
    DO WHILE ( error > tol .AND. iter <= iter_max )
        error = 0.0
        !$acc parallel loop present(a,a_new,rhs) copy(error)
        DO iy = iy_start, iy_end
            DO ix = ix_start, ix_end
                a_new(ix,iy) = -0.25 * (rhs(ix,iy) - ( a(ix+1,iy) + a(ix-1,iy) + a(ix,iy-1) + a(ix,iy+1) ))
                error = MAX( error, ABS( a_new(ix,iy) - a(ix,iy) ) )
            END DO
        END DO
        !$acc end parallel loop
        !Calculate global error across all ranks
        globalerror = 0.0
        call MPI_Allreduce( error, globalerror, 1, MPI_REAL_TYPE, MPI_MAX, MPI_COMM_WORLD, ierror )
        error = globalerror
        
        !TODO: Split into halo and bulk part
        !$acc parallel loop present(a,a_new)
        DO iy = iy_start, iy_end
            DO ix = ix_start, ix_end
                a(ix,iy) = a_new(ix,iy)
            END DO
        END DO
        !$acc end parallel loop
        !TODO: Start bulk part asynchronously
        
        !Handle periodic boundary conditions and halo exchange with MPI
        left = mpi_rank-1
        IF ( mpi_rank == 0 ) THEN
            left = mpi_size-1
        END IF
        right = mpi_rank+1
        IF ( mpi_rank == mpi_size-1 ) THEN
            right = 0
        END IF
        
        mpi_start_time = MPI_WTIME()
        !$acc host_data use_device( a )
            !1. Sent column iy_start (first modified column) to left receive right boundary (iy_end+1) from right
            CALL MPI_Sendrecv( a(ix_start,iy_start), (ix_end-ix_start)+1, MPI_REAL_TYPE, left   , 0, &
                               a(ix_start,iy_end+1), (ix_end-ix_start)+1, MPI_REAL_TYPE, right, 0, &
                               MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierror )

            !2. Sent column iy_end (last modified column) to right receive left boundary (iy_start-1) from left
            CALL MPI_Sendrecv( a(ix_start,iy_end), (ix_end-ix_start)+1, MPI_REAL_TYPE, right, 0, &
                               a(ix_start,(iy_start-1)), (ix_end-ix_start)+1, MPI_REAL_TYPE, left   , 0, &
                               MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierror )
        !$acc end host_data
        mpi_time = (MPI_WTIME() - mpi_start_time) + mpi_time
        !TODO: wait for bulk part
        
        !$acc parallel loop present(a)
        DO iy = iy_start, iy_end
            a(1,iy) = a(nx-1,iy)
            a(nx,iy) = a(2,iy)
        END DO
        !$acc end parallel loop

        IF ( mpi_rank == 0 .AND. ( iter == 1 .OR. MOD( iter, 100 ) == 0 ) ) THEN
            WRITE(*,"('  ',I4,' ',F10.6)") iter, error
        END IF
        
        iter = iter+1
    END DO
    !$acc update self(a(1:nx,iy_start:iy_end))
    !Wait for all processes to ensure correct timing of the parallel version
    CALL MPI_Barrier( MPI_COMM_WORLD, ierror )
    finish = MPI_WTIME()
    runtime = finish-start
    
    errors = .FALSE.
    IF ( check_results( mpi_rank, ix_start, ix_end, iy_start, iy_end, nx, ny, tol, a, a_ref ) ) THEN
        IF ( mpi_rank == 0 ) THEN
            WRITE(*,*) 'Num GPUs: ', mpi_size
            WRITE(*,"(I4,'x',I4,': 1 GPU: ',F8.4,' s ',I1,' GPUs: ',F8.4,' s, speedup: ',F8.2,' efficiency: ',F8.2)"), &
                  nx,ny,runtime_serial,mpi_size,runtime,runtime_serial/runtime,runtime_serial/(mpi_size*runtime)*100
            WRITE(*,"('MPI time: 'F8.4' s, inter GPU BW: 'F8.2' GiB/s')"), &
                  mpi_time,(iter*4*(ix_end-ix_start)*SIZEOF(a(1,1)))/(1024*1024*1024*mpi_time)
        END IF
    ELSE
        errors = .TRUE.
    END IF
    
    !$acc exit data delete(a,a_ref,a_new,rhs)
    CALL MPI_Finalize(ierror)
    
    DEALLOCATE( rhs )
    DEALLOCATE( a_new )
    DEALLOCATE( a_ref )
    DEALLOCATE( a )
    IF ( errors ) THEN
        STOP -1
    END IF
END PROGRAM poisson2d
