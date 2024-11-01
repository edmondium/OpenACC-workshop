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

SUBROUTINE poisson2d_serial(nx, ny, iter_max, mpi_rank,tol,a_ref, a_new,rhs)
    IMPLICIT NONE
    INTEGER, INTENT(IN) :: nx, ny, iter_max, mpi_rank
    REAL, INTENT(IN) :: tol
    REAL, DIMENSION(nx,ny), INTENT(INOUT) :: a_ref, a_new
    REAL, DIMENSION(nx,ny), INTENT(IN) :: rhs
    INTEGER :: ix,iy, iter
    REAL :: error
    
    iter = 1
    error = 1.0
    !$acc data present(a_ref,rhs,a_new)
    !$acc update device(a_ref,rhs)
    DO WHILE ( error > tol .AND. iter <= iter_max )
        error = 0.0
        !$acc parallel loop copy(error)
        DO iy = 2, ny-1
            DO ix = 2, nx-1
                a_new(ix,iy) = -0.25 * (rhs(ix,iy) - ( a_ref(ix+1,iy) + a_ref(ix-1,iy) + a_ref(ix,iy-1) + a_ref(ix,iy+1) ))
                error = MAX( error, ABS( a_new(ix,iy) - a_ref(ix,iy) ) )
            END DO
        END DO
        !$acc end parallel loop
        
        !$acc parallel loop
        DO iy = 2, ny-1
            DO ix = 2, nx-1
                a_ref(ix,iy) = a_new(ix,iy)
            END DO
        END DO
        !$acc end parallel loop
        
        !$acc parallel loop
        DO ix = 2, nx-1
            a_ref(ix,1) = a_ref(ix,ny-1)
            a_ref(ix,ny) = a_ref(ix,2)
        END DO
        !$acc end parallel loop
        
        !$acc parallel loop
        DO iy = 2, ny-1
            a_ref(1,iy) = a_ref(nx-1,iy)
            a_ref(nx,iy) = a_ref(2,iy)
        END DO
        !$acc end parallel loop
        
        IF ( mpi_rank == 0 .AND. ( iter == 1 .OR. MOD( iter, 100 ) == 0 ) ) THEN
            WRITE(*,"('  ',I4,' ',F10.6)") iter, error
        END IF
        
        iter = iter+1
    END DO
    !$acc update self(a_ref)
    !$acc end data
END SUBROUTINE poisson2d_serial

LOGICAL FUNCTION check_results( mpi_rank, ix_start, ix_end, iy_start, iy_end, nx, ny, tol, a, a_ref )
    USE mpi
    IMPLICIT NONE
    INTEGER, INTENT(IN) :: mpi_rank, ix_start, ix_end, iy_start, iy_end, nx, ny
    REAL, INTENT(IN) :: tol
    REAL, DIMENSION(nx,ny), INTENT(IN) :: a, a_ref
    INTEGER :: ix,iy,ierror
    LOGICAL :: no_errors, global_no_errors, mpi_is_initialized
    no_errors = .TRUE.
    iy = iy_start
    ix = ix_start
    DO WHILE ( iy <= iy_end .AND. no_errors )
        DO WHILE ( ix <= ix_end .AND. no_errors )
            IF ( ABS( a_ref(ix,iy) - a(ix,iy)) >= tol ) THEN
                WRITE(*,"('[MPI',I1,'] ERROR: a(',I4,',',I4,') = ',F8.6,' does not match ',F8.6,' (reference)')") &
                    mpi_rank,ix,iy,a(ix,iy),a_ref(ix,iy)
                no_errors = .FALSE.
            END IF
            ix = ix + 1
        END DO
        iy = iy + 1
    END DO
    
    CALL MPI_Initialized(mpi_is_initialized, ierror)
    IF ( mpi_is_initialized ) THEN
        global_no_errors = .FALSE.
        CALL MPI_ALLREDUCE(no_errors, global_no_errors, 1, MPI_LOGICAL, MPI_LAND, MPI_COMM_WORLD, ierror)
        no_errors = global_no_errors
    END IF
    check_results = no_errors
END FUNCTION check_results

SUBROUTINE size_to_2Dsize(mpi_size, mpi_sizex, mpi_sizey)
    IMPLICIT NONE
    INTEGER, INTENT(IN) :: mpi_size
    INTEGER, INTENT(OUT) :: mpi_sizex, mpi_sizey
    INTEGER, DIMENSION(2,16), PARAMETER :: size_to_size2d_map = reshape( &
        (/ 1,1 , 2,1 , 3,1  , 2,2 , &
           5,1 , 3,2 , 7,1  , 4,2 , &
           3,3 , 5,2 , 11,1 , 6,2 , &
          13,1 , 7,2 , 5,3  , 4,4 /), (/ 2, 16 /) )
    mpi_sizex = size_to_size2d_map(2,mpi_size)
    mpi_sizey = size_to_size2d_map(1,mpi_size)
END SUBROUTINE size_to_2Dsize
