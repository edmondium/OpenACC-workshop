!
!  Copyright 2015 NVIDIA Corporation
!
!  Licensed under the Apache License, Version 2.0 (the "License");
!  you may not use this file except in compliance with the License.
!  You may obtain a copy of the License at
!
!      http://www.apache.org/licenses/LICENSE-2.0
!
!  Unless required by applicable law or agreed to in writing, software
!  distributed under the License is distributed on an "AS IS" BASIS,
!  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
!  See the License for the specific language governing permissions and
!  limitations under the License.
!

  double precision function Wtime()
!$  double precision, external :: omp_get_wtime
    call cpu_time(Wtime)
!$  Wtime = omp_get_wtime()
  endfunction ! Wtime



integer function poisson2d_reference(iter_max, tol, Anew, Aref, NM, NN) result(num_openmp_threads)
implicit none
  integer, intent(in) :: NM, NN, iter_max
  real(kind=8), intent(in) :: tol
  real(kind=8), dimension(NN,NM), intent(inout) :: Anew, Aref

  integer :: i, j, iter
  real(kind=8) :: error
!$  integer, external :: omp_get_num_threads

  num_openmp_threads = 1

  error = 1.
!$omp parallel
!$omp master
!$  num_openmp_threads = omp_get_num_threads()
!$omp end master
!$omp end parallel

  iter = 0

  do while ( (error > tol) .and. (iter < iter_max) )
  
!$omp parallel
    error = 0.d0
    !$omp do private(i,j) reduction(max:error)
    do j = 2, NM-1
      do i = 2, NN-1
      
        Anew(i,j) = 0.25 * ( Aref(i+1,j) + Aref(i-1,j) + Aref(i,j-1) + Aref(i,j+1) )

        error = max( error, abs(Anew(i,j) - Aref(i,j)) )
      end do ! i
    end do ! j
    !$omp end do
    
    !$omp barrier
    
    !$omp do private(i,j)
    do j = 2, NM-1 
      do i = 2, NN-1
        Aref(i,j) = Anew(i,j)
      end do ! i
    end do ! j
    !$omp end do
    
    !$omp barrier

    ! periodic boundary conditions
    !$omp do private(i)
    do i = 2, NN-1
      Aref(i,1)  = Aref(i,NM-1)
      Aref(i,NM) = Aref(i,2)
    end do ! i
    !$omp end do
!$omp end parallel

    if(mod(iter,100) == 0) write(*,fmt="(2x,i4,2x,f0.6)") iter,error

    iter = iter + 1

  end do ! while

end function ! laplace2d_openmp

logical function check_results(Aref, A, NM, NN, tol) result(no_errors)
  implicit none
  integer, intent(in) :: NM, NN
  real(kind=8), dimension(NN,NM), intent(in) :: A, Aref
  real(kind=8), intent(in) :: tol

  integer :: ix, iy
  no_errors = .true.
  do iy = 1, NN-1
    do ix = 2, NM-1
      if ( abs( Aref(ix,iy) - A(ix,iy)) >= tol ) then
        write(*,fmt="('ERROR: A(',I0,',',I0,') = ',f0.6,' does not match ',f0.6,' (reference)')") ix, iy, A(ix,iy), Aref(ix,iy)
        no_errors = .false.
        return
      end if
    end do ! while
  end do ! while

end function ! check_results
