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

program main
! use openacc
! use omp_lib
implicit none

double precision, external :: WTime ! from common.f90
integer, external :: poisson2d_reference
logical, external :: check_results

integer, parameter :: NN = 2048, NM = 2048

real(8), allocatable :: A(:,:), Aref(:,:), Anew(:,:)

real(8) :: t1, t2, t1_ref, t2_ref, runtime, runtime_ref, pi, y0, tol, error

integer :: i, j, num_openmp_threads, iter, iter_max, ist

  iter_max = 500
  pi    = 2.d0 * asin(1.d0)
  tol   = 1.d-5
  error = 1.d0

  allocate(A(NN,NM), Aref(NN,NM), Anew(NN,NM), stat=ist)

  if (ist /= 0) stop "failed to allocate arrays!"


! set boundary conditions
  do j = 1, NM
    y0 = sin( 2.0 * pi * j / (NM-1))

    A(1,j)    = y0
    A(NN,j)   = y0

    Aref(1,j) = y0
    Aref(NN,j)= y0

    Anew(1,j) = y0
    Anew(NN,j)= y0

  end do ! j

  write(*,fmt="('Jacobi relaxation Calculation: ',i0,' x ',i0,' mesh')") NN,NM

  write(*,fmt="('Calculate reference solution and time CPU execution.')")
  t1_ref = Wtime()
  num_openmp_threads = poisson2d_reference(iter_max, tol, Anew, Aref, NM, NN)
  t2_ref = Wtime()
  runtime_ref = t2_ref-t1_ref
  write(*,fmt="('GPU execution.')")

!$acc init device_type(acc_device_nvidia)


  t1 = Wtime()

  iter = 0

do while ( (error > tol) .and. (iter < iter_max) )

  error = 0.d0
!! TODO: Add copy clause
!$acc parallel loop reduction(max:error)
  do j = 2, NM-1
    do i = 2, NN-1

      Anew(i,j) = 0.25 * ( A(i+1,j) + A(i-1,j) + A(i,j-1) + A(i,j+1) )

      error = max( error, abs(Anew(i,j) - A(i,j)) )
    end do ! i
  end do ! j
!$acc end parallel loop

! Swap input/output
!! TODO: Add copy clause
!$acc parallel loop
  do j = 2, NM-1
    do i = 2, NN-1
      A(i,j) = Anew(i,j)
    end do ! i
  end do ! j
!$acc end parallel loop

! Periodic boundary conditions
!! TODO: Add copy clause
!$acc parallel loop
  do i = 2, NN-1
    A(i,1)  = A(i,NM-1)
    A(i,NM) = A(i,2)
  end do ! i
!$acc end parallel loop

  if(mod(iter,100) == 0) write(*,fmt="(2x,i4,2x,f0.6)") iter,error

  iter = iter + 1
end do ! while

t2 = Wtime()
runtime = t2-t1
if (check_results(Aref, A, NM, NN, tol)) write(*,fmt="(i0,' x ',i0,': ','1 GPU: ',f0.4,'s, ',i0,' CPU cores: ',f0.4,'s',', speedup: ',f0.2)") NN, NM, runtime, num_openmp_threads,runtime_ref,runtime_ref/runtime



end program ! main
