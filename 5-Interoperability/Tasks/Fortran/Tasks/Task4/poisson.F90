#define Nx 512
#define Ny Nx
#define Size (Nx*Ny)
#define eps 1e-8
#define pi 3.14159265359

double precision function getTime()
  implicit none
    call CPU_TIME(getTime)
endfunction getTime


!! Initial charge distribution !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
subroutine mkCharge(rho)
  implicit none
  double complex, intent(out) :: rho(Ny,*)
  double precision :: dx, dy
  integer :: ix, iy

  dx = 1./Nx
  dy = 1./Ny

  do ix = 1, Nx
    do iy = 1, Ny
      rho(iy,ix) = cmplx(cos(4*pi*(ix-1)*dx)*sin(2*pi*(iy-1)*dy), 0.)
    end do  ! iy
  end do  ! ix
end subroutine mkCharge

!! Fourier space solution !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
subroutine solveKSpace(rhoHat)
  implicit none
  double complex, intent(inout) :: rhoHat(Ny,*)
  double precision :: kx, ky, ik2
  integer :: ix, iy
    !! TODO: Parallelize these loops
    !! Hints:
    !!   - The compiler needs to know that rhoHat is already on the device
    !!   - There are two loops to be parallelized
    do ix = 1, Nx
      do iy = 1, Ny
        if (ix - 1 < Nx/2) then
          kx = (ix - 1) * 2 * pi
        else
          kx = ((ix - 1) - Nx) * 2 * pi
        end if
        if ((iy - 1) < Ny/2) then
          ky = (iy - 1) * 2 * pi
        else
          ky = ((iy - 1) - Ny) * 2 * pi
        end if
        ik2 = 1./(kx*kx + ky*ky + eps)            !! NB. Fuzzy factor to avoid div by 0
        rhoHat(iy,ix) = rhoHat(iy,ix) * ik2       !! NB. No operator overloads allowed
      enddo ! iy
    enddo ! ix
    rhoHat(1,1) = dcmplx(0.d0, 0.d0)              !! Force average value to zero
end subroutine

!! Normalise result !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
subroutine normalise(phi)
  implicit none
  double complex, intent(inout) :: phi(*)
  double precision, parameter :: s = 1./dble(Ny*Nx);      !! ifft(fft(x)) = N*x
  integer :: idx
  ! TODO: Parallelize this loop
  ! Hint: phi is already on the GPU
  do idx = 1, Ny*Nx !! flat loop over 2D array
    phi(idx) = phi(idx)*s
  end do ! idx
  ! TODO: Remember to add closing statements
end subroutine normalise

!! Real space solution !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
subroutine solveRSpace(rho, phi)
  use cufft
  use openacc
  implicit none
  double complex, intent(in)  :: rho(Ny,*)
  double complex, intent(out) :: phi(Ny,*)

  integer :: plan, ierr
  integer :: stream
  !! TODO: Move rho to the device and phi from it
  ierr = cufftPlan2d(plan, Nx, Ny, CUFFT_Z2Z)           !! Build cuFFT plan [Note:CufftPlan]
  stream = 0                                            !! TODO: Obtain OpenACC stream, see [Note:Stream]
  ierr = cufftSetStream(plan, stream)                   !! Move cuFFT to OpenACC stream
  !! TODO: Tell OpenACC to pass the correct addresses of rho and phi 
  ierr = cufftExecZ2Z(plan, rho, phi, CUFFT_FORWARD)    !! FFT
  call solveKSpace(phi)                                 !! div grad ~ -k^2
    !! TODO: Tell OpenACC to pass the correct address of phi
    ierr = cufftExecZ2Z(plan, phi, phi, CUFFT_INVERSE)  !! iFFT inplace
    call normalise(phi)                                 !! Scale result
    ierr = cufftDestroy(plan)                           !! Destroy plan

end subroutine solveRSpace

!! Check result !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
subroutine dump(nme, arr)
  character(len=*), intent(in) :: nme
  double complex, intent(in) :: arr(:,:)
  integer, parameter :: of = 123
  integer :: ix
  open(of, file=nme, action="write");
  do ix = 1, size(arr, 2)
    write(of, "(9999f15.10)") dreal(arr(:,ix)) !! print only the real part
  end do ! ix
  close(of)
end subroutine dump

subroutine report(rho, phi, t0, t1)
  implicit none
  double complex,   intent(in) :: rho(*), phi(*)
  double precision, intent(in) :: t0, t1
  double precision :: err, tmp
  integer :: idx
  !! Write out fields for plotting
  ! call dump("phi.txt", phi)
  ! call dump("rho.txt", rho)
  !! Compute mean square error
  err = 0.
  do idx = 1, Nx*Ny !! flat loop over 2D array
    tmp = dreal(rho(idx)) - 20*pi*pi*dreal(phi(idx))        !! Compare to analytical solution
    if (tmp*tmp > eps) &
      write(*, "(i0,' ',f0.6,' ',f0.6)") idx, dreal(rho(idx)), dreal(phi(idx))*20*pi*pi
    err = err + tmp*tmp
  enddo ! idx
  if (err < eps) then                                       !! Report Time and mean error
    write(*, "(2a,f7.3,2a,f9.6,9a)") &
     "#########################################################\n", &
     "## Accumulated square error:               ", err, "   ##\n", &
     "## Time to compute result:                 ", t1 - t0, "s  ##\n", &
     "#########################################################"
  else                                                      !! Error out of bounds
    write(*, *) &
     "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", &
     "!! ERROR            Result incorrect             ERROR !!", &
     "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
  end if
end subroutine report

program poisson
  use cufft
  implicit none

  double complex, allocatable :: rho(:,:), phi(:,:)
  double precision :: t0, t1
  double precision, external :: getTime

  allocate(rho(Ny,Nx))                  !! Allocate charge density
  allocate(phi(Ny,Nx))                  !! Allocate potential

  call mkCharge(rho)                    !! Setup density.
  t0 = getTime()
  call solveRSpace(rho, phi)            !! Compute potential from charge density.
  t1 = getTime()
  call report(rho, phi, t0, t1)         !! Print timings and errors, dump fields.

  deallocate(rho)
  deallocate(phi)
end program poisson



! Notes !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

! CufftPlan *******************************************************************
!
! A CUFFT/FFTW plan is an opaque handle specifying one type of FFT operation.
! It encapsulates the data types, array ranks and extents.
! This allows for specialised (read fast) executions by synthesing the algorithm
! from smaller building blocks and is especially useful for multiple executions.
!
! In this case we have a 2d plan operating on double precision complex
! numbers. The arrays are of size Nx x Ny.
!

! Stream **********************************************************************
!
! OpenACC, at least /this/ implementation, runs per default on some stream which
! is not the default CUDA stream (0). CUFFT however runs on the the default
! stream. So we need to
!    a. obtain the OpenACC default stream by calling a OpenACC API method (not directive!) and
!    b. force CUFFT on this stream via cufftSetStream
! Using
!    #pragma acc wait
! does not help as it ignores CUDA stream 0!
!
