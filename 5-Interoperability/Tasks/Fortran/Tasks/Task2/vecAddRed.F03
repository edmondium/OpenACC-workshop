attributes(global) subroutine cudaVecAdd(a, b, c, n)
    real(8), dimension(*), intent(in) :: a, b
    real(8), dimension(*), intent(out) :: c
    integer, value :: n
    integer :: i

    i = (blockidx%x - 1) * blockdim%x + threadidx%x
    if (i <= n) then
        ! TODO: Calculate c = a + b for element i
    end if
end subroutine cudaVecAdd

program main
    implicit none
    ! Size of vectors
    integer, parameter :: n = 1000000
 
    ! Input vectors
    real(8), dimension(:), allocatable :: a
    real(8), dimension(:), allocatable :: b
    ! Output vector
    real(8), dimension(:), allocatable :: c

    ! CUDA
    integer :: nThreads, nBlocks
 
    integer :: i
    real(8) :: sum
    real(8) :: diff

    ! Initialize sum
    sum = 0

    ! Allocate memory for each vector
    allocate(a(n))
    allocate(b(n))
    allocate(c(n))

    ! Initialize content of input vectors, vector a[i] = sin(i)^2, vector b[i] = cos(i)^2
    do i = 1, n
        a(i) = sin(i*1D0)*sin(i*1D0)
        b(i) = cos(i*1D0)*cos(i*1D0)
        c(i) = 0
    enddo
 
    ! Init CUDA
    nThreads = 256;
    nBlocks = int(ceiling(real(n)/real(nThreads)));

    ! Sum a and b component wise, reduce
    !$acc data copyin(a(1:n),b(1:n)) create(c(1:n))
    ! TODO: Add clause for using device addresses of a and b
    ! TODO: Call cudaVecAdd kernel with nBlocks blocks and nThreads threads
    ! TODO: Add directive for ending usage of device-addresses
    !$acc parallel loop reduction(+:sum)
    do i=1,n
        sum = sum + c(i)
    enddo
    !$acc end parallel loop
    !$acc end data
 
    ! Divide the sum by n, the final result should be 1.0 within error
    sum = sum/n
    write(*,*) "Result: ", sum

    diff = abs( sum - 1.)

    if (diff > 1e-16) then
        write(*,*) "Result differs from 1 by", diff
        write(*,*) "Something went wrong!"
    end if
 
    ! Release memory
    deallocate(a)
    deallocate(b)
    deallocate(c)
end program
