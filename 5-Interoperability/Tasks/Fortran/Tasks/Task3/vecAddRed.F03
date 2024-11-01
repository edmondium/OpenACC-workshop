program main
    use thrust
    implicit none
    ! Size of vectors
    integer, parameter :: n = 1000000
 
    ! Input vectors
    real(8), dimension(:), allocatable :: a
    real(8), dimension(:), allocatable :: b
    ! Output vector
    real(8), dimension(:), allocatable :: c
 
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
 
    ! Sum a and b component wise, save result into c, reduce
    !$acc data copyin(a(1:n),b(1:n)), create(c(1:n))
    !$acc parallel loop
    do i=1,n
        c(i) = a(i) + b(i)
    enddo
    !$acc end parallel loop
    ! TODO: Use device address for c
    call reduce(c, n, sum)
    ! TODO: Stop using device address for c
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
