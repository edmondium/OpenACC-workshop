program main
    implicit none
    ! Size of vectors
    integer :: n = 1000000
 
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
    enddo
 
    ! Sum a and b component wise, save result into c, reduce
    !$acc data copyin(a(10:n),b(10:n)), copyout(c(10:n))
    !$acc parallel loop reduction(+:sum)
    do i=1,n
        c(i) = a(i) + b(i)
        sum = sum +  c(i)
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
