program main
    use cublas
    implicit none
    ! Size of vectors
    integer, parameter :: n = 1000000
 
    ! Input vectors
    real(8), dimension(:), allocatable :: a
    real(8), dimension(:), allocatable :: b
    ! Output vector
    real(8), dimension(:), allocatable :: c

    ! cuBLAS
    type(cublasHandle) :: handle
    integer :: status
    real(8) :: alpha = 1.0
 
    integer :: i
    real(8) :: sum
    real(8) :: diff

    ! Allocate memory for each vector
    allocate(a(n))
    allocate(b(n))
    allocate(c(n))

    ! Initialize sum
    sum = 0
    
    ! Initialize cuBLAS
    status = cublasCreate(handle)

    ! Initialize content of input vectors, vector a[i] = sin(i)^2, vector b[i] = cos(i)^2
    do i = 1, n
        a(i) = sin(i*1D0)*sin(i*1D0)
        b(i) = cos(i*1D0)*cos(i*1D0)
    enddo
 
    ! Sum a and b component wise, reduce
    !$acc data copyin(a(1:n),b(1:n))
    !$acc host_data use_device(a, b)
    status = cublasDaxpy_v2(handle, n, alpha, a, 1, b, 1);
    ! call cublasDaxpy(n, alpha, a, 1, b, 1);  ! Old API, we are not using this, but it would remove all interaction with the handle
    !$acc end host_data
    !$acc parallel loop reduction(+:sum)
    do i=1,n
        sum = sum + b(i)
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

    ! cuBLAS
    status = cublasDestroy(handle)
end program
