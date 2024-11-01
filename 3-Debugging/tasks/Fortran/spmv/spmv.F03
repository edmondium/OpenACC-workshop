PROGRAM spmv
    USE OMP_LIB
    IMPLICIT NONE
    INTEGER, PARAMETER                  :: n = 200
    INTEGER, PARAMETER                  :: num_repetitions = 10
    INTEGER                             :: num_rows
    INTEGER, DIMENSION(:), ALLOCATABLE  :: row_ptr,col_ptr
    REAL, DIMENSION(:), ALLOCATABLE     :: val, row_sums, x, y
    REAL*8                              :: runtime, start_time, stop_time, y_tmp, reference, diff
    INTEGER                             :: i, row, row_start, row_end, col_idx, num_errors
    
    CALL allocate_3d_poission_matrix(row_ptr, col_ptr, val, num_rows, row_sums, n)
    
    ALLOCATE( x(num_rows) )
    ALLOCATE( y(num_rows) )
    
    DO row = 1,num_rows
        x( row ) = 1.0
        y( row ) = 0.0
    END DO
    
    !$acc data copyin( row_ptr, col_ptr, val, x ) copy( y )
    
    start_time = omp_get_wtime()
    DO i = 1,num_repetitions
        !$acc parallel loop
        DO row = 1,num_rows
            y_tmp = 0.0
            row_start = row_ptr(row)
            row_end = row_ptr(row+1)-1
            DO col_idx = row_start, row_end
                y_tmp = y_tmp + val( col_idx ) * x ( col_idx )
            END DO
            y( row ) = y_tmp
        END DO
        !$acc end parallel loop
    END DO
    stop_time = omp_get_wtime()
    runtime = (stop_time - start_time)/num_repetitions;
    
    !$acc end data

    num_errors = 0
    DO row = 1,num_rows
        reference = row_sums( row )
        diff = ABS( y( row ) - reference )
        IF ( diff > 1.0E-16 ) THEN
            if ( num_errors == 0 ) THEN
                WRITE(*,"('ERROR in row 'I4': 'F4.1' != 'F4.1' (abs diff 'F4.1')')"), row, y( row ), reference, diff
            END IF
            num_errors = num_errors + 1
        END IF
    END DO
    
    IF ( num_errors > 0 ) THEN
        WRITE(*,"('Total 'I4' errors.')"), num_errors
    ELSE
        WRITE(*,"('Runtime: 'F8.6' s.')"), runtime
    END IF
    
    DEALLOCATE( y )
    DEALLOCATE( x )
    DEALLOCATE( row_ptr )
    DEALLOCATE( col_ptr )
    DEALLOCATE( val )
    DEALLOCATE( row_sums )
    
    IF ( num_errors > 0 ) THEN
        STOP num_errors
    END IF
CONTAINS

SUBROUTINE allocate_3d_poission_matrix(row_ptr, col_ptr, val, num_rows, row_sums, n)
    IMPLICIT NONE
    INTEGER, DIMENSION(:), ALLOCATABLE, INTENT(OUT) :: row_ptr, col_ptr
    REAL, DIMENSION(:), ALLOCATABLE, INTENT(OUT)    :: val
    INTEGER, INTENT(OUT)                            :: num_rows
    REAL, DIMENSION(:), ALLOCATABLE, INTENT(OUT)    :: row_sums
    INTEGER, INTENT(IN)                             :: n
    INTEGER                                         :: num_vals
    INTEGER                                         :: z_stride, y_stride, coefs_pos, z, y, x, pos, row, j, n_tmp
    INTEGER, DIMENSION(27)                          :: offsets
    REAL, DIMENSION(27)                             :: coefs
    REAL                                            :: row_sum
    
    num_rows = (n+1)*(n+1)*(n+1)
    num_vals = 27*num_rows

    ALLOCATE( row_ptr(num_rows+1) )
    ALLOCATE( col_ptr(num_vals) )
    ALLOCATE( val(num_vals) )
    ALLOCATE( row_sums(num_rows) )
    
    z_stride = n*n
    y_stride = n
    
    coefs_pos=1
    DO z = -1,1
        DO y = -1,1
            DO x = -1,1
            offsets( coefs_pos ) = z_stride*z+y_stride*y+x
            IF ( x == 0 .AND. y == 0 .AND. z == 0 ) THEN
                coefs( coefs_pos ) = 27.0
            ELSE
                coefs( coefs_pos ) = -1.0
            END IF
            coefs_pos=coefs_pos+1
            END DO
        END DO
    END DO
    
    pos=1
    DO row = 1,num_rows
        row_ptr( row ) = pos
        row_sum = 0.0
        DO j = 1,27
            n_tmp = row + offsets(j)
            IF ( n_tmp >= 1 .AND. n_tmp <= num_rows ) THEN
                col_ptr( pos ) = n_tmp
                val( pos ) = coefs( j )
                pos = pos + 1
                row_sum = row_sum + coefs( j )
            END IF
        END DO
        row_sums( row ) = row_sum
    END DO
    row_ptr( num_rows+1 ) = pos
END SUBROUTINE allocate_3d_poission_matrix

END PROGRAM
