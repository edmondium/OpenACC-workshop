#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <omp.h>

void allocate_3d_poission_matrix(
    int** row_ptr, 
    int** col_ptr, 
    double** val,
    int* n_rows,
    double** row_sums,
    const int n);

int main()
{
    const int n = 200; // => num_rows ~= n*n*n;
    int num_rows;
    int* restrict row_ptr;
    int* restrict col_ptr;
    double* restrict val;
    double* row_sums;
    allocate_3d_poission_matrix( (int**)&row_ptr, (int**)&col_ptr, (double**)&val, &num_rows, (double**)&row_sums, n );
    
    double* restrict const x = (double*)malloc(num_rows*sizeof(double));
    double* restrict const y = (double*)malloc(num_rows*sizeof(double));
    for (int row=0; row<num_rows; ++row)
    {
        x[row] = 1.0;
        y[row] = 0.0;
    }
    
    int num_repetitions = 10;
    double runtime = 0.0;
    const int num_vals = row_ptr[num_rows];
    #pragma acc data copyin( row_ptr[0:num_rows+1], col_ptr[0:num_vals], val[0:num_vals], x[0:num_rows]) copy( y[0:num_rows] )
    {
    
    double start = omp_get_wtime();
    for (int i=0; i<num_repetitions; ++i)
    {
        #pragma acc parallel loop
        for (int row=0; row<num_rows; ++row)
        {
            double y_tmp = 0.0;
            const int row_start = row_ptr[row];
            const int row_end = row_ptr[row+1];
            for (int col_idx = row_start; col_idx < row_end; ++col_idx)
            {
                y_tmp += val[col_idx] * x[col_idx];
            }
            y[row] = y_tmp;
        }
    }
    double stop = omp_get_wtime();
    runtime = (stop - start)/num_repetitions;
    }
    
    int num_errors = 0;
    for (int row=0; row<num_rows; ++row)
    {
        const double reference = row_sums[row];
        const double diff = fabs( y[row] - reference );
        if ( diff > 1E-16 )
        {
            if ( num_errors == 0 )
            {
                printf("ERROR in row %d: %f != %f (abs diff %f)\n", row, y[row],reference,diff);
            }
            ++num_errors;
        }
    }
    
    if (num_errors == 0)
    {
        printf("Runtime %f s.\n", runtime);
    }
    else
    {
        printf("Total %d errors.\n", num_errors);
    }
    
    free(y);
    free(x);
    free( row_sums );
    free( val );
    free( col_ptr );
    free( row_ptr );
    return num_errors;
}

void allocate_3d_poission_matrix(
    int** row_ptr, 
    int** col_ptr, 
    double** val,
    int* n_rows,
    double** row_sums,
    const int n)
{
    const int num_rows = (n+1)*(n+1)*(n+1);
    const int num_vals = 27*num_rows;
    *n_rows = num_rows;
    
    *row_ptr = (int*) malloc( (num_rows+1)*sizeof(int) );
    //slightly over allocating col_ptr and val to simplify matrix construction
    *col_ptr = (int*) malloc( num_vals*sizeof(int) );
    *val = (double*) malloc( num_vals*sizeof(double) );
    
    *row_sums = (double*) malloc( (num_rows)*sizeof(double) );
    
    int offsets[27];
    double coefs[27];
    const int z_stride = n*n;
    const int y_stride = n;
    
    int coefs_pos=0;
    for(int z=-1;z<=1;++z)
    {
        for(int y=-1;y<=1;++y)
        {
            for(int x=-1;x<=1;++x)
            {
                offsets[coefs_pos]=z_stride*z+y_stride*y+x;
                if(x==0 && y==0 && z==0)
                    coefs[coefs_pos]=27.0;
                else
                    coefs[coefs_pos]=-1.0;
                ++coefs_pos;
            }
        }
    }
    
    int pos = 0;
    for(int row=0;row<num_rows;++row)
    {
        (*row_ptr)[row]=pos;
        
        double row_sum = 0.0;
        for(int j=0;j<27;++j)
        {
            int n=row+offsets[j];
            if(n>=0 && n<num_rows)
            {
                (*col_ptr)[pos]=n;
                (*val)[pos]=coefs[j];
                ++pos;
                row_sum += coefs[j];
            }
        }
        (*row_sums)[row] = row_sum;
    }
    (*row_ptr)[num_rows] = pos;
}
