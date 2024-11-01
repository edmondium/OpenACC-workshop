#include <cuda.h>

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

extern "C" void thrust_reduce_wrapper(
	double * __restrict__ c,
	int n,
	double * sum
) {
	thrust::device_ptr<double> c_ptr = thrust::device_pointer_cast(c);
	*sum = thrust::reduce(c_ptr, c_ptr + n);
}
