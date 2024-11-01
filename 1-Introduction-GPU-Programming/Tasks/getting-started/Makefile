CC = icc
NVCC = nvcc
MPICC = mpicc

MKLFLAGS = -qopenmp -I${MKLROOT}/include -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a -Wl,--end-group -lpthread -lm -ldl
CUBLASFLAGS = -I${CUDA_ROOT}/include -L${CUDA_ROOT}/lib64 -lcublas -lcudart
CFLAGS = -g -O3
CXXFLAGS = -std=c++11
NVFLAGS = --cudart=static -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_80,code=sm_80 -std=c++11

.phony: all

all: dgemm ddot mandelbrot mandelbrot_cpu
	
dgemm: dgemm_mkl.cpp
	$(CC) -o $@ $< $(MKLFLAGS) $(CUBLASFLAGS) $(CFLAGS)
	
ddot: ddot.cpp
	$(CC) -o $@ $< $(MKLFLAGS) $(CUBLASFLAGS) $(CFLAGS)
	
mandelbrot: mandelbrot.cu
	$(NVCC) -o $@ $< $(NVFLAGS)

mandelbrot_cpu: mandelbrot.cpp
	$(CXX) -o $@ $< $(CXXFLAGS) $(CFLAGS) -fopenmp

.phony: clean veryclean

clean:
	rm -f *.o dgemm ddot mandelbrot mandelbrot_cpu

veryclean: clean
	rm -f *.pdf

.PHONY: run gen_montage display

run: all
	./bench_all.sh

gen_montage display: all
gen_montage display: MNTG=montage dgemm_bench.png nbody_bench.png mandelbrot_bench.png ddot_bench.png -tile 2x2 -geometry +0+0
gen_montage:
	$(MNTG) benchmarks.png
display: 
	$(MNTG) - | display -
