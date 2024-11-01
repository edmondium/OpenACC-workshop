# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
COMPILER ?= NVHPC-tesla
FC = mpifort
ifeq ($(COMPILER),GCC)
FCFLAGS = -freal-4-real-8 -DMPI_REAL_TYPE=MPI_REAL8
else
FCFLAGS = -r8 -DMPI_REAL_TYPE=MPI_REAL8
endif
ifeq ($(COMPILER),GCC)
	FCFLAGS += -march=native -O3 -lm
else ifeq ($(COMPILER),NVHPC-tesla)
	FCFLAGS += -Minfo=accel -fast -acc=gpu -gpu=cc80
else ifeq ($(COMPILER),NVHPC-multicore)
	FCFLAGS += -Minfo=accel -fast -acc=multicore
endif

NP ?= 4
JSC_SUBMIT_CMD ?= srun --ntasks-per-node 4

all: run

poisson2d_serial.o: poisson2d_serial.F03 Makefile
	$(FC) -c $(FCFLAGS) poisson2d_serial.F03 -o poisson2d_serial.o

poisson2d: poisson2d.F03 poisson2d_serial.o Makefile
	$(FC) $(FCFLAGS) poisson2d.F03 poisson2d_serial.o -o poisson2d

poisson2d.solution: poisson2d.solution.F03 poisson2d_serial.o Makefile
	$(FC) $(FCFLAGS) poisson2d.solution.F03 poisson2d_serial.o -o poisson2d.solution

clean:
	rm -f poisson2d poisson2d.solution poisson2d*.o poisson2d.*.nsys-rep

run: poisson2d
	$(JSC_SUBMIT_CMD) -n $(NP) ./poisson2d

profile: poisson2d
	$(JSC_SUBMIT_CMD) -n $(NP) nsys profile --trace=mpi,cuda,openacc,nvtx -o poisson2d.%q{SLURM_PROCID} ./poisson2d

solution: poisson2d.solution
	$(JSC_SUBMIT_CMD) -n $(NP) ./poisson2d.solution

profile.solution: poisson2d.solution
	$(JSC_SUBMIT_CMD) -n $(NP) nsys profile --trace=mpi,cuda,openacc,nvtx -o poisson2d.solution.%q{SLURM_PROCID} ./poisson2d.solution
