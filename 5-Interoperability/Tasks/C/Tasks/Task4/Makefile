CUDA_PATH ?= $(CUDA_HOME)

INCDIRs = -I$(CUDA_PATH)/include
LIBDIRS = -L$(CUDA_PATH)/lib64

LIBS = -lcufft

COMPOPT = -fast -O3
ACCOPT = -acc=gpu -gpu=cc80 -Minfo=accel -Xlinker -z noexecstack

JSC_SUBMIT_CMD ?= srun --partition=dc-gpu --gres=gpu:1 

.PHONY: all
all: poisson.bin

poisson.bin: poisson.c Makefile
	pgc++ $(COMPOPT) $(ACCOPT) poisson.c $(LIBDIRS) $(LIBS) -o poisson.bin

.PHONY: run clean
run: poisson.bin
	$(JSC_SUBMIT_CMD) ./poisson.bin
	
clean:
	rm -f *.bin *.o
