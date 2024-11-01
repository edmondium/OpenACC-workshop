CUDA_PATH ?= $(CUDA_HOME)

# INCDIRS = -I$(CUDA_PATH)/include
LIBDIRS = -L$(CUDA_PATH)/lib64
LIBS = -lcudart -lstdc++ -lgcc_s

COMPOPT = -Mcuda
ACCOPT = -acc=gpu -gpu=cc80 -Minfo=accel -Xlinker -z noexecstack

JSC_SUBMIT_CMD ?= srun --partition=dc-gpu --gres=gpu:1 

.PHONY: all
all: vecAddRed.bin

vecAddRed.bin: vecAddRed.F03 fortranthrust.F03 thrustWrapper.o Makefile
	nvfortran $(COMPOPT) $(ACCOPT) fortranthrust.F03 vecAddRed.F03 $(LIBDIRS) $(LIBS) thrustWrapper.o -o vecAddRed.bin

thrustWrapper.o: thrustWrapper.cu Makefile
	nvcc thrustWrapper.cu -c

.PHONY: run clean
run: vecAddRed.bin
	$(JSC_SUBMIT_CMD) ./vecAddRed.bin

clean:
	rm -f vecAddRed.bin *.o
