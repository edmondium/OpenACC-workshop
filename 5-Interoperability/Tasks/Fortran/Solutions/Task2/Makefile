CUDA_PATH ?= $(CUDA_HOME)

# INCDIRS = -I$(CUDA_PATH)/include
# LIBDIRS = -L$(CUDA_PATH)/lib64
# Not needed for Fortran

COMPOPT = -Mcuda
ACCOPT = -acc=gpu -ta=tesla:cc80 -Minfo=accel -Xlinker -z noexecstack

JSC_SUBMIT_CMD ?= srun --partition=dc-gpu --gres=gpu:1 

.PHONY: all
all: vecAddRed.bin

vecAddRed.bin: vecAddRed.F03 Makefile
	nvfortran $(COMPOPT) $(ACCOPT) vecAddRed.F03 $(LIBDIRS) $(LIBS) -o vecAddRed.bin

.PHONY: debug
debug: vecAddRed.F03 Makefile
	nvfortran $(COMPOPT) -g $(ACCOPT) vecAddRed.F03 $(LIBDIRS) $(LIBS) -o vecAddRed.bin

.PHONY: run clean
run: vecAddRed.bin
	$(JSC_SUBMIT_CMD) ./vecAddRed.bin


clean:
	rm -f vecAddRed.bin *.o
