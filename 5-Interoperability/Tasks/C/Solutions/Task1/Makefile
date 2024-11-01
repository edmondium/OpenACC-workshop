CUDA_PATH ?= $(CUDA_HOME)

INCDIRS = -I$(CUDA_PATH)/include
LIBDIRS = -L$(CUDA_PATH)/lib64

LIBS = -lcublas

COMPOPT = 
ACCOPT = -acc=gpu -gpu=cc80 -Minfo=accel -Xlinker -z noexecstack

JSC_SUBMIT_CMD ?= srun --partition=dc-gpu --gres=gpu:1 

.PHONY: all
all: vecAddRed.bin

vecAddRed.bin: vecAddRed.c Makefile
	nvc++ $(COMPOPT) $(ACCOPT) vecAddRed.c $(LIBDIRS) $(LIBS) -o vecAddRed.bin

.PHONY: run clean
run: vecAddRed.bin
	$(JSC_SUBMIT_CMD) ./vecAddRed.bin
	
clean:
	rm -f vecAddRed.bin *.o
