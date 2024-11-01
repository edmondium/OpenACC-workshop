LIBS = -lcufft

COMPOPT = -Mcuda
ACCOPT = -acc=gpu -gpu=cc80 -Minfo=accel -Xlinker -z noexecstack

JSC_SUBMIT_CMD ?= srun --partition=dc-gpu --gres=gpu:1 

.PHONY: all
all: poisson.bin

poisson.bin: poisson.F90 Makefile
	nvfortran $(COMPOPT) $(ACCOPT) poisson.F90 $(LIBDIRS) $(LIBS) -o poisson.bin

.PHONY: debug
debug: poisson.F90 Makefile
	nvfortran $(COMPOPT) -g $(ACCOPT) poisson.F90 $(LIBDIRS) $(LIBS) -o poisson.bin

.PHONY: run clean
run: poisson.bin
	$(JSC_SUBMIT_CMD) ./poisson.bin

clean:
	rm -f poisson.bin *.o
