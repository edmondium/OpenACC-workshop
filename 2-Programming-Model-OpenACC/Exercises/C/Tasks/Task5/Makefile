# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
COMPILER ?= NVHPC-manual
CC = nvc
CFLAGS = -DUSE_DOUBLE
ifeq ($(COMPILER),GCC)
	CFLAGS += -std=c99 -march=native -O3 -lm
else ifeq ($(COMPILER),NVHPC)
	CFLAGS += -O1
else ifeq ($(COMPILER),NVHPC-tesla)
	CFLAGS += -Minfo=accel -O1 -acc=gpu -gpu=managed,cc80
else ifeq ($(COMPILER),NVHPC-manual)
	CFLAGS += -Minfo=accel -O1 -acc=gpu -gpu=cc80
else ifeq ($(COMPILER),NVHPC-multicore)
	CFLAGS += -Minfo=accel -O1 -acc=multicore
endif

JSC_SUBMIT_CMD ?= srun --partition=gpus --gres=gpu:1 

all: poisson2d

poisson2d_reference.o: poisson2d_reference.c common.h  Makefile
	$(CC) -c $(CFLAGS) poisson2d_reference.c -o $@

poisson2d: poisson2d.c common.h poisson2d_reference.o Makefile
	$(CC) $(CFLAGS) poisson2d.c poisson2d_reference.o -o $@

.PHONY: clean run profile
clean:
	rm -f poisson2d poisson2d*.o

run: poisson2d
	$(JSC_SUBMIT_CMD) ./poisson2d

profile: poisson2d
	$(JSC_SUBMIT_CMD) nsys nvprof ./poisson2d 10
profile_tofile: poisson2d
	$(JSC_SUBMIT_CMD) nsys profile -o poisson2d.qdrep ./poisson2d 10

.PHONY: task1 task1_profile task4
task1: poisson2d_reference.c poisson2d.c common.h Makefile
	gcc -g -pg -DUSE_DOUBLE -c -o poisson2d_reference.o poisson2d_reference.c
	gcc -g -pg -DUSE_DOUBLE -lm poisson2d_reference.o poisson2d.c -o poisson2d

task1_profile:
	-rm gmon.out
	$(JSC_SUBMIT_CMD) ./poisson2d 10
	gprof -p -l ./poisson2d gmon.out
