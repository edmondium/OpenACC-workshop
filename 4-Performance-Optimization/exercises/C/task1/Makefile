# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
CC=nvc
NCU?=ncu
SW?=3
SM?=80
CFLAGS=-gopt -fast -Minfo=accel
TASK?=$(notdir $(shell pwd))
RUN?=$(JSC_SUBMIT_CMD) --cpu-bind=none
OMP_NUM_THREADS?=256

export OMP_NUM_THREADS

run: conv_gpu
	$(RUN) ./conv_gpu $(SW)

ncu: conv_gpu
	ncu -c1 -s3 ./conv_gpu $(SW)

profile: conv_gpu
	$(RUN) $(NCU) -c1 -s3 --import-source on --set full -f -o conv_$(TASK)_sw$(SW) ./conv_gpu $(SW)

run_par: all
	$(RUN) ./conv_multicore $(SW)
	$(RUN) ./conv_gpu $(SW)

run_all: all	
	$(RUN) ./conv_serial $(SW)
	$(RUN) ./conv_multicore $(SW)
	$(RUN) ./conv_gpu $(SW)

all: conv_serial conv_gpu conv_multicore

clean:
	rm -f conv_serial conv_gpu conv_multicore

conv_serial: conv.c
	$(CC) $(CFLAGS) -acc=host $^ -o $@

conv_gpu: conv.c
	$(CC) $(CFLAGS) -acc=gpu -gpu=cc$(SM) $^ -o $@

conv_multicore: conv.c
	$(CC) $(CFLAGS) -acc=multicore $^ -o $@

.PHONY: all clean run_all run_par profile
