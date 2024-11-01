#!/usr/bin/make -f

.PHONY: all clean clean-dat clean-pdf
all: nbody_bench.pdf

nbody_bench.dat: FORCE
	CUDA_VISIBLE_DEVICES=0,1,2,3; \
	for n in 1024 2048 4096 8192 16384 32768 65536 131072; do \
		./nbody -hostmem -benchmark -numbodies=$$n -numdevices=1 | grep GFLOP; \
		./nbody -benchmark -numbodies=$$n -numdevices=2 | grep GFLOP; \
		./nbody -benchmark -numbodies=$$n -numdevices=4 | grep GFLOP; \
		./nbody -fp64 -benchmark -numbodies=$$n -numdevices=1 | grep GFLOP; \
		./nbody -fp64 -benchmark -numbodies=$$n -numdevices=2 | grep GFLOP; \
		./nbody -fp64 -benchmark -numbodies=$$n -numdevices=4 | grep GFLOP; \
	done | tee nbody_bench.dat

nbody_bench.pdf: nbody_bench.dat
	python3 nbody_bench.py

clean: clean-dat clean-pdf
clean-dat:
	rm nbody_bench.dat
clean-pdf:
	rm nbody_bench.pdf

FORCE:
