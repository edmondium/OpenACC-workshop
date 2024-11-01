#!/usr/bin/env bash

[ -z "$JSCCOURSE_TASK1_MODULES_SOURCED" ] && echo "## Please first call \`source setup.sh\` to prepare the environment!" && exit 1;

[ -z "$CUDA_HOME" ] && echo "## Please load CUDA before invoking the script!" && exit 1;

make

if [ -z "$JSC_SUBMIT_CMD" ]; then
	echo ""
	echo "## Please source course script first!"
	exit
fi

#OMP_NUM_THREADS is set to physical cores
# export OMP_NUM_THREADS=$(lscpu -p | egrep -v '^#' | sort -u -t, -k 2,4 | wc -l)
export OMP_NUM_THREADS=48
export SLURM_CPUS_PER_TASK=$OMP_NUM_THREADS
export CUDA_VISIBLE_DEVICES=0,1,2,3
for benchmark in dgemm_bench.sh nbody_bench.mk ddot_bench.sh mandelbrot_bench.sh; do
	echo "# Running benchmark $benchmark"
	$JSC_SUBMIT_CMD -n 1 ./$benchmark
done