#!/usr/bin/env bash
export MP_TASK_AFFINITY=1
for n in 64 256 1024 4096 8192 16384 32768; 
do 
    ./mandelbrot $n
    ./mandelbrot_cpu $n $n
done | tee mandelbrot_bench.dat
#python Mandelbrot.py gpu &
python3 mandelbrot_bench.py
