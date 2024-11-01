#!/usr/bin/env bash
export MP_TASK_AFFINITY=1
for n in 32 64 128 256 512 1024 2048 4096 8192 16384; do ./dgemm $n|grep took; done | tee dgemm_bench.dat
python3 dgemm_bench.py
