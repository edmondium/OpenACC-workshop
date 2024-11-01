import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

datafile = "dgemm_bench.dat"
posFlops = 4
n = np.array([2**i for i in range(5, 15, 1 )])
GFLOP = 2. * n ** 3 / 10 ** 9
data = np.array([float(l.split()[posFlops]) for l in open(datafile)])
plt.plot(n[:len(data[::2])], GFLOP[:len(data[::2])]/data[::2], 'o-', label="CPU")
plt.plot(n[:len(data[1::2])], GFLOP[:len(data[1::2])]/data[1::2],'s-',  label="GPU")
plt.legend(loc='upper left')
plt.xlabel("Size of Square Matrix")
plt.ylabel("GFLOP/s")
plt.title("DGEMM Benchmark")
plt.savefig("dgemm_bench.png")
plt.savefig("dgemm_bench.svg")
plt.savefig("dgemm_bench.pdf")
# plt.show()
