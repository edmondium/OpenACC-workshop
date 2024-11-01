import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

datafile = "nbody_bench.dat"
posFlops = 1
n = np.array([2**i for i in range(10, 18, 1)])
data = np.array([float(l.split()[posFlops]) for l in open(datafile)])
plt.plot(n, data[::6], 'o-', label="1 GPU SP")
plt.plot(n, data[1::6],'s-',  label="2 GPUs SP")
plt.plot(n, data[2::6],'p-',  label="4 GPUs SP")
plt.plot(n, data[3::6],'d--', label="1 GPU DP")
plt.plot(n, data[4::6],'x--', label="2 GPUs DP")
plt.plot(n, data[5::6],'h--', label="4 GPUs DP")
plt.axis([0, 132000, 0, 25000])
plt.legend(loc='upper left')
plt.xlabel("Number of Particles")
plt.ylabel("GFLOP/s")
plt.title("N-Body Benchmark")
plt.savefig("nbody_bench.png")
plt.savefig("nbody_bench.pdf")
# plt.show()
