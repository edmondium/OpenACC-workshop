import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

datafile = "mandelbrot_bench.dat"
posFlops = 11

data = np.array([[l.split()[posFlops - 3], l.split()[posFlops]] for l in open("mandelbrot_bench.dat")],dtype=float)
#plt.loglog(data[2::3,0], data[2::3,1], 'o-', label="CPU (OpenCL)")
plt.plot(data[::2,0], 1e-6 * data[::2, 0] ** 2 / data[::2,1], 'o-', label="GPU")
plt.plot(data[1::2,0], 1e-6 * data[1::2, 0] ** 2 / data[1::2,1], 'o-', label="CPU")
#plt.loglog(data[1::3,0], data[1::3,1], 'o-', label="GPU (OpenCL)")
#plt.axis([60, 10000, 10**-4, 1])
plt.legend(loc='upper left')
plt.xlabel("Width of Image")
plt.ylabel("MPixel/s")
plt.title("Mandelbrot Benchmark")
plt.savefig("mandelbrot_bench.png")
plt.savefig("mandelbrot_bench.svg")
plt.savefig("mandelbrot_bench.pdf")
# plt.show()
