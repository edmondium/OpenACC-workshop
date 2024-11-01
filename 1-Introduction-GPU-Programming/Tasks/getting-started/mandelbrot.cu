#include <iostream>
#include <string>
#include <chrono>

__host__ __device__
int iterate(double real, double imaginary, int max_iterations){
	double zr {real};
	double zi {imaginary};

	for (int i = 1; i < max_iterations; i++){
		if ((zr * zr + zi * zi) > 4) {
			return i;
		}
		double tzr = zr * zr - zi * zi + real;
		zi = 2 * zr * zi + imaginary;
		zr = tzr;
	}
	return max_iterations;
}

__global__
void mandelbrot(double xmin, double xmax, double ymin, double ymax, int max_iterations, int* iterations){
	int nx = gridDim.x * blockDim.x;
	int ny = gridDim.y * blockDim.y;
	double dx = (xmax - xmin) / nx;
	double dy = (ymax - ymin) / ny;
	double x = (blockIdx.x * blockDim.x + threadIdx.x) * dx + xmin;
	double y = (blockIdx.y * blockDim.y + threadIdx.y) * dy + ymin;

    iterations[(blockIdx.y * blockDim.y + threadIdx.y) * nx + blockIdx.x * blockDim.x + threadIdx.x] = iterate(x, y, max_iterations);
}

void asciiMandelbrot(int width, int height, int max_iter, int* iterations){
	for (int i = 0; i < height; ++i){
		for (int j = 0; j < width; j++){
			if (iterations[i * width + j] >= max_iter){
				std::cout << "+";
			}else{
				std::cout << " ";
			}
		}
		std::cout << std::endl;
	}
}

void iterationsToStandardOut(int width, int height, int max_iter, int* iterations){
	for (int i = 0; i < height; ++i){
		for (int j = 0; j < width; j++){
			std::cout << iterations[i * width +j] << " ";
		}
		std::cout << std::endl;
	}
}


int main(int argc, char** argv){
	int height {64};
	if (argc > 1){
		height = std::stoi(argv[1]);
	}
	int width = {height};

	double xmin {-2.13};
	double xmax {0.77};
	double ymin {-1.45};
	double ymax { 1.45};

	dim3 block {16, 16};
	unsigned int bbdim = (height % block.y == 0) ? height / block.y : height / block.y + 1;
	dim3 grid {bbdim, bbdim};

	int* iterations;
	cudaMallocManaged(&iterations, bbdim * bbdim * block.y * block.y * sizeof(int));
	auto start_time = std::chrono::high_resolution_clock::now();
	mandelbrot<<<grid, block>>>(xmin, xmax, ymin, ymax, 1000, iterations);
	cudaDeviceSynchronize();
	auto end_time = std::chrono::high_resolution_clock::now();
	std::cout << "Calculating the Mandelbrot set on a " << width << " by " << height << " grid took ";
	std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() * 1e-6 << " s" << std::endl;

//	asciiMandelbrot(width, height, 1000, iterations);
}
