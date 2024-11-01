#include <iostream>
#include <chrono>
#include <vector>

std::vector<int> mandelbrot(double xmin=-2.5, double xmax=1.0, double ymin=-1.1, double ymax=1.1, 
        int width=1920, int height=1080, int maxNumberOfIterations=10){
    std::vector<int> image(width * height, maxNumberOfIterations);
    auto dx = (xmax - xmin) / width;
    auto dy = (ymax - ymin) / height;
#pragma omp parallel for
    for(int h = 0; h < height; ++h){
        for(int w = 0; w < height; ++w){
            auto cr = xmin + w * dx;
            auto ci = ymin + h * dy;
            auto zr = 0;
            auto zi = 0;
            for(int i = 0; i < maxNumberOfIterations; ++i){
                auto tzr = zr * zr - zi * zi + cr;
                auto tzi = 2 * zr * zi + ci;
                if((tzr * tzr + tzi * tzi) > 4){
                    image[h * width + w] = i;
                    cr = 0; ci = 0;
                    tzr = 0; tzi = 0;
//                     break;
                }
                zr = tzr; zi = tzi;
            }
        }
    }
    return image;
}

int main(int argc, char** argv){
    int width = 1920;
    int height = 1080;
    if(argc >= 3){
        width = std::stoi(argv[1]);
        height = std::stoi(argv[2]);
    }
    auto start_time = std::chrono::system_clock::now();
    auto image = mandelbrot(-2.5, 1.0, -1.1, 1.1, width, height, 10);
    auto end_time = std::chrono::system_clock::now();
    std::cout << "Calculating the Mandelbrot set on a " << width << " by " << height << " grid took ";
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count()*1e-6 << " s" << std::endl;
}

