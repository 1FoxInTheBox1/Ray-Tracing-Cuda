#ifndef RTWEEKEND_H
#define RTWEEKEND_H

#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <stdio.h>
#include <curand_kernel.h>

// C++ Std Usings

using std::make_shared;
using std::shared_ptr;

// Macros

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

// Constants

__device__ const double infinity = std::numeric_limits<double>::infinity();
__device__ const double pi = 3.1415926535897932385;

// Utility Functions

__device__ inline double degrees_to_radians(double degrees) {
    return degrees * pi / 180.0;
}

// Common Headers

#include "color.cuh"
#include "interval.cuh"
#include "ray.cuh"
#include "vec3.cuh"

#endif