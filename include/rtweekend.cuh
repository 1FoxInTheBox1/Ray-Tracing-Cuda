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

__device__ const float infinity = std::numeric_limits<float>::infinity();
__device__ const float pi = 3.1415926535897932385f;

// Utility Functions

__device__ inline float degrees_to_radians(double degrees) {
    return degrees * pi / 180.0f;
}

// Returns a random float in (0, 1]
__device__ inline float random_float(curandState &rand_state) {
    return curand_uniform(&rand_state);
}

// Returns a random float in [min, max)
__device__ inline float random_float(float min, float max, curandState rand_state) {
    return min + (max-min) * random_float(rand_state);
}

// Common Headers

#include "interval.cuh"
#include "color.cuh"
#include "ray.cuh"
#include "vec3.cuh"

#endif