#include "../include/ray.cuh"

__device__ point3 ray::at(double t) const {
    return orig + t*dir;
}