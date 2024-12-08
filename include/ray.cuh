#ifndef RAY_H
#define RAY_H

#include "vec3.cuh"

class ray
{
public:
    __device__ ray() {};
    __device__ ray(const point3 &origin, const vec3 &direction) : orig(origin), dir(direction) {}

    __device__ const point3 &ray::origin() const { return orig; }
    __device__ const vec3 &ray::direction() const { return dir; }

    __device__ point3 ray::at(double t) const
    {
        return orig + t * dir;
    }

private:
    point3 orig;
    vec3 dir;
};

#endif