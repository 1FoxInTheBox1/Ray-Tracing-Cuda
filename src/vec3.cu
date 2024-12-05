#include "../include/vec3.cuh"

__host__ __device__ vec3 vec3::operator-() const { return vec3(-e[0], -e[1], -e[2]); }
__host__ __device__ float vec3::operator[](int i) const { return e[i]; }
__host__ __device__ float &vec3::operator[](int i) { return e[i]; }

__host__ __device__ vec3 &vec3::operator+=(const vec3 &v)
{
    e[0] += v.e[0];
    e[1] += v.e[1];
    e[2] += v.e[2];
    return *this;
}

__host__ __device__ vec3 &vec3::operator*=(float t)
{
    e[0] *= t;
    e[1] *= t;
    e[2] *= t;
    return *this;
}

__host__ __device__ vec3 &vec3::operator/=(float t)
{
    return *this *= 1 / t;
}
