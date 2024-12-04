#ifndef VEC3_H
#define VEC3_H

#include <cmath>
#include <iostream>

class vec3
{
public:
    float e[3];

    __host__ __device__ vec3() : e{0, 0, 0} {}
    __host__ __device__ vec3(float e0, float e1, float e2) : e{e0, e1, e2} {}

    __host__ __device__ float x() const;
    __host__ __device__ float y() const;
    __host__ __device__ float z() const;

    __host__ __device__ vec3 operator-() const;
    __host__ __device__ float operator[](int i) const;
    __host__ __device__ float &operator[](int i);

    __host__ __device__ vec3 &operator+=(const vec3 &v);
    __host__ __device__ vec3 &operator*=(float t);
    __host__ __device__ vec3 &operator/=(float t);
    __host__ __device__ float length() const;
    __host__ __device__ float length_squared() const;
};

// point3 is just an alias for vec3, but useful for geometric clarity in the code.
using point3 = vec3;

// Vector Utility Functions
inline std::ostream &operator<<(std::ostream &out, const vec3 &v);
__host__ __device__ inline vec3 operator+(const vec3 &u, const vec3 &v);
__host__ __device__ inline vec3 operator-(const vec3 &u, const vec3 &v);
__host__ __device__ inline vec3 operator*(const vec3 &u, const vec3 &v);
__host__ __device__ inline vec3 operator*(float t, const vec3 &v);
__host__ __device__ inline vec3 operator*(const vec3 &v, float t);
__host__ __device__ inline vec3 operator/(const vec3 &v, float t);
__host__ __device__ inline float dot(const vec3 &u, const vec3 &v);
__host__ __device__ inline vec3 cross(const vec3 &u, const vec3 &v);
__host__ __device__ inline vec3 unit_vector(const vec3 &v);

#endif