#ifndef INTERVAL_H
#define INTERVAL_H

class interval
{
public:
    double min, max;

    __device__ interval() : min(+infinity), max(-infinity) {} // Default interval is empty

    __device__ interval(double min, double max) : min(min), max(max) {}

    __device__ double size() const
    {
        return max - min;
    }

    __device__ bool contains(double x) const
    {
        return min <= x && x <= max;
    }

    __device__ bool surrounds(double x) const
    {
        return min < x && x < max;
    }

    static const interval empty, universe;
};

const interval interval::empty = interval(+infinity, -infinity);
const interval interval::universe = interval(-infinity, +infinity);

#endif