#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "hittable.cuh"

#include <vector>

class hittable_list : public hittable
{
public:
    hittable **objects;
    int size;

    hittable_list() {}
    __device__ hittable_list(hittable **objects, int size) : objects(objects), size(size) {}

    __device__ void free_all()
    {
        for (int i = 0; i < size; i++)
        {
            delete objects[i];
        }
    }

    __device__ void add(hittable *object)
    {
        objects[size] = object;
        size++;
    }

    __device__ bool hit(const ray &r, interval ray_t, hit_record &rec) const override
    {
        hit_record temp_rec;
        bool hit_anything = false;
        auto closest_so_far = ray_t.max;

        for (int i = 0; i < size; i++)
        {
            hittable *object = objects[i];
            if (object->hit(r, interval(ray_t.min, closest_so_far), temp_rec))
            {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }

        return hit_anything;
    }
};

#endif