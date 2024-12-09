#ifndef CAMERA_H
#define CAMERA_H

#include <chrono>

#include "hittable_list.cuh"
#include "hittable.cuh"

class camera
{
public:
    float aspect_ratio = 1.0f;  // Ratio of image width over height
    int image_width = 100;      // Rendered image width in pixel count
    int image_height = 100;     // Rendered image height in pixel count
    int samples_per_pixel = 10; // Count of random samples for each pixel
    color *fb;                  // Frame buffer to render image to
    curandState *rand_state;    // RNG for the camera

    __device__ void start_render(hittable_list **d_world)
    {
        initialize();

        render(d_world);

        // cleanup();
    }

private:
    float pixel_samples_scale; // Color scale factor for a sum of pixel samples
    point3 center;             // Camera center
    point3 pixel00_loc;        // Location of pixel 0, 0
    vec3 pixel_delta_u;        // Offset to pixel to the right
    vec3 pixel_delta_v;        // Offset to pixel below

    // Initialize rand_states
    __device__ void init_rand_states()
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;
        if (i >= image_width || j >= image_height)
            return;
        int pixel_index = i + image_width * j;

        curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
    }

    __device__ void initialize()
    {
        pixel_samples_scale = 1.0 / samples_per_pixel;
        center = point3(0, 0, 0);

        // Determine viewport dimensions
        auto focal_length = 1.0f;
        auto viewport_height = 2.0f;
        auto viewport_width = viewport_height * (float(image_width) / image_height);

        // Calculate the vectors across the horizontal and down the vertical viewport edges.
        auto viewport_u = vec3(viewport_width, 0, 0);
        auto viewport_v = vec3(0, -viewport_height, 0);

        // Calculate the horizontal and vertical delta vectors from pixel to pixel.
        pixel_delta_u = viewport_u / image_width;
        pixel_delta_v = viewport_v / image_height;

        // Calculate the location of the upper left pixel.
        auto viewport_upper_left = center - vec3(0, 0, focal_length) - viewport_u / 2 - viewport_v / 2;
        pixel00_loc = viewport_upper_left + 0.5f * (pixel_delta_u + pixel_delta_v);

        // Set up RNG
        init_rand_states();
    }

    __device__ void render(hittable_list **world)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;
        if (i >= image_width || j >= image_height)
            return;
        int pixel_index = i + image_width * j;
        curandState local_rand_state = rand_state[pixel_index];

        color pixel_color(0, 0, 0);
        for (int sample = 0; sample < samples_per_pixel; sample++)
        {
            ray r = get_ray(i, j, local_rand_state);
            // printf("ray: %f %f %f\n", r.direction().x(), r.direction().y(), r.direction().z());
            pixel_color += ray_color(r, *world);
        }

        // printf("cooler: %f %f %f\n", pixel_color.x(), pixel_color.y(), pixel_color.z());
        color final = pixel_samples_scale * pixel_color;
        // printf("scale: %f\n", pixel_samples_scale);
        // printf("final: %f %f %f\n", final.x(), final.y(), final.z());
        fb[pixel_index] = final;
    }

    __device__ color ray_color(const ray &r, const hittable *world)
    {
        hit_record rec;
        if (world->hit(r, interval(0, infinity), rec))
        {
            return 0.5 * (rec.normal + color(1, 1, 1));
        }

        vec3 unit_direction = unit_vector(r.direction());
        auto a = 0.5f * (unit_direction.y() + 1.0f);
        return (1.0f - a) * color(1.0f, 1.0f, 1.0f) + a * color(0.5f, 0.7f, 1.0f);
    }

    __device__ ray get_ray(int i, int j, curandState &rand_state) const
    {
        // Construct a camera ray originating from the origin and directed at randomly sampled
        // point around the pixel location i, j.

        auto offset = sample_square(rand_state);
        // printf("offset: %f %f %f\n", offset.x(), offset.y(), offset.z());
        
        auto pixel_sample = pixel00_loc + ((i + offset.x()) * pixel_delta_u) + ((j + offset.y()) * pixel_delta_v);

        auto ray_origin = center;
        // printf("pix sample: %f %f %f\n", pixel_sample.x(), pixel_sample.y(), pixel_sample.z());
        // printf("origin: %f %f %f\n", ray_origin.x(), ray_origin.y(), ray_origin.z());
        auto ray_direction = pixel_sample - ray_origin;
        // printf("direction: %f %f %f\n", ray_direction.x(), ray_direction.y(), ray_direction.z());

        return ray(ray_origin, ray_direction);
    }

    // Returns the vector to a random point in the [-.5,-.5]-[+.5,+.5] unit square.
    __device__ vec3 sample_square(curandState &rand_state) const
    {
        return vec3(random_float(rand_state) - 0.5f, random_float(rand_state) - 0.5f, 0);
    }
};

#endif