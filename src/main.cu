
#include "../include/rtweekend.cuh"

#include "../include/camera.cuh"
#include "../include/hittable.cuh"
#include "../include/hittable_list.cuh"
#include "../include/sphere.cuh"

__global__ void build_world(hittable_list **d_world, hittable **d_objects)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i == 0 && j == 0)
    {
        d_objects[0] = new sphere(point3(0, 0, -1), 0.5);
        d_objects[1] = new sphere(point3(0, -100.5, -1), 100);
        *d_world = new hittable_list(d_objects, 2);
    }
}

// __global__ void free_world(hittable **d_objects, hittable_list **d_world)
// {
//     int i = threadIdx.x + blockIdx.x * blockDim.x;
//     int j = threadIdx.y + blockIdx.y * blockDim.y;
//     if (i == 0 && j == 0)
//     {
//         delete *(d_objects);
//         delete *(d_objects + 1);
//         // (*d_world)->free_all();
//         delete *d_world;
//     }
// }

__global__ void render(camera *d_cam, hittable_list **d_world)
{
    d_cam->start_render(d_world);
}

void writeImage(color *fb, int image_width, int image_height)
{
    std::cout
        << "P3\n"
        << image_width << ' ' << image_height << "\n255\n";

    for (int j = 0; j < image_height; j++)
    {
        for (int i = 0; i < image_width; i++)
        {
            int pixel_index = i + image_width * j;
            color pixel_color = fb[pixel_index];
            write_color(std::cout, pixel_color);
        }
    }
}

void print_mem_data()
{
    size_t free_byte;
    size_t total_byte;
    cudaError_t cuda_status = cudaMemGetInfo(&free_byte, &total_byte);
    double free_db = (double)free_byte;
    double total_db = (double)total_byte;
    double used_db = total_db - free_db;
    std::clog << "GPU memory usage: used = " << used_db / 1024.0 / 1024.0
              << ", free = " << free_db / 1024.0 / 1024.0 << " MB, total = " << total_db / 1024.0 / 1024.0 << " MB\n";
}

int main()
{
    float aspect_ratio = 16.0 / 9.0;
    int image_width = 400;
    int image_height = int(image_width / aspect_ratio);
    image_height = (image_height < 1) ? 1 : image_height;
    int num_pixels = image_height * image_width;
    size_t fb_size = 3 * num_pixels * sizeof(color);

    int tx = 8;
    int ty = 8;
    dim3 blocks(image_width / tx + 1, image_height / ty + 1);
    dim3 threads(tx, ty);

    // World
    hittable_list **d_world;
    hittable **d_objects;
    checkCudaErrors(cudaMallocManaged(&d_world, sizeof(hittable_list *)));
    checkCudaErrors(cudaMallocManaged(&d_objects, sizeof(hittable *) * 2));
    build_world<<<1, 1>>>(d_world, d_objects);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Camera
    camera *d_cam;
    checkCudaErrors(cudaMallocManaged(&d_cam, sizeof(camera *)));

    d_cam->aspect_ratio = aspect_ratio;
    d_cam->image_width = image_width;
    d_cam->image_height = image_height;
    checkCudaErrors(cudaMallocManaged(&d_cam->fb, fb_size));
    checkCudaErrors(cudaMallocManaged(&d_cam->rand_state, num_pixels * sizeof(curandState)));

    // Render
    std::clog << " Beginning Rendering\n";
    auto render_start = std::chrono::high_resolution_clock::now();

    render<<<blocks, threads>>>(d_cam, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    auto render_stop = std::chrono::high_resolution_clock::now();
    auto render_duration = std::chrono::duration_cast<std::chrono::milliseconds>(render_stop - render_start);
    std::clog << " Rendering complete in " << render_duration.count() << " milliseconds\n";
    std::clog << "\r Done.           \n";

    color *fb = (color *)malloc(fb_size);

    cudaMemcpy(fb, d_cam->fb, fb_size, cudaMemcpyDeviceToHost);

    writeImage(fb, image_width, image_height);

    // Clean up
    free(fb);
    checkCudaErrors(cudaFree(d_cam));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_objects));
    return 0;
}