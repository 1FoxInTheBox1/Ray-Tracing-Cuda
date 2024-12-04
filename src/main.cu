#include "../include/color.cuh"
#include "../include/vec3.cuh"

#include <iostream>

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

__global__ void render(color *fb, int image_width, int image_height)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= image_width || j >= image_height)
        return;
    auto r = float(i) / (image_width - 1);
    auto g = float(j) / (image_height - 1);
    auto b = 0.0f;

    int pixel_index = i + image_width * j;
    color c = color(r, g, b);

    fb[pixel_index] = c;
}

int main()
{
    // Image
    int image_width = 256;
    int image_height = 256;
    int tx = 8;
    int ty = 8;
    int num_pixels = image_height * image_width;
    size_t fb_size = 3 * num_pixels * sizeof(color);

    color *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    // Render
    dim3 blocks(image_width / tx + 1, image_height / ty + 1);
    dim3 threads(tx, ty);
    render<<<blocks, threads>>>(fb, image_width, image_height);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    std::cout << "P3\n"
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

    std::clog << "\r Done.           \n";

    checkCudaErrors(cudaFree(fb));
}