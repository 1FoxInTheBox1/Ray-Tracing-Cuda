cmake_minimum_required(VERSION 3.18)
project(RayTracing LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 75)

# include_directories(include)

add_executable(${PROJECT_NAME} src/main.cu)