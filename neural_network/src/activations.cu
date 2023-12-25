#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <limits.h>

extern "C"
{
#include "initializers.h"
#include "activations.cuh"
#include "utils.h"
}

void activation_present()
{
    printf("Activation Present\n");
}

__device__ float sigmoid(float x)
{
    return 1.0 / (1.0 + exp(-x));
}

__global__ void sigmoid_kernel(float *input, float *output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        output[idx] = sigmoid(input[idx]);
    }
}

extern "C" tensor sigmoid_activation_gpu(tensor rw)
{
    int total_data = rw.size[0] * rw.size[1];
    unsigned int num_threads = NUM_THREADS;
    unsigned int num_blocks = ceil((float)total_data / num_threads);

    float *dvector, *dres_vector, *vector = convert2DTo1D(rw.matrix, rw.size[0], rw.size[1], true);
    cudaMalloc(&dvector, total_data * sizeof(float));
    cudaMalloc(&dres_vector, total_data * sizeof(float));

    cudaMemcpy(dvector, vector, total_data * sizeof(float), cudaMemcpyHostToDevice);
    sigmoid_kernel<<<num_blocks, num_threads>>>(dvector, dres_vector, total_data);
    cudaMemcpy(vector, dres_vector, total_data * sizeof(float), cudaMemcpyDeviceToHost);

    rw.matrix = convert1DTo2D(vector, rw.size[0], rw.size[1], true);

    cudaFree(dvector);
    cudaFree(dres_vector);
    return rw;
}

extern "C" tensor sigmoid_activation(tensor rw)
{
    for (int i = 0; i < rw.size[0]; i++)
    {
        for (int j = 0; j < rw.size[1]; j++)
        {
            rw.matrix[i][j] = 1.0 / (1.0 + expf(-rw.matrix[i][j])); // Sigmoid activation
        }
    }
    return rw;
}

__device__ float relu(float x)
{
    return fmaxf(0.0, x);
}

__global__ void relu_kernel(float *input, float *output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        output[idx] = relu(input[idx]);
    }
}

extern "C" tensor relu_activation_gpu(tensor rw)
{
    int total_data = rw.size[0] * rw.size[1];
    unsigned int num_threads = 1024;
    unsigned int num_blocks = ceil((float)total_data / num_threads);

    float *dvector, *dres_vector, *vector = convert2DTo1D(rw.matrix, rw.size[0], rw.size[1], true);
    cudaMalloc(&dvector, total_data * sizeof(float));
    cudaMalloc(&dres_vector, total_data * sizeof(float));

    cudaMemcpy(dvector, vector, total_data * sizeof(float), cudaMemcpyHostToDevice);
    relu_kernel<<<num_blocks, num_threads>>>(dvector, dres_vector, total_data);
    cudaMemcpy(vector, dres_vector, total_data * sizeof(float), cudaMemcpyDeviceToHost);

    rw.matrix = convert1DTo2D(vector, rw.size[0], rw.size[1], true);

    cudaFree(dvector);
    cudaFree(dres_vector);
    return rw;
}

extern "C" tensor relu_activation(tensor rw)
{
    for (int i = 0; i < rw.size[0]; i++)
    {
        for (int j = 0; j < rw.size[1]; j++)
        {
            rw.matrix[i][j] = fmaxf(0.0, rw.matrix[i][j]); // ReLU activation
        }
    }
    return rw;
}