/*
 * @Author: Shuvrajeet Das 
 * @Date: 2023-12-28 13:45:51 
 * @Last Modified by:   shuvrajeet 
 * @Last Modified time: 2023-12-28 13:45:51 
 */
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

// Sigmoid Functions

__device__ float sigmoid_device(float x)
{
    return 1.0 / (1.0 + exp(-x));
}

__device__ float sigmoid_gradient_device(float x)
{
    return (1 - x) * x;
}

__global__ void sigmoid_kernel(float *input, float *output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        output[idx] = sigmoid_device(input[idx]);
    }
}

__global__ void sigmoid_gradient_kernel(float *input, float *output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        output[idx] = sigmoid_gradient_device(input[idx]);
    }
}

extern "C" float sigmoid_host(float x)
{
    return 1.0 / (1.0 + exp(-x));
}

extern "C" float sigmoid_gradient_host(float x)
{
    return (1 - x) * x;
}

extern "C" tensor sigmoid_activation_gpu(tensor rw)
{
    int total_data = rw.size[0] * rw.size[1];
    int num_threads = NUM_THREADS;
    int num_blocks = ceil((float)total_data / num_threads);

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
            rw.matrix[i][j] = sigmoid_host(rw.matrix[i][j]);
        }
    }
    return rw;
}

extern "C" tensor sigmoid_gradient_gpu(tensor rw)
{
    int total_data = rw.size[0] * rw.size[1];
    int num_threads = NUM_THREADS;
    int num_blocks = ceil((float)total_data / num_threads);

    float *dvector, *dres_vector, *vector = convert2DTo1D(rw.matrix, rw.size[0], rw.size[1], true);
    cudaMalloc(&dvector, total_data * sizeof(float));
    cudaMalloc(&dres_vector, total_data * sizeof(float));

    cudaMemcpy(dvector, vector, total_data * sizeof(float), cudaMemcpyHostToDevice);
    sigmoid_gradient_kernel<<<num_blocks, num_threads>>>(dvector, dres_vector, total_data);
    cudaMemcpy(vector, dres_vector, total_data * sizeof(float), cudaMemcpyDeviceToHost);

    rw.matrix = convert1DTo2D(vector, rw.size[0], rw.size[1], true);

    cudaFree(dvector);
    cudaFree(dres_vector);
    return rw;
}

extern "C" tensor sigmoid_gradient(tensor rw)
{
    for (int i = 0; i < rw.size[0]; i++)
    {
        for (int j = 0; j < rw.size[1]; j++)
        {
            rw.matrix[i][j] = sigmoid_gradient_host(rw.matrix[i][j]);
        }
    }
    return rw;
}

// ReLU Fucntions

__device__ float relu_device(float x)
{
    return fmaxf(0.0, x);
}

__device__ float relu_gradient_device(float x)
{
    return (x > 0);
}

__global__ void relu_kernel(float *input, float *output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        output[idx] = relu_device(input[idx]);
    }
}

__global__ void relu_gradient_kernel(float *input, float *output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        output[idx] = relu_gradient_device(input[idx]);
    }
}

extern "C" float relu_host(float x)
{
    return fmaxf(0.0, x);
}

extern "C" float relu_gradient_host(float x)
{
    return (x > 0);
}

extern "C" tensor relu_activation_gpu(tensor rw)
{
    int total_data = rw.size[0] * rw.size[1];
    int num_threads = NUM_THREADS;
    int num_blocks = ceil((float)total_data / num_threads);

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
            rw.matrix[i][j] = relu_host(rw.matrix[i][j]); // ReLU activation
        }
    }
    return rw;
}

extern "C" tensor relu_gradient_gpu(tensor rw)
{
    int total_data = rw.size[0] * rw.size[1];
    int num_threads = NUM_THREADS;
    int num_blocks = ceil((float)total_data / num_threads);

    float *dvector, *dres_vector, *vector = convert2DTo1D(rw.matrix, rw.size[0], rw.size[1], true);
    cudaMalloc(&dvector, total_data * sizeof(float));
    cudaMalloc(&dres_vector, total_data * sizeof(float));

    cudaMemcpy(dvector, vector, total_data * sizeof(float), cudaMemcpyHostToDevice);
    relu_gradient_kernel<<<num_blocks, num_threads>>>(dvector, dres_vector, total_data);
    cudaMemcpy(vector, dres_vector, total_data * sizeof(float), cudaMemcpyDeviceToHost);

    rw.matrix = convert1DTo2D(vector, rw.size[0], rw.size[1], true);

    cudaFree(dvector);
    cudaFree(dres_vector);
    return rw;
}

extern "C" tensor relu_gradient(tensor rw)
{
    for (int i = 0; i < rw.size[0]; i++)
    {
        for (int j = 0; j < rw.size[1]; j++)
        {
            rw.matrix[i][j] = relu_gradient_host(rw.matrix[i][j]); // ReLU activation
        }
    }
    return rw;
}

// Tanh Functions

__device__ float tanh_device(float x)
{
    return (2.f / (1 + expf(-2 * x)) - 1);
}

__device__ float tanh_gradient_device(float x)
{
    return 1 - x * x;
}

__global__ void tanh_kernel(float *input, float *output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        output[idx] = tanh_device(input[idx]);
    }
}

__global__ void tanh_gradient_kernel(float *input, float *output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        output[idx] = tanh_gradient_device(input[idx]);
    }
}

extern "C" float tanh_host(float x)
{
    return (2.f / (1 + expf(-2 * x)) - 1);
}

extern "C" float tanh_gradient_host(float x)
{
    return 1 - x * x;
}

extern "C" tensor tanh_activation_gpu(tensor rw)
{
    int total_data = rw.size[0] * rw.size[1];
    int num_threads = NUM_THREADS;
    int num_blocks = ceil((float)total_data / num_threads);

    float *dvector, *dres_vector, *vector = convert2DTo1D(rw.matrix, rw.size[0], rw.size[1], true);
    cudaMalloc(&dvector, total_data * sizeof(float));
    cudaMalloc(&dres_vector, total_data * sizeof(float));

    cudaMemcpy(dvector, vector, total_data * sizeof(float), cudaMemcpyHostToDevice);
    tanh_kernel<<<num_blocks, num_threads>>>(dvector, dres_vector, total_data);
    cudaMemcpy(vector, dres_vector, total_data * sizeof(float), cudaMemcpyDeviceToHost);

    rw.matrix = convert1DTo2D(vector, rw.size[0], rw.size[1], true);

    cudaFree(dvector);
    cudaFree(dres_vector);
    return rw;
}

extern "C" tensor tanh_activation(tensor rw)
{
    for (int i = 0; i < rw.size[0]; i++)
    {
        for (int j = 0; j < rw.size[1]; j++)
        {
            rw.matrix[i][j] = tanh_host(rw.matrix[i][j]); // tanh activation
        }
    }
    return rw;
}

extern "C" tensor tanh_gradient_gpu(tensor rw)
{
    int total_data = rw.size[0] * rw.size[1];
    int num_threads = NUM_THREADS;
    int num_blocks = ceil((float)total_data / num_threads);

    float *dvector, *dres_vector, *vector = convert2DTo1D(rw.matrix, rw.size[0], rw.size[1], true);
    cudaMalloc(&dvector, total_data * sizeof(float));
    cudaMalloc(&dres_vector, total_data * sizeof(float));

    cudaMemcpy(dvector, vector, total_data * sizeof(float), cudaMemcpyHostToDevice);
    tanh_gradient_kernel<<<num_blocks, num_threads>>>(dvector, dres_vector, total_data);
    cudaMemcpy(vector, dres_vector, total_data * sizeof(float), cudaMemcpyDeviceToHost);

    rw.matrix = convert1DTo2D(vector, rw.size[0], rw.size[1], true);

    cudaFree(dvector);
    cudaFree(dres_vector);
    return rw;
}

extern "C" tensor tanh_gradient(tensor rw)
{
    for (int i = 0; i < rw.size[0]; i++)
    {
        for (int j = 0; j < rw.size[1]; j++)
        {
            rw.matrix[i][j] = tanh_gradient_host(rw.matrix[i][j]); // ReLU activation
        }
    }
    return rw;
}

// LeakyReLU Functions

__device__ float leaky_relu_device(float x)
{
    return (x > 0) ? x : .1f * x;
}

__device__ float leaky_relu_gradient_device(float x)
{
    return (x > 0) ? 1 : .1f;
}

__global__ void leaky_relu_kernel(float *input, float *output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        output[idx] = leaky_relu_device(input[idx]);
    }
}

__global__ void leaky_relu_gradient_kernel(float *input, float *output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        output[idx] = leaky_relu_gradient_device(input[idx]);
    }
}

extern "C" float leaky_relu_host(float x)
{
    return (x > 0) ? x : .1f * x;
}

extern "C" float leaky_relu_gradient_host(float x)
{
    return (x > 0) ? 1 : .1f;
}

extern "C" tensor leaky_relu_activation_gpu(tensor rw)
{
    int total_data = rw.size[0] * rw.size[1];
    int num_threads = NUM_THREADS;
    int num_blocks = ceil((float)total_data / num_threads);

    float *dvector, *dres_vector, *vector = convert2DTo1D(rw.matrix, rw.size[0], rw.size[1], true);
    cudaMalloc(&dvector, total_data * sizeof(float));
    cudaMalloc(&dres_vector, total_data * sizeof(float));

    cudaMemcpy(dvector, vector, total_data * sizeof(float), cudaMemcpyHostToDevice);
    leaky_relu_kernel<<<num_blocks, num_threads>>>(dvector, dres_vector, total_data);
    cudaMemcpy(vector, dres_vector, total_data * sizeof(float), cudaMemcpyDeviceToHost);

    rw.matrix = convert1DTo2D(vector, rw.size[0], rw.size[1], true);

    cudaFree(dvector);
    cudaFree(dres_vector);
    return rw;
}

extern "C" tensor leaky_relu_activation(tensor rw)
{
    for (int i = 0; i < rw.size[0]; i++)
    {
        for (int j = 0; j < rw.size[1]; j++)
        {
            rw.matrix[i][j] = leaky_relu_host(rw.matrix[i][j]); // leaky_relu activation
        }
    }
    return rw;
}

extern "C" tensor leaky_relu_gradient_gpu(tensor rw)
{
    int total_data = rw.size[0] * rw.size[1];
    int num_threads = NUM_THREADS;
    int num_blocks = ceil((float)total_data / num_threads);

    float *dvector, *dres_vector, *vector = convert2DTo1D(rw.matrix, rw.size[0], rw.size[1], true);
    cudaMalloc(&dvector, total_data * sizeof(float));
    cudaMalloc(&dres_vector, total_data * sizeof(float));

    cudaMemcpy(dvector, vector, total_data * sizeof(float), cudaMemcpyHostToDevice);
    leaky_relu_gradient_kernel<<<num_blocks, num_threads>>>(dvector, dres_vector, total_data);
    cudaMemcpy(vector, dres_vector, total_data * sizeof(float), cudaMemcpyDeviceToHost);

    rw.matrix = convert1DTo2D(vector, rw.size[0], rw.size[1], true);

    cudaFree(dvector);
    cudaFree(dres_vector);
    return rw;
}

extern "C" tensor leaky_relu_gradient(tensor rw)
{
    for (int i = 0; i < rw.size[0]; i++)
    {
        for (int j = 0; j < rw.size[1]; j++)
        {
            rw.matrix[i][j] = leaky_relu_gradient_host(rw.matrix[i][j]); // ReLU activation
        }
    }
    return rw;
}

// SeLU Functions

__device__ float selu_device(float x)
{
    return (x >= 0) * 1.0507f * x + (x < 0) * 1.0507f * 1.6732f * (expf(x) - 1);
}

__device__ float selu_gradient_device(float x)
{
    return (x >= 0) * 1.0507 + (x < 0) * (x + 1.0507 * 1.6732);
}

__global__ void selu_kernel(float *input, float *output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        output[idx] = selu_device(input[idx]);
    }
}

__global__ void selu_gradient_kernel(float *input, float *output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        output[idx] = selu_gradient_device(input[idx]);
    }
}

extern "C" float selu_host(float x)
{
    return (x >= 0) * 1.0507f * x + (x < 0) * 1.0507f * 1.6732f * (expf(x) - 1);
}

extern "C" float selu_gradient_host(float x)
{
    return (x >= 0) * 1.0507 + (x < 0) * (x + 1.0507 * 1.6732);
}

extern "C" tensor selu_activation_gpu(tensor rw)
{
    int total_data = rw.size[0] * rw.size[1];
    int num_threads = NUM_THREADS;
    int num_blocks = ceil((float)total_data / num_threads);

    float *dvector, *dres_vector, *vector = convert2DTo1D(rw.matrix, rw.size[0], rw.size[1], true);
    cudaMalloc(&dvector, total_data * sizeof(float));
    cudaMalloc(&dres_vector, total_data * sizeof(float));

    cudaMemcpy(dvector, vector, total_data * sizeof(float), cudaMemcpyHostToDevice);
    selu_kernel<<<num_blocks, num_threads>>>(dvector, dres_vector, total_data);
    cudaMemcpy(vector, dres_vector, total_data * sizeof(float), cudaMemcpyDeviceToHost);

    rw.matrix = convert1DTo2D(vector, rw.size[0], rw.size[1], true);

    cudaFree(dvector);
    cudaFree(dres_vector);
    return rw;
}

extern "C" tensor selu_activation(tensor rw)
{
    for (int i = 0; i < rw.size[0]; i++)
    {
        for (int j = 0; j < rw.size[1]; j++)
        {
            rw.matrix[i][j] = selu_host(rw.matrix[i][j]); // selu activation
        }
    }
    return rw;
}

extern "C" tensor selu_gradient_gpu(tensor rw)
{
    int total_data = rw.size[0] * rw.size[1];
    int num_threads = NUM_THREADS;
    int num_blocks = ceil((float)total_data / num_threads);

    float *dvector, *dres_vector, *vector = convert2DTo1D(rw.matrix, rw.size[0], rw.size[1], true);
    cudaMalloc(&dvector, total_data * sizeof(float));
    cudaMalloc(&dres_vector, total_data * sizeof(float));

    cudaMemcpy(dvector, vector, total_data * sizeof(float), cudaMemcpyHostToDevice);
    selu_gradient_kernel<<<num_blocks, num_threads>>>(dvector, dres_vector, total_data);
    cudaMemcpy(vector, dres_vector, total_data * sizeof(float), cudaMemcpyDeviceToHost);

    rw.matrix = convert1DTo2D(vector, rw.size[0], rw.size[1], true);

    cudaFree(dvector);
    cudaFree(dres_vector);
    return rw;
}

extern "C" tensor selu_gradient(tensor rw)
{
    for (int i = 0; i < rw.size[0]; i++)
    {
        for (int j = 0; j < rw.size[1]; j++)
        {
            rw.matrix[i][j] = selu_gradient_host(rw.matrix[i][j]); // ReLU activation
        }
    }
    return rw;
}

// eLU Functions

__device__ float elu_device(float x)
{
    return (x >= 0) * x + (x < 0) * (expf(x) - 1);
}

__device__ float elu_gradient_device(float x)
{
    return (x >= 0) + (x < 0) * (x + 1);
}

__global__ void elu_kernel(float *input, float *output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        output[idx] = elu_device(input[idx]);
    }
}

__global__ void elu_gradient_kernel(float *input, float *output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        output[idx] = elu_gradient_device(input[idx]);
    }
}

extern "C" float elu_host(float x)
{
    return (x >= 0) * x + (x < 0) * (expf(x) - 1);
}

extern "C" float elu_gradient_host(float x)
{
    return (x >= 0) + (x < 0) * (x + 1);
}

extern "C" tensor elu_activation_gpu(tensor rw)
{
    int total_data = rw.size[0] * rw.size[1];
    int num_threads = NUM_THREADS;
    int num_blocks = ceil((float)total_data / num_threads);

    float *dvector, *dres_vector, *vector = convert2DTo1D(rw.matrix, rw.size[0], rw.size[1], true);
    cudaMalloc(&dvector, total_data * sizeof(float));
    cudaMalloc(&dres_vector, total_data * sizeof(float));

    cudaMemcpy(dvector, vector, total_data * sizeof(float), cudaMemcpyHostToDevice);
    elu_kernel<<<num_blocks, num_threads>>>(dvector, dres_vector, total_data);
    cudaMemcpy(vector, dres_vector, total_data * sizeof(float), cudaMemcpyDeviceToHost);

    rw.matrix = convert1DTo2D(vector, rw.size[0], rw.size[1], true);

    cudaFree(dvector);
    cudaFree(dres_vector);
    return rw;
}

extern "C" tensor elu_activation(tensor rw)
{
    for (int i = 0; i < rw.size[0]; i++)
    {
        for (int j = 0; j < rw.size[1]; j++)
        {
            rw.matrix[i][j] = elu_host(rw.matrix[i][j]); // elu activation
        }
    }
    return rw;
}

extern "C" tensor elu_gradient_gpu(tensor rw)
{
    int total_data = rw.size[0] * rw.size[1];
    int num_threads = NUM_THREADS;
    int num_blocks = ceil((float)total_data / num_threads);

    float *dvector, *dres_vector, *vector = convert2DTo1D(rw.matrix, rw.size[0], rw.size[1], true);
    cudaMalloc(&dvector, total_data * sizeof(float));
    cudaMalloc(&dres_vector, total_data * sizeof(float));

    cudaMemcpy(dvector, vector, total_data * sizeof(float), cudaMemcpyHostToDevice);
    elu_gradient_kernel<<<num_blocks, num_threads>>>(dvector, dres_vector, total_data);
    cudaMemcpy(vector, dres_vector, total_data * sizeof(float), cudaMemcpyDeviceToHost);

    rw.matrix = convert1DTo2D(vector, rw.size[0], rw.size[1], true);

    cudaFree(dvector);
    cudaFree(dres_vector);
    return rw;
}

extern "C" tensor elu_gradient(tensor rw)
{
    for (int i = 0; i < rw.size[0]; i++)
    {
        for (int j = 0; j < rw.size[1]; j++)
        {
            rw.matrix[i][j] = elu_gradient_host(rw.matrix[i][j]); // ReLU activation
        }
    }
    return rw;
}

// miscellaneous
__device__ float lhtan_activate_kernel(float x)
{
    if (x < 0)
        return .001f * x;
    if (x > 1)
        return .001f * (x - 1.f) + 1.f;
    return x;
}
__device__ float lhtan_gradient_kernel(float x)
{
    if (x > 0 && x < 1)
        return 1;
    return .001;
}

__device__ float hardtan_activate_kernel(float x)
{
    if (x < -1)
        return -1;
    if (x > 1)
        return 1;
    return x;
}
__device__ float loggy_activate_kernel(float x) { return 2.f / (1.f + expf(-x)) - 1; }

__device__ float relie_activate_kernel(float x) { return (x > 0) ? x : .01f * x; }
__device__ float ramp_activate_kernel(float x) { return x * (x > 0) + .1f * x; }

__device__ float plse_activate_kernel(float x)
{
    if (x < -4)
        return .01f * (x + 4);
    if (x > 4)
        return .01f * (x - 4) + 1;
    return .125f * x + .5f;
}
__device__ float stair_activate_kernel(float x)
{
    int n = floorf(x);
    if (n % 2 == 0)
        return floorf(x / 2);
    else
        return (x - n) + floorf(x / 2);
}

__device__ float hardtan_gradient_kernel(float x)
{
    if (x > -1 && x < 1)
        return 1;
    return 0;
}
__device__ float loggy_gradient_kernel(float x)
{
    float y = (x + 1) / 2;
    return 2 * (1 - y) * y;
}

__device__ float relie_gradient_kernel(float x) { return (x > 0) ? 1 : .01f; }
__device__ float ramp_gradient_kernel(float x) { return (x > 0) + .1f; }
__device__ float leaky_gradient_kernel(float x) { return (x > 0) ? 1 : .1f; }

__device__ float plse_gradient_kernel(float x) { return (x < 0 || x > 1) ? .01f : .125f; }
__device__ float stair_gradient_kernel(float x)
{
    if (floorf(x) == x)
        return 0;
    return 1;
}