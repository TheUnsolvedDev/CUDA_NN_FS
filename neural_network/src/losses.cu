/*
 * @Author: Shuvrajeet Das
 * @Date: 2023-12-28 13:46:33
 * @Last Modified by: shuvrajeet
 * @Last Modified time: 2023-12-28 19:12:17
 */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda.h>

extern "C"
{
#include "initializers.h"
#include "utils.h"
}

extern "C" float mean(tensor array)
{
    int bar = 0;
    for (int i = 0; i < array.size[0]; i++)
    {
        for (int j = 0; j < array.size[1]; j++)
        {

            bar += array.matrix[i][j];
        }
    }
    return (float)bar / (float)(array.size[0] * array.size[1]);
}

extern "C" void mean_vector(tensor *array, tensor *result, int axis)
{
    if (array->size[0] != result->size[0] || array->size[1] != result->size[1])
    {
        printf("Illegal dimension! please check!!\n");
        exit(EXIT_FAILURE);
    }

    if (axis == 0)
    {
        for (int i = 0; i < array->size[0]; i++)
        {
            float rowSum = 0.0;
            for (int j = 0; j < array->size[1]; j++)
            {
                rowSum += array->matrix[i][j];
            }
            for (int j = 0; j < array->size[1]; j++)
            {
                result->matrix[i][j] = rowSum / array->size[1];
            }
        }
    }
    else if (axis == 1)
    {
        for (int j = 0; j < array->size[1]; j++)
        {
            float colSum = 0.0f;
            for (int i = 0; i < array->size[0]; i++)
            {
                colSum += array->matrix[i][j];
            }
            for (int i = 0; i < array->size[0]; i++)
            {
                result->matrix[i][j] = colSum / array->size[0];
            }
        }
    }
}

extern "C" void std_dev_vector(tensor *array, tensor *result, int axis)
{
    if (array->size[0] != result->size[0] || array->size[1] != result->size[1])
    {
        printf("Illegal dimension! please check!!\n");
        exit(EXIT_FAILURE);
    }

    if (axis == 0)
    {
        for (int i = 0; i < array->size[0]; i++)
        {
            float mean = 0.0;
            for (int j = 0; j < array->size[1]; j++)
            {
                mean += array->matrix[i][j];
            }
            mean /= array->size[1];

            float variance = 0.0;
            for (int j = 0; j < array->size[1]; j++)
            {
                variance += powf(array->matrix[i][j] - mean, 2.0f);
            }
            variance /= array->size[1];

            for (int j = 0; j < array->size[1]; j++)
            {
                result->matrix[i][j] = sqrt(variance);
            }
        }
    }
    else if (axis == 1)
    {
        for (int j = 0; j < array->size[1]; j++)
        {
            float mean = 0.0;
            for (int i = 0; i < array->size[0]; i++)
            {
                mean += array->matrix[i][j];
            }
            mean /= array->size[0];

            float variance = 0.0;
            for (int i = 0; i < array->size[0]; i++)
            {
                variance += powf(array->matrix[i][j] - mean, 2.0f);
            }
            variance /= array->size[0];

            for (int i = 0; i < array->size[0]; i++)
            {
                result->matrix[i][j] = sqrt(variance);
            }
        }
    }
}

__device__ float squared_difference_device(float a, float b)
{
    return powf(a - b, 2.0f);
}

__global__ void squared_difference_kernel(float *a, float *b, float *result, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        result[idx] = squared_difference_device(a[idx], b[idx]);
    }
}

extern "C" float squared_difference_host(float a, float b)
{
    return powf(a - b, 2.0f);
}

extern "C" void squared_difference_gpu(tensor *a, tensor *b, tensor *c)
{
    if (a->size[0] != b->size[0] || a->size[1] != b->size[1])
    {
        printf("Illegal dimension! please check!!\n");
        exit(EXIT_FAILURE);
    }

    int m = a->size[0], n = a->size[1];
    int total_data = a->size[0] * a->size[1];
    float *vector_a = convert2DTo1D(a->matrix, a->size[0], a->size[1], true);
    float *vector_b = convert2DTo1D(b->matrix, b->size[0], b->size[1], true);
    float *result = convert2DTo1D(c->matrix, c->size[0], c->size[1], true);
    float *dvector_a, *dvector_b, *dresult;

    cudaMalloc(&dvector_a, total_data * sizeof(float));
    cudaMalloc(&dvector_b, total_data * sizeof(float));
    cudaMalloc(&dresult, total_data * sizeof(float));

    cudaMemcpy(dvector_a, vector_a, total_data * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dvector_b, vector_b, total_data * sizeof(float), cudaMemcpyHostToDevice);

    unsigned int num_threads = NUM_THREADS;
    unsigned int num_blocks = ceil((float)total_data / num_threads);

    squared_difference_kernel<<<num_blocks, num_threads>>>(dvector_a, dvector_b, dresult, m * n);
    cudaMemcpy(result, dresult, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    a->matrix = convert1DTo2D(vector_a, m, n, true);
    a->size[0] = m;
    a->size[1] = n;

    b->matrix = convert1DTo2D(vector_b, m, n, true);
    b->size[0] = m;
    b->size[1] = n;

    c->matrix = convert1DTo2D(result, m, n, true);
    c->size[0] = m;
    c->size[1] = n;

    cudaFree(dvector_a);
    cudaFree(dvector_b);
    cudaFree(dresult);
}

extern "C" void squared_difference(tensor *a, tensor *b, tensor *c)
{
    if (a->size[0] != b->size[0] || a->size[1] != b->size[1])
    {
        printf("Illegal dimension! please check!!\n");
        exit(EXIT_FAILURE);
    }

    float **matrix_a = a->matrix;
    float **matrix_b = b->matrix;
    float **matrix_result = c->matrix;

    for (int i = 0; i < a->size[0]; i++)
    {
        for (int j = 0; j < a->size[1]; j++)
        {
            matrix_result[i][j] = squared_difference_host(matrix_a[i][j], matrix_b[i][j]);
        }
    }
}

__device__ float logistic_device(float y_true, float y_pred)
{
    return y_true * log(y_pred + 0.0000001) + (1 - y_true) * log(1 - y_pred + 0.0000001);
}

__global__ void logistic_kernel(float *a, float *b, float *result, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        result[idx] = logistic_device(a[idx], b[idx]);
    }
}

extern "C" float logistic_host(float y_true, float y_pred)
{
    return y_true * log(y_pred + 0.0000001) + (1 - y_true) * log(1 - y_pred + 0.0000001);
}

extern "C" void logistic_gpu(tensor *a, tensor *b, tensor *c)
{
    if (a->size[0] != b->size[0] || a->size[1] != b->size[1])
    {
        printf("Illegal dimension! please check!!\n");
        exit(EXIT_FAILURE);
    }

    int m = a->size[0], n = a->size[1];
    int total_data = a->size[0] * a->size[1];
    float *vector_a = convert2DTo1D(a->matrix, a->size[0], a->size[1], true);
    float *vector_b = convert2DTo1D(b->matrix, b->size[0], b->size[1], true);
    float *result = convert2DTo1D(c->matrix, c->size[0], c->size[1], true);
    float *dvector_a, *dvector_b, *dresult;

    cudaMalloc(&dvector_a, total_data * sizeof(float));
    cudaMalloc(&dvector_b, total_data * sizeof(float));
    cudaMalloc(&dresult, total_data * sizeof(float));

    cudaMemcpy(dvector_a, vector_a, total_data * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dvector_b, vector_b, total_data * sizeof(float), cudaMemcpyHostToDevice);

    unsigned int num_threads = NUM_THREADS;
    unsigned int num_blocks = ceil((float)total_data / num_threads);

    logistic_kernel<<<num_blocks, num_threads>>>(dvector_a, dvector_b, dresult, m * n);
    cudaMemcpy(result, dresult, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    a->matrix = convert1DTo2D(vector_a, m, n, true);
    a->size[0] = m;
    a->size[1] = n;

    b->matrix = convert1DTo2D(vector_b, m, n, true);
    b->size[0] = m;
    b->size[1] = n;

    c->matrix = convert1DTo2D(result, m, n, true);
    c->size[0] = m;
    c->size[1] = n;

    cudaFree(dvector_a);
    cudaFree(dvector_b);
    cudaFree(dresult);
}

extern "C" void logistic(tensor *a, tensor *b, tensor *c)
{
    if (a->size[0] != b->size[0] || a->size[1] != b->size[1])
    {
        printf("Illegal dimension! please check!!\n");
        exit(EXIT_FAILURE);
    }

    float **matrix_a = a->matrix;
    float **matrix_b = b->matrix;
    float **matrix_result = c->matrix;

    for (int i = 0; i < a->size[0]; i++)
    {
        for (int j = 0; j < a->size[1]; j++)
        {
            matrix_result[i][j] = logistic_host(matrix_a[i][j], matrix_b[i][j]);
        }
    }
}
