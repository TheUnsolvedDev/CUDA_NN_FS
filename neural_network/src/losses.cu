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
    if (axis < 0 || axis >= 2)
    {
        printf("Invalid axis. Axis must be 0 (for mean along rows) or 1 (for mean along columns).\n");
        return;
    }

    if (axis == 0)
    {
        for (int j = 0; j < array->size[1]; j++)
        {
            float colSum = 0.0;
            for (int i = 0; i < array->size[0]; i++)
            {
                colSum += array->matrix[i][j];
            }
            result->matrix[j][0] = colSum / array->size[0];
        }
    }
    else if (axis == 1)
    {
        for (int j = 0; j < array->size[1]; j++)
        {
            float rowSum = 0.0;
            for (int i = 0; i < array->size[0]; i++)
            {
                rowSum += array->matrix[i][j];
            }
            result->matrix[0][j] = rowSum / array->size[1];
        }
    }
}

__device__ float squared_diff(float a, float b)
{
    return pow(a - b, 2);
}

__global__ void squared_difference_kernel(float *a, float *b, float *result, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        result[idx] = squared_diff(a[idx], b[idx]);
    }
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
            matrix_result[i][j] = pow(matrix_a[i][j] - matrix_b[i][j], 2);
        }
    }
}