// kernel.cu
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

extern "C"
{
#include "initializers.h"
#include "matmul.cuh"
#include "utils.h"
}

__global__ void matmul_present_kernel()
{
    printf("Matmul present\n");
}

extern "C" void matmul_present()
{
    matmul_present_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
}

__global__ void matrix_multiply_kernel(float *a, float *b, float *result, int m, int n, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < k)
    {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i)
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        result[row * k + col] = sum;
    }
}

extern "C" void matrix_multiply_gpu(tensor *a, tensor *b, tensor *c)
{
    int m = a->size[0];
    int n = a->size[1];
    int k = b->size[1];

    if (a->size[1] != b->size[0])
    {
        printf("Illegal dimension %d != %d\n", a->size[1], b->size[0]);
        exit(EXIT_FAILURE);
    }

    float *vector_a = convert2DTo1D(a->matrix, m, n, true);
    float *vector_b = convert2DTo1D(b->matrix, n, k, true);
    float *result = convert2DTo1D(c->matrix, m, k, true);
    float *dvector_a, *dvector_b, *dresult;

    cudaMalloc(&dvector_a, m * n * sizeof(float));
    cudaMalloc(&dvector_b, n * k * sizeof(float));
    cudaMalloc(&dresult, m * k * sizeof(float));

    cudaMemcpy(dvector_a, vector_a, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dvector_b, vector_b, n * k * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dim_grid((k - 1) / NUM_2D_THREADS + 1, (m - 1) / NUM_2D_THREADS + 1);
    dim3 dim_block(NUM_2D_THREADS, NUM_2D_THREADS);

    matrix_multiply_kernel<<<dim_grid, dim_block>>>(dvector_a, dvector_b, dresult, m, n, k);
    cudaMemcpy(result, dresult, m * k * sizeof(float), cudaMemcpyDeviceToHost);

    a->matrix = convert1DTo2D(vector_a, m, n, true);
    a->size[0] = m;
    a->size[1] = n;

    b->matrix = convert1DTo2D(vector_b, n, k, true);
    b->size[0] = n;
    b->size[1] = k;

    c->matrix = convert1DTo2D(result, m, k, true);
    c->size[0] = m;
    c->size[1] = k;

    cudaFree(dvector_a);
    cudaFree(dvector_b);
    cudaFree(dresult);
}

extern "C" void matrix_multiply(tensor *a, tensor *b, tensor *c)
{
    int m = a->size[0];
    int n = a->size[1];
    int k = b->size[1];

    if (a->size[1] != b->size[0])
    {
        printf("Illegal dimension %d != %d\n", a->size[1], b->size[0]);
        exit(EXIT_FAILURE);
    }

    float **matrix_a = a->matrix;
    float **matrix_b = b->matrix;
    float **matrix_result = c->matrix;

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < k; j++)
        {
            matrix_result[i][j] = 0.0;
            for (int x = 0; x < n; x++)
            {
                matrix_result[i][j] += matrix_a[i][x] * matrix_b[x][j];
            }
        }
    }

    c->size[0] = m;
    c->size[1] = k;
}

__global__ void hadamard_kernel(float *a, float *b, float *c, int m)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m)
    {
        c[idx] = a[idx] * b[idx];
    }
}

extern "C" void hadamard_gpu(tensor *a, tensor *b, tensor *c)
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

    hadamard_kernel<<<num_blocks, num_threads>>>(dvector_a, dvector_b, dresult, m * n);
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

extern "C" void hadamard(tensor *a, tensor *b, tensor *c)
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
            matrix_result[i][j] = matrix_a[i][j] * matrix_b[i][j];
        }
    }
}
