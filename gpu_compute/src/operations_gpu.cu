#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

extern "C" 
{
    #include "matrix.h"
    #include "utils.h"
    #include "operations_gpu.cuh"    
}

__global__ void matrix_multiply_kernel(float *a, float *b, float *c, int m, int n, int k)
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
        
        c[row * k + col] = sum;
    }
}

extern "C" void matrix_multiplication_gpu(tensor2d* t2d1, tensor2d* t2d2, tensor2d* t2d3)
{
    if (t2d1->col_size != t2d2->row_size)
    {
        fprintf(stderr, "Matrix multiplication error: t2d1 columns (%d) != t2d2 rows (%d)\n", t2d1->col_size, t2d2->row_size);
        exit(EXIT_FAILURE);
    }
    if (t2d3->row_size != t2d1->row_size || t2d3->col_size != t2d2->col_size)
    {
        fprintf(stderr, "Matrix multiplication output error: t2d3 shape (%dx%d) does not match expected (%dx%d)\n",
                t2d3->row_size, t2d3->col_size, t2d1->row_size, t2d2->col_size);
        exit(EXIT_FAILURE);
    }
    
    int m = t2d1->row_size;
    int n = t2d1->col_size;
    int k = t2d2->col_size;
    
    tensor1d* a = create_tensor1d(m*n);
    tensor1d* b = create_tensor1d(n*k);
    tensor1d* c = create_tensor1d(m*k);
    
    convert_matrix_to_vector(t2d1,a);
    convert_matrix_to_vector(t2d2, b);
    // convert_matrix_to_vector(t2d3,c);
    
    float *dvector_a, *dvector_b, *dvector_c;
    cudaMalloc((void**)&dvector_a, m * n * sizeof(float));
    cudaMalloc((void**)&dvector_b, n * k * sizeof(float));
    cudaMalloc((void**)&dvector_c, m * k * sizeof(float));
    
    cudaMemcpy(dvector_a, a->data, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dvector_b, b->data, n * k * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 gridSize(ceil(float(m) / 32.0f), ceil(float(k) / 32.0f), 1);
    dim3 blockSize(32, 32, 1);
    
    matrix_multiply_kernel<<<gridSize,blockSize>>>(dvector_a,dvector_b,dvector_c,m,n,k);
    cudaDeviceSynchronize();
    cudaMemcpy(c->data, dvector_c, m * k * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(dvector_a);
    cudaFree(dvector_b);
    cudaFree(dvector_c);
    
    convert_vector_to_matrix(c, t2d3);
    
    free_tensor1d(a);
    free_tensor1d(b);
    free_tensor1d(c);
}

__global__ void hadamard_product_kernel(float *a, float *b, float *c, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols)
    {
        int index = row * cols + col;
        c[index] = a[index] * b[index];
    }
}

extern "C" void hadamard_product_gpu(tensor2d* t2d1, tensor2d* t2d2, tensor2d* t2d3)
{
    if (t2d1->row_size != t2d2->row_size || t2d1->col_size != t2d2->col_size)
    {
        fprintf(stderr, "Hadamard product error: t2d1 shape (%dx%d) != t2d2 shape (%dx%d)\n",
                t2d1->row_size, t2d1->col_size, t2d2->row_size, t2d2->col_size);
        exit(EXIT_FAILURE);
    }
    if (t2d3->row_size != t2d1->row_size || t2d3->col_size != t2d1->col_size)
    {
        fprintf(stderr, "Hadamard product output error: t2d3 shape (%dx%d) does not match expected (%dx%d)\n",
                t2d3->row_size, t2d3->col_size, t2d1->row_size, t2d1->col_size);
        exit(EXIT_FAILURE);
    }

    int rows = t2d1->row_size;
    int cols = t2d1->col_size;

    tensor1d* a = create_tensor1d(rows * cols);
    tensor1d* b = create_tensor1d(rows * cols);
    tensor1d* c = create_tensor1d(rows * cols);

    convert_matrix_to_vector(t2d1, a);
    convert_matrix_to_vector(t2d2, b);

    float *dvector_a, *dvector_b, *dvector_c;
    cudaMalloc((void**)&dvector_a, rows * cols * sizeof(float));
    cudaMalloc((void**)&dvector_b, rows * cols * sizeof(float));
    cudaMalloc((void**)&dvector_c, rows * cols * sizeof(float));

    cudaMemcpy(dvector_a, a->data, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dvector_b, b->data, rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    dim3 gridSize(ceil(float(cols) / 32.0f), ceil(float(rows) / 32.0f), 1);
    dim3 blockSize(32, 32, 1);

    hadamard_product_kernel<<<gridSize, blockSize>>>(dvector_a, dvector_b, dvector_c, rows, cols);
    cudaDeviceSynchronize();
    cudaMemcpy(c->data, dvector_c, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dvector_a);
    cudaFree(dvector_b);
    cudaFree(dvector_c);

    convert_vector_to_matrix(c, t2d3);

    free_tensor1d(a);
    free_tensor1d(b);
    free_tensor1d(c);
}

__global__ void vector_addition_kernel(float *a, float *b, float *c, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
    {
        c[index] = a[index] + b[index];
    }
}

extern "C" void vector_addition_gpu(tensor1d* t1d1, tensor1d* t1d2, tensor1d* t1d3)
{
    if (t1d1->size != t1d2->size)
    {
        fprintf(stderr, "Vector addition error: t1d1 size (%d) != t1d2 size (%d)\n", t1d1->size, t1d2->size);
        exit(EXIT_FAILURE);
    }
    if (t1d3->size != t1d1->size)
    {
        fprintf(stderr, "Vector addition output error: t1d3 size (%d) does not match t1d1 size (%d)\n", t1d3->size, t1d1->size);
        exit(EXIT_FAILURE);
    }

    int size = t1d1->size;

    float *dvector_a, *dvector_b, *dvector_c;

    // Allocate memory on the GPU
    cudaMalloc((void**)&dvector_a, size * sizeof(float));
    cudaMalloc((void**)&dvector_b, size * sizeof(float));
    cudaMalloc((void**)&dvector_c, size * sizeof(float));

    // Copy input data to the GPU
    cudaMemcpy(dvector_a, t1d1->data, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dvector_b, t1d2->data, size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    vector_addition_kernel<<<gridSize, blockSize>>>(dvector_a, dvector_b, dvector_c, size);
    cudaDeviceSynchronize();

    // Copy the result back to the host
    cudaMemcpy(t1d3->data, dvector_c, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(dvector_a);
    cudaFree(dvector_b);
    cudaFree(dvector_c);
}

__global__ void vector_subtraction_kernel(float *a, float *b, float *c, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
    {
        c[index] = a[index] - b[index];
    }
}

extern "C" void vector_subtraction_gpu(tensor1d* t1d1, tensor1d* t1d2, tensor1d* t1d3)
{
    if (t1d1->size != t1d2->size)
    {
        fprintf(stderr, "Vector subtraction error: t1d1 size (%d) != t1d2 size (%d)\n", t1d1->size, t1d2->size);
        exit(EXIT_FAILURE);
    }
    if (t1d3->size != t1d1->size)
    {
        fprintf(stderr, "Vector subtraction output error: t1d3 size (%d) does not match t1d1 size (%d)\n", t1d3->size, t1d1->size);
        exit(EXIT_FAILURE);
    }

    int size = t1d1->size;

    float *dvector_a, *dvector_b, *dvector_c;

    // Allocate memory on GPU
    cudaMalloc((void**)&dvector_a, size * sizeof(float));
    cudaMalloc((void**)&dvector_b, size * sizeof(float));
    cudaMalloc((void**)&dvector_c, size * sizeof(float));

    // Copy data to GPU
    cudaMemcpy(dvector_a, t1d1->data, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dvector_b, t1d2->data, size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    vector_subtraction_kernel<<<gridSize, blockSize>>>(dvector_a, dvector_b, dvector_c, size);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(t1d3->data, dvector_c, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(dvector_a);
    cudaFree(dvector_b);
    cudaFree(dvector_c);
}

__global__ void vector_product_kernel(float *a, float *b, float *c, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
    {
        c[index] = a[index] * b[index];
    }
}

extern "C" void vector_product_gpu(tensor1d* t1d1, tensor1d* t1d2, tensor1d* t1d3)
{
    if (t1d1->size != t1d2->size)
    {
        fprintf(stderr, "Vector product error: t1d1 size (%d) != t1d2 size (%d)\n", t1d1->size, t1d2->size);
        exit(EXIT_FAILURE);
    }
    if (t1d3->size != t1d1->size)
    {
        fprintf(stderr, "Vector product output error: t1d3 size (%d) does not match t1d1 size (%d)\n", t1d3->size, t1d1->size);
        exit(EXIT_FAILURE);
    }

    int size = t1d1->size;

    float *dvector_a, *dvector_b, *dvector_c;

    // Allocate memory on GPU
    cudaMalloc((void**)&dvector_a, size * sizeof(float));
    cudaMalloc((void**)&dvector_b, size * sizeof(float));
    cudaMalloc((void**)&dvector_c, size * sizeof(float));

    // Copy data to GPU
    cudaMemcpy(dvector_a, t1d1->data, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dvector_b, t1d2->data, size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    vector_product_kernel<<<gridSize, blockSize>>>(dvector_a, dvector_b, dvector_c, size);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(t1d3->data, dvector_c, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(dvector_a);
    cudaFree(dvector_b);
    cudaFree(dvector_c);
}

__global__ void matrix_addition_kernel(float *a, float *b, float *c, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols)
    {
        int index = row * cols + col;
        c[index] = a[index] + b[index];
    }
}

extern "C" void matrix_addition_gpu(tensor2d* t2d1, tensor2d* t2d2, tensor2d* t2d3)
{
    if (t2d1->row_size != t2d2->row_size || t2d1->col_size != t2d2->col_size)
    {
        fprintf(stderr, "Error: Input matrices have different dimensions (%d x %d and %d x %d).\n",
                t2d1->row_size, t2d1->col_size, t2d2->row_size, t2d2->col_size);
        exit(1);
    }
    if (t2d3->row_size != t2d1->row_size || t2d3->col_size != t2d1->col_size)
    {
        fprintf(stderr, "Error: Output matrix dimensions (%d x %d) do not match input matrix dimensions (%d x %d).\n",
                t2d3->row_size, t2d3->col_size, t2d1->row_size, t2d1->col_size);
        exit(1);
    }

    int rows = t2d1->row_size;
    int cols = t2d1->col_size;
    int size = rows * cols;

    float *d_a, *d_b, *d_c;

    // Allocate GPU memory
    cudaMalloc((void**)&d_a, size * sizeof(float));
    cudaMalloc((void**)&d_b, size * sizeof(float));
    cudaMalloc((void**)&d_c, size * sizeof(float));

    // Copy data to GPU
    cudaMemcpy(d_a, t2d1->data[0], size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, t2d2->data[0], size * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);

    // Launch kernel
    matrix_addition_kernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, rows, cols);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(t2d3->data[0], d_c, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

__global__ void matrix_subtraction_kernel(float *a, float *b, float *c, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols)
    {
        int index = row * cols + col;
        c[index] = a[index] - b[index];
    }
}

extern "C" void matrix_subtraction_gpu(tensor2d* t2d1, tensor2d* t2d2, tensor2d* t2d3)
{
    if (t2d1->row_size != t2d2->row_size || t2d1->col_size != t2d2->col_size)
    {
        fprintf(stderr, "Error: Input matrices have different dimensions (%d x %d and %d x %d).\n",
                t2d1->row_size, t2d1->col_size, t2d2->row_size, t2d2->col_size);
        exit(1);
    }
    if (t2d3->row_size != t2d1->row_size || t2d3->col_size != t2d1->col_size)
    {
        fprintf(stderr, "Error: Output matrix dimensions (%d x %d) do not match input matrix dimensions (%d x %d).\n",
                t2d3->row_size, t2d3->col_size, t2d1->row_size, t2d1->col_size);
        exit(1);
    }

    int rows = t2d1->row_size;
    int cols = t2d1->col_size;
    int size = rows * cols;

    float *d_a, *d_b, *d_c;

    // Allocate GPU memory
    cudaMalloc((void**)&d_a, size * sizeof(float));
    cudaMalloc((void**)&d_b, size * sizeof(float));
    cudaMalloc((void**)&d_c, size * sizeof(float));

    // Copy data to GPU
    cudaMemcpy(d_a, t2d1->data[0], size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, t2d2->data[0], size * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);

    // Launch kernel
    matrix_subtraction_kernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, rows, cols);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(t2d3->data[0], d_c, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

__global__ void scalar_addition_kernel(float *input, float scalar, float *output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        output[idx] = input[idx] + scalar;
    }
}

extern "C" void scalar_addition_vector_gpu(tensor1d* t1d1, float scalar, tensor1d* t1d3)
{
    if (t1d3->size != t1d1->size)
    {
        fprintf(stderr, "Error: Output vector size (%d) does not match input vector size (%d).\n", t1d3->size, t1d1->size);
        exit(1);
    }

    int size = t1d1->size;
    float *d_input, *d_output;

    // Allocate memory on GPU
    cudaMalloc((void**)&d_input, size * sizeof(float));
    cudaMalloc((void**)&d_output, size * sizeof(float));

    // Copy data to GPU
    cudaMemcpy(d_input, t1d1->data, size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    scalar_addition_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, scalar, d_output, size);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(t1d3->data, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_input);
    cudaFree(d_output);
}


__global__ void scalar_subtraction_kernel(float *input, float scalar, float *output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        output[idx] = input[idx] - scalar;
    }
}

extern "C" void scalar_subtraction_vector_gpu(tensor1d* t1d1, float scalar, tensor1d* t1d3)
{
    if (t1d3->size != t1d1->size)
    {
        fprintf(stderr, "Error: Output vector size (%d) does not match input vector size (%d).\n", t1d3->size, t1d1->size);
        exit(1);
    }

    int size = t1d1->size;
    float *d_input, *d_output;

    // Allocate memory on GPU
    cudaMalloc((void**)&d_input, size * sizeof(float));
    cudaMalloc((void**)&d_output, size * sizeof(float));

    // Copy data to GPU
    cudaMemcpy(d_input, t1d1->data, size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    scalar_subtraction_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, scalar, d_output, size);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(t1d3->data, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_input);
    cudaFree(d_output);
}


__global__ void scalar_multiplication_kernel(float *input, float scalar, float *output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        output[idx] = input[idx] * scalar;
    }
}

extern "C" void scalar_multiplication_vector_gpu(tensor1d* t1d1, float scalar, tensor1d* t1d3)
{
    if (t1d3->size != t1d1->size)
    {
        fprintf(stderr, "Error: Output vector size (%d) does not match input vector size (%d).\n", t1d3->size, t1d1->size);
        exit(1);
    }

    int size = t1d1->size;
    float *d_input, *d_output;

    // Allocate memory on GPU
    cudaMalloc((void**)&d_input, size * sizeof(float));
    cudaMalloc((void**)&d_output, size * sizeof(float));

    // Copy data to GPU
    cudaMemcpy(d_input, t1d1->data, size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    scalar_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, scalar, d_output, size);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(t1d3->data, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_input);
    cudaFree(d_output);
}

__global__ void scalar_division_kernel(float *input, float scalar, float *output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        output[idx] = input[idx] / scalar;
    }
}

extern "C" void scalar_division_vector_gpu(tensor1d* t1d1, float scalar, tensor1d* t1d3)
{
    if (t1d3->size != t1d1->size)
    {
        fprintf(stderr, "Error: Output vector size (%d) does not match input vector size (%d).\n", t1d3->size, t1d1->size);
        exit(1);
    }

    int size = t1d1->size;
    float *d_input, *d_output;

    // Allocate memory on GPU
    cudaMalloc((void**)&d_input, size * sizeof(float));
    cudaMalloc((void**)&d_output, size * sizeof(float));

    // Copy data to GPU
    cudaMemcpy(d_input, t1d1->data, size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    scalar_division_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, scalar, d_output, size);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(t1d3->data, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_input);
    cudaFree(d_output);
}

