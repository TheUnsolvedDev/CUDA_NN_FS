#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "matrix.h"
#include "utils.h"
#include "operations.h"
#include "operations_gpu.cuh"
#include "linear_regression.h"

void gpu_cpu_test()
{
    int m = 4000, n = 2000, k = 3000;
    tensor2d *t2d1 = create_tensor2d(m, n);
    tensor2d *t2d2 = create_tensor2d(n, k);
    tensor2d *t2d3 = create_tensor2d(m, k);
    tensor2d *t2d4 = create_tensor2d(m, k);
    
    // Initialize tensors with random data
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            t2d1->data[i][j] = (float)rand() / RAND_MAX;
        }
    }
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            t2d2->data[i][j] = (float)rand() / RAND_MAX;
        }
    }
    
    clock_t start_cpu = clock(); 
    matrix_multiplication(t2d1, t2d2, t2d3);
    clock_t end_cpu = clock();  
    double cpu_time = ((double)(end_cpu - start_cpu)) / CLOCKS_PER_SEC; 
    printf("\nCPU matrix multiplication time: %.6f seconds\n", cpu_time);
    

    clock_t start_gpu = clock(); 
    matrix_multiplication_gpu(t2d1, t2d2, t2d4);
    clock_t end_gpu = clock();   
    double gpu_time = ((double)(end_gpu - start_gpu)) / CLOCKS_PER_SEC;
    printf("GPU matrix multiplication time: %.6f seconds\n", gpu_time);
    
    // Free tensors
    free_tensor2d(t2d1);
    free_tensor2d(t2d2);
    free_tensor2d(t2d3); 
    free_tensor2d(t2d4); 
    
    float performance = cpu_time / gpu_time;
    printf("Performance ratio: %.6f\n", performance);
}


int main() 
{
    for (int i = 0; i < 10; i++)
    {
        gpu_cpu_test();
    }
    // int n_samples = 4;
    // int n_features = 2;

    // tensor2d* X = create_tensor2d(n_samples, n_features);
    // tensor1d* y = create_tensor1d(n_samples);

    // float input_data[4][2] = {
    //     {1, 2},
    //     {2, 3},
    //     {3, 4},
    //     {4, 5}
    // };
    // float output_data[4] = {5, 7, 9, 11};

    // for (int i = 0; i < n_samples; i++) 
    // {
    //     for (int j = 0; j < n_features; j++) 
    //     {
    //         X->data[i][j] = input_data[i][j];
    //     }
    //     y->data[i] = output_data[i];
    // }

    // int iterations = 1000;
    // float learning_rate = 0.01;
    // linear_regression(X, y, iterations, learning_rate);

    // free_tensor2d(X);
    // free_tensor1d(y);

    return 0;
}
