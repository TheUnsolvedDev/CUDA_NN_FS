/*
 * @Author: Shuvrajeet Das 
 * @Date: 2023-12-28 13:45:58 
 * @Last Modified by:   shuvrajeet 
 * @Last Modified time: 2023-12-28 13:45:58 
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "initializers.h"
#include "operations.cuh"

void cpu_test(int m, int n, int k)
{
    tensor a = allocate_normal_values(m, n);
    tensor b = allocate_normal_values(n, k);
    tensor out = allocate_zero_values(m, k);
    matrix_multiply(&a, &b, &out);
    free_tensor(a);
    free_tensor(b);
    free_tensor(out);
}

void gpu_test(int m, int n, int k)
{
    tensor a = allocate_normal_values(m, n);
    tensor b = allocate_normal_values(n, k);
    tensor out = allocate_zero_values(m, k);
    matrix_multiply_gpu(&a, &b, &out);
    free_tensor(a);
    free_tensor(b);
    free_tensor(out);
}

void time_test(int iterations, int m, int n, int k)
{
    clock_t start_time, end_time;
    double elapsed_time_cpu;
    double elapsed_time_gpu;
    double elapsed_avg_time_cpu;
    double elapsed_avg_time_gpu;

    for (int i = 0; i < iterations; i++)
    {
        start_time = clock();
        cpu_test(m, n, k);
        end_time = clock();
        elapsed_time_cpu = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
        elapsed_avg_time_cpu += elapsed_time_cpu;
        printf("Elapsed time: %.4f seconds on CPU ", elapsed_time_cpu);

        start_time = clock();
        gpu_test(m, n, k);
        clock_t end_time = clock();
        elapsed_time_gpu = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
        elapsed_avg_time_gpu += elapsed_time_gpu;
        printf("and %.4f seconds on GPU \n", elapsed_time_gpu);
    }

    printf("\n**** Average Result: on %dx%d and %dx%d ****\n", m, n, n, k);
    printf("CPU:\t %.4f \n", elapsed_avg_time_cpu / iterations);
    printf("GPU:\t %.4f \n", elapsed_avg_time_gpu / iterations);
    printf("%.2fx times performance gain\n", elapsed_avg_time_cpu / elapsed_avg_time_gpu);
}
