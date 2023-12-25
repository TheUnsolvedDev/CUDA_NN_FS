#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "initializers.h"
#include "activations.cuh"
#include "matmul.cuh"
#include "utils.h"

void init()
{
    initialization_present();
    activation_present();
    matmul_present();
}

void cpu_test()
{
    int layer_structure[3] = {1000, 10000, 500};
    tensor layer1 = allocate_one_weights(layer_structure[0], layer_structure[1]);
    tensor layer2 = allocate_one_weights(layer_structure[1], layer_structure[2]);
    tensor out = allocate_zero_weights(layer_structure[0], layer_structure[2]);

    layer1 = relu_activation(layer1);
    layer2 = relu_activation(layer2);
    matrix_multiply(&layer1, &layer2, &out);

    free_weights(out);
    free_weights(layer1);
    free_weights(layer2);
}

void gpu_test()
{
    int layer_structure[3] = {1000, 10000, 500};
    tensor layer1 = allocate_one_weights(layer_structure[0], layer_structure[1]);
    tensor layer2 = allocate_one_weights(layer_structure[1], layer_structure[2]);
    tensor out = allocate_zero_weights(layer_structure[0], layer_structure[2]);

    layer1 = relu_activation_gpu(layer1);
    layer2 = relu_activation_gpu(layer2);
    matrix_multiply_gpu(&layer1, &layer2, &out);

    free_weights(out);
    free_weights(layer1);
    free_weights(layer2);
}

void linear_regression(tensor data, tensor labels)
{
}

int main(int argc, char **argv)
{
    int iterations = atoi(argv[1]);

    clock_t start_time, end_time;
    double elapsed_time;
    for (int i = 0; i < iterations; i++)
    {
        start_time = clock();
        cpu_test();
        end_time = clock();
        elapsed_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
        printf("Elapsed time: %.4f seconds on CPU \n", elapsed_time);

        start_time = clock();
        gpu_test();
        clock_t end_time = clock();
        elapsed_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
        printf("Elapsed time: %.4f seconds on GPU \n", elapsed_time);
    }

    tensor l1 = allocate_normal_weights(10, 10);
    tensor l2 = allocate_one_weights(10, 10);
    tensor l3 = allocate_zero_weights(10, 10);

    hadamard_gpu(&l1, &l2, &l3);
    print_weights(l1);
    print_weights(l2);
    print_weights(l3);

    free_weights(l1);
    free_weights(l2);
    free_weights(l3);
    return 0;
}