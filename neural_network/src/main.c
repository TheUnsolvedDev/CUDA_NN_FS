/**
 * @ Author: Shuvrajeet Das
 * @ Create Time: 2023-12-15 23:09:00
 * @ Modified by: Your name
 * @ Modified time: 2023-12-27 11:01:04
 * @ Description: main file for calling all the necessary outputs
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "initializers.h"
#include "activations.cuh"
#include "operations.cuh"
#include "losses.cuh"
#include "utils.h"
#include "benchmark.h"

void linear_regression(float **dataset, int num_rows, int num_cols, int batch_size, int epochs, float alpha)
{
    int x_shape[2] = {num_rows, num_cols - 1};
    int y_shape[2] = {num_rows, 1};

    int true_weights[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    printf("\nTrue weights:");
    print_1d_array(true_weights, 10);

    float **data = array2d_slice(dataset, 0, x_shape[0], 0, x_shape[1]);
    float **label = array2d_slice(dataset, 0, y_shape[0], x_shape[1], x_shape[1] + 1);

    int list_size = floor(num_rows / batch_size);
    int *batch_indices = (int *)malloc(list_size * sizeof(int));
    for (int i = 0; i < list_size; i++)
        batch_indices[i] = i;

    int batch_start, batch_end;
    tensor weights = allocate_zero_values(x_shape[1], 1);
    tensor temp_weights = allocate_zero_values(x_shape[1], 1);
    tensor gradient_weights = allocate_zero_values(x_shape[1], 1);
    float bias = 0.1f;

    printf("Weights before training\n");
    print_tensor(weights);

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        fisher_yates_shuffle(batch_indices, list_size);
        float mean_batch_loss = 0;
        for (int batch = 0; batch < list_size; batch++)
        {
            batch_start = batch_indices[batch] * batch_size;
            batch_end = (batch_indices[batch] + 1) * batch_size;
            if (batch_end > num_rows)
                batch_end = num_rows;

            tensor x_batch, y_batch;
            x_batch.matrix = array2d_slice(data, batch_start, batch_end, 0, x_shape[1]);
            x_batch.size[0] = batch_size, x_batch.size[1] = x_shape[1];
            y_batch.matrix = array2d_slice(label, batch_start, batch_end, 0, y_shape[1]);
            y_batch.size[0] = batch_size, y_batch.size[1] = y_shape[1];

            // loss calulation
            tensor y_pred = allocate_zero_values(batch_size, y_shape[1]);
            tensor losses = allocate_zero_values(batch_size, y_shape[1]);

            matrix_multiply_gpu(&x_batch, &weights, &y_pred);
            y_pred = matrix_scalar_add_gpu(bias, y_pred);
            squared_difference_gpu(&y_pred, &y_batch, &losses);

            float loss = mean(losses);
            mean_batch_loss += loss;

            // calculation of gradients
            tensor loss_broadcast = allocate_matrix(x_batch.size[0], x_batch.size[1], 0);
            tensor diff_times_x = allocate_matrix(x_batch.size[0], x_batch.size[1], 0);

            matrix_sub_gpu(&y_batch, &y_pred, &losses);
            tensor_broadcast(&losses, &loss_broadcast);
            hadamard_gpu(&loss_broadcast, &x_batch, &diff_times_x);
            mean_vector(&diff_times_x, &gradient_weights, 0);
            gradient_weights = matrix_scalar_multiply_gpu(0.5, gradient_weights);
            gradient_weights = matrix_scalar_multiply_gpu(alpha, gradient_weights);

            matrix_add(&weights, &gradient_weights, &temp_weights);
            copy_tensor(&temp_weights, &weights);

            free_tensor(diff_times_x);
            free_tensor(loss_broadcast);
            free_tensor(losses);
            free_tensor(y_pred);
            free_tensor(x_batch);
            free_tensor(y_batch);
        }
        if (epoch > epochs / 2)
        {
            alpha *= 0.995;
        }

        printf("Mean Batch loss at epoch [%d/%d]: %.4f\n", epoch + 1, epochs, mean_batch_loss / list_size);
    }

    printf("Weights after training\n");
    print_tensor(weights);

    free_tensor(weights);
    free_tensor(temp_weights);
    free_tensor(gradient_weights);
    free(batch_indices);
    free_2d_array(data, num_rows);
    free_2d_array(label, num_rows);
    free_2d_array(dataset, num_rows);
}

int main(int argc, char **argv)
{
    int iterations;
    if (argc <= 1)
    {
        iterations = 100;
    }
    iterations = atoi(argv[1]);

    const char *filename = "linear_data.csv";
    int num_rows, num_columns;

    float **data_array = read_csv(filename, &num_rows, &num_columns);
    linear_regression(data_array, num_rows, num_columns, 512, iterations, 0.01);

    return 0;
}