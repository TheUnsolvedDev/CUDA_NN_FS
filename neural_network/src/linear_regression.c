/*
 * @Author: Shuvrajeet Das
 * @Date: 2023-12-28 13:46:28
 * @Last Modified by: shuvrajeet
 * @Last Modified time: 2023-12-28 19:26:34
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "initializers.h"
#include "utils.h"
#include "optimizers.h"
#include "save_weights.h"
#include "metrics.h"

#include "operations.cuh"
#include "activations.cuh"
#include "operations.cuh"
#include "losses.cuh"

void linear_regression(float **dataset, int num_rows, int num_cols, int batch_size, int epochs, float alpha)
{
    int x_shape[2] = {num_rows, num_cols - 1};
    int y_shape[2] = {num_rows, 1};

    float **data = array2d_slice(dataset, 0, x_shape[0], 0, x_shape[1]);
    float **label = array2d_slice(dataset, 0, y_shape[0], x_shape[1], x_shape[1] + 1);

    int list_size = floor(num_rows / batch_size);
    int *batch_indices = (int *)malloc(list_size * sizeof(int));
    for (int i = 0; i < list_size; i++)
        batch_indices[i] = i;

    int batch_start, batch_end;
    tensor weights = allocate_zero_values(x_shape[1], 1);
    tensor gradient_weights = allocate_zero_values(x_shape[1], 1);

    tensor mean_x = allocate_matrix(batch_size, x_shape[1], 1);
    tensor std_x = allocate_matrix(batch_size, x_shape[1], 1);
    tensor x_minus_mean = allocate_zero_values(batch_size, x_shape[1]);
    tensor x_minus_mean_std = allocate_zero_values(batch_size, x_shape[1]);
    tensor y_pred = allocate_zero_values(batch_size, y_shape[1]);
    tensor losses = allocate_zero_values(batch_size, y_shape[1]);
    tensor loss_broadcast = allocate_matrix(batch_size, x_shape[1], 0);
    tensor y_times_x = allocate_zero_values(batch_size, x_shape[1]);
    tensor temp_grad = allocate_matrix(batch_size, x_shape[1], 0);

    float bias = 0.0f;

    printf("Weights and bias before training\n");
    printf("bias:%f \n", bias);
    print_tensor(weights);

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        float r2_score_value = 0.0f;
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

            // standardize
            mean_vector(&x_batch, &mean_x, 1);
            std_dev_vector(&x_batch, &std_x, 1);

            matrix_sub_gpu(&x_batch, &mean_x, &x_minus_mean);
            matrix_div_gpu(&x_batch, &std_x, &x_minus_mean_std);

            // loss calulation
            matrix_multiply_gpu(&x_minus_mean_std, &weights, &y_pred);
            y_pred = matrix_scalar_add_gpu(bias, y_pred);
            squared_difference_gpu(&y_pred, &y_batch, &losses);

            float loss = mean(losses);
            mean_batch_loss += loss;
            r2_score_value += r2_score(y_batch, y_pred);

            // calculation of gradients
            matrix_sub_gpu(&y_batch, &y_pred, &losses);
            tensor_broadcast(&losses, &loss_broadcast);
            hadamard_gpu(&loss_broadcast, &x_minus_mean_std, &y_times_x);
            mean_vector(&y_times_x, &temp_grad, 1);

            for (int i = 0; i < x_batch.size[1]; i++)
            {
                gradient_weights.matrix[i][0] = temp_grad.matrix[0][i];
            }
            sgd_optimizer(&weights, &gradient_weights, &weights, -0.5 * alpha);
            bias -= -0.5 * alpha * mean(losses) / batch_size;

            free_tensor(x_batch);
            free_tensor(y_batch);
        }
        alpha *= 0.1 * (epoch % 100 == 0) + (epoch % 100 != 0);

        if ((epoch + 1) % 5 == 0)
        {

            printf("Mean Batch loss at epoch [%d/%d]: %.4f\t", epoch + 1, epochs, mean_batch_loss / list_size);
            printf("The R2 score is :%.6f\n", r2_score_value / list_size);
        }
    }

    printf("Weights and bias after training\n");
    printf("bias:%f \n", bias);
    print_tensor(weights);
    dump_tensor_to_file(&weights, "trained_weights/linear_regression.weights");

    free_tensor(temp_grad);
    free_tensor(y_times_x);
    free_tensor(loss_broadcast);
    free_tensor(losses);
    free_tensor(y_pred);
    free_tensor(x_minus_mean);
    free_tensor(x_minus_mean_std);
    free_tensor(mean_x);
    free_tensor(std_x);

    free_tensor(weights);
    free_tensor(gradient_weights);
    free(batch_indices);
    free_2d_array(data, num_rows);
    free_2d_array(label, num_rows);
    free_2d_array(dataset, num_rows);
}
