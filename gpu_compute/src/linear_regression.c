#include <stdio.h>
#include <stdlib.h>

#include "matrix.h"
#include "utils.h"
#include "operations.h"
#include "operations_gpu.cuh"
#include "activations_cpu.h"

#include "linear_regression.h"

float calculate_mse(tensor1d* predictions, tensor1d* targets) 
{
    tensor1d* diff = create_tensor1d(targets->size);
    tensor1d* squared_diff = create_tensor1d(targets->size);

    vector_subtraction(predictions, targets, diff);
    vector_product(diff, diff, squared_diff);

    float mse = reduce_sum_vector(squared_diff) / targets->size;

    free_tensor1d(diff);
    free_tensor1d(squared_diff);
    return mse;
}

void linear_regression(tensor2d* X, tensor1d* y, int iterations, float learning_rate) 
{
    int n_samples = X->row_size, n_features = X->col_size;
    tensor1d* weights = create_tensor1d(n_features);
    tensor1d* gradients = create_tensor1d(n_features);
    tensor1d* y_pred = create_tensor1d(n_samples);
    tensor1d* errors = create_tensor1d(n_samples);
    float bias = 0.0;

    for (int i = 0; i < n_features; i++) 
    {
        weights->data[i] = 0.0;
    }

    for (int iter = 0; iter < iterations; iter++) 
    {
        dot_product_matrix_vector(X, weights, y_pred);
        scalar_addition_vector(y_pred, bias, y_pred);
        vector_subtraction(y_pred, y, errors);
        
        for (int j = 0; j < n_features; j++) {
            gradients->data[j] = 0.0;
            for (int i = 0; i < n_samples; i++) {
                gradients->data[j] += errors->data[i] * X->data[i][j];
            }
            gradients->data[j] /= n_samples;
        }

        float grad_bias = reduce_sum_vector(errors) / n_samples;

        // Update weights and bias
        for (int j = 0; j < n_features; j++) {
            weights->data[j] -= learning_rate * gradients->data[j];
        }
        bias -= learning_rate * grad_bias;

        // Calculate and print loss
        float loss = calculate_mse(y_pred, y);
        printf("Iteration %d: Loss = %.6f\n", iter + 1, loss);
    }

    printf("Final Weights: ");
    for (int j = 0; j < n_features; j++) 
    {
        printf("%.6f ", weights->data[j]);
    }
    printf("\nFinal Bias: %.6f\n", bias);

    free_tensor1d(weights);
    free_tensor1d(gradients);
    free_tensor1d(y_pred);
    free_tensor1d(errors);
}