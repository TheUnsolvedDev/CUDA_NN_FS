/*
 * @Author: Shuvrajeet Das
 * @Date: 2024-01-07 03:08:31
 * @Last Modified by: shuvrajeet
 * @Last Modified time: 2024-01-07 03:22:19
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "initializers.h"
#include "neural_network.h"
#include "operations.cuh"
#include "activations.cuh"
#include "losses.cuh"
#include "optimizers.h"

void neural_network(float **dataset, int num_rows, int num_cols, int batch_size, int epochs, float alpha)
{
    int num_hidden_layers = 3;
    int num_hidden_units = 32;
    tensor layers_weights[num_hidden_layers + 1];
    tensor layer_biases[num_hidden_layers + 1];
    layers_weights[0] = allocate_matrix(num_rows, num_hidden_units, 0.1);
    layers_weights[0] = allocate_matrix(num_rows, 1, 0.1);
    
    

    for (int i = 0; i < num_hidden_layers; i++)
    {
    }

    for (int i = 0; i <= num_hidden_layers; i++)
    {
        free_tensor(layers_weights[i]);
        free_tensor(layer_biases[i]);
    }
}