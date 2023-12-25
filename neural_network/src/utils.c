#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include "utils.h"
#include "initializers.h"

tensor copy_tensor(tensor tensor_a, tensor tensor_b)
{
    // Allocate memory for tensor_b
    tensor_b.size[0] = tensor_a.size[0];
    tensor_b.size[1] = tensor_a.size[1];

    tensor_b.matrix = (float **)malloc(tensor_b.size[0] * sizeof(float *));
    for (int i = 0; i < tensor_b.size[0]; ++i)
    {
        tensor_b.matrix[i] = (float *)malloc(tensor_b.size[1] * sizeof(float));
    }

    // Copy the contents
    for (int i = 0; i < tensor_a.size[0]; ++i)
    {
        for (int j = 0; j < tensor_a.size[1]; ++j)
        {
            tensor_b.matrix[i][j] = tensor_a.matrix[i][j];
        }
    }
    return tensor_b;
}

float *convert2DTo1D(float **arr2D, int rows, int cols, bool free_data)
{
    float *arr1D = (float *)malloc(rows * cols * sizeof(float));
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            arr1D[i * cols + j] = arr2D[i][j];
        }
    }
    if (free_data)
    {
        for (int i = 0; i < rows; i++)
        {
            free(arr2D[i]);
        }
        free(arr2D);
    }
    return arr1D;
}

float **convert1DTo2D(float *arr1D, int rows, int cols, bool free_data)
{
    float **arr2D = (float **)malloc(rows * sizeof(float *));
    for (int i = 0; i < rows; i++)
    {
        arr2D[i] = (float *)malloc(cols * sizeof(float));
        for (int j = 0; j < cols; j++)
        {
            arr2D[i][j] = arr1D[i * cols + j];
        }
    }
    if (free_data)
        free(arr1D);
    return arr2D;
}