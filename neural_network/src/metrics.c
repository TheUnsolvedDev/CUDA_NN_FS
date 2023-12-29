/*
 * @Author: Shuvrajeet Das
 * @Date: 2023-12-28 17:08:49
 * @Last Modified by: shuvrajeet
 * @Last Modified time: 2023-12-28 17:12:33
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "initializers.h"

float r2_score(tensor y_true, tensor y_pred)
{
    if (y_true.size[0] != y_pred.size[0] || y_true.size[1] != y_pred.size[1])
    {
        printf("Error: Incompatible tensor dimensions for R2 score calculation.\n");
        return -1.0;
    }

    float sum_squared_residuals = 0.0;
    float sum_squared_total = 0.0;
    float y_true_mean = 0.0;

    for (int i = 0; i < y_true.size[0]; i++)
    {
        for (int j = 0; j < y_true.size[1]; j++)
        {
            y_true_mean += y_true.matrix[i][j];
        }
    }
    y_true_mean /= (y_true.size[0] * y_true.size[1]);

    for (int i = 0; i < y_true.size[0]; i++)
    {
        for (int j = 0; j < y_true.size[1]; j++)
        {
            float residual = y_true.matrix[i][j] - y_pred.matrix[i][j];
            sum_squared_residuals += residual * residual;
            float total_difference = y_true.matrix[i][j] - y_true_mean;
            sum_squared_total += total_difference * total_difference;
        }
    }

    float r2 = 1.0 - (sum_squared_residuals / sum_squared_total);

    return r2;
}

float binary_accuracy(tensor y_true, tensor y_pred)
{
    if (y_true.size[0] != y_pred.size[0] || y_true.size[1] != y_pred.size[1])
    {
        printf("Error: Incompatible tensor dimensions for binary accuracy calculation.\n");
        return -1.0;
    }

    int correct_predictions = 0;

    for (int i = 0; i < y_true.size[0]; i++)
    {
        for (int j = 0; j < y_true.size[1]; j++)
        {
            if ((y_true.matrix[i][j] >= 0.5 && y_pred.matrix[i][j] >= 0.5) ||
                (y_true.matrix[i][j] < 0.5 && y_pred.matrix[i][j] < 0.5))
            {
                correct_predictions++;
            }
        }
    }

    float accuracy = (float)correct_predictions / (y_true.size[0] * y_true.size[1]);

    return accuracy;
}