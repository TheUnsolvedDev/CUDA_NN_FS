/*
 * @Author: Shuvrajeet Das
 * @Date: 2023-12-28 13:44:25
 * @Last Modified by: shuvrajeet
 * @Last Modified time: 2023-12-28 19:17:49
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "utils.h"
#include "benchmark.h"
#include "initializers.h"
#include "save_weights.h"
#include "linear_regression.h"
#include "logistic_regression.h"

int main()
{
    int iterations, batch_size;
    float alpha;
    printf("Enter number of iterations: ");
    scanf("%d", &iterations);
    printf("Enter batch size: ");
    scanf("%d", &batch_size);
    printf("Enter alpha: ");
    scanf("%f", &alpha);

    int num_rows, num_columns;
    float **data_array;

    time_test(10, 1000, 500, 1000);

    const char *filename_linear_reg = "dataset/linear_data.csv";
    data_array = read_csv(filename_linear_reg, &num_rows, &num_columns);
    linear_regression(data_array, num_rows, num_columns, batch_size, iterations, alpha);

    const char *filename_logistic_reg = "dataset/logistic_data.csv";
    data_array = read_csv(filename_logistic_reg, &num_rows, &num_columns);
    logistic_regression(data_array, num_rows, num_columns, batch_size, iterations, alpha);

    tensor *loadedTensor = read_tensor_from_file("trained_weights/linear_regression.weights");
    print_tensor(*loadedTensor);
    free_tensor(*loadedTensor);
    return 0;
}
