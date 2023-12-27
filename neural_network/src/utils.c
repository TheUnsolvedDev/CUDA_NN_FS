#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>

#include "utils.h"
#include "initializers.h"

void copy_tensor(tensor *tensor_a, tensor *tensor_b)
{
    if (tensor_a->size[0] != tensor_b->size[0] || tensor_a->size[1] != tensor_b->size[1])
    {
        printf("Illegal dimension! please check!!\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < tensor_a->size[0]; ++i)
    {
        for (int j = 0; j < tensor_a->size[1]; ++j)
        {
            tensor_b->matrix[i][j] = tensor_a->matrix[i][j];
        }
    }
}

void tensor_broadcast(tensor *input, tensor *result)
{
    if (input->size[0] == 1)
    {
        for (int i = 0; i < result->size[0]; i++)
        {
            for (int j = 0; j < result->size[1]; j++)
            {
                result->matrix[i][j] = input->matrix[0][j];
            }
        }
    }
    else if (input->size[1] == 1)
    {
        for (int i = 0; i < result->size[0]; i++)
        {
            for (int j = 0; j < result->size[1]; j++)
            {
                result->matrix[i][j] = input->matrix[i][0];
            }
        }
    }
    else
    {
        printf("Invalid input sizes for broadcasting.\n");
    }
}

tensor convert_tensor(float **array, int rows, int cols)
{
    tensor result;
    result.size[0] = rows;
    result.size[1] = cols;
    result.matrix = (float **)malloc(rows * sizeof(float *));

    for (int i = 0; i < rows; ++i)
    {
        result.matrix[i] = (float *)malloc(cols * sizeof(float));
    }

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            result.matrix[i][j] = array[i][j];
        }
    }
    return result;
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

float **read_csv(const char *filename, int *num_rows, int *num_columns)
{
    FILE *file = fopen(filename, "r");
    if (!file)
    {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    *num_rows = 0;
    *num_columns = 0;
    char line[MAX_LINE_LENGTH];

    if (fgets(line, sizeof(line), file))
    {
        char *token = strtok(line, ",");
        while (token != NULL)
        {
            (*num_columns)++;
            token = strtok(NULL, ",");
        }
        (*num_rows)++;
    }

    while (fgets(line, sizeof(line), file) != NULL)
    {
        (*num_rows)++;
    }

    fseek(file, 0, SEEK_SET);
    float **data_array = (float **)malloc((*num_rows - 1) * sizeof(float *));

    for (int i = 0; i < *num_rows - 1; i++)
    {
        data_array[i] = (float *)malloc(*num_columns * sizeof(float));
    }

    for (int i = 0; i < *num_rows; i++)
    {
        fgets(line, sizeof(line), file);
        char *token = strtok(line, ",");
        if (i == 0)
            continue;
        for (int j = 0; j < *num_columns; j++)
        {
            data_array[i - 1][j] = atof(token);
            token = strtok(NULL, ",");
        }
    }
    *num_rows -= 1;
    fclose(file);
    return data_array;
}

float **array2d_slice(float **array, int start_row, int end_row, int start_col, int end_col)
{
    int num_rows = end_row - start_row;
    float **result = (float **)malloc(num_rows * sizeof(float *));

    for (int i = 0; i < num_rows; ++i)
    {
        result[i] = (float *)malloc((end_col - start_col) * sizeof(float));
    }

    for (int i = start_row; i < end_row; ++i)
    {
        for (int j = start_col; j < end_col; ++j)
        {
            result[i - start_row][j - start_col] = array[i][j];
        }
    }

    return result;
}

void print_2d_array(float **data_array, int num_rows, int num_columns)
{
    for (int i = 0; i < num_rows; i++)
    {
        for (int j = 0; j < num_columns; j++)
        {
            printf("%.4f ", data_array[i][j]);
        }
        printf("\n");
    }
}

void print_1d_array(int *array, int size)
{
    for (int i = 0; i < size; i++)
    {
        printf("%d ", array[i]);
    }
    printf("\n");
}

void free_2d_array(float **data_array, int num_rows)
{
    for (int i = 0; i < num_rows; i++)
    {
        free(data_array[i]);
    }
    free(data_array);
}

void swap(int *a, int *b)
{
    int temp = *a;
    *a = *b;
    *b = temp;
}

void fisher_yates_shuffle(int *array, int size)
{
    srand((unsigned int)time(NULL));

    for (int i = size - 1; i > 0; --i)
    {
        int j = rand() % (i + 1);
        swap(&array[i], &array[j]);
    }
}
