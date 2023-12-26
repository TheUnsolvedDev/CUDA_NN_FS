#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "initializers.h"
#include "utils.h"

void initialization_present()
{
    printf("Initialization Present\n");
}

tensor allocate_zero_values(int rows, int cols)
{
    tensor rw;
    rw.size[0] = rows;
    rw.size[1] = cols;

    rw.matrix = (float **)malloc(rows * sizeof(float *));
    if (rw.matrix == NULL)
    {
        perror("Memory allocation failed");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < rows; i++)
    {
        rw.matrix[i] = (float *)malloc(cols * sizeof(float));
        if (rw.matrix[i] == NULL)
        {
            perror("Memory allocation failed");
            exit(EXIT_FAILURE);
        }
        else
        {
            for (int j = 0; j < cols; j++)
            {
                rw.matrix[i][j] = (float)0.0;
            }
        }
    }
    return rw;
}

tensor allocate_one_values(int rows, int cols)
{
    tensor rw;
    rw.size[0] = rows;
    rw.size[1] = cols;

    rw.matrix = (float **)malloc(rows * sizeof(float *));
    if (rw.matrix == NULL)
    {
        perror("Memory allocation failed");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < rows; i++)
    {
        rw.matrix[i] = (float *)malloc(cols * sizeof(float));
        if (rw.matrix[i] == NULL)
        {
            perror("Memory allocation failed");
            exit(EXIT_FAILURE);
        }
        else
        {
            for (int j = 0; j < cols; j++)
            {
                rw.matrix[i][j] = (float)1.0;
            }
        }
    }
    return rw;
}

tensor allocate_uniform_values(int rows, int cols)
{
    tensor rw;
    rw.size[0] = rows;
    rw.size[1] = cols;

    rw.matrix = (float **)malloc(rows * sizeof(float *));
    if (rw.matrix == NULL)
    {
        perror("Memory allocation failed");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < rows; i++)
    {
        rw.matrix[i] = (float *)malloc(cols * sizeof(float));
        if (rw.matrix[i] == NULL)
        {
            perror("Memory allocation failed");
            exit(EXIT_FAILURE);
        }
        else
        {
            for (int j = 0; j < cols; j++)
            {
                rw.matrix[i][j] = (float)(rand() % 10000) / 10000.0;
            }
        }
    }
    return rw;
}

float randn()
{
    float u1 = rand() / (float)RAND_MAX;
    float u2 = rand() / (float)RAND_MAX;
    return sqrtf(-2 * logf(u1)) * cosf(2 * M_PI * u2);
}

tensor allocate_normal_values(int rows, int cols)
{
    tensor rw;
    rw.size[0] = rows;
    rw.size[1] = cols;

    rw.matrix = (float **)malloc(rows * sizeof(float *));
    if (rw.matrix == NULL)
    {
        perror("Memory allocation failed");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < rows; i++)
    {
        rw.matrix[i] = (float *)malloc(cols * sizeof(float));
        if (rw.matrix[i] == NULL)
        {
            perror("Memory allocation failed");
            exit(EXIT_FAILURE);
        }
        else
        {
            for (int j = 0; j < cols; j++)
            {
                rw.matrix[i][j] = randn();
            }
        }
    }
    return rw;
}

tensor allocate_matrix(int rows, int cols, float value)
{
    tensor rw;
    rw.size[0] = rows;
    rw.size[1] = cols;

    rw.matrix = (float **)malloc(rows * sizeof(float *));
    if (rw.matrix == NULL)
    {
        perror("Memory allocation failed");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < rows; i++)
    {
        rw.matrix[i] = (float *)malloc(cols * sizeof(float));
        if (rw.matrix[i] == NULL)
        {
            perror("Memory allocation failed");
            exit(EXIT_FAILURE);
        }
        else
        {
            for (int j = 0; j < cols; j++)
            {
                rw.matrix[i][j] = (float)value;
            }
        }
    }
    return rw;
}

void free_tensor(tensor rw)
{
    for (int i = 0; i < rw.size[0]; i++)
    {
        free(rw.matrix[i]);
    }
    free(rw.matrix);
}

void print_tensor(tensor rw)
{
    printf("\nmatrix:\n");
    for (int i = 0; i < rw.size[0]; i++)
    {
        for (int j = 0; j < rw.size[1]; j++)
        {
            printf("%.4f\t", rw.matrix[i][j]);
        }
        printf("\n");
    }
    printf("shape:(%d,%d)\n", rw.size[0], rw.size[1]);
}
