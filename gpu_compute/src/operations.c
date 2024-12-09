#include <stdio.h>
#include <stdlib.h>

#include "matrix.h"
#include "operations.h"


void matrix_multiplication(tensor2d* t2d1, tensor2d* t2d2, tensor2d* t2d3)
{
    if (t2d1->col_size != t2d2->row_size)
    {
        fprintf(stderr, "Matrix multiplication error: t2d1 columns (%d) != t2d2 rows (%d)\n", t2d1->col_size, t2d2->row_size);
        exit(EXIT_FAILURE);
    }
    if (t2d3->row_size != t2d1->row_size || t2d3->col_size != t2d2->col_size)
    {
        fprintf(stderr, "Matrix multiplication output error: t2d3 shape (%dx%d) does not match expected (%dx%d)\n",
                t2d3->row_size, t2d3->col_size, t2d1->row_size, t2d2->col_size);
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < t2d1->row_size; i++)
    {
        for (int j = 0; j < t2d2->col_size; j++)
        {
            t2d3->data[i][j] = 0;
            for (int k = 0; k < t2d1->col_size; k++)
            {
                t2d3->data[i][j] += t2d1->data[i][k] * t2d2->data[k][j];
            }
        }
    }
}


void hadamard_product(tensor2d* t2d1, tensor2d* t2d2, tensor2d* t2d3)
{
    if (t2d1->row_size != t2d2->row_size || t2d1->col_size != t2d2->col_size)
    {
        fprintf(stderr, "Hadamard product error: t2d1 shape (%dx%d) != t2d2 shape (%dx%d)\n",
                t2d1->row_size, t2d1->col_size, t2d2->row_size, t2d2->col_size);
        exit(EXIT_FAILURE);
    }
    if (t2d3->row_size != t2d1->row_size || t2d3->col_size != t2d1->col_size)
    {
        fprintf(stderr, "Hadamard product output error: t2d3 shape (%dx%d) does not match expected (%dx%d)\n",
                t2d3->row_size, t2d3->col_size, t2d1->row_size, t2d1->col_size);
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < t2d1->row_size; i++)
    {
        for (int j = 0; j < t2d1->col_size; j++)
        {
            t2d3->data[i][j] = t2d1->data[i][j] * t2d2->data[i][j];
        }
    }
}

void vector_addition(tensor1d* t1d1, tensor1d* t1d2, tensor1d* t1d3)
{
    if (t1d1->size != t1d2->size)
    {
        fprintf(stderr, "Vector addition error: t1d1 size (%d) != t1d2 size (%d)\n", t1d1->size, t1d2->size);
        exit(EXIT_FAILURE);
    }
    if (t1d3->size != t1d1->size)
    {
        fprintf(stderr, "Vector addition output error: t1d3 size (%d) does not match t1d1 size (%d)\n", t1d3->size, t1d1->size);
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < t1d1->size; i++)
    {
        t1d3->data[i] = t1d1->data[i] + t1d2->data[i];
    }
}

void vector_subtraction(tensor1d* t1d1, tensor1d* t1d2, tensor1d* t1d3)
{
    if (t1d1->size != t1d2->size)
    {
        fprintf(stderr, "Vector subtraction error: t1d1 size (%d) != t1d2 size (%d)\n", t1d1->size, t1d2->size);
        exit(EXIT_FAILURE);
    }
    if (t1d3->size != t1d1->size)
    {
        fprintf(stderr, "Vector subtraction output error: t1d3 size (%d) does not match t1d1 size (%d)\n", t1d3->size, t1d1->size);
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < t1d1->size; i++)
    {
        t1d3->data[i] = t1d1->data[i] - t1d2->data[i];
    }
}

void vector_product(tensor1d* t1d1, tensor1d* t1d2, tensor1d* t1d3)
{
    if (t1d1->size != t1d2->size)
    {
        fprintf(stderr, "Error: Input vectors have different sizes (%d and %d).\n", t1d1->size, t1d2->size);
        exit(1);
    }
    if (t1d3->size != t1d1->size)
    {
        fprintf(stderr, "Error: Output vector size (%d) does not match input vector size (%d).\n", t1d3->size, t1d1->size);
        exit(1);
    }
    for (int i = 0; i < t1d1->size; i++)
    {
        t1d3->data[i] = t1d1->data[i] * t1d2->data[i];
    }
}

void matrix_addition(tensor2d* t2d1, tensor2d* t2d2, tensor2d* t2d3)
{
    if (t2d1->row_size != t2d2->row_size || t2d1->col_size != t2d2->col_size)
    {
        fprintf(stderr, "Error: Input matrices have different dimensions (%d x %d and %d x %d).\n",
                t2d1->row_size, t2d1->col_size, t2d2->row_size, t2d2->col_size);
        exit(1);
    }
    if (t2d3->row_size != t2d1->row_size || t2d3->col_size != t2d1->col_size)
    {
        fprintf(stderr, "Error: Output matrix dimensions (%d x %d) do not match input matrix dimensions (%d x %d).\n",
                t2d3->row_size, t2d3->col_size, t2d1->row_size, t2d1->col_size);
        exit(1);
    }
    for (int i = 0; i < t2d1->row_size; i++)
    {
        for (int j = 0; j < t2d1->col_size; j++)
        {
            t2d3->data[i][j] = t2d1->data[i][j] + t2d2->data[i][j];
        }
    }
}

void matrix_subtraction(tensor2d* t2d1, tensor2d* t2d2, tensor2d* t2d3)
{
    if (t2d1->row_size != t2d2->row_size || t2d1->col_size != t2d2->col_size)
    {
        fprintf(stderr, "Error: Input matrices have different dimensions (%d x %d and %d x %d).\n",
                t2d1->row_size, t2d1->col_size, t2d2->row_size, t2d2->col_size);
        exit(1);
    }
    if (t2d3->row_size != t2d1->row_size || t2d3->col_size != t2d1->col_size)
    {
        fprintf(stderr, "Error: Output matrix dimensions (%d x %d) do not match input matrix dimensions (%d x %d).\n",
                t2d3->row_size, t2d3->col_size, t2d1->row_size, t2d1->col_size);
        exit(1);
    }
    for (int i = 0; i < t2d1->row_size; i++)
    {
        for (int j = 0; j < t2d1->col_size; j++)
        {
            t2d3->data[i][j] = t2d1->data[i][j] - t2d2->data[i][j];
        }
    }
}

void matrix_transpose(tensor2d* t2d1, tensor2d* t2d2)
{
    if (t2d2->row_size != t2d1->col_size || t2d2->col_size != t2d1->row_size)
    {
        fprintf(stderr, "Error: Output matrix dimensions (%d x %d) do not match transposed input matrix dimensions (%d x %d).\n",
                t2d2->row_size, t2d2->col_size, t2d1->col_size, t2d1->row_size);
        exit(1);
    }
    for (int i = 0; i < t2d1->row_size; i++)
    {
        for (int j = 0; j < t2d1->col_size; j++)
        {
            t2d2->data[j][i] = t2d1->data[i][j];
        }
    }
}

void scalar_addition_vector(tensor1d* t1d1, float scalar, tensor1d* t1d3)
{
    if (t1d3->size != t1d1->size)
    {
        fprintf(stderr, "Error: Output vector size (%d) does not match input vector size (%d).\n", t1d3->size, t1d1->size);
        exit(1);
    }
    for (int i = 0; i < t1d1->size; i++)
    {
        t1d3->data[i] = t1d1->data[i] + scalar;
    }
}

void scalar_addition_matrix(tensor2d* t2d1, float scalar, tensor2d* t2d3)
{
    if (t2d3->row_size != t2d1->row_size || t2d3->col_size != t2d1->col_size)
    {
        fprintf(stderr, "Error: Output matrix dimensions (%d x %d) do not match input matrix dimensions (%d x %d).\n",
                t2d3->row_size, t2d3->col_size, t2d1->row_size, t2d1->col_size);
        exit(1);
    }
    for (int i = 0; i < t2d1->row_size; i++)
    {
        for (int j = 0; j < t2d1->col_size; j++)
        {
            t2d3->data[i][j] = t2d1->data[i][j] + scalar;
        }
    }
}

void scalar_subtraction_vector(tensor1d* t1d1, float scalar, tensor1d* t1d3)
{
    if (t1d3->size != t1d1->size)
    {
        fprintf(stderr, "Error: Output vector size (%d) does not match input vector size (%d).\n", t1d3->size, t1d1->size);
        exit(1);
    }
    for (int i = 0; i < t1d1->size; i++)
    {
        t1d3->data[i] = t1d1->data[i] - scalar;
    }
}

void scalar_subtraction_matrix(tensor2d* t2d1, float scalar, tensor2d* t2d3)
{
    if (t2d3->row_size != t2d1->row_size || t2d3->col_size != t2d1->col_size)
    {
        fprintf(stderr, "Error: Output matrix dimensions (%d x %d) do not match input matrix dimensions (%d x %d).\n",
                t2d3->row_size, t2d3->col_size, t2d1->row_size, t2d1->col_size);
        exit(1);
    }
    for (int i = 0; i < t2d1->row_size; i++)
    {
        for (int j = 0; j < t2d1->col_size; j++)
        {
            t2d3->data[i][j] = t2d1->data[i][j] - scalar;
        }
    }
}

void scalar_multiplication_vector(tensor1d* t1d1, float scalar, tensor1d* t1d3)
{
    if (t1d3->size != t1d1->size)
    {
        fprintf(stderr, "Error: Output vector size (%d) does not match input vector size (%d).\n", t1d3->size, t1d1->size);
        exit(1);
    }
    for (int i = 0; i < t1d1->size; i++)
    {
        t1d3->data[i] = t1d1->data[i] * scalar;
    }
}

void scalar_multiplication_matrix(tensor2d* t2d1, float scalar, tensor2d* t2d3)
{
    if (t2d3->row_size != t2d1->row_size || t2d3->col_size != t2d1->col_size)
    {
        fprintf(stderr, "Error: Output matrix dimensions (%d x %d) do not match input matrix dimensions (%d x %d).\n",
                t2d3->row_size, t2d3->col_size, t2d1->row_size, t2d1->col_size);
        exit(1);
    }
    for (int i = 0; i < t2d1->row_size; i++)
    {
        for (int j = 0; j < t2d1->col_size; j++)
        {
            t2d3->data[i][j] = t2d1->data[i][j] * scalar;
        }
    }
}


void scalar_division_vector(tensor1d* t1d1, float scalar, tensor1d* t1d3)
{
    if (t1d3->size != t1d1->size)
    {
        fprintf(stderr, "Error: Output vector size (%d) does not match input vector size (%d).\n", t1d3->size, t1d1->size);
        exit(1);
    }
    if (scalar == 0)
    {
        fprintf(stderr, "Error: Division by zero is undefined.\n");
        exit(1);
    }
    for (int i = 0; i < t1d1->size; i++)
    {
        t1d3->data[i] = t1d1->data[i] / scalar;
    }
}

void scalar_division_matrix(tensor2d* t2d1, float scalar, tensor2d* t2d3)
{
    if (t2d3->row_size != t2d1->row_size || t2d3->col_size != t2d1->col_size)
    {
        fprintf(stderr, "Error: Output matrix dimensions (%d x %d) do not match input matrix dimensions (%d x %d).\n",
                t2d3->row_size, t2d3->col_size, t2d1->row_size, t2d1->col_size);
        exit(1);
    }
    if (scalar == 0)
    {
        fprintf(stderr, "Error: Division by zero is undefined.\n");
        exit(1);
    }
    for (int i = 0; i < t2d1->row_size; i++)
    {
        for (int j = 0; j < t2d1->col_size; j++)
        {
            t2d3->data[i][j] = t2d1->data[i][j] / scalar;
        }
    }
}

void matrix_invert(tensor2d* t2d1, tensor2d* t2d3)
{
    if (t2d3->row_size != t2d1->row_size || t2d3->col_size != t2d1->col_size)
    {
        fprintf(stderr, "Error: Output matrix dimensions (%d x %d) do not match input matrix dimensions (%d x %d).\n",
                t2d3->row_size, t2d3->col_size, t2d1->row_size, t2d1->col_size);
        exit(1);
    }
    if (t2d1->row_size != t2d1->col_size)
    {
        fprintf(stderr, "Error: Input matrix is not square (dimensions: %d x %d).\n", t2d1->row_size, t2d1->col_size);
        exit(1);
    }
    for (int i = 0; i < t2d1->row_size; i++)
    {
        for (int j = 0; j < t2d1->col_size; j++)
        {
            if (t2d1->data[j][i] == 0)
            {
                fprintf(stderr, "Error: Cannot invert matrix element at (%d, %d), division by zero.\n", j, i);
                exit(1);
            }
            t2d3->data[i][j] = 1 / t2d1->data[j][i];
        }
    }
}

void vector_invert(tensor1d* t1d1, tensor1d* t1d3)
{
    if (t1d3->size != t1d1->size)
    {
        fprintf(stderr, "Error: Output vector size (%d) does not match input vector size (%d).\n", t1d3->size, t1d1->size);
        exit(1);
    }
    for (int i = 0; i < t1d1->size; i++)
    {
        if (t1d1->data[i] == 0)
        {
            fprintf(stderr, "Error: Cannot invert vector element at index %d, division by zero.\n", i);
            exit(1);
        }
        t1d3->data[i] = 1 / t1d1->data[i];
    }
}

float reduce_sum_vector(tensor1d* t1d1)
{
    float sum = 0;
    for (int i = 0; i < t1d1->size; i++)
    {
        sum += t1d1->data[i];
    }
    return sum;
}

float reduce_sum_matrix(tensor2d* t2d1)
{
    float sum = 0;
    for (int i = 0; i < t2d1->row_size; i++)
    {
        for (int j = 0; j < t2d1->col_size; j++)
        {
            sum += t2d1->data[i][j];
        }
    }
    return sum;
}

void reduce_sum_matrix_row(tensor2d* t2d1, tensor1d* t1d1)
{
    if (t1d1->size != t2d1->row_size)
    {
        fprintf(stderr, "Error: Output vector size (%d) does not match the number of rows in input matrix (%d).\n",
                t1d1->size, t2d1->row_size);
        exit(1);
    }
    for (int i = 0; i < t2d1->row_size; i++)
    {
        float sum = 0;
        for (int j = 0; j < t2d1->col_size; j++)
        {
            sum += t2d1->data[i][j];
        }
        t1d1->data[i] = sum;
    }
}

void reduce_sum_matrix_column(tensor2d* t2d1, tensor1d* t1d1)
{
    if (t1d1->size != t2d1->col_size)
    {
        fprintf(stderr, "Error: Output vector size (%d) does not match the number of columns in input matrix (%d).\n",
                t1d1->size, t2d1->col_size);
        exit(1);
    }
    for (int i = 0; i < t2d1->col_size; i++)
    {
        float sum = 0;
        for (int j = 0; j < t2d1->row_size; j++)
        {
            sum += t2d1->data[j][i];
        }
        t1d1->data[i] = sum;
    }
}

void dot_product_matrix_vector(tensor2d* t2d1, tensor1d* t1d2, tensor1d* t1d3)
{
    if (t1d2->size != t2d1->col_size)
    {
        fprintf(stderr, "Error: Input vector size (%d) does not match the number of columns in matrix (%d).\n",
                t1d2->size, t2d1->col_size);
        exit(1);
    }
    if (t1d3->size != t2d1->row_size)
    {
        fprintf(stderr, "Error: Output vector size (%d) does not match the number of rows in matrix (%d).\n",
                t1d3->size, t2d1->row_size);
        exit(1);
    }
    for (int i = 0; i < t2d1->row_size; i++)
    {
        float sum = 0;
        for (int j = 0; j < t2d1->col_size; j++)
        {
            sum += t2d1->data[i][j] * t1d2->data[j];
        }
        t1d3->data[i] = sum;
    }
}
