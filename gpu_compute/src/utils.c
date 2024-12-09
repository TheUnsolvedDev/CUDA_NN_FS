#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "matrix.h"
#include "utils.h"

void print_tensor1d(tensor1d* t1d)
{
    printf("\n");
    for(int i = 0;i<t1d->size;i++)
    {
        printf("%f\n",t1d->data[i]);
    }
}

void print_tensor2d(tensor2d* t2d)
{
    printf("\n");
    for(int i = 0;i<t2d->row_size;i++)
    {
        for(int j = 0;j<t2d->col_size;j++)
        {
            printf("%f ",t2d->data[i][j]);
        }
        printf("\n");
    }
}

void convert_vector_to_matrix(tensor1d* t1d,tensor2d* t2d)
{
    if (t1d->size != t2d->row_size*t2d->col_size)
    {
        printf("Error: Size mismatch\n");
        exit(1);
    }
    for(int i = 0;i<t1d->size;i++)
    {
        t2d->data[i/t2d->col_size][i%t2d->col_size] = t1d->data[i];
    }
}

void convert_matrix_to_vector(tensor2d* t2d,tensor1d* t1d)
{
    if (t1d->size != t2d->row_size*t2d->col_size)
    {
        printf("Error: Size mismatch\n");
        exit(1);
    }
    for(int i = 0;i<t1d->size;i++)
    {
        t1d->data[i] = t2d->data[i/t2d->col_size][i%t2d->col_size];
    }
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