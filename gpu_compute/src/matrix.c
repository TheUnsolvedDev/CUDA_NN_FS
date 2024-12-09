#include <stdio.h>
#include <stdlib.h>

#include "matrix.h"

void initilization_file_matrix()
{
    printf("matrix.c is initialized\n");
}

tensor1d* create_tensor1d(int size)
{
    tensor1d* t1d = malloc(sizeof(tensor1d));
    t1d->size = size;
    t1d->data = malloc(sizeof(float)*size);
    return t1d;
}

tensor2d* create_tensor2d(int row_size, int col_size)
{
    tensor2d* t2d = malloc(sizeof(tensor2d));
    t2d->row_size = row_size;
    t2d->col_size = col_size;
    t2d->data = malloc(sizeof(float*)*row_size);
    for(int i = 0;i<row_size;i++)
    {
        t2d->data[i] = malloc(sizeof(float)*col_size);
    }
    return t2d;
}

void free_tensor1d(tensor1d* t1d)
{
    free(t1d->data);
    free(t1d);
}

void free_tensor2d(tensor2d* t2d)
{
    for(int i = 0;i<t2d->row_size;i++)
    {
        free(t2d->data[i]);
    }
    free(t2d->data);
    free(t2d);
}