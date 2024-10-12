#ifndef INITIALIZERS_H
#define INITIALIZERS_H

typedef struct tensor_2d
{
    float **matrix;
    int size[2];
    float **grads;
} tensor_2d;

#endif
