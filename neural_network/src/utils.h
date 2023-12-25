#ifndef UTILS_H

#include <stdbool.h>
#include "initializers.h"

#define UTILS_H
#define NUM_THREADS 1024
#define NUM_2D_THREADS 32

float *convert2DTo1D(float **arr2D, int rows, int cols, bool free_data);
float **convert1DTo2D(float *arr1D, int rows, int cols, bool free_data);
tensor copy_tensor(tensor tensor_a, tensor tensor_b);

#endif