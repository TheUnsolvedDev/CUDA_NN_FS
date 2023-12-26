#ifndef UTILS_H

#include <stdbool.h>
#include "initializers.h"

#define UTILS_H
#define NUM_THREADS 1024
#define NUM_2D_THREADS 32
#define MAX_LINE_LENGTH 1024 * 32

float *convert2DTo1D(float **arr2D, int rows, int cols, bool free_data);
float **convert1DTo2D(float *arr1D, int rows, int cols, bool free_data);
float **array2d_slice(float **array, int start_row, int end_row, int start_col, int end_col);
tensor copy_tensor(tensor tensor_a, tensor tensor_b);
tensor convert_tensor(float **array, int rows, int cols);
float **read_csv(const char *filename, int *num_rows, int *num_columns);
void print_2d_array(float **data_array, int num_rows, int num_columns);
void print_1d_array(int *array, int size);
void free_2d_array(float **data_array, int num_rows);
void swap(int *a, int *b);
void fisher_yates_shuffle(int *array, int size);

#endif