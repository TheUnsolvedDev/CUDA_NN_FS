/*
 * @Author: Shuvrajeet Das
 * @Date: 2023-12-28 13:45:12
 * @Last Modified by: shuvrajeet
 * @Last Modified time: 2023-12-28 19:00:48
 */
#ifndef LOSSES_CUH_
#define LOSSES_CUH_

float mean(tensor array);
void mean_vector(tensor *array, tensor *result, int axis);
void std_dev_vector(tensor *array, tensor *result, int axis);
void squared_difference_gpu(tensor *a, tensor *b, tensor *c);
void squared_difference(tensor *a, tensor *b, tensor *c);
void logistic_gpu(tensor *a, tensor *b, tensor *c);
void logistic(tensor *a, tensor *b, tensor *c);
#endif