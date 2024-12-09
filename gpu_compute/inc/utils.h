#ifndef UTILS_H
#define UTILS_H

void print_tensor1d(tensor1d* t1d);
void print_tensor2d(tensor2d* t2d);
void convert_vector_to_matrix(tensor1d* t1d,tensor2d* t2d);
void convert_matrix_to_vector(tensor2d* t2d,tensor1d* t1d);
void fisher_yates_shuffle(int *array, int size);

#endif 