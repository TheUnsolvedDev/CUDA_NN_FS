#ifndef OPERATIONS_CUH_
#define OPERATIONS_CUH_

void matrix_multiplication_gpu(tensor2d* t2d1, tensor2d* t2d2, tensor2d* t2d3);
void hadamard_product_gpu(tensor2d* t2d1, tensor2d* t2d2, tensor2d* t2d3);
void vector_addition_gpu(tensor1d* t1d1, tensor1d* t1d2, tensor1d* t1d3);
void vector_subtraction_gpu(tensor1d* t1d1, tensor1d* t1d2, tensor1d* t1d3);
void vector_product_gpu(tensor1d* t1d1, tensor1d* t1d2, tensor1d* t1d3);
void matrix_addition_gpu(tensor2d* t2d1, tensor2d* t2d2, tensor2d* t2d3);
void matrix_subtraction_gpu(tensor2d* t2d1, tensor2d* t2d2, tensor2d* t2d3);
void scalar_addition_vector_gpu(tensor1d* t1d1, float scalar, tensor1d* t1d3);
void scalar_subtraction_vector_gpu(tensor1d* t1d1, float scalar, tensor1d* t1d3);
void scalar_multiplication_vector_gpu(tensor1d* t1d1, float scalar, tensor1d* t1d3);
void scalar_division_vector_gpu(tensor1d* t1d1, float scalar, tensor1d* t1d3);

#endif