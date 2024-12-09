#ifndef OPERATIONS_H
#define OPERATIONS_H

void matrix_multiplication(tensor2d* t2d1,tensor2d* t2d2,tensor2d* t2d3);
void hadamard_product(tensor2d* t2d1,tensor2d* t2d2,tensor2d* t2d3);
void vector_addition(tensor1d* t1d1,tensor1d* t1d2,tensor1d* t1d3);
void vector_subtraction(tensor1d* t1d1,tensor1d* t1d2,tensor1d* t1d3);
void vector_product(tensor1d* t1d1,tensor1d* t1d2,tensor1d* t1d3);
void matrix_addition(tensor2d* t2d1,tensor2d* t2d2,tensor2d* t2d3);
void matrix_subtraction(tensor2d* t2d1,tensor2d* t2d2,tensor2d* t2d3);
void matrix_transpose(tensor2d* t2d1,tensor2d* t2d2);
void scalar_addition_vector(tensor1d* t1d1,float scalar,tensor1d* t1d2);
void scalar_addition_matrix(tensor2d* t2d1,float scalar,tensor2d* t2d2);
void scalar_subtraction_vector(tensor1d* t1d1,float scalar,tensor1d* t1d2);
void scalar_subtraction_matrix(tensor2d* t2d1,float scalar,tensor2d* t2d2);
void scalar_multiplication_vector(tensor1d* t1d1,float scalar,tensor1d* t1d2);
void scalar_multiplication_matrix(tensor2d* t2d1,float scalar,tensor2d* t2d2);
void scalar_division_vector(tensor1d* t1d1,float scalar,tensor1d* t1d2);
void scalar_division_matrix(tensor2d* t2d1,float scalar,tensor2d* t2d2);
void matrix_invert(tensor2d* t2d1,tensor2d* t2d2);
void vector_invert(tensor1d* t1d1,tensor1d* t1d2);
float reduce_sum_vector(tensor1d* t1d1);
float reduce_sum_matrix(tensor2d* t2d1);
void reduce_sum_matrix_row(tensor2d* t2d1, tensor1d* t1d1);
void reduce_sum_matrix_column(tensor2d* t2d1, tensor1d* t1d1);
void dot_product_matrix_vector(tensor2d* t2d1,tensor1d* t1d2,tensor1d* t1d3);

#endif // !OPERATIONS_H