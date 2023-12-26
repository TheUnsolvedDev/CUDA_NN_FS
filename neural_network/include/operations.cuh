#ifndef _OPERATIONS_CUH_

#define _OPERATIONS_CUH_

void matmul_present();

void matrix_multiply(tensor *a, tensor *b, tensor *c);
void matrix_multiply_gpu(tensor *a, tensor *b, tensor *c);

void hadamard(tensor *a, tensor *b, tensor *c);
void hadamard_gpu(tensor *a, tensor *b, tensor *c);

void matrix_add(tensor *a, tensor *b, tensor *c);
void matrix_add_gpu(tensor *a, tensor *b, tensor *c);

void matrix_sub(tensor *a, tensor *b, tensor *c);
void matrix_sub_gpu(tensor *a, tensor *b, tensor *c);

tensor matrix_scalar_add(float a, tensor b);
tensor matrix_scalar_add_gpu(float a, tensor b);

tensor matrix_scalar_multiply(float a, tensor b);
tensor matrix_scalar_multiply_gpu(float a, tensor b);

#endif