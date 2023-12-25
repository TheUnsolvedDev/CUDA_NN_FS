#ifndef _MATMUL_CUH_

#define _MATMUL_CUH_

void matmul_present();

void matrix_multiply(tensor *a, tensor *b, tensor *c);
void matrix_multiply_gpu(tensor *a, tensor *b, tensor *c);

void hadamard(tensor *a, tensor *b, tensor *c);
void hadamard_gpu(tensor *a, tensor *b, tensor *c);

#endif