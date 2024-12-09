#ifndef MATRIX_H
#define MATRIX_H

#define M_PI 3.14159265358979323846

typedef struct tensor1d
{
    int size;
    float* data;
}tensor1d;

typedef struct tensor2d
{
    int row_size,col_size;
    float** data;
}tensor2d;

void initilization_file_matrix();

tensor1d* create_tensor1d(int size);
tensor2d* create_tensor2d(int row_size,int col_size);
void free_tensor2d(tensor2d* t2d);
void free_tensor1d(tensor1d* t1d);

#endif 