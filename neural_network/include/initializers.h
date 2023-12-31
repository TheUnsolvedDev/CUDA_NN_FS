/*
 * @Author: Shuvrajeet Das 
 * @Date: 2023-12-28 13:45:00 
 * @Last Modified by:   shuvrajeet 
 * @Last Modified time: 2023-12-28 13:45:00 
 */
#ifndef INITIALIZERS_H_
#define INITIALIZERS_H_

#define M_PI 3.14159265358979323846

typedef struct tensor
{
    float **matrix;
    int size[2];

} tensor;

tensor allocate_zero_values(int rows, int cols);
tensor allocate_one_values(int rows, int cols);
tensor allocate_uniform_values(int rows, int cols);
tensor allocate_normal_values(int rows, int cols);
tensor allocate_matrix(int rows, int cols, float value);

void free_tensor(tensor rw);
void print_tensor(tensor rw);

#endif // INITIALIZERS_H_
