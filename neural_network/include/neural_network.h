/*
 * @Author: Shuvrajeet Das
 * @Date: 2024-01-07 03:08:18
 * @Last Modified by: shuvrajeet
 * @Last Modified time: 2024-01-07 03:15:32
 */
#ifndef NEURAL_NETWORK_H_
#define NEURAL_NETWORK_H_

#include "initializers.h"

void neural_network(float **dataset, int num_rows, int num_cols, int batch_size, int epochs, float alpha);

#endif // NEURAL_NETWORK_H_