/*
 * @Author: Shuvrajeet Das 
 * @Date: 2023-12-28 13:45:25 
 * @Last Modified by:   shuvrajeet 
 * @Last Modified time: 2023-12-28 13:45:25 
 */
#ifndef OPTIMIZERS_H_
#define OPTIMIZERS_H_

#include "initializers.h"

void sgd_optimizer(tensor *weight, tensor *gradients, tensor *updated_weights, float learning_rate);
void momentum_optimizer(tensor *weight, tensor *gradients, tensor *momentum, tensor *updated_weights, float learning_rate, float momentum_factor);
void rmsprop_optimizer(tensor *weight, tensor *gradients, tensor *squared_gradients, tensor *updated_weights, float learning_rate, float decay_rate, float epsilon);
void adam_optimizer(tensor *weight, tensor *gradients, tensor *m, tensor *v, int t, float learning_rate, float beta1, float beta2, float epsilon, tensor *updated_weights);
void adagrad_optimizer(tensor *weight, tensor *gradients, tensor *accumulated_gradients, float learning_rate, float epsilon);

#endif // OPTIMIZER_H_