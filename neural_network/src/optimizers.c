/*
 * @Author: Shuvrajeet Das 
 * @Date: 2023-12-28 13:46:47 
 * @Last Modified by:   shuvrajeet 
 * @Last Modified time: 2023-12-28 13:46:47 
 */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "initializers.h"
#include "utils.h"

void sgd_optimizer(tensor *weight, tensor *gradients, tensor *updated_weights, float learning_rate)
{
    if (weight->size[0] != gradients->size[0] || weight->size[1] != gradients->size[1] ||
        weight->size[0] != updated_weights->size[0] || weight->size[1] != updated_weights->size[1])
    {
        printf("Error: Incompatible tensor dimensions for SGD optimization\n");
        return;
    }

    for (int i = 0; i < weight->size[0]; i++)
    {
        for (int j = 0; j < weight->size[1]; j++)
        {
            updated_weights->matrix[i][j] = weight->matrix[i][j] - learning_rate * gradients->matrix[i][j];
        }
    }
}

void momentum_optimizer(tensor *weight, tensor *gradients, tensor *momentum, tensor *updated_weights, float learning_rate, float momentum_factor)
{
    if (weight->size[0] != gradients->size[0] || weight->size[1] != gradients->size[1] ||
        weight->size[0] != momentum->size[0] || weight->size[1] != momentum->size[1] ||
        weight->size[0] != updated_weights->size[0] || weight->size[1] != updated_weights->size[1])
    {
        printf("Error: Incompatible tensor dimensions for momentum optimization\n");
        return;
    }

    for (int i = 0; i < weight->size[0]; i++)
    {
        for (int j = 0; j < weight->size[1]; j++)
        {
            momentum->matrix[i][j] = momentum_factor * momentum->matrix[i][j] +
                                     learning_rate * gradients->matrix[i][j];
            updated_weights->matrix[i][j] = weight->matrix[i][j] - momentum->matrix[i][j];
        }
    }
}

void rmsprop_optimizer(tensor *weight, tensor *gradients, tensor *squared_gradients, tensor *updated_weights, float learning_rate, float decay_rate, float epsilon)
{
    if (weight->size[0] != gradients->size[0] || weight->size[1] != gradients->size[1] ||
        weight->size[0] != squared_gradients->size[0] || weight->size[1] != squared_gradients->size[1] ||
        weight->size[0] != updated_weights->size[0] || weight->size[1] != updated_weights->size[1])
    {
        printf("Error: Incompatible tensor dimensions for RMSprop optimization\n");
        return;
    }

    for (int i = 0; i < weight->size[0]; i++)
    {
        for (int j = 0; j < weight->size[1]; j++)
        {
            squared_gradients->matrix[i][j] = decay_rate * squared_gradients->matrix[i][j] +
                                              (1 - decay_rate) * gradients->matrix[i][j] * gradients->matrix[i][j];

            updated_weights->matrix[i][j] = weight->matrix[i][j] -
                                            (learning_rate / (sqrt(squared_gradients->matrix[i][j]) + epsilon)) * gradients->matrix[i][j];
        }
    }
}

void adam_optimizer(tensor *weight, tensor *gradients, tensor *m, tensor *v, int t, float learning_rate, float beta1, float beta2, float epsilon, tensor *updated_weights)
{
    if (weight->size[0] != gradients->size[0] || weight->size[1] != gradients->size[1] ||
        weight->size[0] != m->size[0] || weight->size[1] != m->size[1] ||
        weight->size[0] != v->size[0] || weight->size[1] != v->size[1] ||
        weight->size[0] != updated_weights->size[0] || weight->size[1] != updated_weights->size[1])
    {
        printf("Error: Incompatible tensor dimensions for Adam optimization\n");
        return;
    }

    float beta1_t = pow(beta1, t);
    float beta2_t = pow(beta2, t);

    for (int i = 0; i < weight->size[0]; i++)
    {
        for (int j = 0; j < weight->size[1]; j++)
        {
            m->matrix[i][j] = beta1 * m->matrix[i][j] + (1 - beta1) * gradients->matrix[i][j];
            v->matrix[i][j] = beta2 * v->matrix[i][j] + (1 - beta2) * gradients->matrix[i][j] * gradients->matrix[i][j];

            float m_hat = m->matrix[i][j] / (1 - beta1_t);
            float v_hat = v->matrix[i][j] / (1 - beta2_t);

            updated_weights->matrix[i][j] = weight->matrix[i][j] - learning_rate * m_hat / (sqrt(v_hat) + epsilon);
        }
    }
}

void adagrad_optimizer(tensor *weight, tensor *gradients, tensor *accumulated_gradients, float learning_rate, float epsilon)
{
    if (weight->size[0] != gradients->size[0] || weight->size[1] != gradients->size[1] ||
        weight->size[0] != accumulated_gradients->size[0] || weight->size[1] != accumulated_gradients->size[1])
    {
        printf("Error: Incompatible tensor dimensions for Adagrad optimization\n");
        return;
    }
    for (int i = 0; i < weight->size[0]; i++)
    {
        for (int j = 0; j < weight->size[1]; j++)
        {
            accumulated_gradients->matrix[i][j] += gradients->matrix[i][j] * gradients->matrix[i][j];
            weight->matrix[i][j] -= learning_rate * gradients->matrix[i][j] / (sqrt(accumulated_gradients->matrix[i][j]) + epsilon);
        }
    }
}
