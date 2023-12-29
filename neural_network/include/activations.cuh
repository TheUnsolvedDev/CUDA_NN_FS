/*
 * @Author: Shuvrajeet Das 
 * @Date: 2023-12-28 13:44:38 
 * @Last Modified by:   shuvrajeet 
 * @Last Modified time: 2023-12-28 13:44:38 
 */
#ifndef ACTIVATIONS_CUH_
#define ACTIVATIONS_CUH_

void activation_present();

tensor sigmoid_activation_gpu(tensor rw);
tensor sigmoid_activation(tensor rw);
tensor sigmoid_gradient_gpu(tensor rw);
tensor sigmoid_gradient(tensor rw);

tensor relu_activation_gpu(tensor rw);
tensor relu_activation(tensor rw);
tensor relu_gradient_gpu(tensor rw);
tensor relu_gradient(tensor rw);

tensor tanh_activation_gpu(tensor rw);
tensor tanh_activation(tensor rw);
tensor tanh_gradient_gpu(tensor rw);
tensor tanh_gradient(tensor rw);

tensor leaky_relu_activation_gpu(tensor rw);
tensor leaky_relu_activation(tensor rw);
tensor leaky_relu_gradient_gpu(tensor rw);
tensor leaky_relu_gradient(tensor rw);

tensor selu_activation_gpu(tensor rw);
tensor selu_activation(tensor rw);
tensor selu_gradient_gpu(tensor rw);
tensor selu_gradient(tensor rw);

tensor elu_activation_gpu(tensor rw);
tensor elu_activation(tensor rw);
tensor elu_gradient_gpu(tensor rw);
tensor elu_gradient(tensor rw);

#endif // ACTIVATIONS_CUH_