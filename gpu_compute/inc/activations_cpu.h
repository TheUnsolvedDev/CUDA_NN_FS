#ifndef ACTIVATIONS_CPU_H
#define ACTIVATIONS_CPU_H

float sigmoid(float x);
float sigmoid_derivative(float x);
tensor1d sigmoid1d_activation(tensor1d t1d);
tensor1d sigmoid1d_activation_derivative(tensor1d t1d);
tensor2d sigmoid2d_activation(tensor2d t2d);
tensor2d sigmoid2d_activation_derivative(tensor2d t2d);

float relu(float x);
float relu_derivative(float x);
tensor1d relu1d_activation(tensor1d t1d);
tensor1d relu1d_activation_derivative(tensor1d t1d);
tensor2d relu2d_activation(tensor2d t2d);
tensor2d relu2d_activation_derivative(tensor2d t2d);

float leaky_relu(float x);
float leaky_relu_derivative(float x);
tensor1d leaky_relu1d_activation(tensor1d t1d);
tensor1d leaky_relu1d_activation_derivative(tensor1d t1d);
tensor2d leaky_relu2d_activation(tensor2d t2d);
tensor2d leaky_relu2d_activation_derivative(tensor2d t2d);

float tanh_(float x);
float tanh_derivative(float x);
tensor1d tanh1d_activation(tensor1d t1d);
tensor1d tanh1d_activation_derivative(tensor1d t1d);
tensor2d tanh2d_activation(tensor2d t2d);
tensor2d tanh2d_activation_derivative(tensor2d t2d);

#endif 