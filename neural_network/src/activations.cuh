#ifndef ACTIVATIONS_CUH

#define ACTIVATIONS_CUH

void activation_present();

tensor sigmoid_activation_gpu(tensor rw);
tensor sigmoid_activation(tensor rw);
tensor relu_activation_gpu(tensor rw);
tensor relu_activation(tensor rw);

#endif