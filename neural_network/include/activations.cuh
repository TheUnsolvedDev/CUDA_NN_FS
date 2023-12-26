#ifndef _ACTIVATIONS_CUH_

#define _ACTIVATIONS_CUH_

void activation_present();

tensor sigmoid_activation_gpu(tensor rw);
tensor sigmoid_activation(tensor rw);
tensor relu_activation_gpu(tensor rw);
tensor relu_activation(tensor rw);

#endif