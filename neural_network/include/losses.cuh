#ifndef _LOSSES_CUH_
#define _LOSSES_CUH_

void squared_difference_gpu(tensor *a, tensor *b, tensor *c);
void squared_difference(tensor *a, tensor *b, tensor *c);
float mean(tensor array);

#endif