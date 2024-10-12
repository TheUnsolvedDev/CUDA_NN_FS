#include <cuda.h>
#include <math.h>
#include <stdlib.h>

extern "C"
{
}

extern "C" __device__ float sigmoid_kernel(float value)
{
    return 1.0 / (1 + exp(-value));
}

extern "C" __device__ float sigmoid_derivative_kernel(float value)
{
    return sigmoid_kernel(value) * sigmoid_kernel(1 - value);
}

extern "C" __device__ float tanh_kernel(float value)
{
    return tanh(value);
}
