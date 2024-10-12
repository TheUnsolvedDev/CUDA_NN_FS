#include <cuda.h>
#include <stdio.h>

extern "C" __host__ void give_gpu_specs()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    printf("Number of CUDA devices: %d\n", deviceCount);

    for (int i = 0; i < deviceCount; ++i)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);

        printf("\nDevice %d: %s\n", i, deviceProp.name);
        printf("  Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("  Total global memory: %lu bytes\n", (unsigned long)deviceProp.totalGlobalMem);
        printf("  Total constant memory: %lu bytes\n", (unsigned long)deviceProp.totalConstMem);
        printf("  Multiprocessors: %d\n", deviceProp.multiProcessorCount);
        printf("  Clock rate: %d kHz\n", deviceProp.clockRate);
        printf("  Memory Clock Rate: %d kHz\n", deviceProp.memoryClockRate);
        printf("  Memory Bus Width: %d bits\n", deviceProp.memoryBusWidth);
    }
}