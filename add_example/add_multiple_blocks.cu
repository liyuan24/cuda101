#include <iostream>
#include <math.h>

// compile the file with command
// nvcc add.cu -o add_cuda
// then run the file with command
// ./add_cuda

// to profile 
// sudo ncu ./add_cuda


// CUDA kernel to add numbers in two arrays
// it will be executed in the GPU
__global__
void add_parallel_multiple_blocks(int n, float *x, float *y)
{
    // we have multiple blocks in this case
    // threadIdx.x is the index of the thread in the block
    // blockDim.x is the number of threads in the block
    // blockIdx.x is the index of the block
    // gridDim.x is the number of blocks in the grid
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index == 286) {
        printf("blockIdx.x: %d, blockDim.x: %d, gridDim.x: %d\n", blockIdx.x, blockDim.x, gridDim.x);
        printf("blockIdx.y: %d, blockDim.y: %d, gridDim.y: %d\n", blockIdx.y, blockDim.y, gridDim.y);
        printf("blockIdx.z: %d, blockDim.z: %d, gridDim.z: %d\n", blockIdx.z, blockDim.z, gridDim.z);
    }
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}

int main(void)
{
    int N = 1 << 20;
    float *x, *y;

    // allocate unified memory that can be accessed by both CPU and GPU
    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));

    // initialize x and y on CPU
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // launch add() kernel on GPU
    int blockSize = 256; // number of threads per block
    int numBlocks = N / blockSize;
    numBlocks = numBlocks + (N % blockSize != 0);
    add_parallel_multiple_blocks<<<numBlocks, blockSize>>>(N, x, y); // Duration: 16.99 us = 0.01699 ms
    // wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++) {
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    }
    std::cout << "Max error: " << maxError << std::endl;

    // free memory
    cudaFree(x);
    cudaFree(y);
    
    return 0;
}