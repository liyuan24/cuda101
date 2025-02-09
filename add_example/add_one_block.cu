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
void add_parallel_one_block(int n, float *x, float *y)
{
    // we only have one block in this case

    // threadIdx.x is the index of the thread in the block
    // blockDim.x is the number of threads in the block
    // each thread in the single block will process part of the array
    // Example
    // Suppose n = 8, x = [1, 2, 3, 4, 5, 6, 7, 8], and y = [0, 1, 2, 3, 4, 5, 6, 7]. If blockDim.x = 4 (4 threads), the threads will process the arrays as follows:
    // Thread 0: i = 0, 4 → y[0] = x[0] + y[0] = 1 + 0 = 1, y[4] = x[4] + y[4] = 5 + 4 = 9
    // Thread 1: i = 1, 5 → y[1] = x[1] + y[1] = 2 + 1 = 3, y[5] = x[5] + y[5] = 6 + 5 = 11
    // Thread 2: i = 2, 6 → y[2] = x[2] + y[2] = 3 + 2 = 5, y[6] = x[6] + y[6] = 7 + 6 = 13
    // Thread 3: i = 3, 7 → y[3] = x[3] + y[3] = 4 + 3 = 7, y[7] = x[7] + y[7] = 8 + 7 = 15
    // The final y array will be [1, 3, 5, 7, 9, 11, 13, 15].
    int index = threadIdx.x;
    int stride = blockDim.x;
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
    add_parallel_one_block<<<1, 256>>>(N, x, y); // Duration: 1.49 ms
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