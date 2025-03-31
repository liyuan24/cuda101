// to compile: nvcc mm.cu -o mm
// to run: ./mm

// references
// https://leimao.github.io/blog/CUDA-Matrix-Multiplication/

#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <typeinfo>

#define BLOCK_SIZE 16

#define checkCuda(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// A: m x n
// B: n x k
// C: m x k
template <typename T>
void matrix_multiplication(T const *A, T const *B, T *C, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            T sum = 0;
            for (int p = 0; p < n; p++) {
                sum += A[i * n + p] * B[p * k + j];
            }
            C[i * k + j] = sum;
        }
    }
}


// A: m x n
// B: n x k
// C: m x k
// each thread will compute one element of C with index (i, j)
// each thread will load one row of A and one column of B, and compute the inner product
template <typename T>
__global__ void matrix_multiplication_kernel(T const *A, T const *B, T *C, int m, int n, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= m || j >= k) return;
    T sum = 0;
    for (int p = 0; p < n; p++) {
        sum += A[i * n + p] * B[p * k + j];
    }
    C[i * k + j] = sum;
}

// divide the rows of A into blocks and use it as blockId.x in the grid
// divide the columns of B into blocks and use it as blockId.y in the grid
// so (blockIdx.x, blockIdx.y) is responsible for computing the block (blockIdx.x, blockIdx.y) of C
// the computation for each (blockIdx.x, blockIdx.y) is done as follows
// divide the columns and rows of A and B into tiles, and load the tiles into shared memory
// threadId.x is specifying the ith row of the tile of A
// threadId.y is specifying the jth column of the tile of B
template <typename T>
__global__ void mm_kernel_optimized(T const* A, T const* B, T* C,
                                    int m, int n, int k)
{
    // shared memory can be accessed by threads in the same block
    __shared__ T A_tile[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ T B_tile[BLOCK_SIZE][BLOCK_SIZE];

    int num_tiles = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    T acc_sum = 0;

    for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        // thread collaboratively load the tile of A
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = tile_idx * blockDim.y + threadIdx.y;

        if (i < m && j < n) {
            A_tile[threadIdx.x][threadIdx.y] = A[i * n + j];
        } else {
            A_tile[threadIdx.x][threadIdx.y] = 0;
        }

        // thread collaboratively load the tile of B
        i = tile_idx * blockDim.x + threadIdx.x;
        j = blockIdx.y * blockDim.y + threadIdx.y;

        if (i < n && j < k) {
            B_tile[threadIdx.x][threadIdx.y] = B[i * k + j];
        } else {
            B_tile[threadIdx.x][threadIdx.y] = 0;
        }

        // wait for all threads to load the tiles
        __syncthreads();

        // each thread computes one element of C and not that this is reading from the shared memory which is faster than reading from the global memory
        for (int p = 0; p < BLOCK_SIZE; ++p) {
            acc_sum += A_tile[threadIdx.x][p] * B_tile[p][threadIdx.y];
        }

        // wait for all threads to compute the inner product
        __syncthreads();
    }

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // load the results to the global memory
    if (i < m && j < k) {
        C[i * k + j] = acc_sum;
    }
}

template<typename T>
void matrix_multiplication_cuda(T const *A, T const *B, T *C, int m, int n, int k, void (*f)(T const*, T const*, T*, int, int, int)) {
    dim3 threads_per_block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks_per_grid(1, 1);
    // since each thread will compute one element of C with index (i, j)
    // the number of blocks in the grid is (m / BLOCK_SIZE, k / BLOCK_SIZE)
    blocks_per_grid.x = (m + threads_per_block.x - 1) / threads_per_block.x;
    blocks_per_grid.y = (k + threads_per_block.y - 1) / threads_per_block.y;
    f<<<blocks_per_grid, threads_per_block>>>(A, B, C, m, n, k);
}

template<typename T>
bool all_close(std::vector<T> const &A, std::vector<T> const &B, T const& abs_tol) {
    if (A.size() != B.size()) return false;
    for (size_t i = 0; i < A.size(); ++i) {
        if (std::abs(A[i] - B[i]) > abs_tol) {
            std::cout << "A[" << i << "] = " << A[i] << ", B[" << i << "] = " << B[i] << std::endl;
            return false;
        }
    }
    return true;
}

template<typename T>
std::vector<T> create_random_vector(int n) {
    std::random_device rd;
    std::default_random_engine gen(rd());
    std::uniform_int_distribution<int> dis(-100, 100);
    std::vector<T> vec(n);
    for (int i = 0; i < n; ++i) {
        vec[i] = static_cast<T>(dis(gen));
    }
    return vec;
}

template<typename T>
bool test_matrix_multiplication_cuda(int m, int n, int k, void (*f)(T const*, T const*, T*, int, int, int)) {
    std::vector<T> A = create_random_vector<T>(m * n);
    std::vector<T> B = create_random_vector<T>(n * k);
    std::vector<T> C_cpu(m * k);
    std::vector<T> C_cuda(m * k);

    T const *mat_A = A.data();
    T const *mat_B = B.data();
    T *mat_C_cpu = C_cpu.data();
    T *mat_C_cuda = C_cuda.data();

    matrix_multiplication(mat_A, mat_B, mat_C_cpu, m, n, k);

    T *d_mat_A, *d_mat_B, *d_mat_C_cuda;
    // allocate memory on device
    checkCuda(cudaMalloc(&d_mat_A, m * n * sizeof(T)));
    checkCuda(cudaMalloc(&d_mat_B, n * k * sizeof(T)));
    checkCuda(cudaMalloc(&d_mat_C_cuda, m * k * sizeof(T)));

    // copy data from host to device
    checkCuda(cudaMemcpy(d_mat_A, mat_A, m * n * sizeof(T), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_mat_B, mat_B, n * k * sizeof(T), cudaMemcpyHostToDevice));

    // launch kernel
    matrix_multiplication_cuda(d_mat_A, d_mat_B, d_mat_C_cuda, m, n, k, f);

    // wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // copy data from device to host
    checkCuda(cudaMemcpy(mat_C_cuda, d_mat_C_cuda, m * k * sizeof(T), cudaMemcpyDeviceToHost));

    // free memory
    checkCuda(cudaFree(d_mat_A));
    checkCuda(cudaFree(d_mat_B));
    checkCuda(cudaFree(d_mat_C_cuda));

    return all_close<T>(C_cpu, C_cuda, 1e-5);
}

template<typename T>
void multiple_matrix_multiplication_cuda(int test_cases, void (*f)(T const*, T const*, T*, int, int, int)) {
    std::random_device rd;
    std::default_random_engine gen(rd());
    std::uniform_int_distribution<int> dis(1, 100);
    int m, n, k;
    bool passed{true};
    for (int i = 0; i < test_cases; ++i) {
        m = dis(gen);
        n = dis(gen);
        k = dis(gen);
        if (!test_matrix_multiplication_cuda<T>(m, n, k, f)) {
            passed = false;
            break;
        }
    }
    if (passed) {
        std::cout << "Test passed!" << typeid(T).name() << std::endl;
    } else {
        std::cout << "Test failed!" << typeid(T).name() << std::endl;
    }
}

template<typename T>
float measure_cuda_latency(int m, int n, int k, int num_tests, int num_warmup, void (*f)(T const*, T const*, T*, int, int, int)) {
    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start));
    checkCuda(cudaEventCreate(&stop));

    T *d_A, *d_B, *d_C;
    // allocate memory on device
    checkCuda(cudaMalloc(&d_A, m * n * sizeof(T)));
    checkCuda(cudaMalloc(&d_B, n * k * sizeof(T)));
    checkCuda(cudaMalloc(&d_C, m * k * sizeof(T)));

    // warmup
    for (int i = 0; i < num_warmup; ++i) {
        matrix_multiplication_cuda(d_A, d_B, d_C, m, n, k, f);
    }

    float total_time = 0.0f;
    checkCuda(cudaEventRecord(start, 0));
    for (int i = 0; i < num_tests; ++i) {
        matrix_multiplication_cuda(d_A, d_B, d_C, m, n, k, f);
    }
    checkCuda(cudaEventRecord(stop, 0));
    checkCuda(cudaEventSynchronize(stop));

    checkCuda(cudaEventElapsedTime(&total_time, start, stop));

    checkCuda(cudaFree(d_A));
    checkCuda(cudaFree(d_B));
    checkCuda(cudaFree(d_C));

    return total_time / num_tests;
}



int main() {
    constexpr int test_cases{10};

    multiple_matrix_multiplication_cuda<float>(test_cases, mm_kernel_optimized<float>);
    multiple_matrix_multiplication_cuda<double>(test_cases, mm_kernel_optimized<double>);
    multiple_matrix_multiplication_cuda<int>(test_cases, mm_kernel_optimized<int>);

    multiple_matrix_multiplication_cuda<float>(test_cases, matrix_multiplication_kernel<float>);
    multiple_matrix_multiplication_cuda<double>(test_cases, matrix_multiplication_kernel<double>);
    multiple_matrix_multiplication_cuda<int>(test_cases, matrix_multiplication_kernel<int>);

    constexpr int num_measurements{100};
    constexpr int num_warmup{10};

    constexpr int m{1024}, n{1024}, k{1024};
    std::cout << "Matrix size: " << m << " x " << n << " x " << k << std::endl;
    std::cout << "Number of measurements: " << num_measurements << std::endl;
    std::cout << "Number of warmup: " << num_warmup << std::endl;
    std::cout << "Latency of float for optimized kernel: " << measure_cuda_latency<float>(m, n, k, num_measurements, num_warmup, mm_kernel_optimized<float>) << " ms" << std::endl;
    std::cout << "Latency of double for optimized kernel: " << measure_cuda_latency<double>(m, n, k, num_measurements, num_warmup, mm_kernel_optimized<double>) << " ms" << std::endl;
    std::cout << "Latency of int for optimized kernel: " << measure_cuda_latency<int>(m, n, k, num_measurements, num_warmup, mm_kernel_optimized<int>) << " ms" << std::endl;
    
    std::cout << "Latency of float for naive kernel: " << measure_cuda_latency<float>(m, n, k, num_measurements, num_warmup, matrix_multiplication_kernel<float>) << " ms" << std::endl;
    std::cout << "Latency of double for naive kernel: " << measure_cuda_latency<double>(m, n, k, num_measurements, num_warmup, matrix_multiplication_kernel<double>) << " ms" << std::endl;
    std::cout << "Latency of int for naive kernel: " << measure_cuda_latency<int>(m, n, k, num_measurements, num_warmup, matrix_multiplication_kernel<int>) << " ms" << std::endl;
}
            