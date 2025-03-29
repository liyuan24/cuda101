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

template<typename T>
void matrix_multiplication_cuda(T const *A, T const *B, T *C, int m, int n, int k) {
    dim3 threads_per_block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks_per_grid(1, 1);
    // since each thread will compute one element of C with index (i, j)
    // the number of blocks in the grid is (m / BLOCK_SIZE, k / BLOCK_SIZE)
    blocks_per_grid.x = (m + threads_per_block.x - 1) / threads_per_block.x;
    blocks_per_grid.y = (k + threads_per_block.y - 1) / threads_per_block.y;
    matrix_multiplication_kernel<T><<<blocks_per_grid, threads_per_block>>>(A, B, C, m, n, k);
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
bool test_matrix_multiplication_cuda(int m, int n, int k) {
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
    matrix_multiplication_cuda(d_mat_A, d_mat_B, d_mat_C_cuda, m, n, k);

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
void multiple_matrix_multiplication_cuda(int test_cases) {
    std::random_device rd;
    std::default_random_engine gen(rd());
    std::uniform_int_distribution<int> dis(1, 100);
    int m, n, k;
    bool passed{true};
    for (int i = 0; i < test_cases; ++i) {
        m = dis(gen);
        n = dis(gen);
        k = dis(gen);
        if (!test_matrix_multiplication_cuda<T>(m, n, k)) {
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
float measure_cuda_latency(int m, int n, int k, int num_tests, int num_warmup) {
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
        matrix_multiplication_cuda(d_A, d_B, d_C, m, n, k);
    }

    float total_time = 0.0f;
    checkCuda(cudaEventRecord(start, 0));
    for (int i = 0; i < num_tests; ++i) {
        matrix_multiplication_cuda(d_A, d_B, d_C, m, n, k);
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

    multiple_matrix_multiplication_cuda<float>(test_cases);
    multiple_matrix_multiplication_cuda<double>(test_cases);
    multiple_matrix_multiplication_cuda<int>(test_cases);

    constexpr int num_measurements{100};
    constexpr int num_warmup{10};

    constexpr int m{1024}, n{1024}, k{1024};
    std::cout << "Matrix size: " << m << " x " << n << " x " << k << std::endl;
    std::cout << "Number of measurements: " << num_measurements << std::endl;
    std::cout << "Number of warmup: " << num_warmup << std::endl;
    std::cout << "Latency of float: " << measure_cuda_latency<float>(m, n, k, num_measurements, num_warmup) << " ms" << std::endl;
    std::cout << "Latency of double: " << measure_cuda_latency<double>(m, n, k, num_measurements, num_warmup) << " ms" << std::endl;
    std::cout << "Latency of int: " << measure_cuda_latency<int>(m, n, k, num_measurements, num_warmup) << " ms" << std::endl;
}
            