// I am using GTX 3090 and the compute capability is 8.6. We need to add this architecture to the compile command otherwise will get compile error
// to compile: nvcc tensor_core.cu -o tensor_core --gpu-architecture=compute_86
// to run: ./tensor_core

#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <typeinfo>
#include <cuda_fp16.h>
#include <mma.h>
#include <algorithm>
#include <type_traits>

#define WARP_SIZE 32

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
// T1: type of A and B
// T2: type of C
// WMMA_M: the tiling size along M dimension
// WMMA_N: the tiling size along N dimension
// WMMA_K: the tiling size along K dimension
// Both A and B are stored in row-major order in the linear memory
// Each warp is handling a tile of C computation
template <typename T1, typename T2, int WMMA_M, int WMMA_N, int WMMA_K>
__global__ void wmma_mm_kernel(T1 const *A, T1 const *B, T2 *C, int m, int n, int k) {
    // see the block and grid specification to understand this
    // they will be used to find the index of the first element of the tile of A and B in the linear memory
    int const warpM = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int const warpK = blockIdx.y * blockDim.y + threadIdx.y;

    // define the fragments to hold the tile of A, B and C
    // note that we use row_major order for A and B
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, T1, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, T1, nvcuda::wmma::row_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, T2> c_frag;

    // initialize the C fragment to 0
    nvcuda::wmma::fill_fragment(c_frag, 0);

    // loop over N dimension
    for (int n1 = 0; n1 < n; n1 += WMMA_N) {
        // the index of the first element of the tile of A and B in the linear memory
        int const row_a_index = warpM * WMMA_M;
        int const col_a_index = n1;
        int const row_b_index = n1;
        int const col_b_index = warpK * WMMA_K;

        // bound check
        if (row_a_index >= m || col_a_index >= n || row_b_index >= n || col_b_index >= k) {
            continue;
        }

        // load the tile of A and B into the fragments
        // the first argument is the fragment, 
        // the second argument is the pointer to the first element of the tile in the linear memory, 
        // the third argument is the leading dimension of A or B
        // since we have defined the major in the fragment, with the first element pointer and the leading dimension, the memory addresses of the tile of A and B are determined
        nvcuda::wmma::load_matrix_sync(a_frag, A + row_a_index * n + col_a_index, n);
        nvcuda::wmma::load_matrix_sync(b_frag, B + row_b_index * k + col_b_index, k);

        // compute the inner product
        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // store the result to the global memory
    // the first argument is the pointer to the first element of the tile of C in the linear memory, 
    // the second argument is the fragment to be stored, 
    // the third argument is the leading dimension of C, since row major, it is number of columns of C
    // the fourth argument is the major of C, we are using row major
    int const row_c_index = warpM * WMMA_M;
    int const col_c_index = warpK * WMMA_K;
    nvcuda::wmma::store_matrix_sync(C + row_c_index * k + col_c_index, c_frag, k, nvcuda::wmma::mem_row_major);
}

template <typename T1, typename T2>
void matrix_multiplication_cuda_wmma(T1 const *A, T1 const *B, T2 *C, int m, int n, int k) {
    // compile time constants
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;
    dim3 gridDim, blockDim;

    int const num_x_warps = 4;
    int const num_y_warps = 4;

    // totally for each block, we have 4 * 4 = 16 warps
    // and totally 4 * 4 * 32 = 512 threads which is also the block size
    blockDim.x = num_x_warps * WARP_SIZE;
    blockDim.y = num_y_warps;

    gridDim.x = (m + (WMMA_M * num_x_warps) - 1) / (WMMA_M * num_x_warps);
    gridDim.y = (k + (WMMA_K * num_y_warps) - 1) / (WMMA_K * num_y_warps);

    wmma_mm_kernel<T1, T2, WMMA_M, WMMA_N, WMMA_K><<<gridDim, blockDim>>>(A, B, C, m, n, k);
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
    // std::random_device rd;
    int seed = 0;
    std::default_random_engine gen(seed);
    std::uniform_int_distribution<int> dis(0, 10);
    std::vector<T> vec(n);
    for (int i = 0; i < n; ++i) {
        vec[i] = static_cast<T>(dis(gen));
    }
    return vec;
}

std::vector<__half> float2half(std::vector<float> const &A) {
    std::vector<__half> B(A.size());
    for (int i = 0; i < A.size(); ++i) {
        B[i] = __float2half(A[i]);
    }
    return B;
}

bool test_matrix_multiplication_cuda_wmma_half_float(int m, int n, int k) {
    std::vector<float> A_float = create_random_vector<float>(m * n);
    std::vector<float> B_float = create_random_vector<float>(n * k);
    
    std::vector<__half> A_half = float2half(A_float);
    std::vector<__half> B_half = float2half(B_float);
    std::vector<float> C_cpu(m * k);
    std::vector<float> C_cuda(m * k);
    
    float const *mat_A_float = A_float.data();
    float const *mat_B_float = B_float.data();
    float *mat_C_cpu = C_cpu.data();
    float *mat_C_cuda = C_cuda.data();
    __half *mat_A_half = A_half.data();
    __half *mat_B_half = B_half.data();

    matrix_multiplication(mat_A_float, mat_B_float, mat_C_cpu, m, n, k);

    __half *d_mat_A, *d_mat_B;
    float *d_mat_C_cuda;
    // allocate memory on device
    checkCuda(cudaMalloc(&d_mat_A, m * n * sizeof(__half)));
    checkCuda(cudaMalloc(&d_mat_B, n * k * sizeof(__half)));
    checkCuda(cudaMalloc(&d_mat_C_cuda, m * k * sizeof(float)));

    // copy data from host to device
    checkCuda(cudaMemcpy(d_mat_A, mat_A_half, m * n * sizeof(__half), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_mat_B, mat_B_half, n * k * sizeof(__half), cudaMemcpyHostToDevice));

    // launch kernel
    matrix_multiplication_cuda_wmma(d_mat_A, d_mat_B, d_mat_C_cuda, m, n, k);

    // wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // copy data from device to host
    checkCuda(cudaMemcpy(mat_C_cuda, d_mat_C_cuda, m * k * sizeof(float), cudaMemcpyDeviceToHost));

    // free memory
    checkCuda(cudaFree(d_mat_A));
    checkCuda(cudaFree(d_mat_B));
    checkCuda(cudaFree(d_mat_C_cuda));

    return all_close<float>(C_cpu, C_cuda, 1e-3);
}

void multiple_matrix_multiplication_cuda_wmma_half_float(int test_cases) {
    int seed = 0;
    std::default_random_engine gen(seed);
    std::uniform_int_distribution<int> dis(1, 100);
    int m, n, k;
    bool passed{true};
    for (int i = 0; i < test_cases; ++i) {
        m = dis(gen);
        n = dis(gen);
        k = dis(gen);
        if (!test_matrix_multiplication_cuda_wmma_half_float(m, n, k)) {
            passed = false;
            break;
        }
    }
    if (passed) {
        std::cout << "Test passed!" << std::endl;
    } else {
        std::cout << "Test failed!" << std::endl;
    }
}

float measure_cuda_latency_wmma_half_float(int m, int n, int k, int num_tests, int num_warmup) {
    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start));
    checkCuda(cudaEventCreate(&stop));

    __half *d_A, *d_B;
    float *d_C;
    // allocate memory on device
    checkCuda(cudaMalloc(&d_A, m * n * sizeof(__half)));
    checkCuda(cudaMalloc(&d_B, n * k * sizeof(__half)));
    checkCuda(cudaMalloc(&d_C, m * k * sizeof(float)));

    // warmup
    for (int i = 0; i < num_warmup; ++i) {
        matrix_multiplication_cuda_wmma(d_A, d_B, d_C, m, n, k);
    }

    float total_time = 0.0f;
    checkCuda(cudaEventRecord(start, 0));
    for (int i = 0; i < num_tests; ++i) {
        matrix_multiplication_cuda_wmma(d_A, d_B, d_C, m, n, k);
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

    multiple_matrix_multiplication_cuda_wmma_half_float(1);

    constexpr int num_measurements{100};
    constexpr int num_warmup{10};

    constexpr int m{1024}, n{1024}, k{1024};

    std::cout << "Latency of float for wmma kernel: " << measure_cuda_latency_wmma_half_float(m, n, k, num_measurements, num_warmup) << " ms" << std::endl;
}
            