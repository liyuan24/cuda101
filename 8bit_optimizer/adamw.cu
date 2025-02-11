#include <cuda_fp16.h>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/cub.cuh>

template <int SIGNED>
__device__ unsigned char quantize(float* quantization_code, int n, float x) {
    /**
    * @brief Use binary search to quantizes a floating-point value to an 8-bit unsigned char.
    *
    * @param quantization_code Pointer to an array of quantization candidates.
    * @param x The floating-point value to be quantized.
    * @return The quantized 8-bit unsigned char representation of the index in the quantization_code.
    */
    int pivot = (n - 1) / 2;
    int right_pivot = n - 1;
    int left_pivot = 0;

    float left = SIGNED ? -1.0f : 0.0f;
    float right = 1.0f;
    float mid;
    float val = quantization_code[pivot];
    for (int i = pivot / 2; i > 0; i >>= 1) {
        if (x > val) {
            left = val;
            left_pivot = pivot;
            pivot += i;
        } else {
            right = val;
            right_pivot = pivot;
            pivot -= i;
        }
        val = quantization_code[pivot];
    }
    if (x > val) {
        mid = (right + val) * 0.5f;
        if (x > mid) {
            return right_pivot;
        } else {
            return pivot;
        }
    } else {
        mid = (left + val) * 0.5f;
        if (x > mid) {
            return pivot;
        } else {
            return left_pivot;
        }
    }
}

template <typename T, int BLOCK_SIZE, int N_PER_TH>
__global__ void
adamw8bitBlockWise(
    T* params,
    T* grads,
    unsigned char* state1,
    unsigned char* state2,
    const float beta1,
    const float beta2,
    const float eps,
    const int step,
    const float weight_decay,
    const float lr,
    float* const quantiles1,
    float* const quantiles2,
    float* absmax1,
    float* absmax2,
    const int n
) {
    /**
     * @brief Performs the AdamW optimization algorithm in 8-bit precision.
     *
     * @param params Pointer to the parameters to be updated.
     * @param grads Pointer to the gradients of the parameters.
     * @param state1 Pointer to the first moment vector (8-bit).
     * @param state2 Pointer to the second moment vector (8-bit).
     * @param beta1 Exponential decay rate for the first moment estimates.
     * @param beta2 Exponential decay rate for the second moment estimates.
     * @param eps Small constant for numerical stability.
     * @param step Current optimization step.
     * @param weight_decay Weight decay coefficient.
     * @param lr Learning rate.
     * @param quantiles1 Pointer to the quantiles for the first moment.
     * @param quantiles2 Pointer to the quantiles for the second moment.
     * @param absmax1 Pointer to the maximum absolute value for the first moment. The length is number of blocks.
     * @param absmax2 Pointer to the maximum absolute value for the second moment. The length is number of blocks.
     * @param n The number of elements in the gradient tensor.
     */
    const float correction1 = 1.0f - __powf(beta1, step);
    const float correction2 = sqrt(1.0f - __powf(beta2, step));
    const float step_size = __fdividef(-lr*correction2, correction1);

    // define a block load from CUB
    // T: the type of the data to load
    // BLOCK_SIZE: the number of threads of the block
    // N_PER_TH: the number of elements to load per thread
    // cub::BLOCK_LOAD_WARP_TRANSPOSE: the transposition of the block load
    // load the gradients and parameters
    typedef cub::BlockLoad<T, BLOCK_SIZE/N_PER_TH, N_PER_TH, cub::BLOCK_LOAD_WARP_TRANSPOSE> LoadT;
    // load the adamw states
    typedef cub::BlockLoad<unsigned char, BLOCK_SIZE/N_PER_TH, N_PER_TH, cub::BLOCK_LOAD_WARP_TRANSPOSE> LoadChar;

    // store the updated parameters back to the global memory
    typedef cub::BlockStore<T, BLOCK_SIZE/N_PER_TH, N_PER_TH, cub::BLOCK_STORE_WARP_TRANSPOSE> StoreT;
    // store the updated adamw states back to the global memory
    typedef cub::BlockStore<unsigned char, BLOCK_SIZE/N_PER_TH, N_PER_TH, cub::BLOCK_STORE_WARP_TRANSPOSE> StoreChar;

    __shared__ typename LoadT::TempStorage loadT_temp_storage1;
    __shared__ typename LoadChar::TempStorage loadChar_temp_storage1;
    __shared__ typename StoreT::TempStorage storeT_temp_storage1;
    __shared__ typename StoreChar::TempStorage storeChar_temp_storage1;

    // define the shared memory for the temp storage of the load, store, and reduce operations
    // with union, the memory can be shared with 4 temp storage to save the precious shared memory
    __shared__ union {
        typename LoadT::TempStorage loadT_temp_storage;
        typename LoadChar::TempStorage loadChar_temp_storage;
        typename StoreT::TempStorage storeT_temp_storage;
        typename StoreChar::TempStorage storeChar_temp_storage;
    } temp_storage;

    __shared__ float shared_mem_quantiles1[257];
    __shared__ float shared_mem_quantiles2[257];

    // get the max of state1 across threads of the block
    typedef cub::BlockReduce<float, BLOCK_SIZE/N_PER_TH> BlockReduce1;
    // get the max of state2 across threads of the block
    typedef cub::BlockReduce<float, BLOCK_SIZE/N_PER_TH> BlockReduce2;
    __shared__ typename BlockReduce1::TempStorage reduce_temp_storage1;
    __shared__ typename BlockReduce2::TempStorage reduce_temp_storage2;

    // define the shared memory for the max of state1 and state2 for the block
    __shared__ float shared_mem_exchange1[1];
    __shared__ float shared_mem_exchange2[1];

    // load quantiles from the global memory to the shared memory
    shared_mem_quantiles1[threadIdx.x] = quantiles1[threadIdx.x];
    shared_mem_quantiles2[threadIdx.x] = quantiles2[threadIdx.x];
    __syncthreads(); // make sure the quantiles are loaded to the shared memory

    // load the gradients and states
    int base_idx = blockIdx.x * BLOCK_SIZE;
    int n_full = gridDim.x * BLOCK_SIZE;
    T g_vals[N_PER_TH];
    T p_vals[N_PER_TH];
    // the states values after dequantization
    float s1_dequantized[N_PER_TH];
    float s2_dequantized[N_PER_TH];
    // the quantized states values
    unsigned char s1_quantized[N_PER_TH];
    unsigned char s2_quantized[N_PER_TH];
    int valid_items = 0;
    float new_local_abs_max1 = -FLT_MAX;
    float new_local_abs_max2 = -FLT_MAX;
    float g_val = 0.0f;
    for (unsigned int i = base_idx; i < n_full; i += gridDim.x * BLOCK_SIZE) {
        valid_items = n - i >= BLOCK_SIZE ? BLOCK_SIZE : n - i;
        // load the gradients
        // &(grads[i]) is address of the gradient at the i-th element
        // g_vals is the array to store the loaded gradients in this thread
        // valid_items is the number of valid items to load for this block
        // (T)0.0f is the value to fill the rest of the array if the number of valid items is less than BLOCK_SIZE
        __syncthreads(); // sync threads and start loading
        LoadT(temp_storage.loadT_temp_storage).Load(&(grads[i]), g_vals, valid_items, (T)0.0f);
        // load the quantized adamw states, s1 is signed, so 128 is 0 in the quantization map
        __syncthreads(); // make sure the gradients are loaded to the shared memory
        LoadChar(temp_storage.loadChar_temp_storage).Load(&(state1[i]), s1_quantized, valid_items, 128);
        // load the quantized adamw states, s2 is unsigned, so 0 is 0 in the quantization map
        __syncthreads(); // make sure the quantized state1 are loaded to the shared memory
        LoadChar(temp_storage.loadChar_temp_storage).Load(&(state2[i]), s2_quantized, valid_items, 0);

        // update the dequantized states
        for (int j = 0; j < N_PER_TH; j++) {
            if (!isnan((float)g_vals[j]) && !isinf((float)g_vals[j])) {
                s1_dequantized[j] = shared_mem_quantiles1[s1_quantized[j]] * absmax1[i/BLOCK_SIZE];
                s2_dequantized[j] = shared_mem_quantiles2[s2_quantized[j]] * absmax2[i/BLOCK_SIZE];
                // use g_vals[j] to update the dequantized states
                g_val = g_vals[j];
                s1_dequantized[j] = s1_dequantized[j] * beta1 + g_val * (1.0f - beta1);
                s2_dequantized[j] = s2_dequantized[j] * beta2 + g_val * g_val * (1.0f - beta2);
            } else {
                s1_dequantized[j] = 0.0f;
                s2_dequantized[j] = 0.0f;
            }
            // update max values
            new_local_abs_max1 = fmaxf(new_local_abs_max1, fabs(s1_dequantized[j]));
            new_local_abs_max2 = fmaxf(new_local_abs_max2, fabs(s2_dequantized[j]));
        }
        // get the max across threads of the block
        new_local_abs_max1 = BlockReduce1(reduce_temp_storage1).Reduce(new_local_abs_max1, cub::Max());
        new_local_abs_max2 = BlockReduce2(reduce_temp_storage2).Reduce(new_local_abs_max2, cub::Max());
        // only the thread 0 will get the block max value
        if (threadIdx.x == 0) {
            shared_mem_exchange1[0] = new_local_abs_max1;
            shared_mem_exchange2[0] = new_local_abs_max2;
        }
        __syncthreads(); // sync threads and start storing
        // update the global absmax and local max value of other threads
        if (threadIdx.x == 0) {
            absmax1[blockIdx.x] = new_local_abs_max1;
            absmax2[blockIdx.x] = new_local_abs_max2;
        } else {
            new_local_abs_max1 = shared_mem_exchange1[0];
            new_local_abs_max2 = shared_mem_exchange2[0];
        }
        __syncthreads(); // sync threads and then start update the parameters
        // load the parameters
        LoadT(temp_storage.loadT_temp_storage).Load(&(params[i]), p_vals, valid_items, (T)0.0f);
        // update the parameters
        for (int j = 0; j < N_PER_TH; j++) {
            if (!isnan((float)p_vals[j]) && !isinf((float)p_vals[j])) {
                p_vals[j] = (T)((float)p_vals[j] + step_size * __fdividef(s1_dequantized[j], (sqrt(s2_dequantized[j]) + eps)));
            }
            if (weight_decay > 0.0f) {
                p_vals[j] = (T)((float)p_vals[j] * (1.0f - lr*weight_decay));
            }
        }
        __syncthreads(); // sync threads and then start storing
        // store the updated parameters
        StoreT(temp_storage.storeT_temp_storage).Store(&(params[i]), p_vals, valid_items);
        // quantize the adamw states
        for (int j = 0; j < N_PER_TH; j++) {
            s1_quantized[j] = quantize<1>(shared_mem_quantiles1, BLOCK_SIZE, __fdividef(s1_dequantized[j], new_local_abs_max1));
            // make sure quantized value and the original value have the same sign
            if (signbit(shared_mem_quantiles1[s1_quantized[j]]) != signbit(s1_dequantized[j])) {
                if (s1_dequantized[j] > 0.0f) {
                    s1_quantized[j] += 1;
                } else {
                    s1_quantized[j] -= 1;
                }
            }
            s2_quantized[j] = quantize<0>(shared_mem_quantiles2, BLOCK_SIZE, __fdividef(s2_dequantized[j], new_local_abs_max2));
        }
        // store the updated adamw states
        __syncthreads(); // sync threads and then start storing
        StoreChar(temp_storage.storeChar_temp_storage).Store(&(state1[i]), s1_quantized, valid_items);
        __syncthreads(); // sync threads and then start storing
        StoreChar(temp_storage.storeChar_temp_storage).Store(&(state2[i]), s2_quantized, valid_items);
    }
}

int main() {
    const int block_size = 32;
    int n = block_size + 3;
    int num_blocks =  (n+block_size-1)/block_size;
    half* parameters;
    half* grads;
    // allocate unified memory that can be accessed by both CPU and GPU
    cudaMallocManaged(&parameters, n * sizeof(half));
    cudaMallocManaged(&grads, n * sizeof(half));

    for (int i = 0; i < n; i++) {
        parameters[i] = __float2half(1.0f);
        grads[i] = __float2half(1.0f);
    }
    unsigned char* state1;
    unsigned char* state2;
    cudaMallocManaged(&state1, n * sizeof(unsigned char));
    cudaMallocManaged(&state2, n * sizeof(unsigned char));
    for (int i = 0; i < n; i++) {
        state1[i] = 1;
        state2[i] = 1;
    }
    float beta1 = 0.9f;
    float beta2 = 0.95f;
    float eps = 0.0f;
    int step = 1;
    float weight_decay = 0.0f;
    float lr = 1.0f;
    float* quantiles1;
    float* quantiles2;
    // allocate unified memory that can be accessed by both CPU and GPU
    cudaMallocManaged(&quantiles1, block_size * sizeof(float));
    cudaMallocManaged(&quantiles2, block_size * sizeof(float));
    quantiles1[0] = -1.0f;
    quantiles1[1] = -0.5f;
    quantiles1[2] = 0.5f;
    quantiles1[3] = 1.0f;
    quantiles2[0] = -1.0f;
    quantiles2[1] = -0.5f;
    quantiles2[2] = 0.5f;
    quantiles2[3] = 1.0f;
    float* absmax1;
    float* absmax2;
    cudaMallocManaged(&absmax1, num_blocks * sizeof(float));
    cudaMallocManaged(&absmax2, num_blocks * sizeof(float));
    for (int i = 0; i < num_blocks; i++) {
        absmax1[i] = 0.0f;
        absmax2[i] = 0.0f;
    }

    // launch the kernel
    adamw8bitBlockWise<half, block_size, 1><<<num_blocks, block_size>>>(
        parameters, grads, state1, state2, beta1, beta2, eps, step, weight_decay, lr, quantiles1, quantiles2, absmax1, absmax2, n);
    cudaDeviceSynchronize();


    // print the results
    for (int i = 0; i < n; i++) {
        printf("%f\n", __half2float(parameters[i]));
    }
}