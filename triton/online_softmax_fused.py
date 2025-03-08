import torch

import triton
import triton.language as tl
from triton.runtime import driver

DEVICE = 'cuda'

properties = driver.active.utils.get_device_properties(0)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()
kernels = {}


# no assumption that each block can read the whole row
# 1. we need to consider the race condition when updating the row-wise max and sum, so probably
#    could launch programs for each row and do the sum and max block by block for each row
# 2. or use atomic operations?
@triton.jit
def softmax_online_add_max_kernel(
    input_ptr,
    max_output_ptr,
    sum_output_ptr,
    input_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    pid_row = tl.program_id(0)
    row_start = pid_row
    row_stride = tl.num_programs(0)
    for i in tl.range(row_start, n_rows, row_stride, num_stages=num_stages):
        # load the max and sum for each row once
        max_start_ptr = max_output_ptr + i
        sum_start_ptr = sum_output_ptr + i
        current_max = tl.load(max_start_ptr)
        current_sum = tl.load(sum_start_ptr)
        for j in tl.range(0, n_cols, BLOCK_SIZE, num_stages=num_stages):
            input_start_ptr = input_ptr + i * input_row_stride + j
            offsets = tl.arange(0, BLOCK_SIZE)
            input_ptrs = input_start_ptr + offsets
            mask = j + offsets < n_cols
            x = tl.load(input_ptrs, mask=mask, other=float('-inf'))
            x_max = tl.max(x, axis=0)
            x_minus_max = x - x_max
            exponential_x_minus_max = tl.exp(x_minus_max)
            exponential_sum = tl.sum(exponential_x_minus_max, axis=0)
            # online softmax: https://arxiv.org/abs/1805.02867
            new_max = tl.maximum(current_max, x_max)
            current_sum = current_sum * tl.exp(current_max - new_max) + exponential_sum * tl.exp(x_max - new_max)
            current_max = new_max
        tl.store(max_start_ptr, current_max)
        tl.store(sum_start_ptr, current_sum)

def softmax_online_add_max(x):
    n_rows, n_cols = x.shape

    # The block size of each loop iteration is the smallest power of two greater than the number of columns in `x`
    BLOCK_SIZE = triton.next_power_of_2(n_cols//4)

    # Another trick we can use is to ask the compiler to use more threads per row by
    # increasing the number of warps (`num_warps`) over which each row is distributed.
    # You will see in the next tutorial how to auto-tune this value in a more natural
    # way so you don't have to come up with manual heuristics yourself.
    
    # at least the number of warps is 4
    num_warps = 4

    # Number of software pipelining stages.
    num_stages = 4 if SIZE_SMEM > 200000 else 2

    # Allocate output
    max_output = torch.full((n_rows, ), float('-inf'), device=x.device)
    sum_output = torch.zeros((n_rows, ), device=x.device)

    # pre-compile kernel to get register usage and compute thread occupancy.
    kernel = softmax_online_add_max_kernel.warmup(x, max_output, sum_output, x.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE,
                                   num_stages=num_stages, num_warps=num_warps, grid=(1, ))
    # May handle things like memory buffers, streams, or execution queues.
    # Ensures the kernel can be launched without further compilation overhead.
    kernel._init_handles()
    n_regs = kernel.n_regs # number of registers allocated for each thread in the kernel
    print(f"n_regs: {n_regs}")
    size_smem = kernel.metadata.shared # the number of bytes of shared memory allocated for each thread block
    print(f"size_smem: {size_smem}")
    occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
    occupancy = min(occupancy, SIZE_SMEM // size_smem) # the number of thread blocks for each SM
    num_programs = NUM_SM * occupancy # total number of thread blocks

    num_programs = min(num_programs, n_rows)
    print(f"num_programs: {num_programs}")
    # Create a number of persistent programs.
    # num_programs: the number of thread blocks on dimension 0
    # 1: the number of thread blocks on dimension 1
    # 1: the number of thread blocks on dimension 2
    softmax_online_add_max_kernel[(num_programs, 1, 1)](x, max_output, sum_output, x.stride(0), n_rows, n_cols, BLOCK_SIZE, num_stages)
    return max_output, sum_output

@triton.jit
def softmax_online_kernel(
    input_ptr,
    max_input_ptr,
    sum_input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)
    row_start = pid_row
    row_step = tl.num_programs(0)
    col_start = pid_col * BLOCK_SIZE
    col_step = tl.num_programs(1) * BLOCK_SIZE
    for i in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        max_ptr = max_input_ptr + i
        max_value = tl.load(max_ptr)
        sum_ptr = sum_input_ptr + i
        sum_value = tl.load(sum_ptr)
        for j in tl.range(col_start, n_cols, col_step, num_stages=num_stages):
            input_start_ptr = input_ptr + i * input_row_stride + j
            offsets = tl.arange(0, BLOCK_SIZE)
            input_ptrs = input_start_ptr + offsets
            mask = j + offsets < n_cols
            x = tl.load(input_ptrs, mask=mask, other=float('-inf'))
            x_minus_max = x - max_value
            exponential_x_minus_max = tl.exp(x_minus_max)
            # to avoid divide by zero
            softmax_res = exponential_x_minus_max / (sum_value + 1e-6)
            output_start_ptr = output_ptr + i * output_row_stride + j
            output_ptrs = output_start_ptr + offsets
            tl.store(output_ptrs, softmax_res, mask=mask)

def softmax_online(x):
    n_rows, n_cols = x.shape
    
    # Another trick we can use is to ask the compiler to use more threads per row by
    # increasing the number of warps (`num_warps`) over which each row is distributed.
    # You will see in the next tutorial how to auto-tune this value in a more natural
    # way so you don't have to come up with manual heuristics yourself.
    
    # at least the number of warps is 4
    num_warps = 4

    # The block size of each loop iteration is the smallest power of two greater than the number of columns in `x`
    BLOCK_SIZE = num_warps * WARP_SIZE
    
    num_col_blocks = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Number of software pipelining stages.
    num_stages = 4 if SIZE_SMEM > 200000 else 2

    # Allocate output
    max_output = torch.full((n_rows, ), float('-inf'), device=x.device)
    sum_output = torch.zeros((n_rows, ), device=x.device)
    # pre-compile kernel to get register usage and compute thread occupancy.
    kernel = softmax_online_add_max_kernel.warmup(x, max_output, sum_output, x.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE,
                                   num_stages=num_stages, num_warps=num_warps, grid=(1, ))
    # May handle things like memory buffers, streams, or execution queues.
    # Ensures the kernel can be launched without further compilation overhead.
    kernel._init_handles()
    n_regs = kernel.n_regs # number of registers allocated for each thread in the kernel
    print(f"n_regs: {n_regs}")
    size_smem = kernel.metadata.shared # the number of bytes of shared memory allocated for each thread block
    print(f"size_smem: {size_smem}")
    occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
    occupancy = min(occupancy, SIZE_SMEM // size_smem) # the number of thread blocks for each SM
    num_programs = NUM_SM * occupancy # total number of thread blocks

    num_programs = min(num_programs, n_rows)
    print(f"num_programs: {num_programs}")
    # get the max and sum with add_max kernel
    softmax_online_add_max_kernel[(num_programs, 1, 1)](x, max_output, sum_output, x.stride(0), n_rows, n_cols, BLOCK_SIZE, num_stages)
    
    # second kernel
    output = torch.empty_like(x)
    kernel = softmax_online_kernel.warmup(x, max_output, sum_output, output, x.stride(0), output.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE,
                                   num_stages=num_stages, num_warps=num_warps, grid=(1, ))
    # May handle things like memory buffers, streams, or execution queues.
    # Ensures the kernel can be launched without further compilation overhead.
    kernel._init_handles()
    n_regs = kernel.n_regs # number of registers allocated for each thread in the kernel
    print(f"n_regs: {n_regs}")
    size_smem = kernel.metadata.shared # the number of bytes of shared memory allocated for each thread block
    print(f"size_smem: {size_smem}")
    occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
    occupancy = min(occupancy, SIZE_SMEM // size_smem) # the number of thread blocks for each SM
    num_programs = NUM_SM * occupancy # total number of thread blocks

    num_programs = min(num_programs, n_rows * num_col_blocks)
    print(f"num_programs: {num_programs}")
    
    softmax_online_kernel[(num_programs // num_col_blocks, num_col_blocks, 1)](x, max_output, sum_output, output, x.stride(0), output.stride(0), n_rows, n_cols, BLOCK_SIZE, num_stages)
    return output
    
            

if __name__ == "__main__":    
    # test online softmax add and sum kernel
    x = torch.randn(10, 32*16, device=DEVICE)
    max_triton, sum_triton = softmax_online_add_max(x)
    max_torch = torch.max(x, axis=1)[0]
    sum_torch = torch.sum(torch.exp(x - max_torch[:, None]), axis=1)
    assert torch.allclose(max_triton, max_torch)
    assert torch.allclose(sum_triton, sum_torch)
    
    # test online softmax kernel
    output_triton = softmax_online(x)
    output_torch = torch.softmax(x, axis=1)
    assert torch.allclose(output_triton, output_torch)




