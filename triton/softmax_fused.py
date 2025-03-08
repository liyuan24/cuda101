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


def naive_softmax(x):
    """Compute row-wise softmax of X using native pytorch

    We subtract the maximum element in order to avoid overflows. Softmax is invariant to
    this shift.
    """
    # read  MN elements ; write M  elements
    x_max = x.max(dim=1, keepdim=True)[0]
    # read MN + M elements ; write MN elements
    z = x - x_max
    # read  MN elements ; write MN elements
    numerator = torch.exp(z)
    # read  MN elements ; write M  elements
    denominator = numerator.sum(dim=1, keepdim=True)
    # read MN + M elements ; write MN elements
    ret = numerator / denominator
    # in total: read 5MN + 2M elements ; wrote 3MN + 2M elements
    return ret

# assume that each block can read the whole row, each block will process the whole row at a time
@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    pid = tl.program_id(0)
    row_stride = tl.num_programs(0)
    row_start = pid
    for i in tl.range(row_start, n_rows, row_stride, num_stages=num_stages):
        row_start_ptr = input_ptr + i * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        x = tl.load(input_ptrs, mask=mask, other=float('-inf'))
        x_max = tl.max(x, axis=0)
        x = x - x_max
        numerator = tl.exp(x)
        denominator = tl.sum(numerator, axis=0)
        output_start_ptr = output_ptr + i * output_row_stride
        output_ptrs = output_start_ptr + col_offsets
        tl.store(output_ptrs, numerator / denominator, mask=mask)


def softmax(x):
    n_rows, n_cols = x.shape

    # The block size of each loop iteration is the smallest power of two greater than the number of columns in `x`
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # Another trick we can use is to ask the compiler to use more threads per row by
    # increasing the number of warps (`num_warps`) over which each row is distributed.
    # You will see in the next tutorial how to auto-tune this value in a more natural
    # way so you don't have to come up with manual heuristics yourself.
    
    # at least the number of warps is 4
    num_warps = 4

    # Number of software pipelining stages.
    num_stages = 4 if SIZE_SMEM > 200000 else 2

    # Allocate output
    y = torch.empty_like(x)

    # pre-compile kernel to get register usage and compute thread occupancy.
    kernel = softmax_kernel.warmup(y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE,
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
    softmax_kernel[(num_programs, 1, 1)](x, y, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE, num_stages)
    return y

if __name__ == "__main__":
    # test softmax kernel
    torch.manual_seed(0)
    x = torch.randn(10, 32*4, device=DEVICE)
    y_triton = softmax(x)
    y_torch = torch.softmax(x, axis=1)
    assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)




