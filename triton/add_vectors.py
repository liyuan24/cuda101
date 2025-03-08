import triton
import triton.language as tl
import torch


@triton.jit
def add_vectors_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
    
def add_vectors(x, y):
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_vectors_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output

x = torch.randn(1000000).to('cuda')
y = torch.randn(1000000).to('cuda')
output = add_vectors(x, y)
print(output)

