import triton
import triton.language as tl
import math
import torch

@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] // args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] // args["BLOCK_N"] == 0,
    }
)
@triton.jit
def _forward_kernel(
    q, # query pointer
    k, # key pointer
    v, # value pointer
    o, # output pointer
    lse, # logsumexp pointer
    stride_qb, # query batch dimension stride
    stride_qh, # query head dimension stride
    stride_qm, # query sequence length dimension stride
    stride_kb, # key batch dimension stride
    stride_kh, # key head dimension stride
    stride_kn, # key sequence length dimension stride
    stride_vb, # value batch dimension stride
    stride_vh, # value head dimension stride
    stride_vn, # value sequence length dimension stride
    stride_ob, # output batch dimension stride
    stride_oh, # output head dimension stride
    stride_om, # output sequence length dimension stride
    nheads, # number of attention heads,
    seqlen_q, # the query sequence length
    seqlen_k, # the key sequence length,
    scale, # the scaling factor for the q, k dot product
    EVEN_M: tl.constexpr, # whether the sequence length of query can be divided by BLOCK_M
    EVEN_N: tl.constexpr, # whether the sequence length of key can be divided by BLOCK_N
    BLOCK_M: tl.constexpr, # number of rows handled by each program
    BLOCK_N: tl.constexpr, # number of columns handled by each program
    HEAD_DIM: tl.constexpr, # the attention head dimension,
    IS_CASUAL: tl.constexpr, # whether the attention is causal
):
    q_seq_start = tl.program_id(0)
    bh_start = tl.program_id(1)
    b_idx = bh_start // nheads
    h_idx = bh_start % nheads
    
    # it decides what part of the query will be loaded
    offsets_m = q_seq_start * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_d = tl.arange(0, HEAD_DIM)
    
    # query block pointers
    q_ptrs = q + b_idx * stride_qb + h_idx * stride_qh + offsets_m[:, None] * stride_qm + offsets_d[None, :]
    
    # load the query and it will stay in the SRAM(l1 cache) throughout the kernel
    if EVEN_M:
        q_block = tl.load(q_ptrs)
    else:
        q_block = tl.load(q_ptrs, mask=offsets_m[:, None] < seqlen_q, other=0.0)
    
    lse_block = tl.zeros([BLOCK_M], dtype=tl.float32)
    max_block = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc_block = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    
    end_n = seqlen_k if IS_CASUAL else tl.minimum((q_seq_start+1) * BLOCK_M, seqlen_k)
    
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N) # hint to the compiler that the start_n is a multiple of BLOCK_N
        offsets_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
        k_ptrs = k + b_idx * stride_kb + h_idx * stride_kh + offsets_n[:, None] * stride_kn + offsets_d[None, :]
        v_ptrs = v + b_idx * stride_vb + h_idx * stride_vh + offsets_n[:, None] * stride_vn + offsets_d[None, :]
        
        # load the key and value blocks
        if EVEN_N:
            k_block = tl.load(k_ptrs)
            v_block = tl.load(v_ptrs)
        else:
            k_block = tl.load(k_ptrs, mask=offsets_n[:, None] < seqlen_k, other=0.0)
            v_block = tl.load(v_ptrs, mask=offsets_n[:, None] < seqlen_k, other=0.0)
        
        # compute the attention scores
        qkT = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        qkT += tl.dot(q_block, k_block.trans())
        
        # apply mask when the the k offsets is out of the k sequence length
        if not EVEN_N:
            qkT += tl.where(offsets_n[None, :] < seqlen_k, 0.0, -float("inf"))
        
        if IS_CASUAL:
            causal_mask = (offsets_m[:, None] >= offsets_n[None, :])
            qkT += tl.where(causal_mask, 0.0, -float("inf"))
            
        # scale the qkT
        qkT = qkT * scale
        
        cur_max = tl.max(qkT, axis=1)
        new_max = tl.maximum(max_block, cur_max)
        p = tl.exp(qkT - new_max[:, None])
        lse_block = lse_block * tl.exp(cur_max - new_max) + tl.sum(p, axis=1)
        p = p.cast(v_block.dtype)
        acc_block = acc_block * tl.exp(cur_max - new_max)[:, None] + tl.dot(p, v_block)
        max_block = new_max
    
    acc_block = acc_block / lse_block[:, None]
    lse_block = tl.log(lse_block) + max_block
    
    # store the acc_block to output block
    o_ptrs = o + b_idx * stride_ob + h_idx * stride_oh + offsets_m[:, None] * stride_om + offsets_d[None, :]
    if EVEN_M:
        tl.store(o_ptrs, acc_block)
    else:
        tl.store(o_ptrs, acc_block, mask=offsets_m[:, None] < seqlen_q)
    
    lse_ptrs = lse + b_idx * stride_ob + h_idx * stride_oh + offsets_m
    if EVEN_M:
        tl.store(lse_ptrs, lse_block)
    else:
        tl.store(lse_ptrs, lse_block, mask=offsets_m < seqlen_q)
        

def _flash_attention_forward(q, k, v, scale: float=None, is_causal: bool=False):
    b, n_heads, q_seq_len, head_dim = q.shape
    _, _ , k_seq_len, _ = k.shape
    
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    assert k.shape == (b, n_heads, k_seq_len, head_dim)
    assert v.shape == (b, n_heads, k_seq_len, head_dim)
    assert q.is_cuda and k.is_cuda and v.is_cuda
    
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
        
    o = torch.empty_like(q)
    lse = torch.empty(b, n_heads, q_seq_len, device=q.device, dtype=torch.float32)
    BLOCK = 128
    num_warps = 4
    num_stages = 1
    
    grid = lambda META: (triton.cdiv(q_seq_len, META["BLOCK_M"]), b * n_heads)
    
    _forward_kernel[grid](
        q,
        k,
        v,
        o,
        lse,
        q.stride(0), 
        q.stride(1), 
        q.stride(2), 
        k.stride(0), 
        k.stride(1), 
        k.stride(2), 
        v.stride(0), 
        v.stride(1), 
        v.stride(2), 
        o.stride(0), 
        o.stride(1), 
        o.stride(2), 
        n_heads, 
        q_seq_len, 
        k_seq_len, 
        scale, 
        BLOCK_M=BLOCK, 
        BLOCK_N=BLOCK, 
        HEAD_DIM=head_dim, 
        IS_CASUAL=is_causal,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    
    return o, lse

if __name__ == "__main__":
    data_type = torch.float32
    q = torch.randn(1, 1, 128, 16, dtype=data_type).cuda()
    k = torch.randn(1, 1, 128, 16, dtype=data_type).cuda()
    v = torch.randn(1, 1, 128, 16, dtype=data_type).cuda()
    
    is_causal = True
    o, lse = _flash_attention_forward(q, k, v, is_causal=is_causal)
    print(o.shape, lse.shape)
    
    # torch attention
    o_torch = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
    print(o_torch.shape)
    
    # # check if the output is the same
    assert torch.allclose(o, o_torch, atol=1e-2)
    
    
    