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
    
    end_n = seqlen_k if not IS_CASUAL else tl.minimum((q_seq_start+1) * BLOCK_M, seqlen_k)
    
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N) # hint to the compiler that the start_n is a multiple of BLOCK_N
        offsets_n = start_n + tl.arange(0, BLOCK_N)
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
    BLOCK = 16
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

@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] // args["BLOCK_M"] == 0,
    }
)
@triton.jit
def _backward_o_do_elementwise_product_kernel(
    o, # the attention output pointer
    do, # the output gradient pointer
    delta, # the element wise product output pointer
    stride_ob, # the output batch dimension stride
    stride_oh, # the output head dimension stride
    stride_om, # the output sequence length dimension stride
    stride_dob, # the output gradient batch dimension stride
    stride_doh, # the output gradient head dimension stride
    stride_dom, # the output gradient sequence length dimension stride
    nheads, # the number of attention heads
    seqlen_q, # the query sequence length
    BLOCK_M: tl.constexpr, # the number of rows handled by each program
    HEAD_DIM: tl.constexpr, # the attention head dimension
    EVEN_M: tl.constexpr, # whether the sequence length of query can be divided by BLOCK_M
):
    start_m = tl.program_id(0)
    start_bh = tl.program_id(1)
    b_idx = start_bh // nheads
    h_idx = start_bh % nheads
    
    offsets_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_d = tl.arange(0, HEAD_DIM)
    
    o_ptrs = o + b_idx * stride_ob + h_idx * stride_oh + offsets_m[:, None] * stride_om + offsets_d[None, :]
    do_ptrs = do + b_idx * stride_dob + h_idx * stride_doh + offsets_m[:, None] * stride_dom + offsets_d[None, :]
    
    # load the attention output and output gradient block
    if EVEN_M:
        o_block = tl.load(o_ptrs).cast(tl.float32)
        do_block = tl.load(do_ptrs).cast(tl.float32)
    else:
        o_block = tl.load(o_ptrs, mask=offsets_m[:, None] < seqlen_q, other=0.0).cast(tl.float32)
        do_block = tl.load(do_ptrs, mask=offsets_m[:, None] < seqlen_q, other=0.0).cast(tl.float32)
    
    # compute the element wise product
    output = tl.sum(o_block * do_block, axis=1)
    
    delta_ptrs = delta + b_idx * stride_ob + h_idx * stride_oh + offsets_m
    
    if EVEN_M:
        tl.store(delta_ptrs, output)
    else:
        tl.store(delta_ptrs, output, mask=offsets_m < seqlen_q)
    
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] // args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] // args["BLOCK_N"] == 0,
    }
)
@triton.jit
def _backward_kernel(
    q, # the query pointer
    k, # the key pointer
    v, # the value pointer
    do, # the attention output gradient pointer
    dq, # the query gradient pointer
    dk, # the key gradient pointer
    dv, # the value gradient pointer
    delta, # the element wise product of attention output and its gradient pointer
    lse, # the logsumexp pointer
    stride_qb, # the query batch dimension stride
    stride_qh, # the query head dimension stride
    stride_qm, # the query sequence length dimension stride
    stride_kb, # the key batch dimension stride
    stride_kh, # the key head dimension stride
    stride_kn, # the key sequence length dimension stride
    stride_vb, # the value batch dimension stride
    stride_vh, # the value head dimension stride
    stride_vn, # the value sequence length dimension stride
    stride_dob, # the output gradient batch dimension stride
    stride_doh, # the output gradient head dimension stride
    stride_dom, # the output gradient sequence length dimension stride
    stride_dqb, # the query gradient batch dimension stride
    stride_dqh, # the query gradient head dimension stride
    stride_dqm, # the query gradient sequence length dimension stride
    stride_dkb, # the key gradient batch dimension stride
    stride_dkh, # the key gradient head dimension stride
    stride_dkn, # the key gradient sequence length dimension stride
    stride_dvb, # the value gradient batch dimension stride
    stride_dvh, # the value gradient head dimension stride
    stride_dvn, # the value gradient sequence length dimension stride
    nheads, # the number of attention heads
    seqlen_q, # the query sequence length
    seqlen_k, # the key sequence length
    scale, # the scaling factor for the q, k dot product
    BLOCK_M: tl.constexpr, # the number of rows handled by each program
    BLOCK_N: tl.constexpr, # the number of columns handled by each program
    HEAD_DIM: tl.constexpr, # the attention head dimension
    IS_CASUAL: tl.constexpr, # whether the attention is causal
    EVEN_M: tl.constexpr, # whether the sequence length of query can be divided by BLOCK_M
    EVEN_N: tl.constexpr, # whether the sequence length of key can be divided by BLOCK_N
):
    start_n = tl.program_id(0)
    if IS_CASUAL:
        begin_m = start_n * BLOCK_N // BLOCK_M * BLOCK_M
    else:
        begin_m = 0
    start_bh = tl.program_id(1)
    b_idx = start_bh // nheads
    h_idx = start_bh % nheads
    
    offsets_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_d = tl.arange(0, HEAD_DIM)
    
    k_ptrs = k + b_idx * stride_kb + h_idx * stride_kh + offsets_n[:, None] * stride_kn + offsets_d[None, :]
    v_ptrs = v + b_idx * stride_vb + h_idx * stride_vh + offsets_n[:, None] * stride_vn + offsets_d[None, :]
    
    # load the key and value blocks and put in SRAM(l1 cache) throughout the kernel
    if EVEN_N:
        k_block = tl.load(k_ptrs)
        v_block = tl.load(v_ptrs)
    else:
        k_block = tl.load(k_ptrs, mask=offsets_n[:, None] < seqlen_k, other=0.0)
        v_block = tl.load(v_ptrs, mask=offsets_n[:, None] < seqlen_k, other=0.0)
        
    # initialize the dk, dv on SRAM
    dk_block = tl.zeros((BLOCK_N, HEAD_DIM), dtype=tl.float32)
    dv_block = tl.zeros((BLOCK_N, HEAD_DIM), dtype=tl.float32)
    
    for start_m in range(begin_m, seqlen_q, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M) # give compiler a hint
        offsets_m = start_m + tl.arange(0, BLOCK_M)
        
        # load the query block
        q_ptrs = q + b_idx * stride_qb + h_idx * stride_qh + offsets_m[:, None] * stride_qm + offsets_d[None, :]
        if EVEN_M:
            q_block = tl.load(q_ptrs)
        else:
            q_block = tl.load(q_ptrs, mask=offsets_m[:, None] < seqlen_q, other=0.0)
            
        # recompute the attention
        qkT = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        qkT += tl.dot(q_block, k_block.trans())
        # apply mask when the the k offsets is out of the k sequence length
        if not EVEN_N:
            qkT += tl.where(offsets_n[None, :] < seqlen_k, 0.0, -float("inf"))
    
        if IS_CASUAL:
            qkT += tl.where(offsets_m[:, None] >= offsets_n[None, :], 0.0, -float("inf"))
        
        # load lse block
        lse_ptrs = lse + b_idx * stride_qb + h_idx * stride_qh + offsets_m
        if EVEN_M:
            lse_block = tl.load(lse_ptrs)
        else:
            lse_block = tl.load(lse_ptrs, mask=offsets_m < seqlen_q, other=0.0)
        
        # compute the attention scores
        p_block = tl.exp(qkT * scale - lse_block[:, None])
        
        # pv = o, and compute dv
        do_ptrs = do + b_idx * stride_dob + h_idx * stride_doh + offsets_m[:, None] * stride_dom + offsets_d[None, :]
        if EVEN_M:
            do_block = tl.load(do_ptrs)
        else:
            do_block = tl.load(do_ptrs, mask=offsets_m[:, None] < seqlen_q, other=0.0)
        dv_block += tl.dot(p_block.cast(do_block.dtype).trans(), do_block)
        
        # compute dp
        dp_block = tl.dot(do_block, v_block.trans())
        
        # compute ds
        # first load delta block
        delta_ptrs = delta + b_idx * stride_qb + h_idx * stride_qh + offsets_m
        if EVEN_M:
            delta_block = tl.load(delta_ptrs)
        else:
            delta_block = tl.load(delta_ptrs, mask=offsets_m < seqlen_q, other=0.0)
        
        # we need to multiply the scale instead of dividing because S = scale * qkT, dL/dq = dL/dS * dS/dq * scale and same for dk
        ds_block = ((dp_block * p_block - delta_block[:, None] * p_block) * scale).cast(q_block.dtype)
        
        # s = scale * qkT
        # load dq block
        dq_ptrs = dq + b_idx * stride_dqb + h_idx * stride_dqh + offsets_m[:, None] * stride_dqm + offsets_d[None, :]
        # if EVEN_M:
        #     dq_block = tl.load(dq_ptrs)
        # else:
        #     dq_block = tl.load(dq_ptrs, mask=offsets_m[:, None] < seqlen_q, other=0.0)
            
        # dq_block += tl.dot(ds_block, k_block)
        # # write back dq block
        # if EVEN_M:
        #     tl.store(dq_ptrs, dq_block)
        # else:
        #     tl.store(dq_ptrs, dq_block, mask=offsets_m[:, None] < seqlen_q)
        dq_block = tl.dot(ds_block, k_block)
        tl.atomic_add(dq_ptrs, dq_block, mask=offsets_m[:, None] < seqlen_q)
            
        dk_block += tl.dot(ds_block.trans(), q_block)
    
    # write back dk and dv
    dk_ptrs = dk + b_idx * stride_dkb + h_idx * stride_dkh + offsets_n[:, None] * stride_dkn + offsets_d[None, :]
    dv_ptrs = dv + b_idx * stride_dvb + h_idx * stride_dvh + offsets_n[:, None] * stride_dvn + offsets_d[None, :]
    
    if EVEN_N:
        tl.store(dk_ptrs, dk_block)
        tl.store(dv_ptrs, dv_block)
    else:
        tl.store(dk_ptrs, dk_block, mask=offsets_n[:, None] < seqlen_k)
        tl.store(dv_ptrs, dv_block, mask=offsets_n[:, None] < seqlen_k)
        

def _flash_attention_backward(q, k, v, o, do, dq, dk, dv, lse, scale: float=None, is_causal: bool=False):
    b, n_heads, q_seq_len, head_dim = q.shape
    _, _ , k_seq_len, _ = k.shape
    
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous() and o.is_contiguous() and do.is_contiguous() and dq.is_contiguous() and dk.is_contiguous() and dv.is_contiguous() and lse.is_contiguous()
    assert k.shape == (b, n_heads, k_seq_len, head_dim)
    assert v.shape == (b, n_heads, k_seq_len, head_dim)
    assert q.is_cuda and k.is_cuda and v.is_cuda and o.is_cuda and do.is_cuda and dq.is_cuda and dk.is_cuda and dv.is_cuda and lse.is_cuda
    
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
        
    delta = torch.empty_like(lse)
    
    BLOCK = 16
    num_warps = 4
    num_stages = 1
    
    grid = lambda META: (triton.cdiv(q_seq_len, META["BLOCK_M"]), b * n_heads)
    
    _backward_o_do_elementwise_product_kernel[grid](
        o, 
        do, 
        delta,      
        o.stride(0), 
        o.stride(1), 
        o.stride(2), 
        do.stride(0), 
        do.stride(1), 
        do.stride(2), 
        n_heads, 
        q_seq_len, 
        BLOCK_M=BLOCK, 
        HEAD_DIM=head_dim,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    
    grid = lambda META: (triton.cdiv(k_seq_len, META["BLOCK_N"]), b * n_heads)
    
    _backward_kernel[grid](
        q, 
        k, 
        v, 
        do, 
        dq, 
        dk, 
        dv, 
        delta, 
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
        do.stride(0), 
        do.stride(1), 
        do.stride(2), 
        dq.stride(0), 
        dq.stride(1), 
        dq.stride(2), 
        dk.stride(0), 
        dk.stride(1), 
        dk.stride(2), 
        dv.stride(0), 
        dv.stride(1), 
        dv.stride(2), 
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
    
    return dq, dk, dv

class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False, scale=None):
        """
        q: (batch_size, seqlen_q, nheads, headdim)
        k, v: (batch_size, seqlen_k, nheads, headdim)
        causal: bool, whether the attention is causal
        scale: float, the scaling factor for the softmax
        """
        # Make sure that the last dimension is contiguous
        q, k, v = [x if x.stride(-1) == 1 else x.contiguous() for x in [q, k, v]]
        o, lse = _flash_attention_forward(
            q, k, v, is_causal=is_causal, scale=scale
        )
        ctx.scale = scale
        ctx.is_causal = is_causal
        ctx.save_for_backward(q, k, v, o, lse)
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        # different from Tri Dao's implementation https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_triton.py#L1044
        # where dq is initialized as torch.empty_like(q)
        # but since we use atomic_add to update dq, if the initial value is not 0, the accumulation for dq will be incorrect
        # so we initialize dq as torch.zeros_like(q)
        dq = torch.zeros_like(q)
        # for dk and dv, since we only store the local value back to the original memory, the initial values are not important
        # so we can use torch.empty_like(k) and torch.empty_like(v)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        _flash_attention_backward(
            q, k, v, o, do, dq, dk, dv, lse, scale=ctx.scale, is_causal=ctx.is_causal
        )
        return dq, dk, dv, None, None
        
        

def test_forward(is_causal: bool=False):
    torch.manual_seed(0)
    data_type = torch.float32
    q = torch.randn(1, 1, 32, 16, dtype=data_type).cuda()
    k = torch.randn(1, 1, 32, 16, dtype=data_type).cuda()
    v = torch.randn(1, 1, 32, 16, dtype=data_type).cuda()
    
    o, lse = _flash_attention_forward(q, k, v, is_causal=is_causal)
    
    # torch attention
    o_torch = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
    
    # # check if the output is the same
    assert torch.allclose(o, o_torch, atol=1e-2)
    
    qkT = q @ k.transpose(-2, -1)
    scale = 1.0 / math.sqrt(q.shape[-1])
    qkT = qkT * scale
    if is_causal:
        mask = torch.where(torch.arange(q.shape[-2])[:, None] >= torch.arange(k.shape[-2])[None, :], 0.0, -float("inf")).to(qkT.device)
        qkT += mask
    max_value = torch.max(qkT, dim=-1).values
    lse_torch = torch.logsumexp(qkT - max_value[..., None], dim=-1) + max_value
    assert torch.allclose(lse, lse_torch, atol=1e-2)
    
    
def test_backward(is_causal: bool=False):
    torch.manual_seed(0)
    data_type = torch.float16
    q = torch.randn(1, 1, 16, 16, dtype=data_type).cuda().requires_grad_(True)
    k = torch.randn(1, 1, 16, 16, dtype=data_type).cuda().requires_grad_(True)
    v = torch.randn(1, 1, 16, 16, dtype=data_type).cuda().requires_grad_(True)
    do = torch.randn_like(q)
    
    # torch attention
    o_torch = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
    o_torch.backward(do)
    
    torch_dv, v.grad = v.grad.clone(), None
    torch_dk, k.grad = k.grad.clone(), None
    torch_dq, q.grad = q.grad.clone(), None
    
    flash_attn = FlashAttnFunc.apply
    o_triton = flash_attn(q, k, v, is_causal)
    o_triton.backward(do)
    
    triton_dv, v.grad = v.grad.clone(), None
    triton_dk, k.grad = k.grad.clone(), None
    triton_dq, q.grad = q.grad.clone(), None
    
    assert torch.allclose(torch_dv, triton_dv, atol=1e-2)
    assert torch.allclose(torch_dk, triton_dk, atol=1e-2)
    assert torch.allclose(torch_dq, triton_dq, atol=1e-2)
    
if __name__ == "__main__":
    test_backward()
    test_backward(is_causal=True)
    test_forward()
    test_forward(is_causal=True)
    
    