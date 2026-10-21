import torch
import triton
import triton.language as tl
import torch.nn.functional as F


def softmax_mult(x, V, dim=-1):
    return F.softmax(x, dim=dim) @ V


@triton.jit
def fused_softmax_kernel(
    x_ptr,
    V_ptr,
    output_ptr,
    stride_xbatch,
    stride_xrow,
    stride_xcol,
    stride_Vbatch,
    stride_Vrow,
    stride_Vcol,
    stride_obatch,
    stride_orow,
    stride_ocol,
    d1: tl.constexpr,
    d2: tl.constexpr,
    d3: tl.constexpr,
    BLOCK_1: tl.constexpr,
    BLOCK_2: tl.constexpr,
):
    tl.static_assert(d1 % BLOCK_1 == 0, "d1 must be divisible by BLOCK_1")
    tl.static_assert(d2 % BLOCK_2 == 0, "d2 must be divisible by BLOCK_2")

    pid_batch = tl.program_id(0)
    pid_row = tl.program_id(1)

    # Block pointers
    x_block = tl.make_block_ptr(
        x_ptr + pid_batch * stride_xbatch,
        shape=(d1, d2),
        strides=(stride_xrow, stride_xcol),
        offsets=(pid_row * BLOCK_1, 0),
        block_shape=(BLOCK_1, BLOCK_2),
        order=(1, 0),
    )

    V_block = tl.make_block_ptr(
        V_ptr + pid_batch * stride_Vbatch,
        shape=(d2, d3),
        strides=(stride_Vrow, stride_Vcol),
        offsets=(0, 0),
        block_shape=(BLOCK_2, d3),
        order=(1, 0),
    )

    Num_blocks = tl.cdiv(d2, BLOCK_2)

    # Initialize m_prev, l_prev and o_prev
    m_prev = tl.full((BLOCK_1,), float("-inf"), dtype=tl.float32)
    l_prev = tl.zeros((BLOCK_1,), dtype=tl.float32)
    out_prev = tl.zeros((BLOCK_1, d3), dtype=tl.float32)

    for _ in range(Num_blocks):
        x = tl.load(x_block, boundary_check=(0, 1), padding_option="zero")
        V = tl.load(V_block, boundary_check=(0, 1), padding_option="zero")

        # Compute running max and sum
        block_max = tl.max(x, axis=1)
        m_curr = tl.maximum(m_prev, block_max)

        exp_x = tl.exp(x - m_curr[:, None])

        l_curr = l_prev * tl.exp(m_prev - m_curr) + tl.sum(exp_x, axis=1)

        scale = (l_prev / l_curr) * tl.exp(m_prev - m_curr)
        normalized = exp_x / l_curr[:, None]
        out_prev = out_prev * scale[:, None] + tl.dot(normalized, V)

        # Update m_prev and l_prev for the next block
        m_prev = m_curr
        l_prev = l_curr

        # Advance block pointers    
        x_block = x_block.advance((0, BLOCK_2))
        V_block = V_block.advance((BLOCK_2, 0))

    # Store final result
    output_block = tl.make_block_ptr(
        output_ptr + pid_batch * stride_obatch,
        shape=(d1, d3),
        strides=(stride_orow, stride_ocol),
        offsets=(pid_row * BLOCK_1, 0),
        block_shape=(BLOCK_1, d3),
        order=(1, 0),
    )
    tl.store(output_block, out_prev.to(output_block.dtype.element_ty), boundary_check=(0, 1))


def fused_softmax(x, V, BLOCK_1=16, BLOCK_2=16):
    """
    Compute fused softmax(x) @ V using Triton.

    Args:
        x: Input tensor of shape (batch, d1, d2)
        V: Value tensor of shape (batch, d2, d3)
        BLOCK_1: Block size for d1 dimension
        BLOCK_2: Block size for d2 dimension

    Returns:
        Output tensor of shape (batch, d1, d3)
    """
    batch_size, d1, d2 = x.shape
    _, _, d3 = V.shape
    softmax_output = torch.empty((batch_size, d1, d3), device=x.device, dtype=x.dtype)

    grid = (batch_size, triton.cdiv(d1, BLOCK_1))

    fused_softmax_kernel[grid](
        x,
        V,
        softmax_output,
        x.stride(0), 
        x.stride(1), 
        x.stride(2),
        V.stride(0), 
        V.stride(1), 
        V.stride(2),
        softmax_output.stride(0), 
        softmax_output.stride(1), 
        softmax_output.stride(2),
        d1=d1,
        d2=d2,
        d3=d3,
        BLOCK_1=BLOCK_1,
        BLOCK_2=BLOCK_2,
    )

    return softmax_output
