import math
import torch
import triton
import triton.language as tl

class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        Flash Attention forward pass using tiled online softmax.

        Args:
            ctx: Context object for saving tensors for backward
            Q: Query tensor of shape (batch, seq_len, head_dim)
            K: Key tensor of shape (batch, seq_len, head_dim)
            V: Value tensor of shape (batch, seq_len, head_dim)
            is_causal: Whether to apply causal masking (default: False)

        Returns:
            O: Output tensor of shape (batch, seq_len, head_dim)
        """
        batch_size, seq_len, head_dim = Q.shape
        sqrt_d = math.sqrt(head_dim)
        block_size = 16
        num_blocks = (seq_len + block_size - 1) // block_size
        device, dtype = Q.device, Q.dtype

        # Initialize
        O = torch.zeros_like(Q)
        L = torch.full((batch_size, seq_len), float("-inf"), device=device, dtype=dtype)

        # Outer loop over query blocks
        start_q = 0
        for _ in range(num_blocks):
            end_q = min(start_q + block_size, seq_len)
            Q_i = Q[:, start_q:end_q, :]

            # Initialize accumulators
            m_i = torch.full((batch_size, end_q - start_q, 1), float("-inf"), device=device, dtype=dtype)
            l_i = torch.zeros((batch_size, end_q - start_q, 1), device=device, dtype=dtype)
            O_i = torch.zeros_like(Q_i)

            # Inner loop over key/value blocks
            start_k = 0
            for _ in range(num_blocks):
                if is_causal and start_k >= end_q:
                    break

                end_k = min(start_k + block_size, seq_len)
                K_j = K[:, start_k:end_k, :]
                V_j = V[:, start_k:end_k, :]

                S = torch.bmm(Q_i, K_j.transpose(1, 2)) / sqrt_d

                # Causal masking handling
                if is_causal:
                    q_idx = torch.arange(start_q, end_q, device=device).unsqueeze(1)
                    k_idx = torch.arange(start_k, end_k, device=device).unsqueeze(0)
                    mask = q_idx >= k_idx
                    S = torch.where(mask.unsqueeze(0), S, torch.tensor(float("-inf"), device=device, dtype=dtype))

                # Update running max and log-sum
                m_ij = torch.max(S, dim=-1, keepdim=True).values
                m_new = torch.maximum(m_i, m_ij)
                l_new = torch.exp(m_i - m_new) * l_i + torch.exp(S - m_new).sum(dim=-1, keepdim=True)
                O_i = (torch.exp(m_i - m_new) * l_i * O_i + torch.bmm(torch.exp(S - m_new), V_j)) / l_new

                m_i = m_new
                l_i = l_new

                # Advance key/value pointer
                start_k += block_size

            O[:, start_q:end_q, :] = O_i
            L[:, start_q:end_q] = (m_i + torch.log(l_i)).squeeze(-1)

            # Advance query pointer
            start_q += block_size  

        # Save tensors for backward
        ctx.save_for_backward(Q, K, V, L, O)
        ctx.is_causal = is_causal
        ctx.sqrt_d = sqrt_d
        
        return O

    @staticmethod
    def backward(ctx, dO):
        """
        Attention backward pass.

        Args:
            ctx: Context object with saved tensors
            dO: Gradient of loss w.r.t. output

        Returns:
            dQ: Gradient w.r.t. Q
            dK: Gradient w.r.t. K
            dV: Gradient w.r.t. V
            None: No gradient for is_causal
        """
        Q, K, V, L, O = ctx.saved_tensors
        dQ, dK, dV, _ = attention_backward_impl(
            Q, K, V, L, O, dO, ctx.sqrt_d, ctx.is_causal
        )
        return dQ, dK, dV, None


def attention_backward_impl(Q, K, V, L, O, dO, sqrt_d, is_causal):
    """
    Backward pass implementation for Flash Attention.

    Uses standard attention gradient formulas with recomputation of P
    from saved log-sum-exp values L.

    Args:
        Q: Query tensor of shape (batch, seq_q, d)
        K: Key tensor of shape (batch, seq_k, d)
        V: Value tensor of shape (batch, seq_k, d)
        L: Log-sum-exp values from forward pass, shape (batch, seq_q)
        O: Output from forward pass, shape (batch, seq_q, d)
        dO: Gradient of loss w.r.t. output, shape (batch, seq_q, d)
        sqrt_d: Square root of head dimension (for scaling)
        is_causal: Whether to apply causal masking

    Returns:
        dQ: Gradient w.r.t. Q
        dK: Gradient w.r.t. K
        dV: Gradient w.r.t. V
        None: Placeholder for compatibility
    """
    # Your code here
    seq_q, seq_k = Q.shape[1], K.shape[1]

    # 1. D = rowsum(O ⊙ dO)
    D = torch.sum(O * dO, dim=-1, keepdim=True)

    # 2. Recompute S = Q @ K^T / sqrt(d)
    S = torch.bmm(Q, K.transpose(1, 2)) / sqrt_d

    # 3. Apply causal mask
    if is_causal:
        causal_mask = torch.triu(torch.ones(seq_q, seq_k, device=Q.device, dtype=torch.bool), diagonal=1)
        S = S.masked_fill(causal_mask, float("-inf"))

    # 4. Recompute P = exp(S - L)
    P = torch.exp(S - L.unsqueeze(-1))

    # 5. dV = P^T @ dO
    dV = torch.bmm(P.transpose(1, 2), dO)

    # 6. dP = dO @ V^T
    dP = torch.bmm(dO, V.transpose(1, 2))

    # 7. dS = P ⊙ (dP - D)
    dS = P * (dP - D)

    # 8. dQ = dS @ K / sqrt(d)
    dQ = torch.bmm(dS, K) / sqrt_d

    # 9. dK = dS^T @ Q / sqrt(d)
    dK = torch.bmm(dS.transpose(1, 2), Q) / sqrt_d

    return dQ, dK, dV, None


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS, scale,
    D: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    is_causal: tl.constexpr,
):
    """
    Flash Attention forward kernel using online softmax algorithm.

    Each program instance processes one query block for one batch element.
    """
    # Get program IDs
    pid_q = tl.program_id(0)
    pid_b = tl.program_id(1)

    # Create block pointers for this batch and query tile
    Q_ptr = Q_ptr + pid_b * stride_qb
    O_ptr = O_ptr + pid_b * stride_ob
    L_ptr = L_ptr + pid_b * stride_lb
    K_ptr = K_ptr + pid_b * stride_kb
    V_ptr = V_ptr + pid_b * stride_vb

    Q_block = tl.make_block_ptr(
        Q_ptr,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(pid_q * BLOCK_Q, 0),
        block_shape=(BLOCK_Q, D),
        order=(1, 0),
    )

    O_block = tl.make_block_ptr(
        O_ptr,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(pid_q * BLOCK_Q, 0),
        block_shape=(BLOCK_Q, D),
        order=(1, 0),
    )

    L_block = tl.make_block_ptr(
        L_ptr,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(pid_q * BLOCK_Q,),
        block_shape=(BLOCK_Q,),
        order=(0,),
    )

    # Create key/value block pointers (later advance)
    K_block = tl.make_block_ptr(
        K_ptr,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(BLOCK_K, D),
        order=(1, 0),
    )
    V_block = tl.make_block_ptr(
        V_ptr,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(BLOCK_K, D),
        order=(1, 0),
    )

    # Initialize 
    m_i = tl.full((BLOCK_Q,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    O_i = tl.zeros((BLOCK_Q, D), dtype=tl.float32)

    Q_i = tl.load(Q_block, boundary_check=(0, 1))

    num_k_blocks = (N_KEYS + BLOCK_K - 1) // BLOCK_K
    for j in range(num_k_blocks):
        K_j = tl.load(K_block, boundary_check=(0, 1), padding_option="zero")
        V_j = tl.load(V_block, boundary_check=(0, 1), padding_option="zero")

        S = tl.dot(Q_i, tl.trans(K_j)) * scale

        if is_causal:
            q_idx = tl.arange(0, BLOCK_Q)[:, None] + pid_q * BLOCK_Q
            k_idx = tl.arange(0, BLOCK_K)[None, :] + j * BLOCK_K
            mask = k_idx > q_idx
            S = tl.where(mask, float("-inf"), S)

        # Update running max and sum
        m_ij = tl.max(S, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        alpha = tl.exp(m_i - m_new)
        P = tl.exp(S - m_new[:, None])

        l_i = alpha * l_i + tl.sum(P, axis=1)
        O_i = O_i * alpha[:, None] + tl.dot(P.to(V_j.dtype), V_j)

        m_i = m_new

        # Advance key/value pointers
        K_block = tl.advance(K_block, (BLOCK_K, 0))
        V_block = tl.advance(V_block, (BLOCK_K, 0))

    # Normalize output
    O_i = O_i / l_i[:, None]

    tl.store(O_block, O_i.to(Q_i.dtype), boundary_check=(0, 1))
    tl.store(L_block, m_i + tl.log(l_i), boundary_check=(0,))


class FlashAttentionTriton(torch.autograd.Function):
    """Flash Attention using Triton kernels for both forward and backward passes."""

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        Forward pass using Triton kernel.

        Args:
            Q: Query tensor of shape (batch, seq_q, d)
            K: Key tensor of shape (batch, seq_k, d)
            V: Value tensor of shape (batch, seq_k, d)
            is_causal: Whether to apply causal masking

        Returns:
            Output tensor of shape (batch, seq_q, d)
        """
        batch_size, seq_len, head_dim = Q.shape
        _, N_KEYS, _ = K.shape

        # Initialize
        O = torch.empty_like(Q)
        L = torch.empty((batch_size, seq_len), device=Q.device, dtype=torch.float32)

        # Choose block sizes and configure grid
        BLOCK_Q = BLOCK_K = 32
        scale = 1.0 / math.sqrt(head_dim)
        grid = (triton.cdiv(seq_len, BLOCK_Q), batch_size)

        # Launch flash_fwd_kernel
        flash_fwd_kernel[grid](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            seq_len, N_KEYS, scale,
            D=head_dim,
            BLOCK_Q=BLOCK_Q,
            BLOCK_K=BLOCK_K,
            is_causal=is_causal,
        )

        # Save tensors for backward
        ctx.save_for_backward(Q, K, V, L, O)
        ctx.is_causal = is_causal
        ctx.sqrt_d = math.sqrt(head_dim)

        return O

    @staticmethod
    def backward(ctx, dO):
        """
        Attention backward pass.

        Args:
            ctx: Context object with saved tensors
            dO: Gradient of loss w.r.t. output

        Returns:
            dQ: Gradient w.r.t. Q
            dK: Gradient w.r.t. K
            dV: Gradient w.r.t. V
            None: No gradient for is_causal
        """
        Q, K, V, L, O = ctx.saved_tensors
        dQ, dK, dV, _ = attention_backward_impl(
            Q, K, V, L, O, dO, ctx.sqrt_d, ctx.is_causal
        )
        return dQ, dK, dV, None