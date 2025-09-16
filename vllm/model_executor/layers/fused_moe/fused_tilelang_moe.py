# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tilelang implementation for fused MoE."""
from typing import Optional

import torch

from vllm.model_executor.layers.fused_moe.fused_moe import moe_align_block_size
from vllm.scalar_type import ScalarType, scalar_types
from vllm.utils import direct_register_custom_op

import tilelang
import tilelang.language as T
from tilelang import tvm as tvm


# Modified from tilelang/examples/dequantize_gemm/example_dequant_groupedgemm_bf16_mxfp4_hopper.py
# Main differences:
# 1. Explicitly pass in output buffer
# 2. use M*topk rather than -1 for padding
# 3. topk_weights is fp32 rather than bf16
# 4. Use symbolic shape M
# 5. Apply topk weights only once
@tilelang.jit
def tl_mxfp4_grouped_gemm(
           N,
           K,
           topk,
           E,
           in_dtype,
           out_dtype,
           accum_dtype,
           source_format='uint',
           num_bits=4,
           scale_size=32,
           fast_dequant=True,
           with_bias=False,
           apply_topk_weights=True,
           block_M=128,
           block_N=256,
           block_K=128,
           num_stages=1,
           threads=512,
           split=1):
    """
    Construct and return a grouped (Mixture-of-Experts) matrix-multiply kernel that multiplies A (shape MxK) by a quantized, expert-grouped B (shape ExNxQK) and writes an output of shape (M, topk, N) in out_dtype.

    The generated kernel accepts:
    - A: dense matrix with element type `in_dtype` and shape (M, K).
    - B: packed quantized matrix for all experts, stored as uint8 with `num_bits` bits per element, shape (E, N, QK), where QK = K / (8/num_bits).
    - Scale: per-expert, per-block scale/exponent information for dequantizing B, shape (E, N, K // scale_size).
    - Bias: per-expert, per-output bias, shape (E, N).
    - topk_weights: router weights for the top-k experts for each token, shape (M, topk).
    - sorted_token_ids: flattened and padded tensor of token indices, shape (padding_M,).
    - expert_ids: expert id for each token in the padded batch, shape (padding_M // block_M,).
    - C: output tensor, shape (M, topk, N).

    The kernel dequantizes B to a working floating format (out_dtype/accum_dtype) using one of two paths:
    - fast_dequant (True): uses an external, hardware/implementation-specific intrinsic group (twiddling) for batch dequantization.
    - fast_dequant (False): uses a simple elementwise dequantization helper.

    Parameters:
        N, K (int): matrix dimensions (A is MxK, result is (M, topk, N)). K must be divisible by (block_K * split).
        topk (int): number of experts selected per token.
        E (int): number of experts.
        padding_M (int): padded number of tokens after grouping and block alignment, must be divisible by block_M.
        in_dtype (str): element type of A (e.g., "bfloat16").
        out_dtype (str): output tensor element type (e.g., "bfloat16").
        accum_dtype (str): accumulation type used for the inner GEMM.
        source_format (str, optional): format string passed to intrinsic selector.
        num_bits (int, optional): number of bits per quantized element in B.
        scale_size (int, optional): number of elements grouped per scale entry.
        fast_dequant (bool, optional): choose the fast intrinsic dequantization path when available.
        apply_bias (bool, optional): whether to apply bias to the output.
        apply_topk_weights (bool, optional): whether to apply topk weights to the output.
        block_M, block_N, block_K (int, optional): tile sizes for M, N, and K dimensions.
        num_stages (int, optional): pipelining stages for K loop.
        threads (int, optional): threads per block used by the kernel.
        split (int, optional): split factor along K used by the scheduler.
        with_bias (bool, optional): whether to add Bias to the output.

    Returns:
        A T.prim_func implementing the grouped, pipelined GEMM that:
        - loads tiled blocks of A and packed B for each expert to shared memory,
        - dequantizes B via the chosen path into a shared dequantized tile,
        - performs a tiled GEMM accumulating into local fragments,
        - applies per-token topk weights and bias,
        - writes the final (M, topk, N) block to the global output tensor.

    Notes:
        - The function queries an intrinsic group to obtain a fast dequantization implementation when fast_dequant is enabled; that intrinsic must supply a valid C source and function name.
        - The kernel layout uses swizzled shared-memory layouts for A, B, and the shared C tile.
        - An assertion enforces that K % (block_K * split) == 0.
    """

    M = tvm.te.var("m")
    padding_M = tvm.te.var("padding_m")

    num_elems_per_byte = 8 // num_bits
    storage_dtype = "uint8"
    QK = K // num_elems_per_byte
    Block_QK = block_K // num_elems_per_byte
    A_shared_shape = (block_M, block_K)
    B_shared_shape = (block_N, Block_QK)
    Bias_shared_shape = (block_N)
    B_dequantize_shared_shape = (block_N, block_K)
    assert K % (block_K * split) == 0

    from tilelang.quantize import get_mxfp_intrin_group
    # fast_dequant_bf16_fp4_twiddling
    mxfp_intrin_info = get_mxfp_intrin_group(
        out_dtype=in_dtype,
        source_format=source_format,
        source_bit=num_bits,
        storage_dtype=storage_dtype,
        use_twiddling=True,
    )
    import_source = mxfp_intrin_info["c_source"]
    func_name = mxfp_intrin_info["func_name"]
    assert import_source is not None, "mxfp_intrin_info is not found"
    assert func_name is not None, "mxfp_intrin_info is not found"
    import_source = import_source

    # the dequant part is the same as in dequant_gemm
    def get_fast_dequant_twiddling_func(in_dtype="fp4", out_dtype="bfloat16"):
        """
        Return a TileLang macro that performs fast dequantization of twiddled FP4-packed data into BF16.
        The returned macro has signature (B_shared, B_dequantize_shared, Scale, k) and:
        - Loads packed FP4 elements from B_shared into per-thread local registers.
        - Calls an external fast dequantization intrinsic (provided via `import_source` / `func_name` in the outer scope) to expand packed FP4 -> BF16 values.
        - Applies a per-block scale factor derived from the Scale tensor (using exponentiation by powers of two).
        - Writes the scaled BF16 results into B_dequantize_shared.

        Notes:
        - This factory only supports in_dtype="fp4" and out_dtype="bfloat16".
        - The macro depends on several names from the enclosing scope (e.g., import_source, func_name, DataType, num_elems_per_byte, storage_dtype, block_N, block_K, threads, scale_size); those must be defined and consistent with the kernel that will use the macro.
        - The macro issues a T.import_source and T.call_extern to invoke the external intrinsic; ensure the external implementation matching `func_name` is available at compilation/runtime.
        """
        assert in_dtype in ["fp4"]
        assert out_dtype in ["bfloat16"]

        # Some variables for dequantization in each thread
        MAX_TRANSACTION_SIZE_BITS = 128
        local_size = MAX_TRANSACTION_SIZE_BITS // 16
        local_compress_size = local_size // num_elems_per_byte

        @T.macro
        def fast_dequant_bf16_fp4_twiddling(B_shared, B_dequantize_shared, Scale_shared, k):
            # import fast_dequantize plugin
            """
            Fast dequantization kernel: convert packed 4-bit quantized values in B_shared to bfloat16
            in B_dequantize_shared using an external intrinsic optimized for twiddled (bit-packed) FP4,
            applying per-block scale factors from Scale.

            This routine is a tiled, thread-parallel helper that:
            - Imports and calls an external dequantization function (via `import_source`/`func_name`)
              to expand compressed uint8-packed FP4 values into BF16 fragments in-thread.
            - Loads the corresponding per-block scale entry, interprets it as an exponent bias
              (applies 2^(Scale - 127)), and multiplies the dequantized BF16 fragment by that factor.
            - Writes the scaled BF16 results back into the shared B_dequantize_shared buffer in-place.

            Parameters:
            - B_shared: read-only shared buffer containing compressed FP4 data (packed uint8 layout).
            - B_dequantize_shared: shared output buffer that is overwritten with BF16 dequantized values.
            - Scale_shared: per-block scale tensor; entries are interpreted such that the multiplicative scale
              = 2^(Scale - 127).
            - k: block index along the K dimension used to select the appropriate Scale entries.

            Side effects:
            - Mutates B_dequantize_shared in shared memory.
            - Calls an external intrinsic function (must be provided by the environment via `import_source`
              and `func_name`) to perform the low-level unpacking/dequantization.
            """
            T.import_source(import_source)

            tx = T.get_thread_binding()

            B_local_thread = T.alloc_local((local_compress_size,), storage_dtype)
            B_dequantize_local_thread = T.alloc_local((local_size,), out_dtype)
            Scale_local_thread = T.alloc_local((1,), storage_dtype)
            Scale_local_thread_exponent = T.alloc_local((1,), out_dtype)

            for i in T.serial(0, block_N * block_K // threads // local_size):
                # First, load data from share memory to register.
                # Prepare for dequant.
                index_base = i * threads * local_compress_size + tx * local_compress_size
                for v in T.vectorized(0, local_compress_size):
                    index = index_base + v
                    B_local_thread[v] = B_shared[index // Block_QK, index % Block_QK]
                index_scale = index_base // (scale_size // num_elems_per_byte)
                si = index_scale // (block_K // scale_size)
                sj = index_scale % (block_K // scale_size)
                Scale_local_thread[0] = Scale_shared[si, k * block_K // scale_size + sj]
                Scale_local_thread_exponent[0] = T.shift_left(1, (Scale_local_thread[0]))

                # Then, dequant.
                T.call_extern(
                    func_name,
                    T.address_of(B_local_thread[0]),
                    T.address_of(B_dequantize_local_thread[0]),
                    1,
                    dtype=out_dtype,
                )

                # Finally, store the dequantized data to shared memory.
                for v in T.Parallel(local_size):
                    B_dequantize_local_thread[v] *= Scale_local_thread_exponent[0]

                for v in T.vectorized(0, local_size):
                    index = i * threads * local_size + tx * local_size + v
                    B_dequantize_shared[index // block_K,
                                        index % block_K] = B_dequantize_local_thread[v]

        return fast_dequant_bf16_fp4_twiddling

    def get_simple_dequant_func(in_dtype="fp4", out_dtype="bfloat16"):

        assert in_dtype in ["fp4"]
        assert out_dtype in ["bfloat16"]

        @T.macro
        def simple_dequant_bf16_fp4(B_shared, B_dequantize_shared, Scale_shared, k):

            B_local = T.alloc_fragment(B_shared_shape, storage_dtype)
            B_dequantize_local = T.alloc_fragment(B_dequantize_shared_shape, out_dtype)

            T.copy(B_shared, B_local)
            for i, j in T.Parallel(block_N, block_K):
                B_dequantize_local[i, j] = _tir_u8_to_f4_to_bf16(
                    num_bits,
                    B_local[i, j // num_elems_per_byte],
                    j % num_elems_per_byte,
                    Scale_shared[
                        i, k * block_K // scale_size + j //
                        scale_size],  # Scale is the exponential part, within the representation of uint8
                    dtype=out_dtype,
                ) * T.shift_left(1, (Scale_shared[i, k * block_K // scale_size + j // scale_size]))
            T.copy(B_dequantize_local, B_dequantize_shared)

        return simple_dequant_bf16_fp4

    @T.prim_func
    def main(
            A: T.Tensor((M, K), in_dtype),
            B: T.Tensor((E, N, QK), storage_dtype),
            Scale: T.Tensor((E, N, K // scale_size), storage_dtype),
            Bias: T.Tensor((E, N), out_dtype),
            # Add fusedmoe tensors
            topk_weights: T.Tensor((M * topk), "float32"),
            sorted_token_ids: T.Tensor((padding_M), "int32"),
            expert_ids: T.Tensor((padding_M // block_M), "int32"),
            C: T.Tensor((M, topk, N), out_dtype),
    ):

        with T.Kernel(
                T.ceildiv(N, block_N), T.ceildiv(padding_M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, storage_dtype)
            B_dequantize_shared = T.alloc_shared(B_dequantize_shared_shape, in_dtype)
            Bias_shared = T.alloc_shared(Bias_shared_shape, out_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_shared = T.alloc_shared((block_M, block_N), out_dtype)
            topk_weights_shared = T.alloc_shared((block_M), "float32")
            sorted_token_ids_shared = T.alloc_shared((block_M), "int32")
            expert_id = T.alloc_local((1), "int32")  # the expert id for the current block
            # To use 1D TMA, the last dim of Scale_shared must have stride=1
            # May use much more shared memory than necessary
            Scale_shared = T.alloc_shared((block_N, K // scale_size), storage_dtype)

            T.annotate_layout({
                A_shared: tilelang.layout.make_swizzled_layout(A_shared),
                B_shared: tilelang.layout.make_swizzled_layout(B_shared),
                C_shared: tilelang.layout.make_swizzled_layout(C_shared),
            })
            T.use_swizzle(10)

            if threads == 512:
                T.disable_warp_group_reg_alloc()

            T.copy(sorted_token_ids[by * block_M:(by + 1) * block_M], sorted_token_ids_shared)
            expert_id[0] = expert_ids[by]

            # Get the topk weights of each token in the current block
            if apply_topk_weights:
                for i in T.Parallel(block_M):
                    if sorted_token_ids_shared[i] != -1:
                        topk_weights_shared[i] = topk_weights[sorted_token_ids_shared[i]]

            # Get bias and scale based on the expert id
            if with_bias:
                T.copy(Bias[expert_id[0], bx * block_N:(bx + 1) * block_N], Bias_shared)
            else:
                T.clear(Bias_shared)

            T.copy(Scale[expert_id[0], bx * block_N:(bx + 1) * block_N, :], Scale_shared)

            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] = Bias_shared[j]

            tx = T.get_thread_binding()

            for k in T.Pipelined(K // block_K, num_stages=num_stages):
                # Each thread copies 4 bytes, local size is 16
                for copy_i in T.serial(block_M * block_K // threads // 16):
                    base = copy_i * threads * 16 + tx * 16
                    if sorted_token_ids_shared[base // block_K] < M * topk:
                        for copy_j in T.vectorized(16):
                            A_shared[base // block_K, base % block_K +
                                     copy_j] = A[sorted_token_ids_shared[base // block_K] // topk,
                                                 k * block_K + base % block_K + copy_j]

                T.copy(B[expert_id[0], bx * block_N, k * block_K // num_elems_per_byte], B_shared)
                if fast_dequant:
                    get_fast_dequant_twiddling_func()(B_shared, B_dequantize_shared, Scale_shared,
                                                      k)
                else:
                    get_simple_dequant_func()(B_shared, B_dequantize_shared, Scale_shared, k)

                T.gemm(A_shared, B_dequantize_shared, C_local, transpose_B=True)

            if apply_topk_weights:
                for i, j in T.Parallel(block_M, block_N):
                    C_local[i, j] = C_local[i, j] * topk_weights_shared[i]

            T.copy(C_local, C_shared)
            for copy_i in T.serial(block_M * block_N // threads // 16):
                base = copy_i * threads * 16 + tx * 16
                if sorted_token_ids_shared[base // block_N] < M * topk:
                    for copy_j in T.vectorized(16):
                        C[sorted_token_ids_shared[base // block_N] // topk,
                          sorted_token_ids_shared[base // block_N] % topk, bx * block_N +
                          base % block_N + copy_j] = C_shared[base // block_N,
                                                              base % block_N + copy_j]

    return main


def fused_tilelang_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    bias1: Optional[torch.Tensor],
    bias2: Optional[torch.Tensor],
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    gating_output: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    quant_type_id: int,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    activation: Optional[str] = "silu",
    expert_map: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    This function computes a Mixture of Experts (MoE) layer using two sets of
    weights, w1 and w2, and top-k gating mechanism.

    Parameters:
    - hidden_states (torch.Tensor): The input tensor to the MoE layer.
    - w1 (torch.Tensor): The first set of expert weights.
    - w2 (torch.Tensor): The second set of expert weights.
    - w1_scale (torch.Tensor): Scale to be used for w1.
    - w2_scale (torch.Tensor): Scale to be used for w2.
    - gating_output (torch.Tensor): The output of the gating operation
        (before softmax).
    - topk_weights (torch.Tensor): Top-k weights.
    - topk_ids (torch.Tensor): Indices of topk-k elements.

    Returns:
    - torch.Tensor: The output tensor after applying the MoE layer.
    """
    quant_type = ScalarType.from_id(quant_type_id)
    assert quant_type == scalar_types.float4_e2m1f
    num_bits = 4

    # model config from gpt-oss-20b
    # hidden_size (K) = 2880, padded to 3072 to be divisible by 256
    # intermediate_size_per_partition (N) = 2880, padded to 2944 to be divisible by 128
    # num_experts (E) = 32
    # topk = 4

    # Check constraints.
    assert hidden_states.shape[0] == gating_output.shape[
        0], "Number of tokens mismatch"
    assert hidden_states.shape[1] // 2 == w1.shape[2], ("Hidden size mismatch")
    assert hidden_states.shape[1] == w2.shape[1], ("Hidden size mismatch w2")
    assert w1.shape[1] == w2.shape[2] * 4, ("shape mismatch w1 and w2")
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert topk_weights.dtype == torch.float32, "Topk weights must be float32"

    M, K = hidden_states.shape
    E = w1.shape[0]
    N = w1.shape[1] // 2
    topk = topk_ids.shape[1]
    
    if global_num_experts == -1:
        global_num_experts = E

    # use heuristic config for now
    block_M, block_N, block_K = 128, 256, 128
    num_stages = 1
    threads = 512
    split = 1

    sorted_token_ids, expert_ids, num_tokens_post_padded = \
        moe_align_block_size(topk_ids, block_M, global_num_experts,
                             expert_map, pad_sorted_ids=True)  # Tilelang grouped gemm required completely padded inputs
    padding_M = sorted_token_ids.shape[0]

    intermediate_cache2 = torch.empty(
        (M * topk, N),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache13 = torch.empty(
        M * topk * max(2 * N, K),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache1 = intermediate_cache13[:M * topk * 2 * N]
    intermediate_cache1 = intermediate_cache1.view(M, topk, 2 * N)
    intermediate_cache3 = intermediate_cache13[:M * topk * K]
    intermediate_cache3 = intermediate_cache3.view(M * topk, 1, K)

    topk_weights = topk_weights.flatten()  # tl kernel requires flattened topk_weights

    fast_dequant = True  # use bits twiddle (requires swizzled quantization)
    with_bias_1 = bias1 is not None
    gemm_kernel_1 = tl_mxfp4_grouped_gemm(
        2 * N, 
        K,
        topk,
        E,
        "bfloat16",
        "bfloat16",
        "float32",
        num_bits=num_bits,
        scale_size=32,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        num_stages=num_stages,
        threads=threads,
        split=split,
        fast_dequant=fast_dequant,
        with_bias=with_bias_1,
        apply_topk_weights=False)
    gemm_kernel_1(
        hidden_states,
        w1,
        w1_scale,
        bias1,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        intermediate_cache1)

    if activation == "silu":
        torch.ops._C.silu_and_mul(intermediate_cache2,
                                  intermediate_cache1.view(-1, 2 * N))
    elif activation == "swigluoai":
        # alpha = 1.702, limit = 7.0
        torch.ops._C.swigluoai_and_mul(intermediate_cache2,
                                       intermediate_cache1.view(-1, 2 * N))
    else:
        raise ValueError(f"Unsupported activation: {activation}. "
                         "Only silu and swigluoai activations are supported.")

    intermediate_cache2 = intermediate_cache2.view(M*topk, N)

    with_bias_2 = bias2 is not None
    gemm_kernel_2 = tl_mxfp4_grouped_gemm(
        K,
        N, 
        1,
        E,
        "bfloat16",
        "bfloat16",
        "float32",
        num_bits=num_bits,
        scale_size=32,  
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        num_stages=num_stages,
        threads=threads,
        split=split,
        fast_dequant=fast_dequant,
        with_bias=with_bias_2,
        apply_topk_weights=True)
    gemm_kernel_2(
        intermediate_cache2,
        w2,
        w2_scale,
        bias2,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        intermediate_cache3,
    )
    output = torch.empty_like(hidden_states)
    return torch.sum(intermediate_cache3.view(M, topk, K),
                     dim=1,
                     out=output)


def fused_tilelang_moe_fake(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    gating_output: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    quant_type_id: int,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    activation: Optional[str] = "silu",
    expert_map: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="fused_tilelang_moe",
    op_func=fused_tilelang_moe,
    mutates_args=[],
    fake_impl=fused_tilelang_moe_fake,
)