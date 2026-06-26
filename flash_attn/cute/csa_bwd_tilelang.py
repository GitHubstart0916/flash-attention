import math
from typing import Optional

import torch
import tilelang
import tilelang.language as T


_LOG2E_CONST = 1.4426950408889634


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def csa_mla_bwd_preprocess_flat(heads, dim):
    batch_size = T.symbolic("batch")
    UQ = T.symbolic("UQ")
    dtype = T.bfloat16
    accum_dtype = T.float32
    shape = [UQ, heads, dim]
    delta_shape = [UQ, heads]
    blk = 32

    @T.prim_func
    def main(
        O_unpad: T.Tensor(shape, dtype),
        dO_unpad: T.Tensor(shape, dtype),
        cu_seqlens_q: T.Tensor([batch_size + 1], T.int32),
        Delta: T.Tensor(delta_shape, accum_dtype),
        max_seqlen_q: T.int32,
    ):
        with T.Kernel(heads, T.ceildiv(max_seqlen_q, blk), batch_size) as (bx, by, bz):
            o = T.alloc_fragment([blk, blk], dtype)
            do = T.alloc_fragment([blk, blk], dtype)
            acc = T.alloc_fragment([blk, blk], accum_dtype)
            delta = T.alloc_fragment([blk], accum_dtype)

            q_start_idx = cu_seqlens_q[bz]
            q_end_idx = cu_seqlens_q[bz + 1]
            q_current_seqlen = q_end_idx - q_start_idx

            T.clear(acc)
            for k in range(T.ceildiv(dim, blk)):
                for i, j in T.Parallel(blk, blk):
                    if by * blk + i < q_current_seqlen and k * blk + j < dim:
                        o[i, j] = O_unpad[q_start_idx + by * blk + i, bx, k * blk + j]
                        do[i, j] = dO_unpad[q_start_idx + by * blk + i, bx, k * blk + j]
                    else:
                        o[i, j] = T.bfloat16(0)
                        do[i, j] = T.bfloat16(0)
                for i, j in T.Parallel(blk, blk):
                    acc[i, j] += o[i, j] * do[i, j]

            T.reduce_sum(acc, delta, 1)
            for i in T.Parallel(blk):
                if by * blk + i < q_current_seqlen:
                    Delta[q_start_idx + by * blk + i, bx] = delta[i]

    return main


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def csa_mla_bwd_dq_flat(
    heads,
    dim,
    window_size,
    block_M=64,
    block_N=64,
    num_stages=2,
    threads=128,
    compress_ratio=4,
):
    batch_size = T.symbolic("batch")
    UQ = T.symbolic("UQ")
    USK = T.symbolic("USK")
    UCK = T.symbolic("UCK")
    sm_scale = (1.0 / dim) ** 0.5
    scale = sm_scale * _LOG2E_CONST
    head_kv = 1
    q_shape = [UQ, heads, dim]
    swa_kv_shape = [USK, head_kv, dim]
    comp_kv_shape = [UCK, head_kv, dim]
    stat_shape = [UQ, heads]
    dtype = T.bfloat16
    accum_dtype = T.float32

    @T.prim_func
    def main(
        Q_unpad: T.Tensor(q_shape, dtype),
        SWA_KV_unpad: T.Tensor(swa_kv_shape, dtype),
        COMP_KV_unpad: T.Tensor(comp_kv_shape, dtype),
        dO_unpad: T.Tensor(q_shape, dtype),
        lse: T.Tensor(stat_shape, accum_dtype),
        Delta: T.Tensor(stat_shape, accum_dtype),
        cu_seqlens_q: T.Tensor([batch_size + 1], T.int32),
        cu_seqlens_swa: T.Tensor([batch_size + 1], T.int32),
        cu_seqlens_comp: T.Tensor([batch_size + 1], T.int32),
        dQ: T.Tensor(q_shape, dtype),
        max_seqlen_q: T.int32,
    ):
        with T.Kernel(heads, T.ceildiv(max_seqlen_q, block_N), batch_size, threads=threads) as (bx, by, bz):
            KV_shared = T.alloc_shared([block_M, dim], dtype)
            q = T.alloc_shared([block_N, dim], dtype)
            do = T.alloc_shared([block_N, dim], dtype)
            dsT_shared = T.alloc_shared([block_M, block_N], dtype)
            qkT = T.alloc_fragment([block_M, block_N], accum_dtype)
            dsT = T.alloc_fragment([block_M, block_N], accum_dtype)
            lse_shared = T.alloc_shared([block_N], accum_dtype)
            delta = T.alloc_shared([block_N], accum_dtype)
            dq = T.alloc_fragment([block_N, dim], accum_dtype)

            head_idx = bx
            kv_head_idx = 0
            q_start_idx = cu_seqlens_q[bz]
            swa_start_idx = cu_seqlens_swa[bz]
            comp_start_idx = cu_seqlens_comp[bz]
            q_end_idx = cu_seqlens_q[bz + 1]
            swa_end_idx = cu_seqlens_swa[bz + 1]
            comp_end_idx = cu_seqlens_comp[bz + 1]
            q_current_seqlen = q_end_idx - q_start_idx
            swa_current_seqlen = swa_end_idx - swa_start_idx
            comp_current_seqlen = comp_end_idx - comp_start_idx
            swa_offset = swa_current_seqlen - q_current_seqlen

            T.copy(
                Q_unpad[
                    q_start_idx + by * block_N:
                    q_start_idx + (by + 1) * block_N,
                    head_idx,
                    :,
                ],
                q,
            )
            T.copy(
                dO_unpad[
                    q_start_idx + by * block_N:
                    q_start_idx + (by + 1) * block_N,
                    head_idx,
                    :,
                ],
                do,
            )
            for i in T.Parallel(block_N):
                if by * block_N + i < q_current_seqlen:
                    lse_shared[i] = lse[q_start_idx + by * block_N + i, head_idx] * _LOG2E_CONST
                    delta[i] = Delta[q_start_idx + by * block_N + i, head_idx]
                else:
                    lse_shared[i] = 0.0
                    delta[i] = 0.0
            T.clear(dq)

            min_swa_visible_k = by * block_N + swa_offset - (window_size - 1)
            max_swa_visible_k = (by + 1) * block_N + swa_offset
            swa_loop_st = T.max(T.floordiv(min_swa_visible_k, block_M), 0)
            swa_loop_ed_raw = T.min(
                T.ceildiv(max_swa_visible_k, block_M),
                T.ceildiv(swa_current_seqlen, block_M),
            )
            swa_loop_ed = T.max(swa_loop_st, swa_loop_ed_raw)
            q_tile_start = by * block_N
            max_comp_visible = T.if_then_else(
                q_tile_start < q_current_seqlen,
                ((by + 1) * block_N + 1) // compress_ratio,
                0,
            )
            comp_loop_ed = T.ceildiv(T.min(max_comp_visible, comp_current_seqlen), block_M)

            total_iters = comp_loop_ed + (swa_loop_ed - swa_loop_st)
            for t in T.Pipelined(0, total_iters, num_stages=num_stages):
                if t < comp_loop_ed:
                    T.copy(
                        COMP_KV_unpad[
                            comp_start_idx + t * block_M:
                            comp_start_idx + (t + 1) * block_M,
                            kv_head_idx,
                            :,
                        ],
                        KV_shared,
                    )
                else:
                    T.copy(
                        SWA_KV_unpad[
                            swa_start_idx + (swa_loop_st + (t - comp_loop_ed)) * block_M:
                            swa_start_idx + (swa_loop_st + (t - comp_loop_ed)) * block_M + block_M,
                            kv_head_idx,
                            :,
                        ],
                        KV_shared,
                    )

                T.clear(qkT)
                T.gemm(KV_shared, q, qkT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                if t < comp_loop_ed:
                    for i, j in T.Parallel(block_M, block_N):
                        qkT[i, j] = T.if_then_else(
                            (t * block_M + i < comp_current_seqlen)
                            and (by * block_N + j < q_current_seqlen)
                            and (t * block_M + i < ((by * block_N + j + 1) // compress_ratio)),
                            T.exp2(qkT[i, j] * scale - lse_shared[j]),
                            0,
                        )
                else:
                    for i, j in T.Parallel(block_M, block_N):
                        qkT[i, j] = T.if_then_else(
                            ((swa_loop_st + (t - comp_loop_ed)) * block_M + i < swa_current_seqlen)
                            and (by * block_N + j < q_current_seqlen)
                            and (
                                by * block_N + j + swa_offset
                                >= (swa_loop_st + (t - comp_loop_ed)) * block_M + i
                            )
                            and (
                                by * block_N + j + swa_offset
                                - ((swa_loop_st + (t - comp_loop_ed)) * block_M + i)
                                < window_size
                            ),
                            T.exp2(qkT[i, j] * scale - lse_shared[j]),
                            0,
                        )

                T.clear(dsT)
                T.gemm(KV_shared, do, dsT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(block_M, block_N):
                    dsT_shared[i, j] = qkT[i, j] * (dsT[i, j] - delta[j]) * sm_scale
                T.gemm(dsT_shared, KV_shared, dq, transpose_A=True)

            for i, d in T.Parallel(block_N, dim):
                if by * block_N + i < q_current_seqlen:
                    dQ[q_start_idx + by * block_N + i, head_idx, d] = dq[i, d]

    return main


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def csa_mla_bwd_dkv_swa_group_reduce_flat(
    heads,
    dim,
    window_size,
    block_M=64,
    block_N=64,
    num_stages=1,
    threads=256,
    heads_per_block=4,
):
    batch_size = T.symbolic("batch")
    UQ = T.symbolic("UQ")
    USK = T.symbolic("USK")
    sm_scale = (1.0 / dim) ** 0.5
    scale = sm_scale * _LOG2E_CONST
    head_kv = 1
    head_blocks = (heads + heads_per_block - 1) // heads_per_block
    q_shape = [UQ, heads, dim]
    kv_shape = [USK, head_kv, dim]
    dkv_partial_shape = [head_blocks, USK, head_kv, dim]
    stat_shape = [UQ, heads]
    dtype = T.bfloat16
    accum_dtype = T.float32

    @T.prim_func
    def main(
        Q_unpad: T.Tensor(q_shape, dtype),
        SWA_KV_unpad: T.Tensor(kv_shape, dtype),
        dO_unpad: T.Tensor(q_shape, dtype),
        lse: T.Tensor(stat_shape, accum_dtype),
        Delta: T.Tensor(stat_shape, accum_dtype),
        cu_seqlens_q: T.Tensor([batch_size + 1], T.int32),
        cu_seqlens_swa: T.Tensor([batch_size + 1], T.int32),
        dSWA_KV: T.Tensor(dkv_partial_shape, accum_dtype),
        max_seqlen_swa: T.int32,
    ):
        with T.Kernel(head_blocks * head_kv, T.ceildiv(max_seqlen_swa, block_M), batch_size, threads=threads) as (bx, by, bz):
            KV_shared = T.alloc_shared([block_M, dim], dtype)
            q = T.alloc_shared([block_N, dim], dtype)
            do = T.alloc_shared([block_N, dim], dtype)
            dsT_shared = T.alloc_shared([block_M, block_N], dtype)
            qkT_cast = T.alloc_shared([block_M, block_N], dtype)
            qkT = T.alloc_fragment([block_M, block_N], accum_dtype)
            dsT = T.alloc_fragment([block_M, block_N], accum_dtype)
            lse_shared = T.alloc_shared([block_N], accum_dtype)
            delta = T.alloc_shared([block_N], accum_dtype)
            dkv = T.alloc_fragment([block_M, dim], accum_dtype)

            head_block_idx = bx % head_blocks
            kv_head_idx = bx // head_blocks
            q_start_idx = cu_seqlens_q[bz]
            swa_start_idx = cu_seqlens_swa[bz]
            q_end_idx = cu_seqlens_q[bz + 1]
            swa_end_idx = cu_seqlens_swa[bz + 1]
            q_current_seqlen = q_end_idx - q_start_idx
            swa_current_seqlen = swa_end_idx - swa_start_idx
            swa_offset = swa_current_seqlen - q_current_seqlen

            for i, d in T.Parallel(block_M, dim):
                if by * block_M + i < swa_current_seqlen:
                    KV_shared[i, d] = SWA_KV_unpad[swa_start_idx + by * block_M + i, kv_head_idx, d]
                else:
                    KV_shared[i, d] = T.bfloat16(0)

            T.clear(dkv)
            loop_st = T.max(T.floordiv(by * block_M - swa_offset, block_N), 0)
            loop_ed = T.min(
                T.ceildiv(q_current_seqlen, block_N),
                T.ceildiv((by + 1) * block_M + window_size - 1 - swa_offset, block_N),
            )

            for head_offset in T.serial(heads_per_block):
                head_idx = head_block_idx * heads_per_block + head_offset
                if head_idx < heads:
                    for k_base in T.Pipelined(loop_st, loop_ed, num_stages=num_stages):
                        for i, d in T.Parallel(block_N, dim):
                            if k_base * block_N + i < q_current_seqlen:
                                q[i, d] = Q_unpad[q_start_idx + k_base * block_N + i, head_idx, d]
                                do[i, d] = dO_unpad[q_start_idx + k_base * block_N + i, head_idx, d]
                            else:
                                q[i, d] = T.bfloat16(0)
                                do[i, d] = T.bfloat16(0)

                        for i in T.Parallel(block_N):
                            if k_base * block_N + i < q_current_seqlen:
                                lse_shared[i] = lse[q_start_idx + k_base * block_N + i, head_idx] * _LOG2E_CONST
                                delta[i] = Delta[q_start_idx + k_base * block_N + i, head_idx]
                            else:
                                lse_shared[i] = 0.0
                                delta[i] = 0.0

                        T.clear(qkT)
                        T.gemm(KV_shared, q, qkT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                        for i, j in T.Parallel(block_M, block_N):
                            qkT[i, j] = T.if_then_else(
                                (by * block_M + i < swa_current_seqlen)
                                and (k_base * block_N + j < q_current_seqlen)
                                and (k_base * block_N + j + swa_offset >= by * block_M + i)
                                and (
                                    k_base * block_N + j + swa_offset - (by * block_M + i)
                                    < window_size
                                ),
                                T.exp2(qkT[i, j] * scale - lse_shared[j]),
                                0,
                            )

                        T.clear(dsT)
                        T.gemm(KV_shared, do, dsT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                        T.copy(qkT, qkT_cast)
                        T.gemm(qkT_cast, do, dkv, policy=T.GemmWarpPolicy.FullRow)

                        for i, j in T.Parallel(block_M, block_N):
                            dsT_shared[i, j] = qkT[i, j] * (dsT[i, j] - delta[j]) * sm_scale
                        T.gemm(dsT_shared, q, dkv, policy=T.GemmWarpPolicy.FullRow)

            for i, d in T.Parallel(block_M, dim):
                if by * block_M + i < swa_current_seqlen:
                    dSWA_KV[head_block_idx, swa_start_idx + by * block_M + i, kv_head_idx, d] = dkv[i, d]

    return main


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def csa_mla_bwd_dkv_comp_group_reduce_flat(
    heads,
    dim,
    block_M=64,
    block_N=64,
    num_stages=1,
    threads=256,
    heads_per_block=4,
    compress_ratio=4,
):
    batch_size = T.symbolic("batch")
    UQ = T.symbolic("UQ")
    UCK = T.symbolic("UCK")
    sm_scale = (1.0 / dim) ** 0.5
    scale = sm_scale * _LOG2E_CONST
    head_kv = 1
    head_blocks = (heads + heads_per_block - 1) // heads_per_block
    q_shape = [UQ, heads, dim]
    kv_shape = [UCK, head_kv, dim]
    dkv_partial_shape = [head_blocks, UCK, head_kv, dim]
    stat_shape = [UQ, heads]
    dtype = T.bfloat16
    accum_dtype = T.float32

    @T.prim_func
    def main(
        Q_unpad: T.Tensor(q_shape, dtype),
        COMP_KV_unpad: T.Tensor(kv_shape, dtype),
        dO_unpad: T.Tensor(q_shape, dtype),
        lse: T.Tensor(stat_shape, accum_dtype),
        Delta: T.Tensor(stat_shape, accum_dtype),
        cu_seqlens_q: T.Tensor([batch_size + 1], T.int32),
        cu_seqlens_comp: T.Tensor([batch_size + 1], T.int32),
        dCOMP_KV: T.Tensor(dkv_partial_shape, accum_dtype),
        max_seqlen_comp: T.int32,
    ):
        with T.Kernel(head_blocks * head_kv, T.ceildiv(max_seqlen_comp, block_M), batch_size, threads=threads) as (bx, by, bz):
            KV_shared = T.alloc_shared([block_M, dim], dtype)
            q = T.alloc_shared([block_N, dim], dtype)
            do = T.alloc_shared([block_N, dim], dtype)
            dsT_shared = T.alloc_shared([block_M, block_N], dtype)
            qkT_cast = T.alloc_shared([block_M, block_N], dtype)
            qkT = T.alloc_fragment([block_M, block_N], accum_dtype)
            dsT = T.alloc_fragment([block_M, block_N], accum_dtype)
            lse_shared = T.alloc_shared([block_N], accum_dtype)
            delta = T.alloc_shared([block_N], accum_dtype)
            dkv = T.alloc_fragment([block_M, dim], accum_dtype)

            head_block_idx = bx % head_blocks
            kv_head_idx = bx // head_blocks
            q_start_idx = cu_seqlens_q[bz]
            comp_start_idx = cu_seqlens_comp[bz]
            q_end_idx = cu_seqlens_q[bz + 1]
            comp_end_idx = cu_seqlens_comp[bz + 1]
            q_current_seqlen = q_end_idx - q_start_idx
            comp_current_seqlen = comp_end_idx - comp_start_idx

            for i, d in T.Parallel(block_M, dim):
                if by * block_M + i < comp_current_seqlen:
                    KV_shared[i, d] = COMP_KV_unpad[comp_start_idx + by * block_M + i, kv_head_idx, d]
                else:
                    KV_shared[i, d] = T.bfloat16(0)

            T.clear(dkv)
            loop_ed = T.ceildiv(q_current_seqlen, block_N)
            first_visible_q = (by * block_M + 1) * compress_ratio - 1
            loop_st_candidate = T.max(T.floordiv(first_visible_q, block_N), 0)
            loop_st = T.if_then_else(
                (by * block_M < comp_current_seqlen) and (comp_current_seqlen > 0),
                T.min(loop_st_candidate, loop_ed),
                loop_ed,
            )

            for head_offset in T.serial(heads_per_block):
                head_idx = head_block_idx * heads_per_block + head_offset
                if head_idx < heads:
                    for k_base in T.Pipelined(loop_st, loop_ed, num_stages=num_stages):
                        for i, d in T.Parallel(block_N, dim):
                            if k_base * block_N + i < q_current_seqlen:
                                q[i, d] = Q_unpad[q_start_idx + k_base * block_N + i, head_idx, d]
                                do[i, d] = dO_unpad[q_start_idx + k_base * block_N + i, head_idx, d]
                            else:
                                q[i, d] = T.bfloat16(0)
                                do[i, d] = T.bfloat16(0)

                        for i in T.Parallel(block_N):
                            if k_base * block_N + i < q_current_seqlen:
                                lse_shared[i] = lse[q_start_idx + k_base * block_N + i, head_idx] * _LOG2E_CONST
                                delta[i] = Delta[q_start_idx + k_base * block_N + i, head_idx]
                            else:
                                lse_shared[i] = 0.0
                                delta[i] = 0.0

                        T.clear(qkT)
                        T.gemm(KV_shared, q, qkT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                        for i, j in T.Parallel(block_M, block_N):
                            qkT[i, j] = T.if_then_else(
                                (by * block_M + i < comp_current_seqlen)
                                and (k_base * block_N + j < q_current_seqlen)
                                and (by * block_M + i < ((k_base * block_N + j + 1) // compress_ratio)),
                                T.exp2(qkT[i, j] * scale - lse_shared[j]),
                                0,
                            )

                        T.clear(dsT)
                        T.gemm(KV_shared, do, dsT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                        T.copy(qkT, qkT_cast)
                        T.gemm(qkT_cast, do, dkv, policy=T.GemmWarpPolicy.FullRow)

                        for i, j in T.Parallel(block_M, block_N):
                            dsT_shared[i, j] = qkT[i, j] * (dsT[i, j] - delta[j]) * sm_scale
                        T.gemm(dsT_shared, q, dkv, policy=T.GemmWarpPolicy.FullRow)

            for i, d in T.Parallel(block_M, dim):
                if by * block_M + i < comp_current_seqlen:
                    dCOMP_KV[head_block_idx, comp_start_idx + by * block_M + i, kv_head_idx, d] = dkv[i, d]

    return main

def csa_mla_varlen_bwd(
    qv: torch.Tensor,
    swa_kv: Optional[torch.Tensor],
    comp_kv: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_swa: torch.Tensor,
    cu_seqlens_comp: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_swa: Optional[int],
    max_seqlen_comp: int,
    swa_window_size: Optional[int],
    learnable_sink: Optional[torch.Tensor],
    compress_ratio: int,
    softmax_scale: Optional[float] = None,
    dlse: Optional[torch.Tensor] = None,
):
    total_q, heads, dim = qv.shape
    expected_scale = 1.0 / math.sqrt(dim)
    if softmax_scale is not None and abs(float(softmax_scale) - expected_scale) > 1e-12:
        raise NotImplementedError("CSA MLA backward currently supports only the default softmax scale")

    if swa_window_size is None:
        swa_window_size = 0
    if max_seqlen_swa is None:
        max_seqlen_swa = 0 if swa_kv is None else max_seqlen_q

    dout = dout.contiguous()
    out = out.contiguous()
    lse = lse.contiguous()

    prep = csa_mla_bwd_preprocess_flat(heads, dim)
    delta = torch.empty(total_q, heads, dtype=torch.float32, device=qv.device)
    prep(out, dout, cu_seqlens_q, delta, max_seqlen_q)
    if dlse is not None:
        delta = delta - dlse.contiguous().float()

    head_kv = comp_kv.shape[1]
    assert head_kv == 1, "CSA MLA backward expects shared MQA KV with one KV head"
    total_comp = comp_kv.shape[0]
    total_swa = 0 if swa_kv is None else swa_kv.shape[0]

    comp_kv_in = comp_kv if total_comp > 0 else comp_kv.new_zeros(1, head_kv, dim)
    swa_kv_in = (
        swa_kv
        if swa_kv is not None and total_swa > 0
        else comp_kv.new_zeros(1, head_kv, dim)
    )

    dq_kernel = csa_mla_bwd_dq_flat(
        heads,
        dim,
        swa_window_size,
        block_M=64,
        block_N=64,
        num_stages=2,
        threads=256,
        compress_ratio=compress_ratio,
    )
    dq = torch.empty(total_q, heads, dim, dtype=qv.dtype, device=qv.device)
    dq_kernel(
        qv,
        swa_kv_in,
        comp_kv_in,
        dout,
        lse,
        delta,
        cu_seqlens_q,
        cu_seqlens_swa,
        cu_seqlens_comp,
        dq,
        max_seqlen_q,
    )

    heads_per_block = 2
    head_blocks = (heads + heads_per_block - 1) // heads_per_block

    if swa_kv is not None:
        if total_swa > 0 and swa_window_size > 0:
            dkv_swa_kernel = csa_mla_bwd_dkv_swa_group_reduce_flat(
                heads,
                dim,
                swa_window_size,
                block_M=64,
                block_N=64,
                num_stages=1,
                threads=512,
                heads_per_block=heads_per_block,
            )
            dswa_partials = torch.empty(
                head_blocks,
                total_swa,
                head_kv,
                dim,
                dtype=torch.float32,
                device=qv.device,
            )
            dkv_swa_kernel(
                qv,
                swa_kv,
                dout,
                lse,
                delta,
                cu_seqlens_q,
                cu_seqlens_swa,
                dswa_partials,
                max_seqlen_swa,
            )
            dswa_kv = dswa_partials.sum(0) if head_blocks > 1 else dswa_partials[0]
        else:
            dswa_kv = torch.zeros_like(swa_kv, dtype=torch.float32)
    else:
        dswa_kv = None

    if total_comp > 0:
        dkv_comp_kernel = csa_mla_bwd_dkv_comp_group_reduce_flat(
            heads,
            dim,
            block_M=64,
            block_N=64,
            num_stages=1,
            threads=512,
            heads_per_block=heads_per_block,
            compress_ratio=compress_ratio,
        )
        dcomp_partials = torch.empty(
            head_blocks,
            total_comp,
            head_kv,
            dim,
            dtype=torch.float32,
            device=qv.device,
        )
        dkv_comp_kernel(
            qv,
            comp_kv,
            dout,
            lse,
            delta,
            cu_seqlens_q,
            cu_seqlens_comp,
            dcomp_partials,
            max_seqlen_comp,
        )
        dcomp_kv = dcomp_partials.sum(0) if head_blocks > 1 else dcomp_partials[0]
    else:
        dcomp_kv = torch.zeros_like(comp_kv, dtype=torch.float32)

    if learnable_sink is not None:
        p_sink = torch.exp(learnable_sink.float().view(1, heads) - lse.float())
        d_sink = -(p_sink * delta).sum(dim=0).to(learnable_sink.dtype)
    else:
        d_sink = None

    return (
        dq.to(qv.dtype),
        None if dswa_kv is None else dswa_kv.to(swa_kv.dtype),
        dcomp_kv.to(comp_kv.dtype),
        d_sink,
    )
