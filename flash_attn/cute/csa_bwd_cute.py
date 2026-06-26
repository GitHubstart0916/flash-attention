import math
import os
from functools import lru_cache
from typing import Optional

import cuda.bindings.driver as cuda
import torch
import triton
import triton.language as tl

import cutlass
import cutlass.cute as cute
from cutlass import Float32
from cutlass.cute.typing import Int32

from flash_attn.cute.cache_utils import get_jit_cache
from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned, to_cute_tensor
from flash_attn.cute import utils
from flash_attn.cute.sm100_hd256_2cta_fmha_backward import (
    _as_bshkrd_tensor,
    _as_shhb_tensor,
)
from flash_attn.cute.sm100_hd256_2cta_fmha_backward_dqkernel import (
    BlackwellFusedMultiHeadAttentionBackwardDQKernel,
)
from flash_attn.cute.sm100_hd256_2cta_fmha_backward_dkdvkernel import (
    BlackwellFusedMultiHeadAttentionBackwardDKDVKernel,
)
from flash_attn.cute.testing import is_fake_mode


class CsaSwaBackwardDQSm100:
    def __init__(self, head_dim: int, window_size: int):
        assert head_dim == 512, "CSA SWA dQ native path is specialized for D=512"
        assert window_size > 0
        self.acc_dtype = cutlass.Float32
        self.dq_kernel = BlackwellFusedMultiHeadAttentionBackwardDQKernel(
            self.acc_dtype,
            (64, 64, head_dim),
            False,  # is_causal
            window_size - 1,
            0,
            False,  # is_persistent
            True,  # split_head
            use_clc_scheduler=False,
        )

    @cute.jit
    def __call__(
        self,
        Q: cute.Tensor,
        K: cute.Tensor,
        V: cute.Tensor,
        dQ: cute.Tensor,
        dO: cute.Tensor,
        lse_log2: cute.Tensor,
        dpsum: cute.Tensor,
        cumulative_s_q: cute.Tensor,
        cumulative_s_k: cute.Tensor,
        scale_softmax: Float32,
        stream: cuda.CUstream = None,
    ):
        varlen = cumulative_s_q is not None or cumulative_s_k is not None
        q_rank = cute.rank(Q.layout)
        k_rank = cute.rank(K.layout)
        if cutlass.const_expr(q_rank == 5):
            h_q = Q.shape[2] * Q.shape[3]
        elif cutlass.const_expr(q_rank == 4):
            h_q = Q.shape[2]
        else:
            h_q = Q.shape[1]
        if cutlass.const_expr(k_rank == 5):
            h_k = K.shape[2]
        elif cutlass.const_expr(k_rank == 4):
            h_k = K.shape[2]
        else:
            h_k = K.shape[1]
        h_r = h_q // h_k
        if cutlass.const_expr(cumulative_s_q is not None):
            b = cumulative_s_q.shape[0] - 1
        elif cutlass.const_expr(cumulative_s_k is not None):
            b = cumulative_s_k.shape[0] - 1
        else:
            b = Q.shape[0]

        Q, K, V, dQ, dO = [assume_tensor_aligned(t) for t in (Q, K, V, dQ, dO)]
        Q = _as_bshkrd_tensor(Q, h_k, h_r, varlen)
        K = _as_bshkrd_tensor(K, h_k, 1, varlen)
        V = _as_bshkrd_tensor(V, h_k, 1, varlen)
        dQ = _as_bshkrd_tensor(dQ, h_k, h_r, varlen)
        dO = _as_bshkrd_tensor(dO, h_k, h_r, varlen)
        scaled_LSE = _as_shhb_tensor(lse_log2, h_k, h_r, b, varlen)
        sum_OdO = _as_shhb_tensor(dpsum, h_k, h_r, b, varlen)

        self.dq_kernel(
            Q,
            K,
            V,
            dQ,
            dO,
            scaled_LSE,
            sum_OdO,
            cumulative_s_q,
            cumulative_s_k,
            scale_softmax,
            stream,
        )


def csa_swa_dq_sm100(
    qv: torch.Tensor,
    swa_kv: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    delta: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_swa: torch.Tensor,
    window_size: int,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    if window_size <= 0 or swa_kv is None or swa_kv.numel() == 0:
        return torch.zeros_like(qv)
    total_q, heads, dim = qv.shape
    assert dim == 512, "CSA SWA dQ native path is specialized for D=512"
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(dim)

    lse_log2 = (lse.float() * math.log2(math.e)).transpose(0, 1).contiguous()
    dpsum = delta.float().transpose(0, 1).contiguous()
    dq = torch.empty_like(qv)
    current_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    compile_key = (
        qv.dtype,
        heads,
        dim,
        window_size,
        cu_seqlens_q is None,
        cu_seqlens_swa is None,
        qv.shape[0],
        swa_kv.shape[0],
    )
    if compile_key not in csa_swa_dq_sm100.compile_cache:
        fa_dq = CsaSwaBackwardDQSm100(dim, window_size)
        csa_swa_dq_sm100.compile_cache[compile_key] = cute.compile(
            fa_dq,
            to_cute_tensor(qv),
            to_cute_tensor(swa_kv),
            to_cute_tensor(swa_kv),
            to_cute_tensor(dq),
            to_cute_tensor(dout),
            to_cute_tensor(lse_log2, assumed_align=4),
            to_cute_tensor(dpsum, assumed_align=4),
            to_cute_tensor(cu_seqlens_q, assumed_align=4),
            to_cute_tensor(cu_seqlens_swa, assumed_align=4),
            softmax_scale,
            current_stream,
            options="--enable-tvm-ffi",
        )
    if not is_fake_mode():
        csa_swa_dq_sm100.compile_cache[compile_key](
            qv.detach(),
            swa_kv.detach(),
            swa_kv.detach(),
            dq,
            dout,
            lse_log2,
            dpsum,
            cu_seqlens_q,
            cu_seqlens_swa,
            softmax_scale,
        )
    return dq


csa_swa_dq_sm100.compile_cache = get_jit_cache("csa_swa_dq_sm100")


class CsaCompBackwardDQSm100:
    def __init__(self, head_dim: int, compress_ratio: int, use_clc_scheduler: bool = False):
        assert head_dim == 512, "CSA comp dQ native path is specialized for D=512"
        assert compress_ratio > 0
        self.acc_dtype = cutlass.Float32
        self.dq_kernel = BlackwellFusedMultiHeadAttentionBackwardDQKernel(
            self.acc_dtype,
            (64, 64, head_dim),
            False,  # is_causal
            None,
            None,
            False,  # is_persistent
            True,  # split_head
            use_clc_scheduler=use_clc_scheduler,
            csa_compress_ratio=compress_ratio,
        )

    @cute.jit
    def __call__(
        self,
        Q: cute.Tensor,
        K: cute.Tensor,
        V: cute.Tensor,
        dQ: cute.Tensor,
        dO: cute.Tensor,
        lse_log2: cute.Tensor,
        dpsum: cute.Tensor,
        cumulative_s_q: cute.Tensor,
        cumulative_s_k: cute.Tensor,
        scale_softmax: Float32,
        stream: cuda.CUstream = None,
    ):
        varlen = cumulative_s_q is not None or cumulative_s_k is not None
        q_rank = cute.rank(Q.layout)
        k_rank = cute.rank(K.layout)
        if cutlass.const_expr(q_rank == 5):
            h_q = Q.shape[2] * Q.shape[3]
        elif cutlass.const_expr(q_rank == 4):
            h_q = Q.shape[2]
        else:
            h_q = Q.shape[1]
        if cutlass.const_expr(k_rank == 5):
            h_k = K.shape[2]
        elif cutlass.const_expr(k_rank == 4):
            h_k = K.shape[2]
        else:
            h_k = K.shape[1]
        h_r = h_q // h_k
        if cutlass.const_expr(cumulative_s_q is not None):
            b = cumulative_s_q.shape[0] - 1
        elif cutlass.const_expr(cumulative_s_k is not None):
            b = cumulative_s_k.shape[0] - 1
        else:
            b = Q.shape[0]

        Q, K, V, dQ, dO = [assume_tensor_aligned(t) for t in (Q, K, V, dQ, dO)]
        Q = _as_bshkrd_tensor(Q, h_k, h_r, varlen)
        K = _as_bshkrd_tensor(K, h_k, 1, varlen)
        V = _as_bshkrd_tensor(V, h_k, 1, varlen)
        dQ = _as_bshkrd_tensor(dQ, h_k, h_r, varlen)
        dO = _as_bshkrd_tensor(dO, h_k, h_r, varlen)
        scaled_LSE = _as_shhb_tensor(lse_log2, h_k, h_r, b, varlen)
        sum_OdO = _as_shhb_tensor(dpsum, h_k, h_r, b, varlen)

        self.dq_kernel(
            Q,
            K,
            V,
            dQ,
            dO,
            scaled_LSE,
            sum_OdO,
            cumulative_s_q,
            cumulative_s_k,
            scale_softmax,
            stream,
        )


def csa_comp_dq_sm100(
    qv: torch.Tensor,
    comp_kv: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    delta: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_comp: torch.Tensor,
    compress_ratio: int,
    softmax_scale: float | None = None,
    use_clc_scheduler: bool = False,
) -> torch.Tensor:
    if comp_kv is None or comp_kv.numel() == 0:
        return torch.zeros_like(qv)
    total_q, heads, dim = qv.shape
    assert dim == 512, "CSA comp dQ native path is specialized for D=512"
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(dim)

    lse_log2 = (lse.float() * math.log2(math.e)).transpose(0, 1).contiguous()
    dpsum = delta.float().transpose(0, 1).contiguous()
    dq = torch.empty_like(qv)
    current_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    compile_key = (
        qv.dtype,
        heads,
        dim,
        compress_ratio,
        use_clc_scheduler,
        cu_seqlens_q is None,
        cu_seqlens_comp is None,
        qv.shape[0],
        comp_kv.shape[0],
    )
    if compile_key not in csa_comp_dq_sm100.compile_cache:
        fa_dq = CsaCompBackwardDQSm100(
            dim,
            compress_ratio,
            use_clc_scheduler=use_clc_scheduler,
        )
        csa_comp_dq_sm100.compile_cache[compile_key] = cute.compile(
            fa_dq,
            to_cute_tensor(qv),
            to_cute_tensor(comp_kv),
            to_cute_tensor(comp_kv),
            to_cute_tensor(dq),
            to_cute_tensor(dout),
            to_cute_tensor(lse_log2, assumed_align=4),
            to_cute_tensor(dpsum, assumed_align=4),
            to_cute_tensor(cu_seqlens_q, assumed_align=4),
            to_cute_tensor(cu_seqlens_comp, assumed_align=4),
            softmax_scale,
            current_stream,
            options="--enable-tvm-ffi",
        )
    if not is_fake_mode():
        csa_comp_dq_sm100.compile_cache[compile_key](
            qv.detach(),
            comp_kv.detach(),
            comp_kv.detach(),
            dq,
            dout,
            lse_log2,
            dpsum,
            cu_seqlens_q,
            cu_seqlens_comp,
            softmax_scale,
        )
    return dq


csa_comp_dq_sm100.compile_cache = get_jit_cache("csa_comp_dq_sm100")


class CsaSwaBackwardDKDVSm100:
    def __init__(self, head_dim: int, head_dim_v: int, window_size: int):
        assert head_dim == 512, "CSA SWA dKV native path is specialized for Q/K D=512"
        assert head_dim_v == 256, "CSA SWA dKV native path splits V/dO into 256-wide halves"
        assert window_size > 0
        self.acc_dtype = cutlass.Float32
        self.dkdv_kernel = BlackwellFusedMultiHeadAttentionBackwardDKDVKernel(
            self.acc_dtype,
            (64, 64, head_dim, head_dim_v, head_dim_v),
            False,  # is_causal
            window_size - 1,
            0,
            use_clc_scheduler=False,
        )

    @cute.jit
    def __call__(
        self,
        Q: cute.Tensor,
        K: cute.Tensor,
        V: cute.Tensor,
        Q_DK: cute.Tensor,
        dK: cute.Tensor,
        dV: cute.Tensor,
        dO: cute.Tensor,
        lse_log2: cute.Tensor,
        dpsum: cute.Tensor,
        cumulative_s_q: cute.Tensor,
        cumulative_s_k: cute.Tensor,
        scale_softmax: Float32,
        dO_DV: cute.Tensor,
        stream: cuda.CUstream = None,
    ):
        varlen = cumulative_s_q is not None or cumulative_s_k is not None
        q_rank = cute.rank(Q.layout)
        k_rank = cute.rank(K.layout)
        if cutlass.const_expr(q_rank == 5):
            h_q = Q.shape[2] * Q.shape[3]
        elif cutlass.const_expr(q_rank == 4):
            h_q = Q.shape[2]
        else:
            h_q = Q.shape[1]
        if cutlass.const_expr(k_rank == 5):
            h_k = K.shape[2]
        elif cutlass.const_expr(k_rank == 4):
            h_k = K.shape[2]
        else:
            h_k = K.shape[1]
        h_r = h_q // h_k
        if cutlass.const_expr(cumulative_s_q is not None):
            b = cumulative_s_q.shape[0] - 1
        elif cutlass.const_expr(cumulative_s_k is not None):
            b = cumulative_s_k.shape[0] - 1
        else:
            b = Q.shape[0]

        Q, K, V, Q_DK, dK, dV, dO, dO_DV = [
            assume_tensor_aligned(t) for t in (Q, K, V, Q_DK, dK, dV, dO, dO_DV)
        ]
        Q = _as_bshkrd_tensor(Q, h_k, h_r, varlen)
        K = _as_bshkrd_tensor(K, h_k, 1, varlen)
        V = _as_bshkrd_tensor(V, h_k, 1, varlen)
        Q_DK = _as_bshkrd_tensor(Q_DK, h_k, h_r, varlen)
        dK = _as_bshkrd_tensor(dK, h_k, 1, varlen)
        dV = _as_bshkrd_tensor(dV, h_k, 1, varlen)
        dO = _as_bshkrd_tensor(dO, h_k, h_r, varlen)
        dO_DV = _as_bshkrd_tensor(dO_DV, h_k, h_r, varlen)
        scaled_LSE = _as_shhb_tensor(lse_log2, h_k, h_r, b, varlen)
        sum_OdO = _as_shhb_tensor(dpsum, h_k, h_r, b, varlen)

        self.dkdv_kernel(
            Q,
            K,
            V,
            dK,
            dV,
            dO,
            scaled_LSE,
            sum_OdO,
            cumulative_s_q,
            cumulative_s_k,
            scale_softmax,
            dO_DV,
            stream,
            Q_DK,
        )


class CsaCompBackwardDKDVSm100:
    def __init__(
        self,
        head_dim: int,
        head_dim_dk: int,
        head_dim_v: int,
        compress_ratio: int,
        head_dim_dp: int | None = None,
        use_clc_scheduler: bool = True,
    ):
        assert head_dim == 512, "CSA comp dKV native path is specialized for Q/K D=512"
        assert head_dim_dk in (256, 512), "CSA comp dKV native path supports dK tiles of 256/512"
        assert head_dim_v in (256, 512), "CSA comp dKV native path supports V/dO tiles of 256/512"
        if head_dim_dp is None:
            head_dim_dp = head_dim_v
        assert head_dim_dp in (256, 512), "CSA comp dKV native path supports dP tiles of 256/512"
        assert compress_ratio > 0
        self.acc_dtype = cutlass.Float32
        self.dkdv_kernel = BlackwellFusedMultiHeadAttentionBackwardDKDVKernel(
            self.acc_dtype,
            (64, 64, head_dim, head_dim_dk, head_dim_v, head_dim_dp),
            False,  # is_causal
            None,
            None,
            use_clc_scheduler=use_clc_scheduler,
            csa_compress_ratio=compress_ratio,
        )

    @cute.jit
    def __call__(
        self,
        Q: cute.Tensor,
        K: cute.Tensor,
        V: cute.Tensor,
        Q_DK: cute.Tensor,
        dK: cute.Tensor,
        dV: cute.Tensor,
        dO: cute.Tensor,
        lse_log2: cute.Tensor,
        dpsum: cute.Tensor,
        cumulative_s_q: cute.Tensor,
        cumulative_s_k: cute.Tensor,
        scale_softmax: Float32,
        dO_DV: cute.Tensor,
        stream: cuda.CUstream = None,
    ):
        varlen = cumulative_s_q is not None or cumulative_s_k is not None
        q_rank = cute.rank(Q.layout)
        k_rank = cute.rank(K.layout)
        if cutlass.const_expr(q_rank == 5):
            h_q = Q.shape[2] * Q.shape[3]
        elif cutlass.const_expr(q_rank == 4):
            h_q = Q.shape[2]
        else:
            h_q = Q.shape[1]
        if cutlass.const_expr(k_rank == 5):
            h_k = K.shape[2]
        elif cutlass.const_expr(k_rank == 4):
            h_k = K.shape[2]
        else:
            h_k = K.shape[1]
        h_r = h_q // h_k
        if cutlass.const_expr(cumulative_s_q is not None):
            b = cumulative_s_q.shape[0] - 1
        elif cutlass.const_expr(cumulative_s_k is not None):
            b = cumulative_s_k.shape[0] - 1
        else:
            b = Q.shape[0]

        Q, K, V, Q_DK, dK, dV, dO, dO_DV = [
            assume_tensor_aligned(t) for t in (Q, K, V, Q_DK, dK, dV, dO, dO_DV)
        ]
        Q = _as_bshkrd_tensor(Q, h_k, h_r, varlen)
        K = _as_bshkrd_tensor(K, h_k, 1, varlen)
        V = _as_bshkrd_tensor(V, h_k, 1, varlen)
        Q_DK = _as_bshkrd_tensor(Q_DK, h_k, h_r, varlen)
        dK = _as_bshkrd_tensor(dK, h_k, 1, varlen)
        dV = _as_bshkrd_tensor(dV, h_k, 1, varlen)
        dO = _as_bshkrd_tensor(dO, h_k, h_r, varlen)
        dO_DV = _as_bshkrd_tensor(dO_DV, h_k, h_r, varlen)
        scaled_LSE = _as_shhb_tensor(lse_log2, h_k, h_r, b, varlen)
        sum_OdO = _as_shhb_tensor(dpsum, h_k, h_r, b, varlen)

        self.dkdv_kernel(
            Q,
            K,
            V,
            dK,
            dV,
            dO,
            scaled_LSE,
            sum_OdO,
            cumulative_s_q,
            cumulative_s_k,
            scale_softmax,
            dO_DV,
            stream,
            Q_DK,
        )


def csa_swa_dkv_sm100(
    qv: torch.Tensor,
    swa_kv: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    delta: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_swa: torch.Tensor,
    window_size: int,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    if window_size <= 0 or swa_kv is None or swa_kv.numel() == 0:
        return torch.zeros_like(swa_kv)
    total_q, heads, dim = qv.shape
    assert dim == 512, "CSA SWA dKV native path is specialized for D=512"
    head_dim_v = dim // 2
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(dim)

    lse_log2 = (lse.float() * math.log2(math.e)).transpose(0, 1).contiguous()
    delta0 = (out[..., :head_dim_v].float() * dout[..., :head_dim_v].float()).sum(-1)
    delta1 = (out[..., head_dim_v:].float() * dout[..., head_dim_v:].float()).sum(-1)
    delta_correction = delta.float() - delta0 - delta1
    delta0 = delta0 + delta_correction
    dpsum0 = delta0.float().transpose(0, 1).contiguous()
    dpsum1 = delta1.float().transpose(0, 1).contiguous()
    dK00 = torch.empty((swa_kv.shape[0], swa_kv.shape[1], head_dim_v), device=swa_kv.device, dtype=swa_kv.dtype)
    dK01 = torch.empty_like(dK00)
    dK10 = torch.empty_like(dK00)
    dK11 = torch.empty_like(dK00)
    dV0 = torch.empty((swa_kv.shape[0], swa_kv.shape[1], head_dim_v), device=swa_kv.device, dtype=swa_kv.dtype)
    dV1 = torch.empty_like(dV0)
    dV_dummy = torch.empty_like(dV0)
    q0 = qv[..., :head_dim_v]
    q1 = qv[..., head_dim_v:]
    v0 = swa_kv[..., :head_dim_v]
    v1 = swa_kv[..., head_dim_v:]
    do0 = dout[..., :head_dim_v]
    do1 = dout[..., head_dim_v:]
    current_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    compile_key = (
        qv.dtype,
        heads,
        dim,
        head_dim_v,
        window_size,
        qv.shape[0],
        swa_kv.shape[0],
    )
    if compile_key not in csa_swa_dkv_sm100.compile_cache:
        fa_dkdv = CsaSwaBackwardDKDVSm100(dim, head_dim_v, window_size)
        csa_swa_dkv_sm100.compile_cache[compile_key] = cute.compile(
            fa_dkdv,
            to_cute_tensor(qv),
            to_cute_tensor(swa_kv),
            to_cute_tensor(v0),
            to_cute_tensor(q0),
            to_cute_tensor(dK00),
            to_cute_tensor(dV0),
            to_cute_tensor(do0),
            to_cute_tensor(lse_log2, assumed_align=4),
            to_cute_tensor(dpsum0, assumed_align=4),
            to_cute_tensor(cu_seqlens_q, assumed_align=4),
            to_cute_tensor(cu_seqlens_swa, assumed_align=4),
            softmax_scale,
            to_cute_tensor(do0),
            current_stream,
            options="--enable-tvm-ffi",
        )
    if not is_fake_mode():
        compiled = csa_swa_dkv_sm100.compile_cache[compile_key]
        compiled(
            qv.detach(),
            swa_kv.detach(),
            v0.detach(),
            q0.detach(),
            dK00,
            dV0,
            do0,
            lse_log2,
            dpsum0,
            cu_seqlens_q,
            cu_seqlens_swa,
            softmax_scale,
            do0,
        )
        compiled(
            qv.detach(),
            swa_kv.detach(),
            v1.detach(),
            q0.detach(),
            dK10,
            dV1,
            do1,
            lse_log2,
            dpsum1,
            cu_seqlens_q,
            cu_seqlens_swa,
            softmax_scale,
            do1,
        )
        compiled(
            qv.detach(),
            swa_kv.detach(),
            v0.detach(),
            q1.detach(),
            dK01,
            dV_dummy,
            do0,
            lse_log2,
            dpsum0,
            cu_seqlens_q,
            cu_seqlens_swa,
            softmax_scale,
            do0,
        )
        compiled(
            qv.detach(),
            swa_kv.detach(),
            v1.detach(),
            q1.detach(),
            dK11,
            dV_dummy,
            do1,
            lse_log2,
            dpsum1,
            cu_seqlens_q,
            cu_seqlens_swa,
            softmax_scale,
            do1,
        )
    dV = torch.empty_like(swa_kv)
    dV[..., :head_dim_v] = dV0
    dV[..., head_dim_v:] = dV1
    dK = torch.empty_like(swa_kv)
    dK[..., :head_dim_v] = (dK00.float() + dK10.float()).to(swa_kv.dtype)
    dK[..., head_dim_v:] = (dK01.float() + dK11.float()).to(swa_kv.dtype)
    return (dK + dV.float()).to(swa_kv.dtype)


csa_swa_dkv_sm100.compile_cache = get_jit_cache("csa_swa_dkv_sm100")


def csa_comp_dkv_sm100(
    qv: torch.Tensor,
    comp_kv: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    delta: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_comp: torch.Tensor,
    compress_ratio: int,
    softmax_scale: float | None = None,
    use_clc_scheduler: bool = False,
) -> torch.Tensor:
    if comp_kv is None or comp_kv.numel() == 0:
        return torch.zeros_like(comp_kv)
    total_q, heads, dim = qv.shape
    assert dim == 512, "CSA comp dKV native path is specialized for D=512"
    head_dim_v = dim // 2
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(dim)

    if (
        os.environ.get("CSA_COMP_DKV_USE_SM100_SMALL") == "1"
        and compress_ratio == 4
        and cu_seqlens_q.numel() == 2
        and cu_seqlens_comp.numel() == 2
        and comp_kv.shape[1] == 1
        and qv.shape[0] >= 256
        and comp_kv.shape[0] >= 64
        and qv.shape[0] % 64 == 0
        and comp_kv.shape[0] % 64 == 0
    ):
        from flash_attn.cute.csa_comp_dkv_sm100_small import csa_comp_dkv_sm100_small

        return csa_comp_dkv_sm100_small(
            qv,
            comp_kv,
            out,
            dout,
            lse,
            delta,
            compress_ratio,
            softmax_scale,
        )

    lse_log2 = (lse.float() * math.log2(math.e)).transpose(0, 1).contiguous()
    q0 = qv[..., :head_dim_v]
    q1 = qv[..., head_dim_v:]
    current_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    if os.environ.get("CSA_COMP_DKV_SM100_DK512") == "1":
        delta0 = (out[..., :head_dim_v].float() * dout[..., :head_dim_v].float()).sum(-1)
        delta1 = (out[..., head_dim_v:].float() * dout[..., head_dim_v:].float()).sum(-1)
        delta_correction = delta.float() - delta0 - delta1
        delta0 = delta0 + delta_correction
        dpsum0 = delta0.float().transpose(0, 1).contiguous()
        dpsum1 = delta1.float().transpose(0, 1).contiguous()
        v0 = comp_kv[..., :head_dim_v]
        v1 = comp_kv[..., head_dim_v:]
        do0 = dout[..., :head_dim_v]
        do1 = dout[..., head_dim_v:]
        dK0 = torch.empty_like(comp_kv)
        dK1 = torch.empty_like(comp_kv)
        dV0 = torch.empty((comp_kv.shape[0], comp_kv.shape[1], head_dim_v), device=comp_kv.device, dtype=comp_kv.dtype)
        dV1 = torch.empty_like(dV0)
        compile_key = (
            "dk512",
            qv.dtype,
            heads,
            dim,
            head_dim_v,
            compress_ratio,
            use_clc_scheduler,
            qv.shape[0],
            comp_kv.shape[0],
        )
        if compile_key not in csa_comp_dkv_sm100.compile_cache:
            fa_dkdv = CsaCompBackwardDKDVSm100(
                dim,
                dim,
                head_dim_v,
                compress_ratio,
                use_clc_scheduler=use_clc_scheduler,
            )
            csa_comp_dkv_sm100.compile_cache[compile_key] = cute.compile(
                fa_dkdv,
                to_cute_tensor(qv),
                to_cute_tensor(comp_kv),
                to_cute_tensor(v0),
                to_cute_tensor(qv),
                to_cute_tensor(dK0),
                to_cute_tensor(dV0),
                to_cute_tensor(do0),
                to_cute_tensor(lse_log2, assumed_align=4),
                to_cute_tensor(dpsum0, assumed_align=4),
                to_cute_tensor(cu_seqlens_q, assumed_align=4),
                to_cute_tensor(cu_seqlens_comp, assumed_align=4),
                softmax_scale,
                to_cute_tensor(do0),
                current_stream,
                options="--enable-tvm-ffi",
            )
        if not is_fake_mode():
            compiled = csa_comp_dkv_sm100.compile_cache[compile_key]
            compiled(
                qv.detach(),
                comp_kv.detach(),
                v0.detach(),
                qv.detach(),
                dK0,
                dV0,
                do0,
                lse_log2,
                dpsum0,
                cu_seqlens_q,
                cu_seqlens_comp,
                softmax_scale,
                do0,
            )
            compiled(
                qv.detach(),
                comp_kv.detach(),
                v1.detach(),
                qv.detach(),
                dK1,
                dV1,
                do1,
                lse_log2,
                dpsum1,
                cu_seqlens_q,
                cu_seqlens_comp,
                softmax_scale,
                do1,
            )
        dV = torch.empty_like(comp_kv)
        dV[..., :head_dim_v] = dV0
        dV[..., head_dim_v:] = dV1
        return (dK0.float() + dK1.float() + dV.float()).to(comp_kv.dtype)

    if os.environ.get("CSA_COMP_DKV_SM100_FULL_V") == "1":
        dpsum = delta.float().transpose(0, 1).contiguous()
        dK0 = torch.empty(
            (comp_kv.shape[0], comp_kv.shape[1], head_dim_v),
            device=comp_kv.device,
            dtype=comp_kv.dtype,
        )
        dK1 = torch.empty_like(dK0)
        dV0 = torch.empty_like(dK0)
        dV1 = torch.empty_like(dK0)
        do0 = dout[..., :head_dim_v]
        do1 = dout[..., head_dim_v:]
        compile_key = (
            "full_dp",
            qv.dtype,
            heads,
            dim,
            head_dim_v,
            compress_ratio,
            use_clc_scheduler,
            qv.shape[0],
            comp_kv.shape[0],
        )
        if compile_key not in csa_comp_dkv_sm100.compile_cache:
            fa_dkdv = CsaCompBackwardDKDVSm100(
                dim,
                head_dim_v,
                head_dim_v,
                compress_ratio,
                head_dim_dp=dim,
                use_clc_scheduler=use_clc_scheduler,
            )
            csa_comp_dkv_sm100.compile_cache[compile_key] = cute.compile(
                fa_dkdv,
                to_cute_tensor(qv),
                to_cute_tensor(comp_kv),
                to_cute_tensor(comp_kv),
                to_cute_tensor(q0),
                to_cute_tensor(dK0),
                to_cute_tensor(dV0),
                to_cute_tensor(dout),
                to_cute_tensor(lse_log2, assumed_align=4),
                to_cute_tensor(dpsum, assumed_align=4),
                to_cute_tensor(cu_seqlens_q, assumed_align=4),
                to_cute_tensor(cu_seqlens_comp, assumed_align=4),
                softmax_scale,
                to_cute_tensor(do0),
                current_stream,
                options="--enable-tvm-ffi",
            )
        if not is_fake_mode():
            compiled = csa_comp_dkv_sm100.compile_cache[compile_key]
            compiled(
                qv.detach(),
                comp_kv.detach(),
                comp_kv.detach(),
                q0.detach(),
                dK0,
                dV0,
                dout.detach(),
                lse_log2,
                dpsum,
                cu_seqlens_q,
                cu_seqlens_comp,
                softmax_scale,
                do0,
            )
            compiled(
                qv.detach(),
                comp_kv.detach(),
                comp_kv.detach(),
                q1.detach(),
                dK1,
                dV1,
                dout.detach(),
                lse_log2,
                dpsum,
                cu_seqlens_q,
                cu_seqlens_comp,
                softmax_scale,
                do1,
            )
        dK = torch.empty_like(comp_kv)
        dK[..., :head_dim_v] = dK0
        dK[..., head_dim_v:] = dK1
        dV = torch.empty_like(comp_kv)
        dV[..., :head_dim_v] = dV0
        dV[..., head_dim_v:] = dV1
        return (dK.float() + dV.float()).to(comp_kv.dtype)

    delta0 = (out[..., :head_dim_v].float() * dout[..., :head_dim_v].float()).sum(-1)
    delta1 = (out[..., head_dim_v:].float() * dout[..., head_dim_v:].float()).sum(-1)
    delta_correction = delta.float() - delta0 - delta1
    delta0 = delta0 + delta_correction
    dpsum0 = delta0.float().transpose(0, 1).contiguous()
    dpsum1 = delta1.float().transpose(0, 1).contiguous()
    dK00 = torch.empty((comp_kv.shape[0], comp_kv.shape[1], head_dim_v), device=comp_kv.device, dtype=comp_kv.dtype)
    dK01 = torch.empty_like(dK00)
    dK10 = torch.empty_like(dK00)
    dK11 = torch.empty_like(dK00)
    dV0 = torch.empty((comp_kv.shape[0], comp_kv.shape[1], head_dim_v), device=comp_kv.device, dtype=comp_kv.dtype)
    dV1 = torch.empty_like(dV0)
    dV_dummy = torch.empty_like(dV0)
    v0 = comp_kv[..., :head_dim_v]
    v1 = comp_kv[..., head_dim_v:]
    do0 = dout[..., :head_dim_v]
    do1 = dout[..., head_dim_v:]

    compile_key = (
        qv.dtype,
        heads,
        dim,
        head_dim_v,
        compress_ratio,
        use_clc_scheduler,
        qv.shape[0],
        comp_kv.shape[0],
    )
    if compile_key not in csa_comp_dkv_sm100.compile_cache:
        fa_dkdv = CsaCompBackwardDKDVSm100(
            dim,
            head_dim_v,
            head_dim_v,
            compress_ratio,
            use_clc_scheduler=use_clc_scheduler,
        )
        csa_comp_dkv_sm100.compile_cache[compile_key] = cute.compile(
            fa_dkdv,
            to_cute_tensor(qv),
            to_cute_tensor(comp_kv),
            to_cute_tensor(v0),
            to_cute_tensor(q0),
            to_cute_tensor(dK00),
            to_cute_tensor(dV0),
            to_cute_tensor(do0),
            to_cute_tensor(lse_log2, assumed_align=4),
            to_cute_tensor(dpsum0, assumed_align=4),
            to_cute_tensor(cu_seqlens_q, assumed_align=4),
            to_cute_tensor(cu_seqlens_comp, assumed_align=4),
            softmax_scale,
            to_cute_tensor(do0),
            current_stream,
            options="--enable-tvm-ffi",
        )
    if not is_fake_mode():
        compiled = csa_comp_dkv_sm100.compile_cache[compile_key]
        compiled(
            qv.detach(),
            comp_kv.detach(),
            v0.detach(),
            q0.detach(),
            dK00,
            dV0,
            do0,
            lse_log2,
            dpsum0,
            cu_seqlens_q,
            cu_seqlens_comp,
            softmax_scale,
            do0,
        )
        compiled(
            qv.detach(),
            comp_kv.detach(),
            v1.detach(),
            q0.detach(),
            dK10,
            dV1,
            do1,
            lse_log2,
            dpsum1,
            cu_seqlens_q,
            cu_seqlens_comp,
            softmax_scale,
            do1,
        )
        compiled(
            qv.detach(),
            comp_kv.detach(),
            v0.detach(),
            q1.detach(),
            dK01,
            dV_dummy,
            do0,
            lse_log2,
            dpsum0,
            cu_seqlens_q,
            cu_seqlens_comp,
            softmax_scale,
            do0,
        )
        compiled(
            qv.detach(),
            comp_kv.detach(),
            v1.detach(),
            q1.detach(),
            dK11,
            dV_dummy,
            do1,
            lse_log2,
            dpsum1,
            cu_seqlens_q,
            cu_seqlens_comp,
            softmax_scale,
            do1,
        )
    dV = torch.empty_like(comp_kv)
    dV[..., :head_dim_v] = dV0
    dV[..., head_dim_v:] = dV1
    dK = torch.empty_like(comp_kv)
    dK[..., :head_dim_v] = (dK00.float() + dK10.float()).to(comp_kv.dtype)
    dK[..., head_dim_v:] = (dK01.float() + dK11.float()).to(comp_kv.dtype)
    return (dK + dV.float()).to(comp_kv.dtype)


csa_comp_dkv_sm100.compile_cache = get_jit_cache("csa_comp_dkv_sm100")


def _as_contiguous(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return None if x is None else x.contiguous()


@triton.jit
def _csa_bwd_dq_kernel(
    Q,
    SWA_KV,
    COMP_KV,
    DO,
    LSE,
    DELTA,
    CU_Q,
    CU_SWA,
    CU_COMP,
    DQ,
    MAX_Q: tl.constexpr,
    MAX_SWA: tl.constexpr,
    MAX_COMP: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    WINDOW: tl.constexpr,
    RATIO: tl.constexpr,
    HAS_SWA: tl.constexpr,
    HAS_COMP: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_KD: tl.constexpr,
    SM_SCALE: tl.constexpr,
    SM_SCALE_LOG2: tl.constexpr,
    LOG2E: tl.constexpr,
):
    q_block = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)

    q_start = tl.load(CU_Q + batch)
    q_end = tl.load(CU_Q + batch + 1)
    q_len = q_end - q_start
    swa_start = tl.load(CU_SWA + batch)
    swa_len = tl.load(CU_SWA + batch + 1) - swa_start
    comp_start = tl.load(CU_COMP + batch)
    comp_len = tl.load(CU_COMP + batch + 1) - comp_start
    swa_offset = swa_len - q_len

    q_offsets = q_block * BLOCK_Q + tl.arange(0, BLOCK_Q)
    q_abs = q_offsets
    q_valid = q_offsets < q_len
    kd_offsets = tl.arange(0, BLOCK_KD)
    k_offsets = tl.arange(0, BLOCK_K)

    lse = tl.load(LSE + (q_start + q_offsets) * HEADS + head, mask=q_valid, other=0.0)
    delta = tl.load(DELTA + (q_start + q_offsets) * HEADS + head, mask=q_valid, other=0.0)

    for db in tl.static_range(0, DIM, BLOCK_D):
        d_offsets = db + tl.arange(0, BLOCK_D)
        d_mask = d_offsets < DIM
        tl.store(
            DQ + (q_start + q_offsets)[:, None] * HEADS * DIM + head * DIM + d_offsets[None, :],
            tl.zeros((BLOCK_Q, BLOCK_D), tl.float32),
            mask=q_valid[:, None] & d_mask[None, :],
        )

    if HAS_COMP:
        q_tile_end = tl.minimum((q_block + 1) * BLOCK_Q, q_len)
        comp_loop_ed = tl.cdiv(tl.minimum(q_tile_end // RATIO, comp_len), BLOCK_K)
        kb = 0
        while kb < comp_loop_ed:
            k_abs = kb * BLOCK_K + k_offsets
            k_valid = k_abs < comp_len
            score = tl.zeros((BLOCK_K, BLOCK_Q), tl.float32)
            dp = tl.zeros((BLOCK_K, BLOCK_Q), tl.float32)
            d0 = 0
            while d0 < DIM:
                kd = d0 + kd_offsets
                q_chunk = tl.load(
                    Q + (q_start + q_offsets)[:, None] * HEADS * DIM + head * DIM + kd[None, :],
                    mask=q_valid[:, None] & (kd[None, :] < DIM),
                    other=0.0,
                )
                do_chunk = tl.load(
                    DO + (q_start + q_offsets)[:, None] * HEADS * DIM + head * DIM + kd[None, :],
                    mask=q_valid[:, None] & (kd[None, :] < DIM),
                    other=0.0,
                )
                kv_chunk = tl.load(
                    COMP_KV + (comp_start + k_abs)[:, None] * DIM + kd[None, :],
                    mask=k_valid[:, None] & (kd[None, :] < DIM),
                    other=0.0,
                )
                score += tl.dot(kv_chunk, tl.trans(q_chunk), out_dtype=tl.float32)
                dp += tl.dot(kv_chunk, tl.trans(do_chunk), out_dtype=tl.float32)
                d0 += BLOCK_KD
            keep = k_valid[:, None] & q_valid[None, :] & (
                k_abs[:, None] < ((q_abs[None, :] + 1) // RATIO)
            )
            prob = tl.exp2(score * SM_SCALE_LOG2 - lse[None, :] * LOG2E)
            prob = tl.where(keep, prob, 0.0)
            ds = prob * (dp - delta[None, :]) * SM_SCALE
            for db in tl.static_range(0, DIM, BLOCK_D):
                d_offsets = db + tl.arange(0, BLOCK_D)
                d_mask = d_offsets < DIM
                kv_d = tl.load(
                    COMP_KV + (comp_start + k_abs)[:, None] * DIM + d_offsets[None, :],
                    mask=k_valid[:, None] & d_mask[None, :],
                    other=0.0,
                )
                dq_ptrs = (
                    DQ
                    + (q_start + q_offsets)[:, None] * HEADS * DIM
                    + head * DIM
                    + d_offsets[None, :]
                )
                cur = tl.load(dq_ptrs, mask=q_valid[:, None] & d_mask[None, :], other=0.0)
                cur += tl.dot(tl.trans(ds.to(tl.bfloat16)), kv_d, out_dtype=tl.float32)
                tl.store(dq_ptrs, cur, mask=q_valid[:, None] & d_mask[None, :])
            kb += 1

    if HAS_SWA:
        min_visible = q_block * BLOCK_Q + swa_offset - (WINDOW - 1)
        max_visible = (q_block + 1) * BLOCK_Q + swa_offset
        kb = tl.maximum(min_visible // BLOCK_K, 0)
        kb_end = tl.minimum(tl.cdiv(max_visible, BLOCK_K), tl.cdiv(swa_len, BLOCK_K))
        while kb < kb_end:
            k_abs = kb * BLOCK_K + k_offsets
            k_valid = k_abs < swa_len
            score = tl.zeros((BLOCK_K, BLOCK_Q), tl.float32)
            dp = tl.zeros((BLOCK_K, BLOCK_Q), tl.float32)
            d0 = 0
            while d0 < DIM:
                kd = d0 + kd_offsets
                q_chunk = tl.load(
                    Q + (q_start + q_offsets)[:, None] * HEADS * DIM + head * DIM + kd[None, :],
                    mask=q_valid[:, None] & (kd[None, :] < DIM),
                    other=0.0,
                )
                do_chunk = tl.load(
                    DO + (q_start + q_offsets)[:, None] * HEADS * DIM + head * DIM + kd[None, :],
                    mask=q_valid[:, None] & (kd[None, :] < DIM),
                    other=0.0,
                )
                kv_chunk = tl.load(
                    SWA_KV + (swa_start + k_abs)[:, None] * DIM + kd[None, :],
                    mask=k_valid[:, None] & (kd[None, :] < DIM),
                    other=0.0,
                )
                score += tl.dot(kv_chunk, tl.trans(q_chunk), out_dtype=tl.float32)
                dp += tl.dot(kv_chunk, tl.trans(do_chunk), out_dtype=tl.float32)
                d0 += BLOCK_KD
            q_pos = q_abs + swa_offset
            keep = (
                k_valid[:, None]
                & q_valid[None, :]
                & (q_pos[None, :] >= k_abs[:, None])
                & ((q_pos[None, :] - k_abs[:, None]) < WINDOW)
            )
            prob = tl.exp2(score * SM_SCALE_LOG2 - lse[None, :] * LOG2E)
            prob = tl.where(keep, prob, 0.0)
            ds = prob * (dp - delta[None, :]) * SM_SCALE
            for db in tl.static_range(0, DIM, BLOCK_D):
                d_offsets = db + tl.arange(0, BLOCK_D)
                d_mask = d_offsets < DIM
                kv_d = tl.load(
                    SWA_KV + (swa_start + k_abs)[:, None] * DIM + d_offsets[None, :],
                    mask=k_valid[:, None] & d_mask[None, :],
                    other=0.0,
                )
                dq_ptrs = (
                    DQ
                    + (q_start + q_offsets)[:, None] * HEADS * DIM
                    + head * DIM
                    + d_offsets[None, :]
                )
                cur = tl.load(dq_ptrs, mask=q_valid[:, None] & d_mask[None, :], other=0.0)
                cur += tl.dot(tl.trans(ds.to(tl.bfloat16)), kv_d, out_dtype=tl.float32)
                tl.store(dq_ptrs, cur, mask=q_valid[:, None] & d_mask[None, :])
            kb += 1


@triton.jit
def _csa_bwd_dkv_kernel(
    Q,
    KV,
    DO,
    LSE,
    DELTA,
    CU_Q,
    CU_KV,
    PARTIAL,
    MAX_Q: tl.constexpr,
    MAX_KV: tl.constexpr,
    TOTAL_KV: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    WINDOW: tl.constexpr,
    RATIO: tl.constexpr,
    IS_SWA: tl.constexpr,
    HEADS_PER_BLOCK: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_KD: tl.constexpr,
    SM_SCALE: tl.constexpr,
    SM_SCALE_LOG2: tl.constexpr,
    LOG2E: tl.constexpr,
):
    kv_block = tl.program_id(0)
    head_block = tl.program_id(1)
    batch = tl.program_id(2)

    q_start = tl.load(CU_Q + batch)
    q_len = tl.load(CU_Q + batch + 1) - q_start
    kv_start = tl.load(CU_KV + batch)
    kv_len = tl.load(CU_KV + batch + 1) - kv_start
    swa_offset = kv_len - q_len

    k_offsets = kv_block * BLOCK_K + tl.arange(0, BLOCK_K)
    k_valid = k_offsets < kv_len
    q_offsets = tl.arange(0, BLOCK_Q)
    kd_offsets = tl.arange(0, BLOCK_KD)

    for db in tl.static_range(0, DIM, BLOCK_D):
        d_offsets = db + tl.arange(0, BLOCK_D)
        d_mask = d_offsets < DIM
        tl.store(
            PARTIAL
            + head_block * TOTAL_KV * DIM
            + (kv_start + k_offsets)[:, None] * DIM
            + d_offsets[None, :],
            tl.zeros((BLOCK_K, BLOCK_D), tl.float32),
            mask=k_valid[:, None] & d_mask[None, :],
        )

    for h_off in tl.static_range(0, HEADS_PER_BLOCK):
        head = head_block * HEADS_PER_BLOCK + h_off
        if head < HEADS:
            if IS_SWA:
                qb = tl.maximum((kv_block * BLOCK_K - swa_offset) // BLOCK_Q, 0)
                qb_end = tl.minimum(
                    tl.cdiv(q_len, BLOCK_Q),
                    tl.cdiv((kv_block + 1) * BLOCK_K + WINDOW - 1 - swa_offset, BLOCK_Q),
                )
            else:
                first_visible_q = (kv_block * BLOCK_K + 1) * RATIO - 1
                qb = tl.minimum(tl.maximum(first_visible_q // BLOCK_Q, 0), tl.cdiv(q_len, BLOCK_Q))
                qb_end = tl.cdiv(q_len, BLOCK_Q)
            while qb < qb_end:
                q_abs = qb * BLOCK_Q + q_offsets
                q_valid = q_abs < q_len
                lse = tl.load(LSE + (q_start + q_abs) * HEADS + head, mask=q_valid, other=0.0)
                delta = tl.load(DELTA + (q_start + q_abs) * HEADS + head, mask=q_valid, other=0.0)
                score = tl.zeros((BLOCK_K, BLOCK_Q), tl.float32)
                dp = tl.zeros((BLOCK_K, BLOCK_Q), tl.float32)
                d0 = 0
                while d0 < DIM:
                    kd = d0 + kd_offsets
                    q_chunk = tl.load(
                        Q + (q_start + q_abs)[:, None] * HEADS * DIM + head * DIM + kd[None, :],
                        mask=q_valid[:, None] & (kd[None, :] < DIM),
                        other=0.0,
                    )
                    do_chunk = tl.load(
                        DO + (q_start + q_abs)[:, None] * HEADS * DIM + head * DIM + kd[None, :],
                        mask=q_valid[:, None] & (kd[None, :] < DIM),
                        other=0.0,
                    )
                    kv_chunk = tl.load(
                        KV + (kv_start + k_offsets)[:, None] * DIM + kd[None, :],
                        mask=k_valid[:, None] & (kd[None, :] < DIM),
                        other=0.0,
                    )
                    score += tl.dot(kv_chunk, tl.trans(q_chunk), out_dtype=tl.float32)
                    dp += tl.dot(kv_chunk, tl.trans(do_chunk), out_dtype=tl.float32)
                    d0 += BLOCK_KD
                if IS_SWA:
                    q_pos = q_abs + swa_offset
                    keep = (
                        k_valid[:, None]
                        & q_valid[None, :]
                        & (q_pos[None, :] >= k_offsets[:, None])
                        & ((q_pos[None, :] - k_offsets[:, None]) < WINDOW)
                    )
                else:
                    keep = k_valid[:, None] & q_valid[None, :] & (
                        k_offsets[:, None] < ((q_abs[None, :] + 1) // RATIO)
                    )
                prob = tl.exp2(score * SM_SCALE_LOG2 - lse[None, :] * LOG2E)
                prob = tl.where(keep, prob, 0.0)
                ds = prob * (dp - delta[None, :]) * SM_SCALE
                for db in tl.static_range(0, DIM, BLOCK_D):
                    d_offsets = db + tl.arange(0, BLOCK_D)
                    d_mask = d_offsets < DIM
                    q_d = tl.load(
                        Q
                        + (q_start + q_abs)[:, None] * HEADS * DIM
                        + head * DIM
                        + d_offsets[None, :],
                        mask=q_valid[:, None] & d_mask[None, :],
                        other=0.0,
                    )
                    do_d = tl.load(
                        DO
                        + (q_start + q_abs)[:, None] * HEADS * DIM
                        + head * DIM
                        + d_offsets[None, :],
                        mask=q_valid[:, None] & d_mask[None, :],
                        other=0.0,
                    )
                    partial_ptrs = (
                        PARTIAL
                        + head_block * TOTAL_KV * DIM
                        + (kv_start + k_offsets)[:, None] * DIM
                        + d_offsets[None, :]
                    )
                    cur = tl.load(partial_ptrs, mask=k_valid[:, None] & d_mask[None, :], other=0.0)
                    cur += tl.dot(prob.to(tl.bfloat16), do_d, out_dtype=tl.float32)
                    cur += tl.dot(ds.to(tl.bfloat16), q_d, out_dtype=tl.float32)
                    tl.store(partial_ptrs, cur, mask=k_valid[:, None] & d_mask[None, :])
                qb += 1


def _run_triton_csa_bwd(
    qv: torch.Tensor,
    swa_kv: Optional[torch.Tensor],
    comp_kv: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    delta: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_swa: torch.Tensor,
    cu_seqlens_comp: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_swa: Optional[int],
    max_seqlen_comp: int,
    swa_window_size: int,
    compress_ratio: int,
):
    total_q, heads, dim = qv.shape
    batch = cu_seqlens_q.numel() - 1
    block_q = 32
    block_k_dq = 128
    block_k_dkv = 64
    block_d_dq = 256
    block_d_dkv = 256
    block_kd = 64
    heads_per_block = 4
    sm_scale = 1.0 / math.sqrt(dim)
    log2e = math.log2(math.e)
    sm_scale_log2 = sm_scale * math.log2(math.e)
    has_swa = swa_kv is not None and swa_kv.numel() > 0 and swa_window_size > 0
    has_comp = comp_kv.numel() > 0

    dq = torch.empty((total_q, heads, dim), dtype=torch.float32, device=qv.device)
    _csa_bwd_dq_kernel[
        (triton.cdiv(max_seqlen_q, block_q), heads, batch)
    ](
        qv,
        swa_kv if has_swa else comp_kv,
        comp_kv,
        dout,
        lse,
        delta,
        cu_seqlens_q,
        cu_seqlens_swa,
        cu_seqlens_comp,
        dq,
        max_seqlen_q,
        0 if max_seqlen_swa is None else max_seqlen_swa,
        max_seqlen_comp,
        heads,
        dim,
        swa_window_size,
        compress_ratio,
        has_swa,
        has_comp,
        block_q,
        block_k_dq,
        block_d_dq,
        block_kd,
        sm_scale,
        sm_scale_log2,
        log2e,
        num_warps=8,
        num_stages=3,
    )

    head_blocks = triton.cdiv(heads, heads_per_block)
    dswa_kv = None
    if has_swa:
        dswa_partials = torch.empty(
            (head_blocks, swa_kv.shape[0], 1, dim), dtype=torch.float32, device=qv.device
        )
        _csa_bwd_dkv_kernel[
            (triton.cdiv(max_seqlen_swa, block_k_dkv), head_blocks, batch)
        ](
            qv,
            swa_kv,
            dout,
            lse,
            delta,
            cu_seqlens_q,
            cu_seqlens_swa,
            dswa_partials,
            max_seqlen_q,
            max_seqlen_swa,
            swa_kv.shape[0],
            heads,
            dim,
            swa_window_size,
            compress_ratio,
            True,
            heads_per_block,
            block_q,
            block_k_dkv,
            block_d_dkv,
            block_kd,
            sm_scale,
            sm_scale_log2,
            log2e,
            num_warps=8,
            num_stages=3,
        )
        dswa_kv = dswa_partials.sum(0).to(swa_kv.dtype)

    if has_comp:
        dcomp_partials = torch.empty(
            (head_blocks, comp_kv.shape[0], 1, dim), dtype=torch.float32, device=qv.device
        )
        _csa_bwd_dkv_kernel[
            (triton.cdiv(max_seqlen_comp, block_k_dkv), head_blocks, batch)
        ](
            qv,
            comp_kv,
            dout,
            lse,
            delta,
            cu_seqlens_q,
            cu_seqlens_comp,
            dcomp_partials,
            max_seqlen_q,
            max_seqlen_comp,
            comp_kv.shape[0],
            heads,
            dim,
            swa_window_size,
            compress_ratio,
            False,
            heads_per_block,
            block_q,
            block_k_dkv,
            block_d_dkv,
            block_kd,
            sm_scale,
            sm_scale_log2,
            log2e,
            num_warps=8,
            num_stages=3,
        )
        dcomp_kv = dcomp_partials.sum(0).to(comp_kv.dtype)
    else:
        dcomp_kv = torch.zeros_like(comp_kv)

    return dq.to(qv.dtype), dswa_kv, dcomp_kv


def _run_cute_full_csa_bwd(
    qv: torch.Tensor,
    swa_kv: Optional[torch.Tensor],
    comp_kv: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    delta: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_swa: torch.Tensor,
    cu_seqlens_comp: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_swa: Optional[int],
    max_seqlen_comp: int,
    swa_window_size: int,
    compress_ratio: int,
):
    from flash_attn.cute.csa_bwd_cute_full import (
        csa_mla_bwd_dkv_full_cute,
        csa_mla_bwd_dq_full_cute,
    )

    has_swa = swa_kv is not None and swa_kv.numel() > 0 and swa_window_size > 0
    has_comp = comp_kv.numel() > 0
    if os.environ.get("CSA_DQ_USE_SM100") != "1":
        dq = csa_mla_bwd_dq_full_cute(
            qv,
            swa_kv,
            comp_kv,
            dout,
            lse,
            delta,
            cu_seqlens_q,
            cu_seqlens_swa,
            cu_seqlens_comp,
            max_seqlen_q,
            max_seqlen_swa,
            max_seqlen_comp,
            swa_window_size,
            compress_ratio,
        )
    else:
        dq = None
        if has_comp:
            dq = csa_comp_dq_sm100(
                qv,
                comp_kv,
                dout,
                lse,
                delta,
                cu_seqlens_q,
                cu_seqlens_comp,
                compress_ratio,
                use_clc_scheduler=os.environ.get("CSA_COMP_DQ_SM100_CLC", "0") == "1",
            )
        if has_swa:
            dq_swa = csa_swa_dq_sm100(
                qv,
                swa_kv,
                dout,
                lse,
                delta,
                cu_seqlens_q,
                cu_seqlens_swa,
                swa_window_size,
            )
            dq = dq_swa if dq is None else dq.add_(dq_swa)
        if dq is None:
            dq = torch.zeros_like(qv)
    if has_swa:
        if os.environ.get("CSA_SWA_DKV_USE_SM100") == "1":
            dswa_kv = csa_swa_dkv_sm100(
                qv,
                swa_kv,
                out,
                dout,
                lse,
                delta,
                cu_seqlens_q,
                cu_seqlens_swa,
                swa_window_size,
            )
        else:
            dswa_kv = csa_mla_bwd_dkv_full_cute(
                qv,
                swa_kv,
                dout,
                lse,
                delta,
                cu_seqlens_q,
                cu_seqlens_swa,
                max_seqlen_q,
                max_seqlen_swa if max_seqlen_swa is not None else max_seqlen_q,
                swa_window_size,
                compress_ratio,
                True,
                heads_per_block=1,
                q_splits=1,
                block_q=32,
                num_threads=256,
            )
    else:
        dswa_kv = None

    if has_comp:
        use_small_comp_dkv = os.environ.get("CSA_COMP_DKV_USE_SM100_SMALL") == "1"
        small_comp_dkv_eligible = (
            compress_ratio == 4
            and cu_seqlens_q.numel() == 2
            and cu_seqlens_comp.numel() == 2
            and comp_kv.shape[1] == 1
            and qv.shape[0] >= 256
            and comp_kv.shape[0] >= 64
            and qv.shape[0] % 64 == 0
            and comp_kv.shape[0] % 64 == 0
        )
        if (
            (use_small_comp_dkv and small_comp_dkv_eligible)
            or (
                not use_small_comp_dkv
                and os.environ.get("CSA_COMP_DKV_USE_SM100") == "1"
            )
        ):
            dcomp_kv = csa_comp_dkv_sm100(
                qv,
                comp_kv,
                out,
                dout,
                lse,
                delta,
                cu_seqlens_q,
                cu_seqlens_comp,
                compress_ratio,
                use_clc_scheduler=os.environ.get("CSA_COMP_DKV_SM100_CLC") == "1",
            )
        else:
            dcomp_kv = csa_mla_bwd_dkv_full_cute(
                qv,
                comp_kv,
                dout,
                lse,
                delta,
                cu_seqlens_q,
                cu_seqlens_comp,
                max_seqlen_q,
                max_seqlen_comp,
                swa_window_size,
                compress_ratio,
                False,
                heads_per_block=2,
                block_q=64,
                num_threads=256,
                use_clc_scheduler=True,
                atom_layout_m_dkv=1,
            )
    else:
        dcomp_kv = torch.zeros_like(comp_kv)
    return dq, dswa_kv, dcomp_kv


@lru_cache(maxsize=None)
def _get_csa_comp_mask_mod(compress_ratio: int):
    @cute.jit
    def _csa_comp_mask_mod(
        batch: cute.TensorSSA,
        head: cute.TensorSSA,
        q_idx: cute.TensorSSA,
        kv_idx: cute.TensorSSA,
        seqlen_info,
        aux_tensors: None,
    ) -> cute.TensorSSA:
        ratio = utils.scalar_to_ssa(compress_ratio, cutlass.Int32)
        one = utils.scalar_to_ssa(1, cutlass.Int32)
        return kv_idx < ((q_idx + one) // ratio)

    return _csa_comp_mask_mod


def _run_shared_kv_fa4_bwd(
    qv: torch.Tensor,
    kv: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: Optional[float],
    dlse: Optional[torch.Tensor],
    window_size_left: Optional[int] = None,
    window_size_right: Optional[int] = None,
    mask_mod=None,
):
    from flash_attn.cute.interface import _flash_attn_bwd

    lse_t = lse.transpose(0, 1).contiguous()
    dlse_t = None if dlse is None else dlse.transpose(0, 1).contiguous()
    dq, dk, dv = _flash_attn_bwd(
        q=qv,
        k=kv,
        v=kv,
        out=out,
        dout=dout,
        lse=lse_t,
        softmax_scale=softmax_scale,
        causal=False,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        mask_mod=mask_mod,
        dlse=dlse_t,
    )
    dkv = (dk.float() + dv.float()).to(kv.dtype)
    return dq, dkv


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

    qv = qv.contiguous()
    comp_kv = comp_kv.contiguous()
    swa_kv = _as_contiguous(swa_kv)
    dout = dout.contiguous()
    out = out.contiguous()
    lse = lse.contiguous()

    head_kv = comp_kv.shape[1]
    assert head_kv == 1, "CSA MLA backward expects shared MQA KV with one KV head"

    delta = (out.float() * dout.float()).sum(dim=-1)
    if dlse is not None:
        delta = delta - dlse.contiguous().float()

    if os.environ.get("CSA_BWD_FORCE_TRITON") == "1":
        dq, dswa_kv, dcomp_kv = _run_triton_csa_bwd(
            qv,
            swa_kv,
            comp_kv,
            out,
            dout,
            lse,
            delta,
            cu_seqlens_q,
            cu_seqlens_swa,
            cu_seqlens_comp,
            max_seqlen_q,
            max_seqlen_swa,
            max_seqlen_comp,
            swa_window_size,
            compress_ratio,
        )
    else:
        dq, dswa_kv, dcomp_kv = _run_cute_full_csa_bwd(
            qv,
            swa_kv,
            comp_kv,
            out,
            dout,
            lse,
            delta,
            cu_seqlens_q,
            cu_seqlens_swa,
            cu_seqlens_comp,
            max_seqlen_q,
            max_seqlen_swa,
            max_seqlen_comp,
            swa_window_size,
            compress_ratio,
        )

    if learnable_sink is not None:
        from flash_attn.cute.csa_bwd_cute_full import csa_mla_bwd_sink_grad_cute

        d_sink = csa_mla_bwd_sink_grad_cute(learnable_sink, lse, delta)
    else:
        d_sink = None

    return (
        dq,
        None if dswa_kv is None else dswa_kv,
        dcomp_kv,
        d_sink,
    )
