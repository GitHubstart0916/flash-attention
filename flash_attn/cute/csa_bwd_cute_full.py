import math
from typing import Optional

import cuda.bindings.driver as cuda
import torch

import cutlass
import cutlass.pipeline as pipeline
import cutlass.cute as cute
from cutlass import Float32, Int32
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.utils import ClcDynamicPersistentTileScheduler

from quack import layout_utils

from flash_attn.cute import ampere_helpers as sm80_utils
from flash_attn.cute import utils
from flash_attn.cute.cache_utils import get_jit_cache
from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned, to_cute_tensor
from flash_attn.cute.testing import is_fake_mode
from flash_attn.cute.tile_scheduler import (
    ClcState,
    Sm100FmhaClcDynamicTileScheduler as FmhaClcDynamicTileScheduler,
    Sm100FmhaClcDynamicTileSchedulerParams as FmhaClcDynamicTileSchedulerParams,
)


LOG2E = math.log2(math.e)


def _as_contiguous(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return None if x is None else x.contiguous()


class CsaMlaBwdDqFullCute:
    """CSA MLA dQ kernel with a full D=512 accumulator per CTA.

    This is deliberately independent from FA4's hd256 two-CTA backward kernels:
    K and V are the same CSA KV tensor, so the kernel keeps one KV tile in smem
    and accumulates the complete [BLOCK_Q, 512] dQ tile in registers.
    """

    def __init__(
        self,
        heads: int,
        head_dim: int,
        window_size: int,
        compress_ratio: int,
        has_swa: bool,
        has_comp: bool,
        block_k: int = 64,
        block_q: int = 32,
        num_threads: int = 256,
        atom_layout_m_sdp: int = 2,
        atom_layout_m_dq: int = 1,
    ):
        assert head_dim == 512
        self.heads = heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.compress_ratio = compress_ratio
        self.has_swa = has_swa
        self.has_comp = has_comp
        self.block_k = block_k
        self.block_q = block_q
        self.num_threads = num_threads
        self.SdP_swapAB = False
        self.dQ_swapAB = False
        self.AtomLayoutMSdP = atom_layout_m_sdp
        self.AtomLayoutMdQ = atom_layout_m_dq

    def _setup_attributes(self):
        self.head_dim_padded = self.head_dim
        sD_layout_atom = sm80_utils.get_smem_layout_atom(self.dtype, self.head_dim_padded)
        self.sQ_layout = cute.tile_to_shape(
            sD_layout_atom, (self.block_q, self.head_dim_padded), (0, 1)
        )
        self.sdO_layout = self.sQ_layout
        self.sKV_layout = cute.tile_to_shape(
            sD_layout_atom, (self.block_k, self.head_dim_padded), (0, 1)
        )
        sPdS_layout_atom = sm80_utils.get_smem_layout_atom(self.dtype, self.block_q)
        self.sPdS_layout = cute.tile_to_shape(
            sPdS_layout_atom, (self.block_q, self.block_k), (0, 1)
        )
        self.sStats_layout = cute.make_layout((self.block_q,))

        universal_copy_bits = 128
        async_copy_elems = universal_copy_bits // self.dtype.width
        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self.dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        atom_store_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), self.dtype, num_bits_per_copy=universal_copy_bits
        )
        tD_shape_dim_1 = sD_layout_atom.outer.shape[1] // async_copy_elems
        assert self.num_threads % tD_shape_dim_1 == 0
        tD_layout = cute.make_ordered_layout(
            (self.num_threads // tD_shape_dim_1, tD_shape_dim_1), order=(1, 0)
        )
        vD_layout = cute.make_layout((1, async_copy_elems))
        self.gmem_tiled_copy_D = cute.make_tiled_copy_tv(atom_async_copy, tD_layout, vD_layout)
        self.gmem_tiled_store_D = cute.make_tiled_copy_tv(atom_store_copy, tD_layout, vD_layout)
        self.is_even_q = self.block_q % tD_layout.shape[0] == 0
        self.is_even_k = self.block_k % tD_layout.shape[0] == 0

    def _get_tiled_mma(self):
        num_mma_warps = self.num_threads // 32
        atom_layout_sdp = (self.AtomLayoutMSdP, num_mma_warps // self.AtomLayoutMSdP, 1)
        tiled_mma_sdp = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self.dtype, cutlass.Float32, (16, 8, 16)),
            atom_layout_sdp,
            permutation_mnk=(atom_layout_sdp[0] * 16, atom_layout_sdp[1] * 16, 16),
        )
        atom_layout_dq = (self.AtomLayoutMdQ, num_mma_warps // self.AtomLayoutMdQ, 1)
        tiled_mma_dq = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self.dtype, cutlass.Float32, (16, 8, 16)),
            atom_layout_dq,
            permutation_mnk=(atom_layout_dq[0] * 16, atom_layout_dq[1] * 16, 16),
        )
        return tiled_mma_sdp, tiled_mma_dq

    def _get_shared_storage_cls(self):
        sQ_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype, cute.cosize(self.sQ_layout)], 1024
        ]
        sdO_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype, cute.cosize(self.sdO_layout)], 1024
        ]
        sKV_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype, cute.cosize(self.sKV_layout)], 1024
        ]
        sPdS_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype, cute.cosize(self.sPdS_layout)], 128
        ]
        sStats_struct = cute.struct.Align[
            cute.struct.MemRange[Float32, cute.cosize(self.sStats_layout)], 128
        ]

        @cute.struct
        class SharedStorage:
            sQ: sQ_struct
            sdO: sdO_struct
            sKV: sKV_struct
            sdS: sPdS_struct
            sLSE: sStats_struct
            sDelta: sStats_struct

        return SharedStorage

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mSwaKV: cute.Tensor,
        mCompKV: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mDelta: cute.Tensor,
        mCuQ: cute.Tensor,
        mCuSwa: cute.Tensor,
        mCuComp: cute.Tensor,
        mdQ: cute.Tensor,
        max_seqlen_q: Int32,
        max_seqlen_swa: Int32,
        max_seqlen_comp: Int32,
        softmax_scale: Float32,
        softmax_scale_log2: Float32,
        stream: cuda.CUstream = None,
    ):
        if cutlass.const_expr(
            not (mQ.element_type == mSwaKV.element_type == mCompKV.element_type == mdO.element_type == mdQ.element_type)
        ):
            raise TypeError("Q/KV/dO/dQ must have the same dtype")
        if cutlass.const_expr(mQ.element_type not in [cutlass.Float16, cutlass.BFloat16]):
            raise TypeError("Only fp16/bf16 are supported")
        if cutlass.const_expr(mLSE.element_type != cutlass.Float32 or mDelta.element_type != cutlass.Float32):
            raise TypeError("LSE and Delta must be fp32")

        mQ, mSwaKV, mCompKV, mdO, mdQ = [
            assume_tensor_aligned(t) for t in (mQ, mSwaKV, mCompKV, mdO, mdQ)
        ]
        self.dtype = mQ.element_type
        self._setup_attributes()
        tiled_mma_sdp, tiled_mma_dq = self._get_tiled_mma()
        SharedStorage = self._get_shared_storage_cls()
        grid = (cute.ceil_div(max_seqlen_q, self.block_q), self.heads, mCuQ.shape[0] - 1)
        self.kernel(
            mQ,
            mSwaKV,
            mCompKV,
            mdO,
            mLSE,
            mDelta,
            mCuQ,
            mCuSwa,
            mCuComp,
            mdQ,
            max_seqlen_q,
            max_seqlen_swa,
            max_seqlen_comp,
            softmax_scale,
            softmax_scale_log2,
            self.sQ_layout,
            self.sdO_layout,
            self.sKV_layout,
            self.sPdS_layout,
            self.sStats_layout,
            self.gmem_tiled_copy_D,
            self.gmem_tiled_store_D,
            tiled_mma_sdp,
            tiled_mma_dq,
            SharedStorage,
        ).launch(
            grid=grid,
            block=[self.num_threads, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mSwaKV: cute.Tensor,
        mCompKV: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mDelta: cute.Tensor,
        mCuQ: cute.Tensor,
        mCuSwa: cute.Tensor,
        mCuComp: cute.Tensor,
        mdQ: cute.Tensor,
        max_seqlen_q: Int32,
        max_seqlen_swa: Int32,
        max_seqlen_comp: Int32,
        softmax_scale: Float32,
        softmax_scale_log2: Float32,
        sQ_layout: cute.ComposedLayout,
        sdO_layout: cute.ComposedLayout,
        sKV_layout: cute.ComposedLayout,
        sPdS_layout: cute.ComposedLayout,
        sStats_layout: cute.Layout,
        gmem_tiled_copy_D: cute.TiledCopy,
        gmem_tiled_store_D: cute.TiledCopy,
        tiled_mma_sdp: cute.TiledMma,
        tiled_mma_dq: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        q_block, head_idx, batch_idx = cute.arch.block_idx()

        q_start = mCuQ[batch_idx]
        q_len = mCuQ[batch_idx + 1] - q_start
        swa_start = mCuSwa[batch_idx]
        swa_len = mCuSwa[batch_idx + 1] - swa_start
        comp_start = mCuComp[batch_idx]
        comp_len = mCuComp[batch_idx + 1] - comp_start
        swa_offset = swa_len - q_len

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sQ = storage.sQ.get_tensor(sQ_layout)
        sdO = storage.sdO.get_tensor(sdO_layout)
        sKV = storage.sKV.get_tensor(sKV_layout)
        sdS = storage.sdS.get_tensor(sPdS_layout)
        sLSE = storage.sLSE.get_tensor(sStats_layout)
        sDelta = storage.sDelta.get_tensor(sStats_layout)
        sKVt = layout_utils.transpose_view(sKV)

        mQ_cur = cute.domain_offset((q_start, 0), mQ[None, head_idx, None])
        mdO_cur = cute.domain_offset((q_start, 0), mdO[None, head_idx, None])
        mdQ_cur = cute.domain_offset((q_start, 0), mdQ[None, head_idx, None])
        gQ = cute.local_tile(mQ_cur, (self.block_q, self.head_dim_padded), (q_block, 0))
        gdO = cute.local_tile(mdO_cur, (self.block_q, self.head_dim_padded), (q_block, 0))
        gdQ = cute.local_tile(mdQ_cur, (self.block_q, self.head_dim_padded), (q_block, 0))

        gmem_thr_copy_D = gmem_tiled_copy_D.get_slice(tidx)
        gmem_thr_store_D = gmem_tiled_store_D.get_slice(tidx)
        tQgQ = gmem_thr_copy_D.partition_S(gQ)
        tQsQ = gmem_thr_copy_D.partition_D(sQ)
        tdOgdO = gmem_thr_copy_D.partition_S(gdO)
        tdOsdO = gmem_thr_copy_D.partition_D(sdO)
        tdQsdQ = gmem_thr_store_D.partition_S(sQ)
        tdQgdQ = gmem_thr_store_D.partition_D(gdQ)

        cQ = cute.make_identity_tensor((self.block_q, self.head_dim_padded))
        tQcQ = gmem_thr_copy_D.partition_S(cQ)
        t0QcQ = gmem_tiled_copy_D.get_slice(0).partition_S(cQ)
        tQpQ = utils.predicate_k(tQcQ, limit=self.head_dim_padded)

        for m in cutlass.range_constexpr(cute.size(tQsQ.shape[1])):
            if self.is_even_q or m < cute.size(tQsQ.shape[1]) - 1 or tQcQ[0, m, 0][0] < self.block_q:
                pred_m = t0QcQ[0, m, 0][0] < q_len - q_block * self.block_q - tQcQ[0][0]
                pred = cute.make_fragment_like(tQpQ[None, 0, None])
                for k in cutlass.range_constexpr(cute.size(pred.shape[1])):
                    for i in cutlass.range_constexpr(cute.size(pred.shape[0])):
                        pred[i, k] = tQpQ[i, m, k] and pred_m
                cute.copy(gmem_tiled_copy_D, tQgQ[None, m, None], tQsQ[None, m, None], pred=pred)
                cute.copy(gmem_tiled_copy_D, tdOgdO[None, m, None], tdOsdO[None, m, None], pred=pred)
        if tidx < self.block_q:
            q_stat_rel = q_block * self.block_q + tidx
            if q_stat_rel < q_len:
                sLSE[tidx] = mLSE[q_start + q_stat_rel, head_idx] * LOG2E
                sDelta[tidx] = mDelta[q_start + q_stat_rel, head_idx]
            else:
                sLSE[tidx] = Float32(0.0)
                sDelta[tidx] = Float32(0.0)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()

        thr_mma_sdp = tiled_mma_sdp.get_slice(tidx)
        thr_mma_dq = tiled_mma_dq.get_slice(tidx)
        tSrQ = utils.mma_make_fragment_A(sQ, thr_mma_sdp, swapAB=self.SdP_swapAB)
        tSrKV = utils.mma_make_fragment_B(sKV, thr_mma_sdp, swapAB=self.SdP_swapAB)
        tdPrdO = utils.mma_make_fragment_A(sdO, thr_mma_sdp, swapAB=self.SdP_swapAB)
        tdQrdS = utils.mma_make_fragment_A(sdS, thr_mma_dq, swapAB=self.dQ_swapAB)
        tdQrKV = utils.mma_make_fragment_B(sKVt, thr_mma_dq, swapAB=self.dQ_swapAB)

        smem_copy_atom = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.dtype
        )
        smem_copy_atom_t = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self.dtype
        )
        smem_thr_copy_Q = utils.make_tiled_copy_A(smem_copy_atom, tiled_mma_sdp, swapAB=self.SdP_swapAB).get_slice(tidx)
        smem_thr_copy_KV = utils.make_tiled_copy_B(smem_copy_atom, tiled_mma_sdp, swapAB=self.SdP_swapAB).get_slice(tidx)
        smem_thr_copy_dS = utils.make_tiled_copy_A(smem_copy_atom, tiled_mma_dq, swapAB=self.dQ_swapAB).get_slice(tidx)
        smem_thr_copy_KVt = utils.make_tiled_copy_B(smem_copy_atom_t, tiled_mma_dq, swapAB=self.dQ_swapAB).get_slice(tidx)
        tSsQ = smem_thr_copy_Q.partition_S(sQ)
        tSsKV = smem_thr_copy_KV.partition_S(sKV)
        tdPsdO = smem_thr_copy_Q.partition_S(sdO)
        tdQsdS = smem_thr_copy_dS.partition_S(sdS)
        tdQsKVt = smem_thr_copy_KVt.partition_S(sKVt)

        r2s_thr_copy_dS = cute.make_tiled_copy_C(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.dtype, num_bits_per_copy=2 * self.dtype.width),
            tiled_mma_sdp,
        ).get_slice(tidx)
        tdSsdS = r2s_thr_copy_dS.partition_D(sdS)

        r2s_thr_copy_dQ = cute.make_tiled_copy_C(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.dtype, num_bits_per_copy=2 * self.dtype.width),
            tiled_mma_dq,
        ).get_slice(tidx)
        tdQacc_sQ = r2s_thr_copy_dQ.partition_D(sQ)

        acc_shape_dq = thr_mma_dq.partition_shape_C((self.block_q, self.head_dim_padded))
        acc_dQ = cute.make_rmem_tensor(acc_shape_dq, cutlass.Float32)
        acc_dQ.fill(0.0)

        cS = cute.make_identity_tensor((self.block_q, self.block_k))
        tScS = layout_utils.reshape_acc_to_mn(thr_mma_sdp.partition_C(cS))

        q_tile_limit = min((q_block + 1) * self.block_q, q_len)
        if cutlass.const_expr(self.has_comp):
            comp_loop_end = cute.ceil_div(min(q_tile_limit // self.compress_ratio, comp_len), self.block_k)
            comp_fast_end = min(
                comp_loop_end,
                (q_block * self.block_q + 1) // (self.block_k * self.compress_ratio),
            )
            for kv_block in cutlass.range(0, comp_fast_end, unroll=1):
                self._compute_dq_one_kv_block(
                    mCompKV,
                    comp_start,
                    comp_len,
                    q_start,
                    q_len,
                    q_block,
                    kv_block,
                    head_idx,
                    False,
                    True,
                    swa_offset,
                    softmax_scale,
                    softmax_scale_log2,
                    sLSE,
                    sDelta,
                    sKV,
                    sdS,
                    tSrQ,
                    tSrKV,
                    tdPrdO,
                    tdQrdS,
                    tdQrKV,
                    tSsQ,
                    tSsKV,
                    tdPsdO,
                    tdQsdS,
                    tdQsKVt,
                    tdSsdS,
                    r2s_thr_copy_dS,
                    smem_thr_copy_KV,
                    smem_thr_copy_Q,
                    smem_thr_copy_dS,
                    smem_thr_copy_KVt,
                    gmem_tiled_copy_D,
                    gmem_thr_copy_D,
                    tiled_mma_sdp,
                    tiled_mma_dq,
                    thr_mma_sdp,
                    thr_mma_dq,
                    tScS,
                    acc_dQ,
                )
            for kv_block in cutlass.range(comp_fast_end, comp_loop_end, unroll=1):
                self._compute_dq_one_kv_block(
                    mCompKV,
                    comp_start,
                    comp_len,
                    q_start,
                    q_len,
                    q_block,
                    kv_block,
                    head_idx,
                    False,
                    False,
                    swa_offset,
                    softmax_scale,
                    softmax_scale_log2,
                    sLSE,
                    sDelta,
                    sKV,
                    sdS,
                    tSrQ,
                    tSrKV,
                    tdPrdO,
                    tdQrdS,
                    tdQrKV,
                    tSsQ,
                    tSsKV,
                    tdPsdO,
                    tdQsdS,
                    tdQsKVt,
                    tdSsdS,
                    r2s_thr_copy_dS,
                    smem_thr_copy_KV,
                    smem_thr_copy_Q,
                    smem_thr_copy_dS,
                    smem_thr_copy_KVt,
                    gmem_tiled_copy_D,
                    gmem_thr_copy_D,
                    tiled_mma_sdp,
                    tiled_mma_dq,
                    thr_mma_sdp,
                    thr_mma_dq,
                    tScS,
                    acc_dQ,
                )

        if cutlass.const_expr(self.has_swa):
            min_visible = q_block * self.block_q + swa_offset - (self.window_size - 1)
            max_visible = (q_block + 1) * self.block_q + swa_offset
            kv_block = max(min_visible // self.block_k, 0)
            kv_block_end = min(cute.ceil_div(max_visible, self.block_k), cute.ceil_div(swa_len, self.block_k))
            for kbi in cutlass.range(kv_block, kv_block_end, unroll=1):
                self._compute_dq_one_kv_block(
                    mSwaKV,
                    swa_start,
                    swa_len,
                    q_start,
                    q_len,
                    q_block,
                    kbi,
                    head_idx,
                    True,
                    False,
                    swa_offset,
                    softmax_scale,
                    softmax_scale_log2,
                    sLSE,
                    sDelta,
                    sKV,
                    sdS,
                    tSrQ,
                    tSrKV,
                    tdPrdO,
                    tdQrdS,
                    tdQrKV,
                    tSsQ,
                    tSsKV,
                    tdPsdO,
                    tdQsdS,
                    tdQsKVt,
                    tdSsdS,
                    r2s_thr_copy_dS,
                    smem_thr_copy_KV,
                    smem_thr_copy_Q,
                    smem_thr_copy_dS,
                    smem_thr_copy_KVt,
                    gmem_tiled_copy_D,
                    gmem_thr_copy_D,
                    tiled_mma_sdp,
                    tiled_mma_dq,
                    thr_mma_sdp,
                    thr_mma_dq,
                    tScS,
                    acc_dQ,
                )

        rdQ = cute.make_fragment_like(acc_dQ, self.dtype)
        rdQ.store(acc_dQ.load().to(self.dtype))
        taccQ = r2s_thr_copy_dQ.retile(rdQ)
        cute.copy(r2s_thr_copy_dQ, taccQ, tdQacc_sQ)
        cute.arch.barrier()

        for m in cutlass.range_constexpr(cute.size(tdQsdQ.shape[1])):
            if self.is_even_q or m < cute.size(tdQsdQ.shape[1]) - 1 or tQcQ[0, m, 0][0] < self.block_q:
                pred_m = t0QcQ[0, m, 0][0] < q_len - q_block * self.block_q - tQcQ[0][0]
                pred = cute.make_fragment_like(tQpQ[None, 0, None])
                for k in cutlass.range_constexpr(cute.size(pred.shape[1])):
                    for i in cutlass.range_constexpr(cute.size(pred.shape[0])):
                        pred[i, k] = tQpQ[i, m, k] and pred_m
                cute.copy(gmem_tiled_store_D, tdQsdQ[None, m, None], tdQgdQ[None, m, None], pred=pred)

    @cute.jit
    def _compute_dq_one_kv_block(
        self,
        mKV: cute.Tensor,
        kv_start: Int32,
        kv_len: Int32,
        q_start: Int32,
        q_len: Int32,
        q_block: Int32,
        kv_block: Int32,
        head_idx: Int32,
        is_swa: cutlass.Constexpr[bool],
        skip_comp_mask: cutlass.Constexpr[bool],
        swa_offset: Int32,
        softmax_scale: Float32,
        softmax_scale_log2: Float32,
        sLSE: cute.Tensor,
        sDelta: cute.Tensor,
        sKV: cute.Tensor,
        sdS: cute.Tensor,
        tSrQ: cute.Tensor,
        tSrKV: cute.Tensor,
        tdPrdO: cute.Tensor,
        tdQrdS: cute.Tensor,
        tdQrKV: cute.Tensor,
        tSsQ: cute.Tensor,
        tSsKV: cute.Tensor,
        tdPsdO: cute.Tensor,
        tdQsdS: cute.Tensor,
        tdQsKVt: cute.Tensor,
        tdSsdS: cute.Tensor,
        r2s_thr_copy_dS: cute.TiledCopy,
        smem_thr_copy_KV: cute.TiledCopy,
        smem_thr_copy_Q: cute.TiledCopy,
        smem_thr_copy_dS: cute.TiledCopy,
        smem_thr_copy_KVt: cute.TiledCopy,
        gmem_tiled_copy_D: cute.TiledCopy,
        gmem_thr_copy_D: cute.TiledCopy,
        tiled_mma_sdp: cute.TiledMma,
        tiled_mma_dq: cute.TiledMma,
        thr_mma_sdp: cute.ThrMma,
        thr_mma_dq: cute.ThrMma,
        tScS: cute.Tensor,
        acc_dQ: cute.Tensor,
    ):
        mKV_cur = cute.domain_offset((kv_start, 0), mKV[None, 0, None])
        gKV = cute.local_tile(mKV_cur, (self.block_k, self.head_dim_padded), (kv_block, 0))
        tKVgKV = gmem_thr_copy_D.partition_S(gKV)
        tKVsKV = gmem_thr_copy_D.partition_D(sKV)
        cKV = cute.make_identity_tensor((self.block_k, self.head_dim_padded))
        tKVcKV = gmem_thr_copy_D.partition_S(cKV)
        t0KVcKV = gmem_tiled_copy_D.get_slice(0).partition_S(cKV)
        tKVpKV = utils.predicate_k(tKVcKV, limit=self.head_dim_padded)
        for m in cutlass.range_constexpr(cute.size(tKVsKV.shape[1])):
            if self.is_even_k or m < cute.size(tKVsKV.shape[1]) - 1 or tKVcKV[0, m, 0][0] < self.block_k:
                pred_m = t0KVcKV[0, m, 0][0] < kv_len - kv_block * self.block_k - tKVcKV[0][0]
                pred = cute.make_fragment_like(tKVpKV[None, 0, None])
                for k in cutlass.range_constexpr(cute.size(pred.shape[1])):
                    for i in cutlass.range_constexpr(cute.size(pred.shape[0])):
                        pred[i, k] = tKVpKV[i, m, k] and pred_m
                cute.copy(gmem_tiled_copy_D, tKVgKV[None, m, None], tKVsKV[None, m, None], pred=pred)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()

        acc_shape_s = thr_mma_sdp.partition_shape_C((self.block_q, self.block_k))
        acc_S = cute.make_rmem_tensor(acc_shape_s, cutlass.Float32)
        acc_S.fill(0.0)
        sm80_utils.gemm(
            thr_mma_sdp,
            acc_S,
            tSrQ,
            tSrKV,
            tSsQ,
            tSsKV,
            smem_thr_copy_Q,
            smem_thr_copy_KV,
            swap_AB=self.SdP_swapAB,
        )
        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S, transpose=self.SdP_swapAB)
        nrow = cutlass.const_expr(cute.size(acc_S_mn.shape[0]))
        ncol = cutlass.const_expr(cute.size(acc_S_mn.shape[1]))
        if cutlass.const_expr(skip_comp_mask):
            for r in cutlass.range_constexpr(nrow):
                q_local = tScS[r, 0][0]
                for c in cutlass.range_constexpr(ncol):
                    lse_val = sLSE[q_local]
                    acc_S_mn[r, c] = cute.math.exp2(
                        acc_S_mn[r, c] * softmax_scale_log2 - lse_val,
                        fastmath=True,
                    )
        elif cutlass.const_expr(is_swa):
            for r in cutlass.range_constexpr(nrow):
                q_local = tScS[r, 0][0]
                q_rel_row = q_block * self.block_q + q_local
                for c in cutlass.range_constexpr(ncol):
                    q_rel = q_rel_row
                    kv_rel = kv_block * self.block_k + tScS[r, c][1]
                    keep = (kv_rel < kv_len) and (q_rel < q_len)
                    q_pos = q_rel + swa_offset
                    keep = keep and (q_pos >= kv_rel) and ((q_pos - kv_rel) < self.window_size)
                    lse_val = sLSE[q_local]
                    prob = cute.math.exp2(acc_S_mn[r, c] * softmax_scale_log2 - lse_val, fastmath=True)
                    acc_S_mn[r, c] = prob if keep else Float32(0.0)
        else:
            for r in cutlass.range_constexpr(nrow):
                q_local = tScS[r, 0][0]
                q_rel_row = q_block * self.block_q + q_local
                for c in cutlass.range_constexpr(ncol):
                    q_rel = q_rel_row
                    kv_rel = kv_block * self.block_k + tScS[r, c][1]
                    keep = (kv_rel < kv_len) and (q_rel < q_len)
                    keep = keep and (q_rel >= ((kv_rel + 1) * self.compress_ratio - 1))
                    lse_val = sLSE[q_local]
                    prob = cute.math.exp2(
                        acc_S_mn[r, c] * softmax_scale_log2 - lse_val,
                        fastmath=True,
                    )
                    acc_S_mn[r, c] = prob if keep else Float32(0.0)

        acc_dP = cute.make_rmem_tensor(acc_shape_s, cutlass.Float32)
        acc_dP.fill(0.0)
        sm80_utils.gemm(
            thr_mma_sdp,
            acc_dP,
            tdPrdO,
            tSrKV,
            tdPsdO,
            tSsKV,
            smem_thr_copy_Q,
            smem_thr_copy_KV,
            swap_AB=self.SdP_swapAB,
        )
        acc_dP_mn = layout_utils.reshape_acc_to_mn(acc_dP, transpose=self.SdP_swapAB)
        for r in cutlass.range_constexpr(nrow):
            q_local = tScS[r, 0][0]
            delta = sDelta[q_local]
            for c in cutlass.range_constexpr(ncol):
                acc_dP_mn[r, c] = acc_S_mn[r, c] * (acc_dP_mn[r, c] - delta) * softmax_scale

        rdS = cute.make_fragment_like(acc_dP, self.dtype)
        rdS.store(acc_dP.load().to(self.dtype))
        tdSrdS = r2s_thr_copy_dS.retile(rdS)
        cute.copy(r2s_thr_copy_dS, tdSrdS, tdSsdS)
        cute.arch.barrier()

        sm80_utils.gemm(
            thr_mma_dq,
            acc_dQ,
            tdQrdS,
            tdQrKV,
            tdQsdS,
            tdQsKVt,
            smem_thr_copy_dS,
            smem_thr_copy_KVt,
            swap_AB=self.dQ_swapAB,
        )
        cute.arch.barrier()


def csa_mla_bwd_dq_full_cute(
    qv: torch.Tensor,
    swa_kv: Optional[torch.Tensor],
    comp_kv: torch.Tensor,
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
    block_k: int = 64,
    block_q: int = 32,
    num_threads: int = 256,
    atom_layout_m_sdp: int = 2,
    atom_layout_m_dq: int = 1,
) -> torch.Tensor:
    total_q, heads, dim = qv.shape
    assert dim == 512
    qv = qv.contiguous()
    dout = dout.contiguous()
    lse = lse.contiguous()
    delta = delta.contiguous()
    comp_kv = comp_kv.contiguous()
    has_swa = swa_kv is not None and swa_kv.numel() > 0 and swa_window_size > 0
    has_comp = comp_kv.numel() > 0
    if not has_swa:
        swa_kv_in = comp_kv if has_comp else qv.new_zeros((1, 1, dim))
        max_seqlen_swa = 0
    else:
        swa_kv_in = swa_kv.contiguous()
        max_seqlen_swa = max_seqlen_q if max_seqlen_swa is None else max_seqlen_swa
    comp_kv_in = comp_kv if has_comp else qv.new_zeros((1, 1, dim))

    dq = torch.empty_like(qv)
    softmax_scale = 1.0 / math.sqrt(dim)
    softmax_scale_log2 = softmax_scale * LOG2E
    current_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    compile_key = (
        qv.dtype,
        heads,
        dim,
        swa_window_size,
        compress_ratio,
        has_swa,
        has_comp,
        qv.shape[0],
        swa_kv_in.shape[0],
        comp_kv_in.shape[0],
        block_k,
        block_q,
        num_threads,
        atom_layout_m_sdp,
        atom_layout_m_dq,
    )
    if compile_key not in csa_mla_bwd_dq_full_cute.compile_cache:
        kernel = CsaMlaBwdDqFullCute(
            heads=heads,
            head_dim=dim,
            window_size=swa_window_size,
            compress_ratio=compress_ratio,
            has_swa=has_swa,
            has_comp=has_comp,
            block_k=block_k,
            block_q=block_q,
            num_threads=num_threads,
            atom_layout_m_sdp=atom_layout_m_sdp,
            atom_layout_m_dq=atom_layout_m_dq,
        )
        csa_mla_bwd_dq_full_cute.compile_cache[compile_key] = cute.compile(
            kernel,
            to_cute_tensor(qv),
            to_cute_tensor(swa_kv_in),
            to_cute_tensor(comp_kv_in),
            to_cute_tensor(dout),
            to_cute_tensor(lse, assumed_align=4),
            to_cute_tensor(delta, assumed_align=4),
            to_cute_tensor(cu_seqlens_q, assumed_align=4),
            to_cute_tensor(cu_seqlens_swa, assumed_align=4),
            to_cute_tensor(cu_seqlens_comp, assumed_align=4),
            to_cute_tensor(dq),
            max_seqlen_q,
            0 if max_seqlen_swa is None else max_seqlen_swa,
            max_seqlen_comp,
            softmax_scale,
            softmax_scale_log2,
            current_stream,
            options="--enable-tvm-ffi",
        )
    if not is_fake_mode():
        csa_mla_bwd_dq_full_cute.compile_cache[compile_key](
            qv.detach(),
            swa_kv_in.detach(),
            comp_kv_in.detach(),
            dout.detach(),
            lse,
            delta,
            cu_seqlens_q,
            cu_seqlens_swa,
            cu_seqlens_comp,
            dq,
            max_seqlen_q,
            0 if max_seqlen_swa is None else max_seqlen_swa,
            max_seqlen_comp,
            softmax_scale,
            softmax_scale_log2,
        )
    return dq


csa_mla_bwd_dq_full_cute.compile_cache = get_jit_cache("csa_mla_bwd_dq_full_cute")


class CsaMlaBwdDkvFullCute:
    """CSA MLA dKV kernel with a full D=512 accumulator per CTA."""

    def __init__(
        self,
        heads: int,
        head_dim: int,
        window_size: int,
        compress_ratio: int,
        is_swa: bool,
        heads_per_block: int = 4,
        q_splits: int = 1,
        q_blocks_per_split: int = 0,
        block_k: int = 64,
        block_q: int = 64,
        num_threads: int = 256,
        use_clc_scheduler: bool = False,
        atom_layout_m_sdp: int = 4,
        atom_layout_m_dkv: int = 2,
        sdp_swap_ab: bool = False,
    ):
        assert head_dim == 512
        self.heads = heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.compress_ratio = compress_ratio
        self.is_swa = is_swa
        self.heads_per_block = heads_per_block
        self.q_splits = q_splits
        self.q_blocks_per_split = q_blocks_per_split
        self.block_k = block_k
        self.block_q = block_q
        self.num_threads = num_threads
        self.use_clc_scheduler = use_clc_scheduler
        self.cluster_shape_mnk = (1, 1, 1)
        self.sched_warp_id = self.num_threads // 32 - 1
        self.num_clc_stage = 1
        self.num_clc_response_bytes = 16
        self.SdP_swapAB = sdp_swap_ab
        self.dKV_swapAB = False
        self.AtomLayoutMSdP = atom_layout_m_sdp
        self.AtomLayoutMdKV = atom_layout_m_dkv

    def _setup_attributes(self):
        self.head_dim_padded = self.head_dim
        sD_layout_atom = sm80_utils.get_smem_layout_atom(self.dtype, self.head_dim_padded)
        self.sQ_layout = cute.tile_to_shape(
            sD_layout_atom, (self.block_q, self.head_dim_padded), (0, 1)
        )
        self.sdO_layout = self.sQ_layout
        self.sKV_layout = cute.tile_to_shape(
            sD_layout_atom, (self.block_k, self.head_dim_padded), (0, 1)
        )
        sPdS_layout_atom = sm80_utils.get_smem_layout_atom(self.dtype, self.block_q)
        self.sPdS_layout = cute.tile_to_shape(
            sPdS_layout_atom, (self.block_k, self.block_q), (0, 1)
        )
        self.sStats_layout = cute.make_layout((self.block_q,))

        universal_copy_bits = 128
        async_copy_elems = universal_copy_bits // self.dtype.width
        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self.dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        tD_shape_dim_1 = sD_layout_atom.outer.shape[1] // async_copy_elems
        assert self.num_threads % tD_shape_dim_1 == 0
        tD_layout = cute.make_ordered_layout(
            (self.num_threads // tD_shape_dim_1, tD_shape_dim_1), order=(1, 0)
        )
        vD_layout = cute.make_layout((1, async_copy_elems))
        self.gmem_tiled_copy_D = cute.make_tiled_copy_tv(atom_async_copy, tD_layout, vD_layout)
        self.is_even_q = self.block_q % tD_layout.shape[0] == 0
        self.is_even_k = self.block_k % tD_layout.shape[0] == 0

    def _get_tiled_mma(self):
        num_mma_warps = self.num_threads // 32
        atom_layout_sdp = (
            (self.AtomLayoutMSdP, num_mma_warps // self.AtomLayoutMSdP, 1)
            if cutlass.const_expr(not self.SdP_swapAB)
            else (num_mma_warps // self.AtomLayoutMSdP, self.AtomLayoutMSdP, 1)
        )
        tiled_mma_sdp = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self.dtype, cutlass.Float32, (16, 8, 16)),
            atom_layout_sdp,
            permutation_mnk=(atom_layout_sdp[0] * 16, atom_layout_sdp[1] * 16, 16),
        )
        atom_layout_dkv = (self.AtomLayoutMdKV, num_mma_warps // self.AtomLayoutMdKV, 1)
        tiled_mma_dkv = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self.dtype, cutlass.Float32, (16, 8, 16)),
            atom_layout_dkv,
            permutation_mnk=(atom_layout_dkv[0] * 16, atom_layout_dkv[1] * 16, 16),
        )
        return tiled_mma_sdp, tiled_mma_dkv

    def _get_shared_storage_cls(self):
        sQ_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype, cute.cosize(self.sQ_layout)], 1024
        ]
        sdO_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype, cute.cosize(self.sdO_layout)], 1024
        ]
        sKV_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype, cute.cosize(self.sKV_layout)], 1024
        ]
        sPdS_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype, cute.cosize(self.sPdS_layout)], 128
        ]
        sStats_struct = cute.struct.Align[
            cute.struct.MemRange[Float32, cute.cosize(self.sStats_layout)], 128
        ]

        @cute.struct
        class SharedStorage:
            clc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            clc_response: cute.struct.MemRange[Int32, 4]
            sQ: sQ_struct
            sdO: sdO_struct
            sKV: sKV_struct
            sP: sPdS_struct
            sdS: sPdS_struct
            sLSE: sStats_struct
            sDelta: sStats_struct

        return SharedStorage

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mKV: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mDelta: cute.Tensor,
        mCuQ: cute.Tensor,
        mCuKV: cute.Tensor,
        mCuKVActual: cute.Tensor,
        mdKVPartial: cute.Tensor,
        max_seqlen_q: Int32,
        max_seqlen_kv: Int32,
        softmax_scale: Float32,
        softmax_scale_log2: Float32,
        stream: cuda.CUstream = None,
    ):
        if cutlass.const_expr(not (mQ.element_type == mKV.element_type == mdO.element_type)):
            raise TypeError("Q/KV/dO must have the same dtype")
        if cutlass.const_expr(mdKVPartial.element_type != cutlass.Float32):
            raise TypeError("dKV partial must be fp32")
        self.dtype = mQ.element_type
        mQ, mKV, mdO = [assume_tensor_aligned(t) for t in (mQ, mKV, mdO)]
        self._setup_attributes()
        tiled_mma_sdp, tiled_mma_dkv = self._get_tiled_mma()
        SharedStorage = self._get_shared_storage_cls()
        head_blocks = cute.ceil_div(self.heads, self.heads_per_block)
        kv_blocks = cute.ceil_div(max_seqlen_kv, self.block_k)
        problem_shape_mbh = (kv_blocks * self.q_splits, mCuQ.shape[0] - 1, head_blocks)
        tile_sched_params = FmhaClcDynamicTileSchedulerParams(
            problem_shape_mbh,
            self.cluster_shape_mnk,
        )
        cluster_layout_vmnk = cute.make_layout(self.cluster_shape_mnk)
        if cutlass.const_expr(self.use_clc_scheduler):
            grid = FmhaClcDynamicTileScheduler.get_grid_shape(tile_sched_params)
        else:
            grid = (kv_blocks * self.q_splits, head_blocks, mCuQ.shape[0] - 1)
        self.kernel(
            mQ,
            mKV,
            mdO,
            mLSE,
            mDelta,
            mCuQ,
            mCuKV,
            mCuKVActual,
            mdKVPartial,
            max_seqlen_q,
            max_seqlen_kv,
            softmax_scale,
            softmax_scale_log2,
            self.sQ_layout,
            self.sdO_layout,
            self.sKV_layout,
            self.sPdS_layout,
            self.sStats_layout,
            self.gmem_tiled_copy_D,
            tiled_mma_sdp,
            tiled_mma_dkv,
            SharedStorage,
            tile_sched_params,
            cluster_layout_vmnk,
        ).launch(
            grid=grid,
            block=[self.num_threads, 1, 1],
            cluster=self.cluster_shape_mnk,
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mKV: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mDelta: cute.Tensor,
        mCuQ: cute.Tensor,
        mCuKV: cute.Tensor,
        mCuKVActual: cute.Tensor,
        mdKVPartial: cute.Tensor,
        max_seqlen_q: Int32,
        max_seqlen_kv: Int32,
        softmax_scale: Float32,
        softmax_scale_log2: Float32,
        sQ_layout: cute.ComposedLayout,
        sdO_layout: cute.ComposedLayout,
        sKV_layout: cute.ComposedLayout,
        sPdS_layout: cute.ComposedLayout,
        sStats_layout: cute.Layout,
        gmem_tiled_copy_D: cute.TiledCopy,
        tiled_mma_sdp: cute.TiledMma,
        tiled_mma_dkv: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
        tile_sched_params: FmhaClcDynamicTileSchedulerParams,
        cluster_layout_vmnk: cute.Layout,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        if cutlass.const_expr(self.use_clc_scheduler):
            self._kernel_clc(
                mQ,
                mKV,
                mdO,
                mLSE,
                mDelta,
                mCuQ,
                mCuKV,
                mCuKVActual,
                mdKVPartial,
                max_seqlen_kv,
                softmax_scale,
                softmax_scale_log2,
                sQ_layout,
                sdO_layout,
                sKV_layout,
                sPdS_layout,
                sStats_layout,
                gmem_tiled_copy_D,
                tiled_mma_sdp,
                tiled_mma_dkv,
                SharedStorage,
                tile_sched_params,
                cluster_layout_vmnk,
            )
            return

        packed_kv_split, head_block, batch_idx = cute.arch.block_idx()
        kv_blocks = cute.ceil_div(max_seqlen_kv, self.block_k)
        q_split = packed_kv_split // kv_blocks
        kv_block = packed_kv_split - q_split * kv_blocks

        q_start = mCuQ[batch_idx]
        q_len = mCuQ[batch_idx + 1] - q_start
        kv_start = mCuKV[batch_idx]
        kv_actual_start = mCuKVActual[batch_idx]
        kv_len = mCuKVActual[batch_idx + 1] - kv_actual_start
        swa_offset = kv_len - q_len

        if cutlass.const_expr(self.is_swa):
            q_block_start = max((kv_block * self.block_k - swa_offset) // self.block_q, 0)
            q_block_end = min(
                cute.ceil_div(q_len, self.block_q),
                cute.ceil_div((kv_block + 1) * self.block_k + self.window_size - 1 - swa_offset, self.block_q),
            )
        else:
            first_visible_q = (kv_block * self.block_k + 1) * self.compress_ratio - 1
            q_block_start = min(max(first_visible_q // self.block_q, 0), cute.ceil_div(q_len, self.block_q))
            q_block_end = cute.ceil_div(q_len, self.block_q)
        if cutlass.const_expr(self.q_splits > 1):
            q_block_split_start = q_block_start + q_split * self.q_blocks_per_split
            q_block_split_end = min(q_block_end, q_block_split_start + self.q_blocks_per_split)
            q_block_start = q_block_split_start
            q_block_end = q_block_split_end

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sQ = storage.sQ.get_tensor(sQ_layout)
        sdO = storage.sdO.get_tensor(sdO_layout)
        sKV = storage.sKV.get_tensor(sKV_layout)
        sP = storage.sP.get_tensor(sPdS_layout)
        sdS = storage.sdS.get_tensor(sPdS_layout)
        sLSE = storage.sLSE.get_tensor(sStats_layout)
        sDelta = storage.sDelta.get_tensor(sStats_layout)
        sQt = layout_utils.transpose_view(sQ)
        sdOt = layout_utils.transpose_view(sdO)

        gmem_thr_copy_D = gmem_tiled_copy_D.get_slice(tidx)
        mKV_cur = cute.domain_offset((kv_start, 0), mKV[None, 0, None])
        gKV = cute.local_tile(mKV_cur, (self.block_k, self.head_dim_padded), (kv_block, 0))
        tKVgKV = gmem_thr_copy_D.partition_S(gKV)
        tKVsKV = gmem_thr_copy_D.partition_D(sKV)
        cKV = cute.make_identity_tensor((self.block_k, self.head_dim_padded))
        tKVcKV = gmem_thr_copy_D.partition_S(cKV)
        t0KVcKV = gmem_tiled_copy_D.get_slice(0).partition_S(cKV)
        tKVpKV = utils.predicate_k(tKVcKV, limit=self.head_dim_padded)
        for m in cutlass.range_constexpr(cute.size(tKVsKV.shape[1])):
            if self.is_even_k or m < cute.size(tKVsKV.shape[1]) - 1 or tKVcKV[0, m, 0][0] < self.block_k:
                pred_m = t0KVcKV[0, m, 0][0] < kv_len - kv_block * self.block_k - tKVcKV[0][0]
                pred = cute.make_fragment_like(tKVpKV[None, 0, None])
                for k in cutlass.range_constexpr(cute.size(pred.shape[1])):
                    for i in cutlass.range_constexpr(cute.size(pred.shape[0])):
                        pred[i, k] = tKVpKV[i, m, k] and pred_m
                cute.copy(gmem_tiled_copy_D, tKVgKV[None, m, None], tKVsKV[None, m, None], pred=pred)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()

        thr_mma_sdp = tiled_mma_sdp.get_slice(tidx)
        thr_mma_dkv = tiled_mma_dkv.get_slice(tidx)
        tSrKV = utils.mma_make_fragment_A(sKV, thr_mma_sdp, swapAB=self.SdP_swapAB)
        tSrQ = utils.mma_make_fragment_B(sQ, thr_mma_sdp, swapAB=self.SdP_swapAB)
        tdPrdO = utils.mma_make_fragment_B(sdO, thr_mma_sdp, swapAB=self.SdP_swapAB)
        tdVrP = utils.mma_make_fragment_A(sP, thr_mma_dkv, swapAB=self.dKV_swapAB)
        tdVrdO = utils.mma_make_fragment_B(sdOt, thr_mma_dkv, swapAB=self.dKV_swapAB)
        tdKrdS = utils.mma_make_fragment_A(sdS, thr_mma_dkv, swapAB=self.dKV_swapAB)
        tdKrQ = utils.mma_make_fragment_B(sQt, thr_mma_dkv, swapAB=self.dKV_swapAB)

        smem_copy_atom = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.dtype
        )
        smem_copy_atom_t = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self.dtype
        )
        smem_thr_copy_KV = utils.make_tiled_copy_A(smem_copy_atom, tiled_mma_sdp, swapAB=self.SdP_swapAB).get_slice(tidx)
        smem_thr_copy_Q = utils.make_tiled_copy_B(smem_copy_atom, tiled_mma_sdp, swapAB=self.SdP_swapAB).get_slice(tidx)
        smem_thr_copy_PdS = utils.make_tiled_copy_A(smem_copy_atom, tiled_mma_dkv, swapAB=self.dKV_swapAB).get_slice(tidx)
        smem_thr_copy_QdOt = utils.make_tiled_copy_B(smem_copy_atom_t, tiled_mma_dkv, swapAB=self.dKV_swapAB).get_slice(tidx)
        tSsKV = smem_thr_copy_KV.partition_S(sKV)
        tSsQ = smem_thr_copy_Q.partition_S(sQ)
        tdPsdO = smem_thr_copy_Q.partition_S(sdO)
        tdVsP = smem_thr_copy_PdS.partition_S(sP)
        tdKsdS = smem_thr_copy_PdS.partition_S(sdS)
        tdVsdOt = smem_thr_copy_QdOt.partition_S(sdOt)
        tdKsQt = smem_thr_copy_QdOt.partition_S(sQt)

        r2s_thr_copy_PdS = cute.make_tiled_copy_C(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.dtype, num_bits_per_copy=2 * self.dtype.width),
            tiled_mma_sdp,
        ).get_slice(tidx)
        tPsP = r2s_thr_copy_PdS.partition_D(sP)
        tdSsdS = r2s_thr_copy_PdS.partition_D(sdS)

        acc_shape_dkv = thr_mma_dkv.partition_shape_C((self.block_k, self.head_dim_padded))
        acc_dKV = cute.make_rmem_tensor(acc_shape_dkv, cutlass.Float32)
        acc_dKV.fill(0.0)
        cS = cute.make_identity_tensor(
            (self.block_q, self.block_k)
            if cutlass.const_expr(self.SdP_swapAB)
            else (self.block_k, self.block_q)
        )
        tScS = layout_utils.reshape_acc_to_mn(thr_mma_sdp.partition_C(cS), transpose=self.SdP_swapAB)

        for head_off in cutlass.range_constexpr(self.heads_per_block):
            head_idx = head_block * self.heads_per_block + head_off
            if head_idx < self.heads:
                for q_block in cutlass.range(q_block_start, q_block_end, unroll=1):
                    self._compute_dkv_one_q_block(
                        mQ,
                        mdO,
                        q_start,
                        q_len,
                        kv_start,
                        kv_len,
                        q_block,
                        kv_block,
                        head_idx,
                        False,
                        swa_offset,
                        softmax_scale,
                        softmax_scale_log2,
                        mLSE,
                        mDelta,
                        sQ,
                        sdO,
                        sP,
                        sdS,
                        sLSE,
                        sDelta,
                        tSrKV,
                        tSrQ,
                        tdPrdO,
                        tdVrP,
                        tdVrdO,
                        tdKrdS,
                        tdKrQ,
                        tSsKV,
                        tSsQ,
                        tdPsdO,
                        tdVsP,
                        tdKsdS,
                        tdVsdOt,
                        tdKsQt,
                        tPsP,
                        tdSsdS,
                        r2s_thr_copy_PdS,
                        smem_thr_copy_KV,
                        smem_thr_copy_Q,
                        smem_thr_copy_PdS,
                        smem_thr_copy_QdOt,
                        gmem_tiled_copy_D,
                        gmem_thr_copy_D,
                        tiled_mma_sdp,
                        tiled_mma_dkv,
                        thr_mma_sdp,
                        thr_mma_dkv,
                        tScS,
                        acc_dKV,
                    )

        mPartial_cur = cute.domain_offset((kv_start, 0), mdKVPartial[q_split, head_block, None, 0, None])
        gPartial = cute.local_tile(mPartial_cur, (self.block_k, self.head_dim_padded), (kv_block, 0))
        tCgPartial = thr_mma_dkv.partition_C(gPartial)
        simt_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.Float32, num_bits_per_copy=cutlass.Float32.width
        )
        cute.copy(simt_atom, acc_dKV, tCgPartial)

    @cute.jit
    def _kernel_clc(
        self,
        mQ: cute.Tensor,
        mKV: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mDelta: cute.Tensor,
        mCuQ: cute.Tensor,
        mCuKV: cute.Tensor,
        mCuKVActual: cute.Tensor,
        mdKVPartial: cute.Tensor,
        max_seqlen_kv: Int32,
        softmax_scale: Float32,
        softmax_scale_log2: Float32,
        sQ_layout: cute.ComposedLayout,
        sdO_layout: cute.ComposedLayout,
        sKV_layout: cute.ComposedLayout,
        sPdS_layout: cute.ComposedLayout,
        sStats_layout: cute.Layout,
        gmem_tiled_copy_D: cute.TiledCopy,
        tiled_mma_sdp: cute.TiledMma,
        tiled_mma_dkv: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
        tile_sched_params: FmhaClcDynamicTileSchedulerParams,
        cluster_layout_vmnk: cute.Layout,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        is_sched_warp = warp_idx == self.sched_warp_id
        kv_blocks = cute.ceil_div(max_seqlen_kv, self.block_k)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sQ = storage.sQ.get_tensor(sQ_layout)
        sdO = storage.sdO.get_tensor(sdO_layout)
        sKV = storage.sKV.get_tensor(sKV_layout)
        sP = storage.sP.get_tensor(sPdS_layout)
        sdS = storage.sdS.get_tensor(sPdS_layout)
        sLSE = storage.sLSE.get_tensor(sStats_layout)
        sDelta = storage.sDelta.get_tensor(sStats_layout)
        sQt = layout_utils.transpose_view(sQ)
        sdOt = layout_utils.transpose_view(sdO)

        clc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        clc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.num_threads,
        )
        clc_response_ptr = storage.clc_response.data_ptr()
        clc = ClcState.create(
            hw_scheduler=ClcDynamicPersistentTileScheduler.create(
                tile_sched_params.clc_hw_params(),
                cute.arch.block_idx(),
                cute.arch.grid_dim(),
                clc_response_ptr,
            ),
            pipeline=pipeline.PipelineClcFetchAsync.create(
                barrier_storage=storage.clc_mbar_ptr.data_ptr(),
                num_stages=self.num_clc_stage,
                producer_group=clc_pipeline_producer_group,
                consumer_group=clc_pipeline_consumer_group,
                tx_count=self.num_clc_response_bytes,
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            ),
            consumer_state=pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                self.num_clc_stage,
            ),
            producer_state=pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                self.num_clc_stage,
            ),
        )
        tile_sched = FmhaClcDynamicTileScheduler.create(
            tile_sched_params,
            cute.arch.block_idx(),
            cute.arch.grid_dim(),
            clc_response_ptr,
            clc,
        )
        pipeline.pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        gmem_thr_copy_D = gmem_tiled_copy_D.get_slice(tidx)
        thr_mma_sdp = tiled_mma_sdp.get_slice(tidx)
        thr_mma_dkv = tiled_mma_dkv.get_slice(tidx)
        tSrKV = utils.mma_make_fragment_A(sKV, thr_mma_sdp, swapAB=self.SdP_swapAB)
        tSrQ = utils.mma_make_fragment_B(sQ, thr_mma_sdp, swapAB=self.SdP_swapAB)
        tdPrdO = utils.mma_make_fragment_B(sdO, thr_mma_sdp, swapAB=self.SdP_swapAB)
        tdVrP = utils.mma_make_fragment_A(sP, thr_mma_dkv, swapAB=self.dKV_swapAB)
        tdVrdO = utils.mma_make_fragment_B(sdOt, thr_mma_dkv, swapAB=self.dKV_swapAB)
        tdKrdS = utils.mma_make_fragment_A(sdS, thr_mma_dkv, swapAB=self.dKV_swapAB)
        tdKrQ = utils.mma_make_fragment_B(sQt, thr_mma_dkv, swapAB=self.dKV_swapAB)

        smem_copy_atom = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.dtype
        )
        smem_copy_atom_t = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self.dtype
        )
        smem_thr_copy_KV = utils.make_tiled_copy_A(
            smem_copy_atom, tiled_mma_sdp, swapAB=self.SdP_swapAB
        ).get_slice(tidx)
        smem_thr_copy_Q = utils.make_tiled_copy_B(
            smem_copy_atom, tiled_mma_sdp, swapAB=self.SdP_swapAB
        ).get_slice(tidx)
        smem_thr_copy_PdS = utils.make_tiled_copy_A(
            smem_copy_atom, tiled_mma_dkv, swapAB=self.dKV_swapAB
        ).get_slice(tidx)
        smem_thr_copy_QdOt = utils.make_tiled_copy_B(
            smem_copy_atom_t, tiled_mma_dkv, swapAB=self.dKV_swapAB
        ).get_slice(tidx)
        tSsKV = smem_thr_copy_KV.partition_S(sKV)
        tSsQ = smem_thr_copy_Q.partition_S(sQ)
        tdPsdO = smem_thr_copy_Q.partition_S(sdO)
        tdVsP = smem_thr_copy_PdS.partition_S(sP)
        tdKsdS = smem_thr_copy_PdS.partition_S(sdS)
        tdVsdOt = smem_thr_copy_QdOt.partition_S(sdOt)
        tdKsQt = smem_thr_copy_QdOt.partition_S(sQt)

        r2s_thr_copy_PdS = cute.make_tiled_copy_C(
            cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.dtype,
                num_bits_per_copy=2 * self.dtype.width,
            ),
            tiled_mma_sdp,
        ).get_slice(tidx)
        tPsP = r2s_thr_copy_PdS.partition_D(sP)
        tdSsdS = r2s_thr_copy_PdS.partition_D(sdS)

        acc_shape_dkv = thr_mma_dkv.partition_shape_C((self.block_k, self.head_dim_padded))
        acc_dKV = cute.make_rmem_tensor(acc_shape_dkv, cutlass.Float32)
        cS = cute.make_identity_tensor(
            (self.block_q, self.block_k)
            if cutlass.const_expr(self.SdP_swapAB)
            else (self.block_k, self.block_q)
        )
        tScS = layout_utils.reshape_acc_to_mn(thr_mma_sdp.partition_C(cS))
        simt_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            cutlass.Float32,
            num_bits_per_copy=cutlass.Float32.width,
        )

        pipeline.pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)
        work_tile = tile_sched.initial_work_tile_info()
        while work_tile.is_valid_tile:
            packed_kv_split = work_tile.tile_idx[0]
            batch_idx = work_tile.tile_idx[2][0]
            head_block = work_tile.tile_idx[2][1]
            q_split = packed_kv_split // kv_blocks
            kv_block = packed_kv_split - q_split * kv_blocks

            if is_sched_warp:
                tile_sched.prefetch_next_work()

            q_start = mCuQ[batch_idx]
            q_len = mCuQ[batch_idx + 1] - q_start
            kv_start = mCuKV[batch_idx]
            kv_actual_start = mCuKVActual[batch_idx]
            kv_len = mCuKVActual[batch_idx + 1] - kv_actual_start
            swa_offset = kv_len - q_len

            if cutlass.const_expr(self.is_swa):
                q_block_start = max((kv_block * self.block_k - swa_offset) // self.block_q, 0)
                q_block_end = min(
                    cute.ceil_div(q_len, self.block_q),
                    cute.ceil_div(
                        (kv_block + 1) * self.block_k + self.window_size - 1 - swa_offset,
                        self.block_q,
                    ),
                )
            else:
                first_visible_q = (kv_block * self.block_k + 1) * self.compress_ratio - 1
                q_block_start = min(
                    max(first_visible_q // self.block_q, 0),
                    cute.ceil_div(q_len, self.block_q),
                )
                q_block_end = cute.ceil_div(q_len, self.block_q)
            if cutlass.const_expr(self.q_splits > 1):
                q_block_split_start = q_block_start + q_split * self.q_blocks_per_split
                q_block_split_end = min(
                    q_block_end,
                    q_block_split_start + self.q_blocks_per_split,
                )
                q_block_start = q_block_split_start
                q_block_end = q_block_split_end

            mKV_cur = cute.domain_offset((kv_start, 0), mKV[None, 0, None])
            gKV = cute.local_tile(
                mKV_cur, (self.block_k, self.head_dim_padded), (kv_block, 0)
            )
            tKVgKV = gmem_thr_copy_D.partition_S(gKV)
            tKVsKV = gmem_thr_copy_D.partition_D(sKV)
            cKV = cute.make_identity_tensor((self.block_k, self.head_dim_padded))
            tKVcKV = gmem_thr_copy_D.partition_S(cKV)
            t0KVcKV = gmem_tiled_copy_D.get_slice(0).partition_S(cKV)
            tKVpKV = utils.predicate_k(tKVcKV, limit=self.head_dim_padded)
            for m in cutlass.range_constexpr(cute.size(tKVsKV.shape[1])):
                if self.is_even_k or m < cute.size(tKVsKV.shape[1]) - 1 or tKVcKV[0, m, 0][0] < self.block_k:
                    pred_m = t0KVcKV[0, m, 0][0] < kv_len - kv_block * self.block_k - tKVcKV[0][0]
                    pred = cute.make_fragment_like(tKVpKV[None, 0, None])
                    for k in cutlass.range_constexpr(cute.size(pred.shape[1])):
                        for i in cutlass.range_constexpr(cute.size(pred.shape[0])):
                            pred[i, k] = tKVpKV[i, m, k] and pred_m
                    cute.copy(
                        gmem_tiled_copy_D,
                        tKVgKV[None, m, None],
                        tKVsKV[None, m, None],
                        pred=pred,
                    )
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()

            acc_dKV.fill(0.0)
            for head_off in cutlass.range_constexpr(self.heads_per_block):
                head_idx = head_block * self.heads_per_block + head_off
                if head_idx < self.heads:
                    if cutlass.const_expr(self.is_swa):
                        for q_block in cutlass.range(q_block_start, q_block_end, unroll=1):
                            self._compute_dkv_one_q_block(
                                mQ,
                                mdO,
                                q_start,
                                q_len,
                                kv_start,
                                kv_len,
                                q_block,
                                kv_block,
                                head_idx,
                                False,
                                swa_offset,
                                softmax_scale,
                                softmax_scale_log2,
                                mLSE,
                                mDelta,
                                sQ,
                                sdO,
                                sP,
                                sdS,
                                sLSE,
                                sDelta,
                                tSrKV,
                                tSrQ,
                                tdPrdO,
                                tdVrP,
                                tdVrdO,
                                tdKrdS,
                                tdKrQ,
                                tSsKV,
                                tSsQ,
                                tdPsdO,
                                tdVsP,
                                tdKsdS,
                                tdVsdOt,
                                tdKsQt,
                                tPsP,
                                tdSsdS,
                                r2s_thr_copy_PdS,
                                smem_thr_copy_KV,
                                smem_thr_copy_Q,
                                smem_thr_copy_PdS,
                                smem_thr_copy_QdOt,
                                gmem_tiled_copy_D,
                                gmem_thr_copy_D,
                                tiled_mma_sdp,
                                tiled_mma_dkv,
                                thr_mma_sdp,
                                thr_mma_dkv,
                                tScS,
                                acc_dKV,
                            )
                    else:
                        q_block_full_visible_start = cute.ceil_div(
                            (kv_block * self.block_k + self.block_k) * self.compress_ratio - 1,
                            self.block_q,
                        )
                        q_block_fast_start = min(max(q_block_start, q_block_full_visible_start), q_block_end)
                        q_block_fast_end = min(max(q_block_fast_start, q_len // self.block_q), q_block_end)
                        for q_block in cutlass.range(q_block_start, q_block_fast_start, unroll=1):
                            self._compute_dkv_one_q_block(
                                mQ,
                                mdO,
                                q_start,
                                q_len,
                                kv_start,
                                kv_len,
                                q_block,
                                kv_block,
                                head_idx,
                                False,
                                swa_offset,
                                softmax_scale,
                                softmax_scale_log2,
                                mLSE,
                                mDelta,
                                sQ,
                                sdO,
                                sP,
                                sdS,
                                sLSE,
                                sDelta,
                                tSrKV,
                                tSrQ,
                                tdPrdO,
                                tdVrP,
                                tdVrdO,
                                tdKrdS,
                                tdKrQ,
                                tSsKV,
                                tSsQ,
                                tdPsdO,
                                tdVsP,
                                tdKsdS,
                                tdVsdOt,
                                tdKsQt,
                                tPsP,
                                tdSsdS,
                                r2s_thr_copy_PdS,
                                smem_thr_copy_KV,
                                smem_thr_copy_Q,
                                smem_thr_copy_PdS,
                                smem_thr_copy_QdOt,
                                gmem_tiled_copy_D,
                                gmem_thr_copy_D,
                                tiled_mma_sdp,
                                tiled_mma_dkv,
                                thr_mma_sdp,
                                thr_mma_dkv,
                                tScS,
                                acc_dKV,
                            )
                        for q_block in cutlass.range(q_block_fast_start, q_block_fast_end, unroll=1):
                            self._compute_dkv_one_q_block(
                                mQ,
                                mdO,
                                q_start,
                                q_len,
                                kv_start,
                                kv_len,
                                q_block,
                                kv_block,
                                head_idx,
                                True,
                                swa_offset,
                                softmax_scale,
                                softmax_scale_log2,
                                mLSE,
                                mDelta,
                                sQ,
                                sdO,
                                sP,
                                sdS,
                                sLSE,
                                sDelta,
                                tSrKV,
                                tSrQ,
                                tdPrdO,
                                tdVrP,
                                tdVrdO,
                                tdKrdS,
                                tdKrQ,
                                tSsKV,
                                tSsQ,
                                tdPsdO,
                                tdVsP,
                                tdKsdS,
                                tdVsdOt,
                                tdKsQt,
                                tPsP,
                                tdSsdS,
                                r2s_thr_copy_PdS,
                                smem_thr_copy_KV,
                                smem_thr_copy_Q,
                                smem_thr_copy_PdS,
                                smem_thr_copy_QdOt,
                                gmem_tiled_copy_D,
                                gmem_thr_copy_D,
                                tiled_mma_sdp,
                                tiled_mma_dkv,
                                thr_mma_sdp,
                                thr_mma_dkv,
                                tScS,
                                acc_dKV,
                            )
                        for q_block in cutlass.range(q_block_fast_end, q_block_end, unroll=1):
                            self._compute_dkv_one_q_block(
                                mQ,
                                mdO,
                                q_start,
                                q_len,
                                kv_start,
                                kv_len,
                                q_block,
                                kv_block,
                                head_idx,
                                False,
                                swa_offset,
                                softmax_scale,
                                softmax_scale_log2,
                                mLSE,
                                mDelta,
                                sQ,
                                sdO,
                                sP,
                                sdS,
                                sLSE,
                                sDelta,
                                tSrKV,
                                tSrQ,
                                tdPrdO,
                                tdVrP,
                                tdVrdO,
                                tdKrdS,
                                tdKrQ,
                                tSsKV,
                                tSsQ,
                                tdPsdO,
                                tdVsP,
                                tdKsdS,
                                tdVsdOt,
                                tdKsQt,
                                tPsP,
                                tdSsdS,
                                r2s_thr_copy_PdS,
                                smem_thr_copy_KV,
                                smem_thr_copy_Q,
                                smem_thr_copy_PdS,
                                smem_thr_copy_QdOt,
                                gmem_tiled_copy_D,
                                gmem_thr_copy_D,
                                tiled_mma_sdp,
                                tiled_mma_dkv,
                                thr_mma_sdp,
                                thr_mma_dkv,
                                tScS,
                                acc_dKV,
                            )

            mPartial_cur = cute.domain_offset(
                (kv_start, 0), mdKVPartial[q_split, head_block, None, 0, None]
            )
            gPartial = cute.local_tile(
                mPartial_cur, (self.block_k, self.head_dim_padded), (kv_block, 0)
            )
            tCgPartial = thr_mma_dkv.partition_C(gPartial)
            cute.copy(simt_atom, acc_dKV, tCgPartial)
            cute.arch.barrier()
            work_tile = tile_sched.advance_to_next_work()

        if is_sched_warp:
            tile_sched.producer_tail()

    @cute.jit
    def _compute_dkv_one_q_block(
        self,
        mQ: cute.Tensor,
        mdO: cute.Tensor,
        q_start: Int32,
        q_len: Int32,
        kv_start: Int32,
        kv_len: Int32,
        q_block: Int32,
        kv_block: Int32,
        head_idx: Int32,
        skip_comp_mask: cutlass.Constexpr[bool],
        swa_offset: Int32,
        softmax_scale: Float32,
        softmax_scale_log2: Float32,
        mLSE: cute.Tensor,
        mDelta: cute.Tensor,
        sQ: cute.Tensor,
        sdO: cute.Tensor,
        sP: cute.Tensor,
        sdS: cute.Tensor,
        sLSE: cute.Tensor,
        sDelta: cute.Tensor,
        tSrKV: cute.Tensor,
        tSrQ: cute.Tensor,
        tdPrdO: cute.Tensor,
        tdVrP: cute.Tensor,
        tdVrdO: cute.Tensor,
        tdKrdS: cute.Tensor,
        tdKrQ: cute.Tensor,
        tSsKV: cute.Tensor,
        tSsQ: cute.Tensor,
        tdPsdO: cute.Tensor,
        tdVsP: cute.Tensor,
        tdKsdS: cute.Tensor,
        tdVsdOt: cute.Tensor,
        tdKsQt: cute.Tensor,
        tPsP: cute.Tensor,
        tdSsdS: cute.Tensor,
        r2s_thr_copy_PdS: cute.TiledCopy,
        smem_thr_copy_KV: cute.TiledCopy,
        smem_thr_copy_Q: cute.TiledCopy,
        smem_thr_copy_PdS: cute.TiledCopy,
        smem_thr_copy_QdOt: cute.TiledCopy,
        gmem_tiled_copy_D: cute.TiledCopy,
        gmem_thr_copy_D: cute.TiledCopy,
        tiled_mma_sdp: cute.TiledMma,
        tiled_mma_dkv: cute.TiledMma,
        thr_mma_sdp: cute.ThrMma,
        thr_mma_dkv: cute.ThrMma,
        tScS: cute.Tensor,
        acc_dKV: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        mQ_cur = cute.domain_offset((q_start, 0), mQ[None, head_idx, None])
        mdO_cur = cute.domain_offset((q_start, 0), mdO[None, head_idx, None])
        gQ = cute.local_tile(mQ_cur, (self.block_q, self.head_dim_padded), (q_block, 0))
        gdO = cute.local_tile(mdO_cur, (self.block_q, self.head_dim_padded), (q_block, 0))
        tQgQ = gmem_thr_copy_D.partition_S(gQ)
        tQsQ = gmem_thr_copy_D.partition_D(sQ)
        tdOgdO = gmem_thr_copy_D.partition_S(gdO)
        tdOsdO = gmem_thr_copy_D.partition_D(sdO)
        cQ = cute.make_identity_tensor((self.block_q, self.head_dim_padded))
        tQcQ = gmem_thr_copy_D.partition_S(cQ)
        t0QcQ = gmem_tiled_copy_D.get_slice(0).partition_S(cQ)
        tQpQ = utils.predicate_k(tQcQ, limit=self.head_dim_padded)
        for m in cutlass.range_constexpr(cute.size(tQsQ.shape[1])):
            if self.is_even_q or m < cute.size(tQsQ.shape[1]) - 1 or tQcQ[0, m, 0][0] < self.block_q:
                pred_m = t0QcQ[0, m, 0][0] < q_len - q_block * self.block_q - tQcQ[0][0]
                pred = cute.make_fragment_like(tQpQ[None, 0, None])
                for k in cutlass.range_constexpr(cute.size(pred.shape[1])):
                    for i in cutlass.range_constexpr(cute.size(pred.shape[0])):
                        pred[i, k] = tQpQ[i, m, k] and pred_m
                cute.copy(gmem_tiled_copy_D, tQgQ[None, m, None], tQsQ[None, m, None], pred=pred)
                cute.copy(gmem_tiled_copy_D, tdOgdO[None, m, None], tdOsdO[None, m, None], pred=pred)
        if tidx < self.block_q:
            q_stat_rel = q_block * self.block_q + tidx
            if q_stat_rel < q_len:
                sLSE[tidx] = mLSE[q_start + q_stat_rel, head_idx] * LOG2E
                sDelta[tidx] = mDelta[q_start + q_stat_rel, head_idx]
            else:
                sLSE[tidx] = Float32(0.0)
                sDelta[tidx] = Float32(0.0)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()

        acc_shape_s = thr_mma_sdp.partition_shape_C(
            (self.block_q, self.block_k)
            if cutlass.const_expr(self.SdP_swapAB)
            else (self.block_k, self.block_q)
        )
        acc_S = cute.make_rmem_tensor(acc_shape_s, cutlass.Float32)
        acc_S.fill(0.0)
        sm80_utils.gemm(
            thr_mma_sdp,
            acc_S,
            tSrKV,
            tSrQ,
            tSsKV,
            tSsQ,
            smem_thr_copy_KV,
            smem_thr_copy_Q,
            swap_AB=self.SdP_swapAB,
        )
        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S)
        nrow = cutlass.const_expr(cute.size(acc_S_mn.shape[0]))
        ncol = cutlass.const_expr(cute.size(acc_S_mn.shape[1]))
        if cutlass.const_expr(skip_comp_mask):
            for r in cutlass.range_constexpr(nrow):
                for c in cutlass.range_constexpr(ncol):
                    if cutlass.const_expr(self.SdP_swapAB):
                        q_local = tScS[r, c][0]
                    else:
                        q_local = tScS[r, c][1]
                    lse_val = sLSE[q_local]
                    acc_S_mn[r, c] = cute.math.exp2(
                        acc_S_mn[r, c] * softmax_scale_log2 - lse_val,
                        fastmath=True,
                    )
        elif cutlass.const_expr(self.is_swa):
            for r in cutlass.range_constexpr(nrow):
                for c in cutlass.range_constexpr(ncol):
                    if cutlass.const_expr(self.SdP_swapAB):
                        q_local = tScS[r, c][0]
                        kv_rel = kv_block * self.block_k + tScS[r, c][1]
                    else:
                        kv_rel = kv_block * self.block_k + tScS[r, 0][0]
                        q_local = tScS[r, c][1]
                    q_rel = q_block * self.block_q + q_local
                    keep = (kv_rel < kv_len) and (q_rel < q_len)
                    q_pos = q_rel + swa_offset
                    keep = keep and (q_pos >= kv_rel) and ((q_pos - kv_rel) < self.window_size)
                    lse_val = sLSE[q_local]
                    prob = cute.math.exp2(acc_S_mn[r, c] * softmax_scale_log2 - lse_val, fastmath=True)
                    acc_S_mn[r, c] = prob if keep else Float32(0.0)
        else:
            for r in cutlass.range_constexpr(nrow):
                for c in cutlass.range_constexpr(ncol):
                    if cutlass.const_expr(self.SdP_swapAB):
                        q_local = tScS[r, c][0]
                        kv_rel = kv_block * self.block_k + tScS[r, c][1]
                    else:
                        kv_rel = kv_block * self.block_k + tScS[r, 0][0]
                        q_local = tScS[r, c][1]
                    q_rel = q_block * self.block_q + q_local
                    keep = (kv_rel < kv_len) and (q_rel < q_len)
                    keep = keep and (q_rel >= ((kv_rel + 1) * self.compress_ratio - 1))
                    lse_val = sLSE[q_local]
                    prob = cute.math.exp2(
                        acc_S_mn[r, c] * softmax_scale_log2 - lse_val,
                        fastmath=True,
                    )
                    acc_S_mn[r, c] = prob if keep else Float32(0.0)

        acc_dP = cute.make_rmem_tensor(acc_shape_s, cutlass.Float32)
        acc_dP.fill(0.0)
        sm80_utils.gemm(
            thr_mma_sdp,
            acc_dP,
            tSrKV,
            tdPrdO,
            tSsKV,
            tdPsdO,
            smem_thr_copy_KV,
            smem_thr_copy_Q,
            swap_AB=self.SdP_swapAB,
        )
        acc_dP_mn = layout_utils.reshape_acc_to_mn(acc_dP)
        for r in cutlass.range_constexpr(nrow):
            for c in cutlass.range_constexpr(ncol):
                if cutlass.const_expr(self.SdP_swapAB):
                    q_local = tScS[r, c][0]
                else:
                    q_local = tScS[r, c][1]
                delta = sDelta[q_local]
                acc_dP_mn[r, c] = acc_S_mn[r, c] * (acc_dP_mn[r, c] - delta) * softmax_scale

        rP = cute.make_fragment_like(acc_S, self.dtype)
        rP.store(acc_S.load().to(self.dtype))
        tPrP = r2s_thr_copy_PdS.retile(rP)
        cute.copy(r2s_thr_copy_PdS, tPrP, tPsP)
        rdS = cute.make_fragment_like(acc_dP, self.dtype)
        rdS.store(acc_dP.load().to(self.dtype))
        tdSrdS = r2s_thr_copy_PdS.retile(rdS)
        cute.copy(r2s_thr_copy_PdS, tdSrdS, tdSsdS)
        cute.arch.barrier()

        sm80_utils.gemm(
            thr_mma_dkv,
            acc_dKV,
            tdVrP,
            tdVrdO,
            tdVsP,
            tdVsdOt,
            smem_thr_copy_PdS,
            smem_thr_copy_QdOt,
            swap_AB=self.dKV_swapAB,
        )
        sm80_utils.gemm(
            thr_mma_dkv,
            acc_dKV,
            tdKrdS,
            tdKrQ,
            tdKsdS,
            tdKsQt,
            smem_thr_copy_PdS,
            smem_thr_copy_QdOt,
            swap_AB=self.dKV_swapAB,
        )
        cute.arch.barrier()


class CsaMlaBwdDkvPartialReduceCute:
    """Reduce fp32 DKV partials across q-splits/head-blocks and cast to output dtype."""

    def __init__(
        self,
        q_splits: int,
        head_blocks: int,
        head_dim: int,
        rows_per_block: int = 2,
        num_threads: int = 256,
    ):
        assert head_dim == 512
        self.q_splits = q_splits
        self.head_blocks = head_blocks
        self.head_dim = head_dim
        self.rows_per_block = rows_per_block
        self.num_threads = num_threads

    @cute.jit
    def __call__(
        self,
        mPartial: cute.Tensor,
        mDKV: cute.Tensor,
        total_kv: Int32,
        stream: cuda.CUstream = None,
    ):
        if cutlass.const_expr(mPartial.element_type != cutlass.Float32):
            raise TypeError("DKV partial tensor must be fp32")
        if cutlass.const_expr(mDKV.element_type not in [cutlass.Float16, cutlass.BFloat16]):
            raise TypeError("DKV output must be fp16/bf16")
        self.dtype = mDKV.element_type
        mPartial = assume_tensor_aligned(mPartial)
        mDKV = assume_tensor_aligned(mDKV)
        grid = (cute.ceil_div(total_kv, self.rows_per_block),)
        self.kernel(mPartial, mDKV, total_kv).launch(
            grid=grid,
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mPartial: cute.Tensor,
        mDKV: cute.Tensor,
        total_kv: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        row_block, _, _ = cute.arch.block_idx()
        elems_per_block = cutlass.const_expr(self.rows_per_block * self.head_dim)
        for elem in cutlass.range(tidx, elems_per_block, self.num_threads, unroll=1):
            row = row_block * self.rows_per_block + elem // self.head_dim
            d = elem - (elem // self.head_dim) * self.head_dim
            if row < total_kv:
                acc = Float32(0.0)
                for split_idx in cutlass.range_constexpr(self.q_splits):
                    for head_block in cutlass.range_constexpr(self.head_blocks):
                        acc += mPartial[split_idx, head_block, row, 0, d]
                mDKV[row, 0, d] = acc.to(self.dtype)


def csa_mla_bwd_dkv_partial_reduce_cute(
    partial: torch.Tensor,
    out: torch.Tensor,
    q_splits: int,
    head_blocks: int,
    rows_per_block: int = 2,
) -> torch.Tensor:
    assert partial.dtype == torch.float32
    assert out.shape[-1] == 512
    current_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    compile_key = (
        out.dtype,
        q_splits,
        head_blocks,
        rows_per_block,
        out.shape[0],
        out.shape[-1],
    )
    if compile_key not in csa_mla_bwd_dkv_partial_reduce_cute.compile_cache:
        kernel = CsaMlaBwdDkvPartialReduceCute(
            q_splits=q_splits,
            head_blocks=head_blocks,
            head_dim=out.shape[-1],
            rows_per_block=rows_per_block,
        )
        csa_mla_bwd_dkv_partial_reduce_cute.compile_cache[compile_key] = cute.compile(
            kernel,
            to_cute_tensor(partial, assumed_align=4),
            to_cute_tensor(out),
            out.shape[0],
            current_stream,
            options="--enable-tvm-ffi",
        )
    if not is_fake_mode():
        csa_mla_bwd_dkv_partial_reduce_cute.compile_cache[compile_key](
            partial,
            out,
            out.shape[0],
        )
    return out


csa_mla_bwd_dkv_partial_reduce_cute.compile_cache = get_jit_cache(
    "csa_mla_bwd_dkv_partial_reduce_cute"
)


class CsaMlaBwdSinkGradPartialCute:
    """Compute partial learnable-sink gradients without materializing p_sink."""

    def __init__(
        self,
        heads: int,
        rows_per_block: int = 256,
        num_threads: int = 256,
    ):
        self.heads = heads
        self.rows_per_block = rows_per_block
        self.num_threads = num_threads
        self.num_warps = num_threads // 32

    def _get_shared_storage_cls(self):
        sWarp_struct = cute.struct.Align[
            cute.struct.MemRange[Float32, self.num_warps], 128
        ]

        @cute.struct
        class SharedStorage:
            sWarp: sWarp_struct

        return SharedStorage

    @cute.jit
    def __call__(
        self,
        mSink: cute.Tensor,
        mLSE: cute.Tensor,
        mDelta: cute.Tensor,
        mPartial: cute.Tensor,
        total_q: Int32,
        stream: cuda.CUstream = None,
    ):
        if cutlass.const_expr(mLSE.element_type != cutlass.Float32 or mDelta.element_type != cutlass.Float32):
            raise TypeError("LSE and Delta must be fp32")
        if cutlass.const_expr(mPartial.element_type != cutlass.Float32):
            raise TypeError("Sink partial must be fp32")
        self.dtype = mSink.element_type
        SharedStorage = self._get_shared_storage_cls()
        grid = (self.heads, cute.ceil_div(total_q, self.rows_per_block))
        self.kernel(mSink, mLSE, mDelta, mPartial, total_q, SharedStorage).launch(
            grid=grid,
            block=[self.num_threads, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mSink: cute.Tensor,
        mLSE: cute.Tensor,
        mDelta: cute.Tensor,
        mPartial: cute.Tensor,
        total_q: Int32,
        SharedStorage: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        head_idx, row_block, _ = cute.arch.block_idx()
        lane = tidx % 32
        warp_idx = tidx // 32
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sWarp = storage.sWarp.get_tensor(cute.make_layout((self.num_warps,)))

        acc = Float32(0.0)
        sink_val = mSink[head_idx].to(Float32)
        for row_off in cutlass.range(tidx, self.rows_per_block, self.num_threads, unroll=1):
            row = row_block * self.rows_per_block + row_off
            if row < total_q:
                prob_sink = cute.math.exp2((sink_val - mLSE[row, head_idx]) * LOG2E, fastmath=True)
                acc += -prob_sink * mDelta[row, head_idx]

        warp_sum = utils.warp_reduction_sum(acc)
        if lane == 0:
            sWarp[warp_idx] = warp_sum
        cute.arch.barrier()

        block_sum = Float32(0.0)
        if warp_idx == 0:
            block_sum = sWarp[lane] if lane < self.num_warps else Float32(0.0)
            block_sum = utils.warp_reduction_sum(block_sum)
        if tidx == 0:
            mPartial[row_block, head_idx] = block_sum


class CsaMlaBwdSinkGradReduceCute:
    """Reduce sink-gradient partials over row blocks and cast to sink dtype."""

    def __init__(
        self,
        row_blocks: int,
        heads: int,
        num_threads: int = 256,
    ):
        self.row_blocks = row_blocks
        self.heads = heads
        self.num_threads = num_threads
        self.num_warps = num_threads // 32

    def _get_shared_storage_cls(self):
        sWarp_struct = cute.struct.Align[
            cute.struct.MemRange[Float32, self.num_warps], 128
        ]

        @cute.struct
        class SharedStorage:
            sWarp: sWarp_struct

        return SharedStorage

    @cute.jit
    def __call__(
        self,
        mPartial: cute.Tensor,
        mSinkGrad: cute.Tensor,
        stream: cuda.CUstream = None,
    ):
        if cutlass.const_expr(mPartial.element_type != cutlass.Float32):
            raise TypeError("Sink partial must be fp32")
        self.dtype = mSinkGrad.element_type
        SharedStorage = self._get_shared_storage_cls()
        self.kernel(mPartial, mSinkGrad, SharedStorage).launch(
            grid=(self.heads,),
            block=[self.num_threads, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mPartial: cute.Tensor,
        mSinkGrad: cute.Tensor,
        SharedStorage: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        head_idx, _, _ = cute.arch.block_idx()
        lane = tidx % 32
        warp_idx = tidx // 32
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sWarp = storage.sWarp.get_tensor(cute.make_layout((self.num_warps,)))

        acc = Float32(0.0)
        for row_block in cutlass.range(tidx, self.row_blocks, self.num_threads, unroll=1):
            acc += mPartial[row_block, head_idx]
        warp_sum = utils.warp_reduction_sum(acc)
        if lane == 0:
            sWarp[warp_idx] = warp_sum
        cute.arch.barrier()

        block_sum = Float32(0.0)
        if warp_idx == 0:
            block_sum = sWarp[lane] if lane < self.num_warps else Float32(0.0)
            block_sum = utils.warp_reduction_sum(block_sum)
        if tidx == 0:
            mSinkGrad[head_idx] = block_sum.to(self.dtype)


def csa_mla_bwd_sink_grad_cute(
    sink: torch.Tensor,
    lse: torch.Tensor,
    delta: torch.Tensor,
    rows_per_block: int = 256,
) -> torch.Tensor:
    assert lse.dtype == torch.float32
    assert delta.dtype == torch.float32
    sink = sink.contiguous()
    lse = lse.contiguous()
    delta = delta.contiguous()
    total_q, heads = delta.shape
    row_blocks = triton_cdiv(total_q, rows_per_block)
    partial = torch.empty((row_blocks, heads), dtype=torch.float32, device=delta.device)
    out = torch.empty_like(sink)
    current_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    partial_key = (sink.dtype, heads, total_q, rows_per_block)
    if partial_key not in csa_mla_bwd_sink_grad_cute.partial_compile_cache:
        kernel = CsaMlaBwdSinkGradPartialCute(
            heads=heads,
            rows_per_block=rows_per_block,
        )
        csa_mla_bwd_sink_grad_cute.partial_compile_cache[partial_key] = cute.compile(
            kernel,
            to_cute_tensor(sink),
            to_cute_tensor(lse, assumed_align=4),
            to_cute_tensor(delta, assumed_align=4),
            to_cute_tensor(partial, assumed_align=4),
            total_q,
            current_stream,
            options="--enable-tvm-ffi",
        )
    reduce_key = (sink.dtype, heads, row_blocks)
    if reduce_key not in csa_mla_bwd_sink_grad_cute.reduce_compile_cache:
        kernel = CsaMlaBwdSinkGradReduceCute(
            row_blocks=row_blocks,
            heads=heads,
        )
        csa_mla_bwd_sink_grad_cute.reduce_compile_cache[reduce_key] = cute.compile(
            kernel,
            to_cute_tensor(partial, assumed_align=4),
            to_cute_tensor(out),
            current_stream,
            options="--enable-tvm-ffi",
        )
    if not is_fake_mode():
        csa_mla_bwd_sink_grad_cute.partial_compile_cache[partial_key](
            sink,
            lse,
            delta,
            partial,
            total_q,
        )
        csa_mla_bwd_sink_grad_cute.reduce_compile_cache[reduce_key](
            partial,
            out,
        )
    return out


csa_mla_bwd_sink_grad_cute.partial_compile_cache = get_jit_cache(
    "csa_mla_bwd_sink_grad_cute_partial"
)
csa_mla_bwd_sink_grad_cute.reduce_compile_cache = get_jit_cache(
    "csa_mla_bwd_sink_grad_cute_reduce"
)


def csa_mla_bwd_dkv_full_cute(
    qv: torch.Tensor,
    kv: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    delta: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    window_size: int,
    compress_ratio: int,
    is_swa: bool,
    heads_per_block: int = 4,
    q_splits: Optional[int] = None,
    block_k: int = 64,
    block_q: int = 64,
    num_threads: int = 256,
    reduce_rows_per_block: int = 2,
    use_clc_scheduler: Optional[bool] = None,
    atom_layout_m_sdp: int = 4,
    atom_layout_m_dkv: int = 2,
    sdp_swap_ab: bool = False,
) -> torch.Tensor:
    total_q, heads, dim = qv.shape
    assert dim == 512
    kv_in, cu_seqlens_kv_storage, unpad_meta = _pad_kv_to_block(kv, cu_seqlens_kv, block_k)
    head_blocks = triton_cdiv(heads, heads_per_block)
    total_q_blocks = triton_cdiv(max_seqlen_q, block_q)
    if q_splits is None:
        q_splits = 1 if is_swa else min(4, max(1, total_q_blocks))
    q_splits = max(1, int(q_splits))
    q_blocks_per_split = triton_cdiv(total_q_blocks, q_splits)
    if use_clc_scheduler is None:
        use_clc_scheduler = not is_swa
    partial = torch.empty((q_splits, head_blocks, kv_in.shape[0], 1, dim), dtype=torch.float32, device=qv.device)
    softmax_scale = 1.0 / math.sqrt(dim)
    softmax_scale_log2 = softmax_scale * LOG2E
    current_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    compile_key = (
        qv.dtype,
        heads,
        dim,
        window_size,
        compress_ratio,
        is_swa,
        heads_per_block,
        q_splits,
        q_blocks_per_split,
        block_k,
        block_q,
        num_threads,
        reduce_rows_per_block,
        use_clc_scheduler,
        atom_layout_m_sdp,
        atom_layout_m_dkv,
        sdp_swap_ab,
        qv.shape[0],
        kv_in.shape[0],
    )
    if compile_key not in csa_mla_bwd_dkv_full_cute.compile_cache:
        kernel = CsaMlaBwdDkvFullCute(
            heads=heads,
            head_dim=dim,
            window_size=window_size,
            compress_ratio=compress_ratio,
            is_swa=is_swa,
            heads_per_block=heads_per_block,
            q_splits=q_splits,
            q_blocks_per_split=q_blocks_per_split,
            block_k=block_k,
            block_q=block_q,
            num_threads=num_threads,
            use_clc_scheduler=use_clc_scheduler,
            atom_layout_m_sdp=atom_layout_m_sdp,
            atom_layout_m_dkv=atom_layout_m_dkv,
            sdp_swap_ab=sdp_swap_ab,
        )
        csa_mla_bwd_dkv_full_cute.compile_cache[compile_key] = cute.compile(
            kernel,
            to_cute_tensor(qv),
            to_cute_tensor(kv_in),
            to_cute_tensor(dout),
            to_cute_tensor(lse, assumed_align=4),
            to_cute_tensor(delta, assumed_align=4),
            to_cute_tensor(cu_seqlens_q, assumed_align=4),
            to_cute_tensor(cu_seqlens_kv_storage, assumed_align=4),
            to_cute_tensor(cu_seqlens_kv, assumed_align=4),
            to_cute_tensor(partial, assumed_align=4),
            max_seqlen_q,
            triton_cdiv(max_seqlen_kv, block_k) * block_k,
            softmax_scale,
            softmax_scale_log2,
            current_stream,
            options="--enable-tvm-ffi",
        )
    if not is_fake_mode():
        csa_mla_bwd_dkv_full_cute.compile_cache[compile_key](
            qv.detach(),
            kv_in.detach(),
            dout.detach(),
            lse,
            delta,
            cu_seqlens_q,
            cu_seqlens_kv_storage,
            cu_seqlens_kv,
            partial,
            max_seqlen_q,
            triton_cdiv(max_seqlen_kv, block_k) * block_k,
            softmax_scale,
            softmax_scale_log2,
        )
    if q_splits > 1:
        dkv_padded = torch.empty_like(kv_in)
        csa_mla_bwd_dkv_partial_reduce_cute(
            partial,
            dkv_padded,
            q_splits,
            head_blocks,
            reduce_rows_per_block,
        )
    else:
        dkv_padded = partial.sum((0, 1)).to(kv.dtype)
    if unpad_meta is None:
        return dkv_padded
    return _unpad_kv_from_block(dkv_padded, kv, cu_seqlens_kv, cu_seqlens_kv_storage)


def triton_cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _pad_kv_to_block(kv: torch.Tensor, cu: torch.Tensor, block: int):
    lens = (cu[1:] - cu[:-1]).detach().cpu().tolist()
    max_padded_len = triton_cdiv(max((int(x) for x in lens), default=0), block) * block
    padded_lens = [max_padded_len for _ in lens]
    if sum(padded_lens) == kv.shape[0] and all(int(x) == int(y) for x, y in zip(lens, padded_lens)):
        return kv.contiguous(), cu, None
    padded = kv.new_zeros((sum(padded_lens), kv.shape[1], kv.shape[2]))
    starts = [0]
    for length in padded_lens:
        starts.append(starts[-1] + length)
    cu_padded = torch.tensor(starts, dtype=cu.dtype, device=cu.device)
    for b, length in enumerate(lens):
        if length:
            src0 = int(cu[b].item())
            padded[starts[b] : starts[b] + int(length)].copy_(kv[src0 : src0 + int(length)])
    return padded.contiguous(), cu_padded, padded_lens


def _unpad_kv_from_block(
    dkv_padded: torch.Tensor,
    kv_ref: torch.Tensor,
    cu: torch.Tensor,
    cu_padded: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty_like(kv_ref)
    batch = cu.numel() - 1
    for b in range(batch):
        src0 = int(cu_padded[b].item())
        dst0 = int(cu[b].item())
        length = int((cu[b + 1] - cu[b]).item())
        if length:
            out[dst0 : dst0 + length].copy_(dkv_padded[src0 : src0 + length])
    return out


csa_mla_bwd_dkv_full_cute.compile_cache = get_jit_cache("csa_mla_bwd_dkv_full_cute")
