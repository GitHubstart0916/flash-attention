import math

import cuda.bindings.driver as cuda
import torch

import cutlass
import cutlass.cute as cute
from cutlass import Float32
from cutlass.cute.typing import Int32
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils as cutlass_utils
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from flash_attn.cute.cache_utils import get_jit_cache
from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned, to_cute_tensor
from flash_attn.cute.testing import is_fake_mode
from flash_attn.cute.tile_scheduler import (
    SM100_TMEM_CAPACITY_COLUMNS,
    make_sm100_thread_cooperative_group as make_thread_cooperative_group,
)
from flash_attn.cute.sm100_hd256_2cta_fmha_backward_dkdvkernel import (
    BlackwellFusedMultiHeadAttentionBackwardDKDVKernel,
)


LAYOUT_RANK_CONSTANT = 3


@cute.jit
def split_wg(t: cute.Tensor, num_warp_groups: Int32, wg_idx: Int32) -> cute.Tensor:
    ret = None
    if cutlass.const_expr(cute.rank(t.layout) == LAYOUT_RANK_CONSTANT):
        p = cute.composition(
            t,
            cute.make_layout(
                (
                    t.shape[0],
                    t.shape[1],
                    (num_warp_groups, cute.size(t, mode=[2]) // num_warp_groups),
                )
            ),
        )
        ret = p[None, None, (wg_idx, None)]
    else:
        p = cute.composition(
            t,
            cute.make_layout(
                (
                    t.shape[0],
                    t.shape[1],
                    t.shape[2],
                    (num_warp_groups, cute.size(t, mode=[3]) // num_warp_groups),
                )
            ),
        )
        ret = p[None, None, None, (wg_idx, None)]
    return ret


class CsaCompDkvSm100Small:
    """CSA comp-DKV-only SM100 path.

    Fixed assumptions:
    - Q/dO are rank-3 unpadded tensors: (S, H, 512)
    - comp KV is MQA: (K, 1, 512) and K == V
    - ratio == 4, batch == 1, no causal/window/varlen scheduler
    - dK and dV are emitted one 256-D half per launch to stay within tcgen05 N <= 256
    """

    def __init__(self):
        self.acc_dtype = cutlass.Float32
        self.cta_tiler = (64, 64, 512, 256, 256, 256)
        self.use_clc_scheduler = False
        self.sched_warp_id = None
        self.csa_compress_ratio = 4
        self.has_csa_compression = True
        self.tile_shape_Q = 64
        self.tile_shape_K = 64
        self.tile_shape_QK = 512
        self.tile_shape_dQ_K = 256
        self.tile_shape_dV_dO = 256
        self.tile_shape_dP_dO = 256
        self.reuse_k_as_v = False
        self.KQ_mma_tiler = (64, 64, 512)
        self.VdO_mma_tiler = (64, 64, 256)
        self.PdO_mma_tiler = (64, 256, 64)
        self.dSQ_mma_tiler = (64, 256, 64)
        self.cluster_shape_mn = (1, 1)
        self.is_causal = False
        self.window_size_left = -1
        self.window_size_right = -1
        self.has_sliding_window = False
        self.compute_warp_id = (0, 1, 2, 3, 4, 5, 6, 7)
        self.mma_warp_id = 8
        self.load_warp_id = 9
        self.empty_warp_id = 10
        self.num_compute_warps = 8
        self.tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS
        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * (self.num_compute_warps + 4)
        self.cta_sync_bar_id = 0
        self.tmem_alloc_sync_bar_id = 1
        self.compute_sync_bar_id = 2
        self.epilogue_sync_bar_id = 3
        self.reduce_sync_bar_id = 4
        self.tmem_dK_offset = 0
        self.tmem_dV_offset = self.Tmemory_offset(0, self.tile_shape_dQ_K // 2)
        self.tmem_S_offset = self.Tmemory_offset(
            0, (self.tile_shape_dQ_K + self.tile_shape_dV_dO) // 2
        )
        self.tmem_dP_offset = self.Tmemory_offset(
            0,
            (self.tile_shape_dQ_K + self.tile_shape_dV_dO) // 2
            + self.cta_tiler[0] // 2,
        )
        self.num_regs_reduce = 152
        self.num_regs_compute = 128
        self.num_regs_mma = 128
        self.num_regs_empty = 96
        self.num_regs_load = 96
        self.buffer_align_bytes = 128

    def _setup_attributes(self):
        self.load_mma_Q_stage = 1
        self.load_mma_K_stage = 1
        self.load_mma_V_stage = 1
        self.load_mma_QT_stage = 1
        self.load_mma_dO_stage = 1
        self.load_compute_LSE_stage = 1
        self.load_compute_sum_OdO_stage = 1
        self.mma_compute_S_stage = 1
        self.mma_compute_dP_stage = 1
        self.compute_mma_P_stage = 1
        self.compute_mma_dS_stage = 1
        self.mma_compute_dKdV_stage = 2

    @staticmethod
    def Tmemory_offset(lane, col):
        return (lane << 16) + col

    mma_2cta = BlackwellFusedMultiHeadAttentionBackwardDKDVKernel.mma_2cta
    reg_to_smem_mma64x64 = BlackwellFusedMultiHeadAttentionBackwardDKDVKernel.reg_to_smem_mma64x64
    quantize = BlackwellFusedMultiHeadAttentionBackwardDKDVKernel.quantize
    store = BlackwellFusedMultiHeadAttentionBackwardDKDVKernel.store
    epilogue_clear = BlackwellFusedMultiHeadAttentionBackwardDKDVKernel.epilogue_clear

    @cute.jit
    def get_Q_block_min_max(
        self,
        seq_Q: Int32,
        seq_K: Int32,
        blk_coord_k: Int32,
        is_2cta: bool,
    ):
        del seq_K, is_2cta
        q_block_max = cute.ceil_div(seq_Q, self.tile_shape_Q)
        first_visible_q = (
            (blk_coord_k * self.tile_shape_K + 1) * self.csa_compress_ratio
            - 1
        )
        q_block_min = first_visible_q // self.tile_shape_Q
        return q_block_min, q_block_max

    @staticmethod
    def _compute_bwd_grid(problem_shape, block_k: int):
        K = problem_shape[1]
        _, H_K = problem_shape[3][0]
        B = problem_shape[3][1]
        return (cute.ceil_div(K, block_k), cute.size(H_K), cute.size(B))

    @cute.jit
    def compute(
        self,
        tSTtST: cute.Tensor,
        tdPTtdPT: cute.Tensor,
        tdVrP: cute.Tensor,
        sP: cute.Tensor,
        sLSE: cute.Tensor,
        sdST: cute.Tensor,
        sdOT: cute.Tensor,
        sSum_OdO: cute.Tensor,
        dK: cute.Tensor,
        dV: cute.Tensor,
        tdKtdK: cute.Tensor,
        tdVtdV: cute.Tensor,
        PdO_tiled_mma: cute.TiledMma,
        dSQ_tiled_mma: cute.TiledMma,
        blk_coord: cute.Coord,
        blk_offset: cute.Shape,
        problem_shape: tuple[Int32, Int32, Int32, tuple[tuple[Int32, Int32], Int32]],
        iter_count: Int32,
        iter_start: Int32,
        iter_end: Int32,
        scale_softmax: cutlass.Float32,
        mma_compute_S_producer,
        mma_compute_S_consumer,
        compute_mma_P_producer,
        compute_mma_P_consumer,
        load_compute_LSE_producer,
        load_compute_LSE_consumer,
        load_compute_sum_OdO_producer,
        load_compute_sum_OdO_consumer,
        mma_compute_dP_producer,
        mma_compute_dP_consumer,
        compute_mma_dS_producer,
        compute_mma_dS_consumer,
        mma_compute_dKdV_producer,
        mma_compute_dKdV_consumer,
        varlen: bool,
        sK: cute.Tensor,
        problem_shape_k_cur_batch: Int32,
    ):
        del varlen, sK
        tidx, _, _ = cute.arch.thread_idx()
        Q, K, _, _ = problem_shape
        _, blk_coord_k, _, _ = blk_coord
        iter_index = iter_start

        tmem_load_op = tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(16))
        tmem_load_atom = cute.make_copy_atom(tmem_load_op, self.acc_dtype)
        tSTtST = tSTtST[(None, None), 0, 0]
        tdPTtdPT = tdPTtdPT[(None, None), 0, 0]
        tSTtST = cute.make_tensor(
            tSTtST.iterator,
            cute.make_layout((16, 32), stride=(1, 65536)),
        )
        tdPTtdPT = cute.make_tensor(
            tdPTtdPT.iterator,
            cute.make_layout((16, 32), stride=(1, 65536)),
        )

        cST = cute.make_identity_tensor((self.tile_shape_K, self.tile_shape_Q))
        cdPT = cute.make_identity_tensor((self.tile_shape_K, self.tile_shape_Q))
        num_warp_groups = self.num_compute_warps // 4
        dp_idx = tidx % 128
        wg_idx = (tidx % (self.num_compute_warps * self.threads_per_warp)) // 128
        tiled_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tSTtST)
        thr_t2r = tiled_t2r.get_slice(dp_idx)

        tTR_cST = split_wg(thr_t2r.partition_D(cST), num_warp_groups, wg_idx)
        tTR_rST = cute.make_rmem_tensor(tTR_cST.shape, self.acc_dtype)
        tTR_tST = split_wg(thr_t2r.partition_S(tSTtST), num_warp_groups, wg_idx)

        tTR_cdPT = split_wg(thr_t2r.partition_D(cdPT), num_warp_groups, wg_idx)
        tTR_rdPT = cute.make_rmem_tensor(tTR_cdPT.shape, self.acc_dtype)
        tTR_tdPT = split_wg(thr_t2r.partition_S(tdPTtdPT), num_warp_groups, wg_idx)

        while iter_count > 0:
            s_handle = mma_compute_S_consumer.wait_and_advance()
            p_handle = compute_mma_P_producer.acquire_and_advance()
            lse_handle = load_compute_LSE_consumer.wait_and_advance()

            cute.copy(tiled_t2r, tTR_tST, tTR_rST)
            for i in cutlass.range(cute.size(tTR_rST), unroll_full=True):
                c_transpose = tTR_cST[i]
                q_pos = cute.get(c_transpose, mode=[1]) + iter_index * self.tile_shape_Q
                k_pos = cute.get(c_transpose, mode=[0]) + blk_coord_k * self.tile_shape_K
                invalid = (k_pos >= ((q_pos + 1) // self.csa_compress_ratio)) or not cute.elem_less(
                    (q_pos, k_pos), (Q, K)
                )
                if invalid:
                    tTR_rST[i] = -cutlass.Float32.inf

            softmax_scale_log2_e = scale_softmax * cutlass.Float32(math.log2(math.e))
            for i in cutlass.range(0, cute.size(tTR_rST), 2, unroll_full=True):
                lse = (
                    -sLSE[cute.get(tTR_cST[i], mode=[1]), lse_handle.index],
                    -sLSE[cute.get(tTR_cST[i + 1], mode=[1]), lse_handle.index],
                )
                tTR_rST[i], tTR_rST[i + 1] = cute.arch.fma_packed_f32x2(
                    (tTR_rST[i], tTR_rST[i + 1]),
                    (softmax_scale_log2_e, softmax_scale_log2_e),
                    lse,
                )
                tTR_rST[i] = cute.math.exp2(tTR_rST[i], fastmath=True)
                tTR_rST[i + 1] = cute.math.exp2(tTR_rST[i + 1], fastmath=True)

            tTR_rPT = self.quantize(tTR_rST, dV.element_type)
            self.reg_to_smem_mma64x64(
                tTR_rPT,
                sP,
                p_handle.index,
                (self.tile_shape_K, self.tile_shape_Q),
                dp_idx,
                wg_idx,
            )
            cute.arch.fence_view_async_shared()
            cute.arch.barrier(
                barrier_id=self.compute_sync_bar_id,
                number_of_threads=self.num_compute_warps * self.threads_per_warp,
            )
            p_handle.commit()
            s_handle.release()
            lse_handle.release()

            sum_odo_handle = load_compute_sum_OdO_consumer.wait_and_advance()
            dp_handle = mma_compute_dP_consumer.wait_and_advance()
            ds_handle = compute_mma_dS_producer.acquire_and_advance()
            cute.copy(tiled_t2r, tTR_tdPT, tTR_rdPT)

            for i in cutlass.range(0, cute.size(tTR_rdPT), 2, unroll_full=True):
                dpsum_0 = -sSum_OdO[cute.get(tTR_cdPT[i], mode=[1]), sum_odo_handle.index]
                dpsum_1 = -sSum_OdO[cute.get(tTR_cdPT[i + 1], mode=[1]), sum_odo_handle.index]
                tTR_rdPT[i], tTR_rdPT[i + 1] = cute.arch.add_packed_f32x2(
                    (tTR_rdPT[i], tTR_rdPT[i + 1]), (dpsum_0, dpsum_1)
                )
                tTR_rdPT[i], tTR_rdPT[i + 1] = cute.arch.mul_packed_f32x2(
                    (tTR_rdPT[i], tTR_rdPT[i + 1]), (tTR_rST[i], tTR_rST[i + 1])
                )
            for i in cutlass.range(cute.size(tTR_rdPT), unroll_full=True):
                c_transpose = tTR_cdPT[i]
                q_pos = cute.get(c_transpose, mode=[1]) + iter_index * self.tile_shape_Q
                k_pos = cute.get(c_transpose, mode=[0]) + blk_coord_k * self.tile_shape_K
                invalid = (k_pos >= ((q_pos + 1) // self.csa_compress_ratio)) or not cute.elem_less(
                    (q_pos, k_pos), (Q, K)
                )
                if invalid:
                    tTR_rdPT[i] = cutlass.Float32(0.0)

            tTR_rdST = self.quantize(tTR_rdPT, dV.element_type)
            cute.arch.fence_view_async_tmem_load()
            dp_handle.release()
            self.reg_to_smem_mma64x64(
                tTR_rdST,
                sdST,
                ds_handle.index,
                (self.tile_shape_K, self.tile_shape_Q),
                dp_idx,
                wg_idx,
            )
            cute.arch.fence_view_async_shared()
            cute.arch.barrier(
                barrier_id=self.compute_sync_bar_id,
                number_of_threads=self.num_compute_warps * self.threads_per_warp,
            )
            ds_handle.commit()
            sum_odo_handle.release()

            iter_count -= 1
            iter_index += 1
            if iter_index == iter_end:
                iter_index = iter_start

        mma_compute_dKdV_consumer = self.epilogue(
            blk_coord,
            blk_offset,
            problem_shape,
            dK,
            dV,
            tdKtdK,
            tdVtdV,
            scale_softmax,
            mma_compute_dKdV_producer,
            mma_compute_dKdV_consumer,
            problem_shape_k_cur_batch,
            False,
            sdOT,
            sP,
        )

        compute_mma_P_producer.tail()
        compute_mma_dS_producer.tail()
        return (
            compute_mma_P_producer,
            compute_mma_dS_producer,
            mma_compute_S_consumer,
            compute_mma_P_consumer,
            load_compute_LSE_consumer,
            load_compute_sum_OdO_consumer,
            mma_compute_dP_consumer,
            compute_mma_dS_consumer,
            mma_compute_dKdV_consumer,
        )

    @cute.jit
    def load(
        self,
        K_in: cute.Tensor,
        V_in: cute.Tensor,
        Q_in: cute.Tensor,
        QT_in: cute.Tensor,
        dO_in: cute.Tensor,
        dOT_in: cute.Tensor,
        LSE_in: cute.Tensor,
        sum_OdO_in: cute.Tensor,
        sK: cute.Tensor,
        sQ: cute.Tensor,
        sQT: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sdOT: cute.Tensor,
        sLSE: cute.Tensor,
        sSum_OdO: cute.Tensor,
        KQ_tiled_mma: cute.TiledMma,
        VdO_tiled_mma: cute.TiledMma,
        PdO_tiled_mma: cute.TiledMma,
        dSQ_tiled_mma: cute.TiledMma,
        tma_atom_K: cute.CopyAtom,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_QT: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        tma_atom_dOT: cute.CopyAtom,
        blk_offset: cute.Shape,
        problem_shape: tuple[Int32, Int32, Int32, tuple[tuple[Int32, Int32], Int32]],
        varlen: bool,
        iter_count: Int32,
        iter_start: Int32,
        iter_end: Int32,
        load_mma_Q_producer,
        load_mma_Q_consumer,
        load_mma_K_producer,
        load_mma_K_consumer,
        load_mma_V_producer,
        load_mma_V_consumer,
        load_compute_LSE_producer,
        load_compute_LSE_consumer,
        load_mma_dO_producer,
        load_mma_dO_consumer,
        load_mma_dOT_producer,
        load_mma_dOT_consumer,
        load_compute_sum_OdO_producer,
        load_compute_sum_OdO_consumer,
        load_mma_QT_producer,
        load_mma_QT_consumer,
        blk_coord_k_override: Int32 = Int32(-1),
        blk_coord_h_k_override: Int32 = Int32(-1),
        blk_coord_b_override: Int32 = Int32(-1),
    ):
        del varlen, blk_offset, load_mma_Q_consumer, load_mma_K_consumer
        del load_mma_V_consumer, load_compute_LSE_consumer, load_mma_dO_consumer
        del load_mma_dOT_consumer, load_compute_sum_OdO_consumer, load_mma_QT_consumer
        del blk_coord_k_override, blk_coord_h_k_override, blk_coord_b_override
        tidx, _, _ = cute.arch.thread_idx()
        blk_coord_k, blk_coord_h_k, blk_coord_b = cute.arch.block_idx()
        blk_coord_h_r = Int32(0)
        blk_coord_h = (blk_coord_h_r, blk_coord_h_k)
        iter_index = iter_start
        mma_tile_coord_v = blk_coord_k % cute.size(KQ_tiled_mma.thr_id.shape)
        mma_tile_coord_m = blk_coord_k // cute.size(KQ_tiled_mma.thr_id.shape)

        gK = cute.local_tile(K_in, cute.select(self.KQ_mma_tiler, mode=[0, 2]), (None, None, None))
        gQ = cute.local_tile(Q_in, cute.select(self.KQ_mma_tiler, mode=[1, 2]), (None, None, None))
        gQT = cute.local_tile(QT_in, cute.select(self.dSQ_mma_tiler, mode=[1, 2]), (None, None, None))
        gV = cute.local_tile(V_in, cute.select(self.VdO_mma_tiler, mode=[0, 2]), (None, None, None))
        gdO = cute.local_tile(dO_in, cute.select(self.VdO_mma_tiler, mode=[1, 2]), (None, None, None))
        gdOT = cute.local_tile(dOT_in, cute.select(self.PdO_mma_tiler, mode=[1, 2]), (None, None, None))

        KQ_thr_mma = KQ_tiled_mma.get_slice(mma_tile_coord_v)
        VdO_thr_mma = VdO_tiled_mma.get_slice(mma_tile_coord_v)
        PdO_thr_mma = PdO_tiled_mma.get_slice(mma_tile_coord_v)
        dSQ_thr_mma = dSQ_tiled_mma.get_slice(mma_tile_coord_v)

        tSTgK = KQ_thr_mma.partition_A(gK)
        tSTgQ = KQ_thr_mma.partition_B(gQ)
        tdKgQT = dSQ_thr_mma.partition_B(gQT)
        tdPTgV = VdO_thr_mma.partition_A(gV)
        tdPTgdO = VdO_thr_mma.partition_B(gdO)
        tdVgdOT = PdO_thr_mma.partition_B(gdOT)

        cta_layout_mnk = cute.make_layout(self.cluster_shape_mnk)
        cta_layout_vmnk = cute.tiled_divide(cta_layout_mnk, (KQ_tiled_mma.thr_id,))
        cta_in_cluster_coord_vmnk = cta_layout_vmnk.get_flat_coord(cute.arch.block_idx_in_cluster())

        tKsK, tKgK_mkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_K,
            cta_in_cluster_coord_vmnk[2],
            cute.make_layout(cute.size(cta_layout_vmnk, mode=[2])),
            cute.group_modes(sK, 0, 3),
            cute.group_modes(tSTgK, 0, 3),
        )
        tQsQ, tQgQ_mkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_Q,
            cta_in_cluster_coord_vmnk[1],
            cute.make_layout(cute.size(cta_layout_vmnk, mode=[1])),
            cute.group_modes(sQ, 0, 3),
            cute.group_modes(tSTgQ, 0, 3),
        )
        tQTsQT, tQTgQT_mkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_QT,
            cta_in_cluster_coord_vmnk[1],
            cute.make_layout(cute.size(cta_layout_vmnk, mode=[1])),
            cute.group_modes(sQT, 0, 3),
            cute.group_modes(tdKgQT, 0, 3),
        )
        tVsV, tVgV_mkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_V,
            cta_in_cluster_coord_vmnk[2],
            cute.make_layout(cute.size(cta_layout_vmnk, mode=[2])),
            cute.group_modes(sV, 0, 3),
            cute.group_modes(tdPTgV, 0, 3),
        )
        tdOsdO, tdOgdO_mkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_dO,
            cta_in_cluster_coord_vmnk[1],
            cute.make_layout(cute.size(cta_layout_vmnk, mode=[1])),
            cute.group_modes(sdO, 0, 3),
            cute.group_modes(tdPTgdO, 0, 3),
        )
        tdOTsdOT, tdOTgdOT_mkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_dOT,
            cta_in_cluster_coord_vmnk[1],
            cute.make_layout(cute.size(cta_layout_vmnk, mode=[1])),
            cute.group_modes(sdOT, 0, 3),
            cute.group_modes(tdVgdOT, 0, 3),
        )

        k_handle = load_mma_K_producer.acquire_and_advance()
        cute.copy(
            tma_atom_K,
            tKgK_mkl[(None, mma_tile_coord_m, 0, (blk_coord_h, blk_coord_b))],
            tKsK[None, 0],
            tma_bar_ptr=k_handle.barrier,
        )

        v_handle = load_mma_V_producer.acquire_and_advance()
        cute.copy(
            tma_atom_V,
            tVgV_mkl[(None, mma_tile_coord_m, 0, (blk_coord_h, blk_coord_b))],
            tVsV[None, 0],
            tma_bar_ptr=v_handle.barrier,
        )

        thread_idx = tidx % self.threads_per_warp
        async_copy_num_elts = sLSE.shape[0] // self.threads_per_warp
        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.ALWAYS),
            self.acc_dtype,
            num_bits_per_copy=self.acc_dtype.width,
        )

        while iter_count > 0:
            if iter_index == iter_end:
                iter_index = iter_start
                blk_coord_h_r += 1
                blk_coord_h = (blk_coord_h_r, blk_coord_h_k)

            q_handle = load_mma_Q_producer.acquire_and_advance()
            cute.copy(
                tma_atom_Q,
                tQgQ_mkl[(None, iter_index, 0, (blk_coord_h, blk_coord_b))],
                tQsQ[None, q_handle.index],
                tma_bar_ptr=q_handle.barrier,
            )

            lse_handle = load_compute_LSE_producer.acquire_and_advance()
            sLSE_for_copy = cute.flat_divide(sLSE, (1,))
            LSE_for_copy = cute.flat_divide(LSE_in, (1,))
            for i in cutlass.range_constexpr(async_copy_num_elts):
                q_idx = self.tile_shape_Q * iter_index + thread_idx + i * self.threads_per_warp
                s_idx = thread_idx + i * self.threads_per_warp
                cute.copy(
                    atom_async_copy,
                    LSE_for_copy[None, q_idx, (blk_coord_h, blk_coord_b)],
                    sLSE_for_copy[None, s_idx, lse_handle.index],
                )
            lse_handle.commit()

            do_handle = load_mma_dO_producer.acquire_and_advance()
            cute.copy(
                tma_atom_dO,
                tdOgdO_mkl[(None, iter_index, 0, (blk_coord_h, blk_coord_b))],
                tdOsdO[None, do_handle.index],
                tma_bar_ptr=do_handle.barrier,
            )

            sum_odo_handle = load_compute_sum_OdO_producer.acquire_and_advance()
            sSum_OdO_for_copy = cute.flat_divide(sSum_OdO, (1,))
            sum_OdO_for_copy = cute.flat_divide(sum_OdO_in, (1,))
            for i in cutlass.range_constexpr(async_copy_num_elts):
                q_idx = self.tile_shape_Q * iter_index + thread_idx + i * self.threads_per_warp
                s_idx = thread_idx + i * self.threads_per_warp
                cute.copy(
                    atom_async_copy,
                    sum_OdO_for_copy[None, q_idx, (blk_coord_h, blk_coord_b)],
                    sSum_OdO_for_copy[None, s_idx, sum_odo_handle.index],
                )
            sum_odo_handle.commit()

            dot_handle = load_mma_dOT_producer.acquire_and_advance()
            cute.copy(
                tma_atom_dOT,
                tdOTgdOT_mkl[(None, 0, iter_index, (blk_coord_h, blk_coord_b))],
                tdOTsdOT[None, dot_handle.index],
                tma_bar_ptr=dot_handle.barrier,
            )

            qt_handle = load_mma_QT_producer.acquire_and_advance()
            cute.copy(
                tma_atom_QT,
                tQTgQT_mkl[(None, 0, iter_index, (blk_coord_h, blk_coord_b))],
                tQTsQT[None, qt_handle.index],
                tma_bar_ptr=qt_handle.barrier,
            )

            iter_count -= 1
            iter_index += 1

        load_mma_K_producer.tail()
        load_mma_V_producer.tail()
        load_mma_Q_producer.tail()
        load_compute_LSE_producer.tail()
        load_mma_dO_producer.tail()
        load_mma_dOT_producer.tail()
        load_compute_sum_OdO_producer.tail()
        load_mma_QT_producer.tail()

        return (
            load_mma_Q_producer,
            load_mma_K_producer,
            load_mma_V_producer,
            load_compute_LSE_producer,
            load_mma_dO_producer,
            load_mma_dOT_producer,
            load_compute_sum_OdO_producer,
            load_mma_QT_producer,
        )

    @cute.jit
    def epilogue(
        self,
        blk_coord: cute.Coord,
        blk_offset: cute.Shape,
        problem_shape: tuple[Int32, Int32, Int32, tuple[tuple[Int32, Int32], Int32]],
        dK: cute.Tensor,
        dV: cute.Tensor,
        tdKtdK: cute.Tensor,
        tdVtdV: cute.Tensor,
        scale_softmax: cutlass.Float32,
        mma_compute_dKdV_producer,
        mma_compute_dKdV_consumer,
        problem_shape_k_cur_batch: Int32,
        varlen: bool,
        sdOT: cute.Tensor,
        sP: cute.Tensor,
    ):
        del mma_compute_dKdV_producer, varlen, sdOT, sP
        tidx, _, _ = cute.arch.thread_idx()
        _, K, _, HB = problem_shape
        _, blk_coord_k, _, blk_coord_batch = blk_coord

        tmem_copy_op = tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32))
        load_op = cute.make_copy_atom(tmem_copy_op, self.acc_dtype)
        num_warp_groups = self.num_compute_warps // 4
        dp_idx = tidx % 128
        wg_idx = (tidx % (self.num_compute_warps * self.threads_per_warp)) // 128

        tdVtdV = tdVtdV[(None, None), 0, 0]
        mdV = cute.make_tensor(
            dV.iterator,
            cute.make_layout((K, self.tile_shape_dV_dO, HB), stride=dV.stride),
        )
        mdV_offset = cute.assume(blk_offset[1] * mdV.stride[0], divby=64)
        mdV = cute.make_tensor(mdV.iterator + mdV_offset, mdV.layout)
        gdV = cute.local_tile(mdV, (self.tile_shape_K, self.tile_shape_dV_dO), (None, None, None))
        gdV = gdV[None, None, blk_coord_k, 0, blk_coord_batch]
        cdV = cute.domain_offset(
            (blk_coord_k * self.tile_shape_K, 0),
            cute.make_identity_tensor((self.tile_shape_K, self.tile_shape_dV_dO)),
        )
        tiled_t2r_dV = tcgen05.make_tmem_copy(load_op, tdVtdV)
        thread_t2r_dV = tiled_t2r_dV.get_slice(dp_idx)
        tTR_cdV = split_wg(thread_t2r_dV.partition_D(cdV), num_warp_groups, wg_idx)
        tTR_gdV = split_wg(thread_t2r_dV.partition_D(gdV), num_warp_groups, wg_idx)
        tTR_rdV = cute.make_rmem_tensor(tTR_cdV.shape, self.acc_dtype)
        tTR_tdV = split_wg(thread_t2r_dV.partition_S(tdVtdV), num_warp_groups, wg_idx)

        dkdv_handle = mma_compute_dKdV_consumer.wait_and_advance()
        if blk_coord_k * self.tile_shape_K < problem_shape_k_cur_batch:
            cute.copy(tiled_t2r_dV, tTR_tdV, tTR_rdV)
            self.store(tTR_gdV, tTR_rdV, tTR_cdV, (K, dV.shape[1]))
        cute.arch.fence_view_async_tmem_load()
        dkdv_handle.release()

        tdKtdK = tdKtdK[(None, None), 0, 0]
        mdK_offset = cute.assume(blk_offset[1] * dK.stride[0], divby=64)
        mdK = cute.make_tensor(
            dK.iterator + mdK_offset,
            cute.make_layout((K, self.tile_shape_dQ_K, HB), stride=dK.stride),
        )
        gdK = cute.local_tile(mdK, (self.tile_shape_K, self.tile_shape_dQ_K), (None, None, None))
        gdK = gdK[None, None, blk_coord_k, 0, blk_coord_batch]
        cdK = cute.domain_offset(
            (blk_coord_k * self.tile_shape_K, 0),
            cute.make_identity_tensor((self.tile_shape_K, self.tile_shape_dQ_K)),
        )
        tiled_t2r_dK = tcgen05.make_tmem_copy(load_op, tdKtdK)
        thread_t2r_dK = tiled_t2r_dK.get_slice(dp_idx)
        tTR_cdK = split_wg(thread_t2r_dK.partition_D(cdK), num_warp_groups, wg_idx)
        tTR_gdK = split_wg(thread_t2r_dK.partition_D(gdK), num_warp_groups, wg_idx)
        tTR_rdK = cute.make_rmem_tensor(tTR_cdK.shape, self.acc_dtype)
        tTR_tdK = split_wg(thread_t2r_dK.partition_S(tdKtdK), num_warp_groups, wg_idx)

        dkdv_handle = mma_compute_dKdV_consumer.wait_and_advance()
        if blk_coord_k * self.tile_shape_K < problem_shape_k_cur_batch:
            cute.copy(tiled_t2r_dK, tTR_tdK, tTR_rdK)
            for i in cutlass.range(cute.size(tTR_rdK), unroll_full=True):
                tTR_rdK[i] = scale_softmax * tTR_rdK[i]
            self.store(tTR_gdK, tTR_rdK, tTR_cdK, (K, dK.shape[1]))
        cute.arch.fence_view_async_tmem_load()
        dkdv_handle.release()
        return mma_compute_dKdV_consumer

    @cute.jit
    def __call__(
        self,
        Q_in: cute.Tensor,
        KV_in: cute.Tensor,
        V_in_raw: cute.Tensor,
        Q_DK_in: cute.Tensor,
        dK_in: cute.Tensor,
        dV_in: cute.Tensor,
        dO_in: cute.Tensor,
        dO_DV_in: cute.Tensor,
        lse_log2_in: cute.Tensor,
        dpsum_in: cute.Tensor,
        scale_softmax: Float32,
        stream: cuda.CUstream = None,
    ):
        Q_in, KV_in, V_in_raw, Q_DK_in, dK_in, dV_in, dO_in, dO_DV_in = [
            assume_tensor_aligned(t)
            for t in (Q_in, KV_in, V_in_raw, Q_DK_in, dK_in, dV_in, dO_in, dO_DV_in)
        ]
        h_r = Q_in.shape[1]
        h_k = 1
        batch = 1
        hb = ((h_r, h_k), batch)

        # (S,H,D) -> (S,D,((H_r,H_k),B)); H_k is fixed to one shared KV head.
        Q = cute.make_tensor(
            Q_in.iterator,
            cute.make_layout(
                (Q_in.shape[0], Q_in.shape[2], hb),
                stride=(
                    cute.assume(Q_in.stride[0], divby=64),
                    Q_in.stride[2],
                    ((Q_in.stride[1], Q_in.stride[1] * h_r), 0),
                ),
            ),
        )
        Q_DK = cute.make_tensor(
            Q_DK_in.iterator,
            cute.make_layout(
                (Q_DK_in.shape[0], Q_DK_in.shape[2], hb),
                stride=(
                    cute.assume(Q_DK_in.stride[0], divby=64),
                    Q_DK_in.stride[2],
                    ((Q_DK_in.stride[1], Q_DK_in.stride[1] * h_r), 0),
                ),
            ),
        )
        K = cute.make_tensor(
            KV_in.iterator,
            cute.make_layout(
                (KV_in.shape[0], KV_in.shape[2], hb),
                stride=(
                    cute.assume(KV_in.stride[0], divby=64),
                    KV_in.stride[2],
                    ((0, KV_in.stride[1]), 0),
                ),
            ),
        )
        V = cute.make_tensor(
            V_in_raw.iterator,
            cute.make_layout(
                (V_in_raw.shape[0], V_in_raw.shape[2], hb),
                stride=(
                    cute.assume(V_in_raw.stride[0], divby=64),
                    V_in_raw.stride[2],
                    ((0, V_in_raw.stride[1]), 0),
                ),
            ),
        )
        dK = cute.make_tensor(
            dK_in.iterator,
            cute.make_layout(
                (dK_in.shape[0], dK_in.shape[2], hb),
                stride=(
                    cute.assume(dK_in.stride[0], divby=64),
                    dK_in.stride[2],
                    ((0, dK_in.stride[1]), 0),
                ),
            ),
        )
        dV = cute.make_tensor(
            dV_in.iterator,
            cute.make_layout(
                (dV_in.shape[0], dV_in.shape[2], hb),
                stride=(
                    cute.assume(dV_in.stride[0], divby=64),
                    dV_in.stride[2],
                    ((0, dV_in.stride[1]), 0),
                ),
            ),
        )
        dO = cute.make_tensor(
            dO_in.iterator,
            cute.make_layout(
                (dO_in.shape[0], dO_in.shape[2], hb),
                stride=(
                    cute.assume(dO_in.stride[0], divby=64),
                    dO_in.stride[2],
                    ((dO_in.stride[1], dO_in.stride[1] * h_r), 0),
                ),
            ),
        )
        dO_DV = cute.make_tensor(
            dO_DV_in.iterator,
            cute.make_layout(
                (dO_DV_in.shape[0], dO_DV_in.shape[2], hb),
                stride=(
                    cute.assume(dO_DV_in.stride[0], divby=64),
                    dO_DV_in.stride[2],
                    ((dO_DV_in.stride[1], dO_DV_in.stride[1] * h_r), 0),
                ),
            ),
        )
        QT = cute.make_tensor(
            Q_DK.iterator,
            cute.make_layout(
                (Q_DK.shape[1], Q_DK.shape[0], Q_DK.shape[2]),
                stride=(Q_DK.stride[1], Q_DK.stride[0], Q_DK.stride[2]),
            ),
        )
        dOT = cute.make_tensor(
            dO_DV.iterator,
            cute.make_layout(
                (dO_DV.shape[1], dO_DV.shape[0], dO_DV.shape[2]),
                stride=(dO_DV.stride[1], dO_DV.stride[0], dO_DV.stride[2]),
            ),
        )
        scaled_LSE = cute.make_tensor(
            lse_log2_in.iterator,
            cute.make_layout(
                (lse_log2_in.shape[1], hb),
                stride=(lse_log2_in.stride[1], ((lse_log2_in.stride[0], lse_log2_in.stride[0] * h_r), 0)),
            ),
        )
        sum_OdO = cute.make_tensor(
            dpsum_in.iterator,
            cute.make_layout(
                (dpsum_in.shape[1], hb),
                stride=(dpsum_in.stride[1], ((dpsum_in.stride[0], dpsum_in.stride[0] * h_r), 0)),
            ),
        )

        self.Q_major_mode = cutlass_utils.LayoutEnum.from_tensor(Q).mma_major_mode()
        self.K_major_mode = cutlass_utils.LayoutEnum.from_tensor(K).mma_major_mode()
        self.dK_major_mode = cutlass_utils.LayoutEnum.from_tensor(dK).mma_major_mode()
        self.V_major_mode = cutlass_utils.LayoutEnum.from_tensor(V).mma_major_mode()
        self.dV_major_mode = cutlass_utils.LayoutEnum.from_tensor(dV).mma_major_mode()
        if cutlass.const_expr(self.Q_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError(f"The layout of q is not supported: {self.Q_major_mode}")
        if cutlass.const_expr(self.K_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of kv is not supported")
        if cutlass.const_expr(self.dK_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of dk is not supported")
        if cutlass.const_expr(self.dV_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of dv is not supported")

        self._setup_attributes()

        cta_group = tcgen05.CtaGroup.ONE
        pt_source = tcgen05.OperandSource.SMEM
        KQ_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            K.element_type,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            cta_group,
            self.KQ_mma_tiler[:2],
        )
        VdO_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            V.element_type,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            cta_group,
            self.VdO_mma_tiler[:2],
        )
        PdO_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            dO.element_type,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.MN,
            self.acc_dtype,
            cta_group,
            self.PdO_mma_tiler[:2],
            pt_source,
        )
        dSQ_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            Q.element_type,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.MN,
            self.acc_dtype,
            cta_group,
            self.dSQ_mma_tiler[:2],
        )
        atom_thr_size = cute.size(KQ_tiled_mma.thr_id.shape)
        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (atom_thr_size,),
        )

        K_smem_layout_staged = sm100_utils.make_smem_layout_a(
            KQ_tiled_mma, self.KQ_mma_tiler, K.element_type, self.load_mma_K_stage
        )
        Q_smem_layout_staged = sm100_utils.make_smem_layout_b(
            KQ_tiled_mma, self.KQ_mma_tiler, Q.element_type, self.load_mma_Q_stage
        )
        V_smem_layout_staged = sm100_utils.make_smem_layout_a(
            VdO_tiled_mma, self.VdO_mma_tiler, V.element_type, self.load_mma_V_stage
        )
        dO_smem_layout_staged = sm100_utils.make_smem_layout_b(
            VdO_tiled_mma, self.VdO_mma_tiler, dO.element_type, self.load_mma_dO_stage
        )
        dST_smem_layout_staged = sm100_utils.make_smem_layout_a(
            dSQ_tiled_mma, self.dSQ_mma_tiler, Q.element_type, self.compute_mma_dS_stage
        )
        QT_smem_layout_staged = sm100_utils.make_smem_layout_b(
            dSQ_tiled_mma, self.dSQ_mma_tiler, Q.element_type, self.load_mma_QT_stage
        )
        P_smem_layout_staged = sm100_utils.make_smem_layout_a(
            PdO_tiled_mma, self.PdO_mma_tiler, Q.element_type, self.compute_mma_P_stage
        )
        dOT_smem_layout_staged = sm100_utils.make_smem_layout_b(
            PdO_tiled_mma, self.PdO_mma_tiler, dO.element_type, self.load_mma_dO_stage
        )
        LSE_smem_layout = cute.make_layout((self.cta_tiler[0], self.load_compute_LSE_stage))
        sum_OdO_smem_layout = cute.make_layout((self.cta_tiler[0], self.load_compute_sum_OdO_stage))

        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)
        K_smem_layout = cute.select(K_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_K, tma_tensor_K = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            K,
            K_smem_layout,
            self.KQ_mma_tiler,
            KQ_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        V_smem_layout = cute.select(V_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_V, tma_tensor_V = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            V,
            V_smem_layout,
            self.VdO_mma_tiler,
            VdO_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        Q_smem_layout = cute.select(Q_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_Q, tma_tensor_Q = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            Q,
            Q_smem_layout,
            self.KQ_mma_tiler,
            KQ_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        QT_smem_layout = cute.select(QT_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_QT, tma_tensor_QT = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            QT,
            QT_smem_layout,
            self.dSQ_mma_tiler,
            dSQ_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        dO_smem_layout = cute.select(dO_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_dO, tma_tensor_dO = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            dO,
            dO_smem_layout,
            self.VdO_mma_tiler,
            VdO_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        dOT_smem_layout = cute.select(dOT_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_dOT, tma_tensor_dOT = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            dOT,
            dOT_smem_layout,
            self.PdO_mma_tiler,
            PdO_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        self.tma_copy_Q_bytes = cute.size_in_bytes(Q.element_type, Q_smem_layout) * atom_thr_size
        self.tma_copy_K_bytes = cute.size_in_bytes(K.element_type, K_smem_layout) * atom_thr_size
        self.tma_copy_V_bytes = (
            0
            if cutlass.const_expr(self.reuse_k_as_v)
            else cute.size_in_bytes(V.element_type, V_smem_layout) * atom_thr_size
        )
        self.tma_copy_dO_bytes = cute.size_in_bytes(dO.element_type, dO_smem_layout) * atom_thr_size
        self.tma_copy_dOT_bytes = cute.size_in_bytes(dO_DV.element_type, dOT_smem_layout) * atom_thr_size

        sP_storage_elems = cute.cosize(P_smem_layout_staged)

        @cute.struct
        class SharedStorage:
            load_mma_Q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            load_mma_K_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            load_mma_V_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            load_mma_QT_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            load_mma_dO_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            load_mma_dOT_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            load_compute_lse_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            load_compute_sum_OdO_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            mma_compute_S_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            mma_compute_dP_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            compute_mma_P_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            compute_mma_dS_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            mma_compute_dKdV_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 4]
            tmem_holding_buf: cutlass.Int32
            tmem_dealloc_mbar_ptr: cutlass.Int64
            sK: cute.struct.Align[
                cute.struct.MemRange[K.element_type, cute.cosize(K_smem_layout_staged)], 128
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[
                    V.element_type,
                    1 if cutlass.const_expr(self.reuse_k_as_v) else cute.cosize(V_smem_layout_staged),
                ],
                128,
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[Q.element_type, cute.cosize(Q_smem_layout_staged)], 128
            ]
            sQT: cute.struct.Align[
                cute.struct.MemRange[Q.element_type, cute.cosize(QT_smem_layout_staged)], 128
            ]
            sdO: cute.struct.Align[
                cute.struct.MemRange[dO.element_type, cute.cosize(dO_smem_layout_staged)], 128
            ]
            sdOT: cute.struct.Align[
                cute.struct.MemRange[dO.element_type, cute.cosize(dOT_smem_layout_staged)], 128
            ]
            sP: cute.struct.Align[cute.struct.MemRange[Q.element_type, sP_storage_elems], 128]
            sdST: cute.struct.Align[
                cute.struct.MemRange[Q.element_type, cute.cosize(dST_smem_layout_staged)], 128
            ]
            sLSE: cute.struct.Align[
                cute.struct.MemRange[self.acc_dtype, cute.cosize(LSE_smem_layout)], 128
            ]
            sSum_OdO: cute.struct.Align[
                cute.struct.MemRange[self.acc_dtype, cute.cosize(sum_OdO_smem_layout)], 128
            ]

        self.shared_storage = SharedStorage
        problem_shape = (Q.shape[0], K.shape[0], Q.shape[1], hb)
        bwd_grid = self._compute_bwd_grid(problem_shape, self.cta_tiler[1])
        bwd_grid = cute.round_up(bwd_grid, self.cluster_shape_mnk)

        self.dkdv_bwd_small_static(
            KQ_tiled_mma,
            VdO_tiled_mma,
            PdO_tiled_mma,
            dSQ_tiled_mma,
            tma_atom_K,
            tma_tensor_K,
            K,
            tma_atom_V,
            tma_tensor_V,
            tma_atom_Q,
            tma_tensor_Q,
            Q,
            tma_atom_QT,
            tma_tensor_QT,
            tma_atom_dO,
            tma_tensor_dO,
            tma_atom_dOT,
            tma_tensor_dOT,
            dK,
            dV,
            scaled_LSE,
            scale_softmax,
            sum_OdO,
            problem_shape,
            None,
            None,
            self.cluster_layout_vmnk,
            K_smem_layout_staged,
            Q_smem_layout_staged,
            V_smem_layout_staged,
            dO_smem_layout_staged,
            dST_smem_layout_staged,
            QT_smem_layout_staged,
            dOT_smem_layout_staged,
            P_smem_layout_staged,
            LSE_smem_layout,
            sum_OdO_smem_layout,
        ).launch(
            grid=bwd_grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def dkdv_bwd_small_static(
        self,
        KQ_tiled_mma: cute.TiledMma,
        VdO_tiled_mma: cute.TiledMma,
        PdO_tiled_mma: cute.TiledMma,
        dSQ_tiled_mma: cute.TiledMma,
        tma_atom_K: cute.CopyAtom,
        K_in: cute.Tensor,
        K_ref: cute.Tensor,
        tma_atom_V: cute.CopyAtom,
        V_in: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        Q_in: cute.Tensor,
        Q_ref: cute.Tensor,
        tma_atom_QT: cute.CopyAtom,
        QT_in: cute.Tensor,
        tma_atom_dO: cute.CopyAtom,
        dO_in: cute.Tensor,
        tma_atom_dOT: cute.CopyAtom,
        dOT_in: cute.Tensor,
        dK: cute.Tensor,
        dV: cute.Tensor,
        LSE: cute.Tensor,
        scale_softmax: cutlass.Float32,
        sum_OdO: cute.Tensor,
        problem_shape: tuple[Int32, Int32, Int32, tuple[tuple[Int32, Int32], Int32]],
        cumulative_s_q: cute.Tensor | None,
        cumulative_s_k: cute.Tensor | None,
        cluster_layout_vmnk: cute.Layout,
        K_smem_layout_staged: cute.ComposedLayout,
        Q_smem_layout_staged: cute.ComposedLayout,
        V_smem_layout_staged: cute.ComposedLayout,
        dO_smem_layout_staged: cute.ComposedLayout,
        dST_smem_layout_staged: cute.ComposedLayout,
        QT_smem_layout_staged: cute.ComposedLayout,
        dOT_smem_layout_staged: cute.ComposedLayout,
        P_smem_layout_staged: cute.ComposedLayout,
        LSE_smem_layout: cute.Layout,
        sum_OdO_smem_layout: cute.Layout,
    ):
        del cumulative_s_q, cumulative_s_k
        bidx, bidy, bidz = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        varlen = False

        if warp_idx == self.load_warp_id:
            cpasync.prefetch_descriptor(tma_atom_K)
            cpasync.prefetch_descriptor(tma_atom_Q)
            cpasync.prefetch_descriptor(tma_atom_QT)
            cpasync.prefetch_descriptor(tma_atom_V)
            cpasync.prefetch_descriptor(tma_atom_dO)
            cpasync.prefetch_descriptor(tma_atom_dOT)

        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        load_mma_Q_producer, load_mma_Q_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.load_mma_Q_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_Q_bytes,
            barrier_storage=storage.load_mma_Q_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        load_mma_K_producer, load_mma_K_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.load_mma_K_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_K_bytes,
            barrier_storage=storage.load_mma_K_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        load_mma_V_producer, load_mma_V_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.load_mma_V_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_V_bytes,
            barrier_storage=storage.load_mma_V_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        load_mma_QT_producer, load_mma_QT_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.load_mma_QT_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_Q_bytes,
            barrier_storage=storage.load_mma_QT_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        load_mma_dO_producer, load_mma_dO_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.load_mma_dO_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_dO_bytes,
            barrier_storage=storage.load_mma_dO_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        load_mma_dOT_producer, load_mma_dOT_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.load_mma_dO_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_dOT_bytes,
            barrier_storage=storage.load_mma_dOT_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        load_compute_LSE_producer, load_compute_LSE_consumer = pipeline.PipelineCpAsync.create(
            num_stages=self.load_compute_LSE_stage,
            producer_group=make_thread_cooperative_group(self.threads_per_warp),
            consumer_group=make_thread_cooperative_group(self.threads_per_warp * self.num_compute_warps),
            barrier_storage=storage.load_compute_lse_mbar_ptr.data_ptr(),
        ).make_participants()
        load_compute_sum_OdO_producer, load_compute_sum_OdO_consumer = (
            pipeline.PipelineCpAsync.create(
                num_stages=self.load_compute_sum_OdO_stage,
                producer_group=make_thread_cooperative_group(self.threads_per_warp),
                consumer_group=make_thread_cooperative_group(self.threads_per_warp * self.num_compute_warps),
                barrier_storage=storage.load_compute_sum_OdO_mbar_ptr.data_ptr(),
            ).make_participants()
        )
        mma_compute_S_producer, mma_compute_S_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.mma_compute_S_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(
                self.num_compute_warps * self.threads_per_warp * cluster_layout_vmnk.shape[0][0]
            ),
            barrier_storage=storage.mma_compute_S_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        mma_compute_dP_producer, mma_compute_dP_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.mma_compute_dP_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(
                self.num_compute_warps * self.threads_per_warp * cluster_layout_vmnk.shape[0][0]
            ),
            barrier_storage=storage.mma_compute_dP_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        compute_mma_P_producer, compute_mma_P_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.compute_mma_P_stage,
            producer_group=make_thread_cooperative_group(
                self.num_compute_warps * self.threads_per_warp * cluster_layout_vmnk.shape[0][0]
            ),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            barrier_storage=storage.compute_mma_P_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        compute_mma_dS_producer, compute_mma_dS_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.compute_mma_dS_stage,
            producer_group=make_thread_cooperative_group(
                self.num_compute_warps * self.threads_per_warp * cluster_layout_vmnk.shape[0][0]
            ),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            barrier_storage=storage.compute_mma_dS_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        mma_compute_dKdV_producer, mma_compute_dKdV_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.mma_compute_dKdV_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(
                self.num_compute_warps * self.threads_per_warp * cluster_layout_vmnk.shape[0][0]
            ),
            barrier_storage=storage.mma_compute_dKdV_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()

        cute.arch.barrier(barrier_id=self.cta_sync_bar_id, number_of_threads=self.threads_per_cta)

        sQ = storage.sQ.get_tensor(Q_smem_layout_staged.outer, swizzle=Q_smem_layout_staged.inner)
        sK = storage.sK.get_tensor(K_smem_layout_staged.outer, swizzle=K_smem_layout_staged.inner)
        if cutlass.const_expr(self.reuse_k_as_v):
            sV = storage.sK.get_tensor(V_smem_layout_staged.outer, swizzle=V_smem_layout_staged.inner)
        else:
            sV = storage.sV.get_tensor(V_smem_layout_staged.outer, swizzle=V_smem_layout_staged.inner)
        sdO = storage.sdO.get_tensor(dO_smem_layout_staged.outer, swizzle=dO_smem_layout_staged.inner)
        sLSE = storage.sLSE.get_tensor(LSE_smem_layout)
        sSum_OdO = storage.sSum_OdO.get_tensor(sum_OdO_smem_layout)
        sQT = storage.sQT.get_tensor(QT_smem_layout_staged.outer, swizzle=QT_smem_layout_staged.inner)
        sdST = storage.sdST.get_tensor(dST_smem_layout_staged.outer, swizzle=dST_smem_layout_staged.inner)
        sP = storage.sP.get_tensor(P_smem_layout_staged.outer, swizzle=P_smem_layout_staged.inner)
        sdOT = storage.sdOT.get_tensor(dOT_smem_layout_staged.outer, swizzle=dOT_smem_layout_staged.inner)

        tSTrK = KQ_tiled_mma.make_fragment_A(sK)
        tSTrQ = KQ_tiled_mma.make_fragment_B(sQ)
        tdPTrV = VdO_tiled_mma.make_fragment_A(sV)
        tdPTrdO = VdO_tiled_mma.make_fragment_B(sdO)
        tdKrdST = dSQ_tiled_mma.make_fragment_A(sdST)
        tdKrQT = dSQ_tiled_mma.make_fragment_B(sQT)

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            num_threads=self.threads_per_cta,
        )
        tmem = cutlass_utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.load_warp_id,
            is_two_cta=False,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
        )
        tmem.allocate(self.tmem_alloc_cols)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

        pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=False)

        tSTtST_shape = KQ_tiled_mma.partition_shape_C(cute.select(self.KQ_mma_tiler, mode=[0, 1]))
        tSTtST = KQ_tiled_mma.make_fragment_C(tSTtST_shape)
        tSTtST = cute.make_tensor(tmem_ptr + self.tmem_S_offset, tSTtST.layout)
        tdPTtdPT_shape = VdO_tiled_mma.partition_shape_C(cute.select(self.VdO_mma_tiler, mode=[0, 1]))
        tdPTtdPT = VdO_tiled_mma.make_fragment_C(tdPTtdPT_shape)
        tdPTtdPT = cute.make_tensor(tmem_ptr + self.tmem_dP_offset, tdPTtdPT.layout)
        tdVrP = PdO_tiled_mma.make_fragment_A(sP)
        tdVrdOT = PdO_tiled_mma.make_fragment_B(sdOT)
        tdKtdK_shape = dSQ_tiled_mma.partition_shape_C(cute.select(self.dSQ_mma_tiler, mode=[0, 1]))
        tdKtdK = dSQ_tiled_mma.make_fragment_C(tdKtdK_shape)
        tdKtdK = cute.make_tensor(tmem_ptr + self.tmem_dK_offset, tdKtdK.layout)
        tdVtdV_shape = PdO_tiled_mma.partition_shape_C(cute.select(self.PdO_mma_tiler, mode=[0, 1]))
        tdVtdV = PdO_tiled_mma.make_fragment_C(tdVtdV_shape)
        tdVtdV = cute.make_tensor(tmem_ptr + self.tmem_dV_offset, tdVtdV.layout)

        blk_coord = (Int32(0), bidx, Int32(0), ((Int32(0), bidy), bidz))
        blk_offset = (Int32(0), Int32(0), Int32(0), ((Int32(0), Int32(0)), Int32(0)))
        seqlen_q_cur_batch = Q_ref.shape[0]
        seqlen_k_cur_batch = K_ref.shape[0]
        iter_start, iter_end = self.get_Q_block_min_max(
            seqlen_q_cur_batch,
            seqlen_k_cur_batch,
            blk_coord[1],
            is_2cta=True,
        )
        pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        iter_count = (iter_end - iter_start) * problem_shape[3][0][0]
        problem_shape_cur_batch = (
            seqlen_q_cur_batch,
            seqlen_k_cur_batch,
            problem_shape[2],
            problem_shape[3],
        )
        if iter_count <= 0:
            if bidx * self.tile_shape_K < seqlen_k_cur_batch:
                self.epilogue_clear(blk_coord, blk_offset, problem_shape_cur_batch, dK, dV)
        elif warp_idx == self.load_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_load)
            self.load(
                K_in,
                V_in,
                Q_in,
                QT_in,
                dO_in,
                dOT_in,
                LSE,
                sum_OdO,
                sK,
                sQ,
                sQT,
                sV,
                sdO,
                sdOT,
                sLSE,
                sSum_OdO,
                KQ_tiled_mma,
                VdO_tiled_mma,
                PdO_tiled_mma,
                dSQ_tiled_mma,
                tma_atom_K,
                tma_atom_Q,
                tma_atom_QT,
                tma_atom_V,
                tma_atom_dO,
                tma_atom_dOT,
                blk_offset,
                problem_shape_cur_batch,
                varlen,
                iter_count,
                iter_start,
                iter_end,
                load_mma_Q_producer,
                load_mma_Q_consumer,
                load_mma_K_producer,
                load_mma_K_consumer,
                load_mma_V_producer,
                load_mma_V_consumer,
                load_compute_LSE_producer,
                load_compute_LSE_consumer,
                load_mma_dO_producer,
                load_mma_dO_consumer,
                load_mma_dOT_producer,
                load_mma_dOT_consumer,
                load_compute_sum_OdO_producer,
                load_compute_sum_OdO_consumer,
                load_mma_QT_producer,
                load_mma_QT_consumer,
            )
        elif warp_idx == self.mma_warp_id:
            cute.arch.warpgroup_reg_alloc(self.num_regs_mma)
            self.mma_2cta(
                KQ_tiled_mma,
                VdO_tiled_mma,
                PdO_tiled_mma,
                dSQ_tiled_mma,
                tSTtST,
                tSTrQ,
                tSTrK,
                tdPTtdPT,
                tdPTrV,
                tdPTrdO,
                tdVtdV,
                tdVrP,
                tdVrdOT,
                tdKrdST,
                tdKtdK,
                tdKrQT,
                iter_count,
                load_mma_Q_consumer,
                load_mma_K_consumer,
                load_mma_V_consumer,
                mma_compute_S_producer,
                load_mma_dO_consumer,
                mma_compute_dP_producer,
                load_mma_dOT_consumer,
                compute_mma_P_consumer,
                compute_mma_dS_consumer,
                load_mma_QT_consumer,
                mma_compute_dKdV_producer,
            )
        elif warp_idx >= self.compute_warp_id[0] and warp_idx <= self.compute_warp_id[-1]:
            cute.arch.warpgroup_reg_alloc(self.num_regs_compute)
            self.compute(
                tSTtST,
                tdPTtdPT,
                tdVrP,
                sP,
                sLSE,
                sdST,
                sdOT,
                sSum_OdO,
                dK,
                dV,
                tdKtdK,
                tdVtdV,
                PdO_tiled_mma,
                dSQ_tiled_mma,
                blk_coord,
                blk_offset,
                problem_shape_cur_batch,
                iter_count,
                iter_start,
                iter_end,
                scale_softmax,
                mma_compute_S_producer,
                mma_compute_S_consumer,
                compute_mma_P_producer,
                compute_mma_P_consumer,
                load_compute_LSE_producer,
                load_compute_LSE_consumer,
                load_compute_sum_OdO_producer,
                load_compute_sum_OdO_consumer,
                mma_compute_dP_producer,
                mma_compute_dP_consumer,
                compute_mma_dS_producer,
                compute_mma_dS_consumer,
                mma_compute_dKdV_producer,
                mma_compute_dKdV_consumer,
                varlen,
                sK,
                seqlen_k_cur_batch,
            )
            cute.arch.barrier(
                barrier_id=self.epilogue_sync_bar_id,
                number_of_threads=self.num_compute_warps * self.threads_per_warp,
            )
        else:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_empty)

        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)


def csa_comp_dkv_sm100_small(
    qv: torch.Tensor,
    comp_kv: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    delta: torch.Tensor,
    compress_ratio: int,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    total_q, heads, dim = qv.shape
    if compress_ratio != 4 or dim != 512 or comp_kv.shape[1] != 1:
        raise NotImplementedError("small SM100 CSA comp-DKV path is fixed to ratio=4, MQA, D=512")
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(dim)

    head_dim_v = dim // 2
    lse_log2 = (lse.float() * math.log2(math.e)).transpose(0, 1).contiguous()
    delta0 = (out[..., :head_dim_v].float() * dout[..., :head_dim_v].float()).sum(-1)
    delta1 = (out[..., head_dim_v:].float() * dout[..., head_dim_v:].float()).sum(-1)
    delta_correction = delta.float() - delta0 - delta1
    delta0 = delta0 + delta_correction
    dpsum0 = delta0.float().transpose(0, 1).contiguous()
    dpsum1 = delta1.float().transpose(0, 1).contiguous()
    q0 = qv[..., :head_dim_v]
    q1 = qv[..., head_dim_v:]
    v0 = comp_kv[..., :head_dim_v]
    v1 = comp_kv[..., head_dim_v:]
    do0 = dout[..., :head_dim_v]
    do1 = dout[..., head_dim_v:]
    dK00 = torch.empty((comp_kv.shape[0], 1, head_dim_v), device=comp_kv.device, dtype=comp_kv.dtype)
    dK01 = torch.empty_like(dK00)
    dK10 = torch.empty_like(dK00)
    dK11 = torch.empty_like(dK00)
    dV0 = torch.empty_like(dK00)
    dV1 = torch.empty_like(dK00)
    dV_dummy = torch.empty_like(dK00)
    current_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    compile_key = (qv.dtype, total_q, comp_kv.shape[0], heads)
    if compile_key not in csa_comp_dkv_sm100_small.compile_cache:
        kernel = CsaCompDkvSm100Small()
        csa_comp_dkv_sm100_small.compile_cache[compile_key] = cute.compile(
            kernel,
            to_cute_tensor(qv),
            to_cute_tensor(comp_kv),
            to_cute_tensor(v0),
            to_cute_tensor(q0),
            to_cute_tensor(dK00),
            to_cute_tensor(dV0),
            to_cute_tensor(do0),
            to_cute_tensor(do0),
            to_cute_tensor(lse_log2, assumed_align=4),
            to_cute_tensor(dpsum0, assumed_align=4),
            softmax_scale,
            current_stream,
            options="--enable-tvm-ffi",
        )
    if not is_fake_mode():
        compiled = csa_comp_dkv_sm100_small.compile_cache[compile_key]
        compiled(
            qv.detach(),
            comp_kv.detach(),
            v0.detach(),
            q0.detach(),
            dK00,
            dV0,
            do0,
            do0,
            lse_log2,
            dpsum0,
            softmax_scale,
        )
        compiled(
            qv.detach(),
            comp_kv.detach(),
            v1.detach(),
            q1.detach(),
            dK11,
            dV1,
            do1,
            do1,
            lse_log2,
            dpsum1,
            softmax_scale,
        )
        compiled(
            qv.detach(),
            comp_kv.detach(),
            v1.detach(),
            q0.detach(),
            dK10,
            dV_dummy,
            do1,
            do1,
            lse_log2,
            dpsum1,
            softmax_scale,
        )
        compiled(
            qv.detach(),
            comp_kv.detach(),
            v0.detach(),
            q1.detach(),
            dK01,
            dV_dummy,
            do0,
            do0,
            lse_log2,
            dpsum0,
            softmax_scale,
        )
    dK = torch.empty_like(comp_kv)
    dK[..., :head_dim_v] = (dK00.float() + dK10.float()).to(comp_kv.dtype)
    dK[..., head_dim_v:] = (dK01.float() + dK11.float()).to(comp_kv.dtype)
    dV = torch.empty_like(comp_kv)
    dV[..., :head_dim_v] = dV0
    dV[..., head_dim_v:] = dV1
    return (dK.float() + dV.float()).to(comp_kv.dtype)


csa_comp_dkv_sm100_small.compile_cache = get_jit_cache("csa_comp_dkv_sm100_small")
