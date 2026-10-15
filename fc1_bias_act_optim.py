from __future__ import annotations

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.nvgpu.warp import MmaF16BF16Op

try:
    from .cuda_compat import CUstream
except ImportError:
    from cuda_compat import CUstream

ACT_SILU = 0
ACT_GELU = 1

SM120_BF16_MMA_INST_MNK = (16, 8, 16)

# Warp assignment for the MMA warp group: ``prod(atom_layout)`` MMA warps plus
# one dedicated TMA producer warp.
#   (4, 2, 1) -> 8 MMA warps + 1 TMA warp = 288 threads  (default)
#   (2, 2, 1) -> 4 MMA warps + 1 TMA warp = 160 threads  (legacy, A/B baseline)
SM120_DEFAULT_ATOM_LAYOUT_MNK = (2, 2, 1)
SM120_SUPPORTED_ATOM_LAYOUTS_MNK = ((4, 4, 1), (4, 2, 1), (2, 2, 1))
SM120_DEFAULT_TILE_SHAPE_MNK = (64, 64, 64)
SM120_SUPPORTED_TILE_SHAPES_MNK = (
    (64, 64, 64),
    (64, 128, 64),
    (128, 64, 64),
    (128, 128, 64),
    (128, 256, 64),
    (128, 128, 128),
)


@dataclass(frozen=True)
class FusedGatedMlpShape:
    batch_tokens: int
    hidden_size: int
    intermediate_size: int


def activation_name(act_id: int) -> str:
    if act_id == ACT_SILU:
        return "silu"
    if act_id == ACT_GELU:
        return "gelu"
    raise ValueError(f"unsupported activation id: {act_id}")


class FC1GatedActSM120BF16Kernel:
    """Static CuTe DSL configuration for the SM120 BF16 fused gated MLP path.

    SM120 GeForce uses warp-level F16/BF16 MMA here.  This is intentionally not
    a tcgen05/WGMMA configuration: the supported BF16 atom is m16n8k16.
    """

    def __init__(
        self,
        shape: FusedGatedMlpShape | None = None,
        *,
        fc1_tile_shape_mnk: tuple[int, int, int] = SM120_DEFAULT_TILE_SHAPE_MNK,
        fc2_tile_shape_mnk: tuple[int, int, int] = SM120_DEFAULT_TILE_SHAPE_MNK,
        atom_layout_mnk: tuple[int, int, int] = SM120_DEFAULT_ATOM_LAYOUT_MNK,
        ab_dtype: type[cutlass.Numeric] = cutlass.Float16,
        acc_dtype: type[cutlass.Numeric] = cutlass.Float32,
        output_dtype: type[cutlass.Numeric] = cutlass.Float16,
    ):
        self.shape = shape
        self.fc1_tile_shape_mnk = tuple(fc1_tile_shape_mnk)
        self.fc2_tile_shape_mnk = tuple(fc2_tile_shape_mnk)
        self.tile_shape_mnk = self.fc1_tile_shape_mnk
        self.atom_layout = tuple(atom_layout_mnk)
        self.cluster_shape_mnk = (1, 1, 1)

        self.ab_dtype = ab_dtype
        self.a_dtype = ab_dtype
        self.b_dtype = ab_dtype
        self.acc_dtype = acc_dtype
        self.output_dtype = output_dtype
        self.c_dtype = output_dtype
        self.bias_dtype = output_dtype

        self.act_id = ACT_SILU

        self.mma_inst_mnk = SM120_BF16_MMA_INST_MNK
        self.permutation_mnk = None
        self.tiled_mma = None
        self.fc1_tiled_mma = None
        self.fc2_tiled_mma = None
        self.mma = None
        self.cta_layout_mnk = None

        self.occupancy = 1
        self.num_mma_warps = self.atom_layout[0] * self.atom_layout[1] * self.atom_layout[2]
        self.num_threads_per_warp = 32
        self.threads_per_cta = (self.num_mma_warps + 1) * self.num_threads_per_warp
        self.buffer_align_bytes = 1024

        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_120")
        self.ab_stage = None
        self.epi_stage = None
        self.epi_tile = None

        self.a_smem_layout_staged = None
        self.b_smem_layout_staged = None
        self.epi_smem_layout_staged = None

        self.shared_storage = None

        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.num_mma_warps * self.num_threads_per_warp,
        )
        self.load_register_requirement = 40
        # Each MMA warp owns two fp32 accumulators (gate + value).  At
        # atom_layout (2, 2, 1) a warp holds 2 x 32 x 32 fp32 values
        # (64 regs/lane); at (4, 2, 1) the per-warp value tile halves to
        # 16 x 16 per repetition, so 2 x 16 x 16 x 2 = 32 regs/lane suffices and
        # the reallocation target can drop.  Register budget at 1 CTA/SM:
        # 8 warps * 32 * 168 + 32 * 40 = 44288 <= 65536.
        self.mma_register_requirement = 224

        self.fc1_output_size = None if shape is None else shape.intermediate_size * 2
        self.fc2_output_size = None if shape is None else shape.hidden_size

        self._validate_static_config()

    def _validate_static_config(self) -> None:
        if self.ab_dtype not in (cutlass.Float16, cutlass.BFloat16):
            raise TypeError("FC1GatedActSM120BF16Kernel requires FP16 or BF16 A/B inputs")
        if self.output_dtype not in (cutlass.Float16, cutlass.BFloat16):
            raise TypeError("FC1GatedActSM120BF16Kernel requires FP16 or BF16 output/bias")
        if self.acc_dtype != cutlass.Float32:
            raise TypeError("SM120 BF16 MMA requires Float32 accumulation")
        if self.atom_layout not in SM120_SUPPORTED_ATOM_LAYOUTS_MNK:
            raise ValueError(
                f"SM120 BF16 path supports atom_layout_mnk in "
                f"{SM120_SUPPORTED_ATOM_LAYOUTS_MNK}, got {self.atom_layout}"
            )
        if self.mma_inst_mnk != SM120_BF16_MMA_INST_MNK:
            raise ValueError("SM120 BF16 warp MMA instruction must be m16n8k16")

        for name, tile_shape in (
            ("fc1_tile_shape_mnk", self.fc1_tile_shape_mnk),
            ("fc2_tile_shape_mnk", self.fc2_tile_shape_mnk),
        ):
            if tile_shape not in SM120_SUPPORTED_TILE_SHAPES_MNK:
                raise ValueError(
                    f"{name}={tile_shape} is not supported by the SM120 GeForce "
                    f"BF16 path; expected one of {SM120_SUPPORTED_TILE_SHAPES_MNK}"
                )

        if self.shape is None:
            return
        if self.shape.batch_tokens <= 0:
            raise ValueError("batch_tokens must be positive")
        if self.shape.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.shape.intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive")
        if self.shape.hidden_size % self.mma_inst_mnk[2] != 0:
            raise ValueError("hidden_size must be a multiple of 16 for m16n8k16 BF16 MMA")
        if self.shape.intermediate_size % self.mma_inst_mnk[2] != 0:
            raise ValueError("intermediate_size must be a multiple of 16 for m16n8k16 BF16 MMA")

    def _setup_attributes(self):
        op = MmaF16BF16Op(
            self.ab_dtype,
            self.acc_dtype,
            self.mma_inst_mnk,
        )
        tC = cute.make_layout(self.atom_layout)
        self.permutation_mnk = (
            self.atom_layout[0] * self.mma_inst_mnk[0],
            self.atom_layout[1] * self.mma_inst_mnk[1] * 2,
            self.atom_layout[2] * self.mma_inst_mnk[2],
        )
        self.tiled_mma = cute.make_tiled_mma(
            op,
            tC,
            permutation_mnk=self.permutation_mnk,
        )
        self.fc1_tiled_mma = self.tiled_mma
        self.fc2_tiled_mma = self.tiled_mma
        self.mma = self.tiled_mma
        self.cta_layout_mnk = cute.make_layout(self.cluster_shape_mnk)

        self.num_mcast_ctas_a = self.cluster_shape_mnk[1]
        self.num_mcast_ctas_b = self.cluster_shape_mnk[0]
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        self.epi_tile = sm90_utils.compute_tile_shape_or_override(
            self.tile_shape_mnk, self.c_dtype, is_cooperative=False
        )

        self.ab_stage, self.epi_stage = self._compute_stages(
            self.tile_shape_mnk,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.smem_capacity,
            self.occupancy,
        )
        if self.ab_stage == 0:
            raise RuntimeError(
                "ab_stage == 0: not enough shared memory for the gated FC1+Act "
                "kernel. Reduce the tile shape or the number of stages."
            )

        (
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
        ) = self._make_smem_layouts(
            self.tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.a_layout,
            self.b_dtype,
            self.b_layout,
            self.ab_stage,
            self.c_dtype,
            self.c_layout,
            self.epi_stage,
        )
        return self

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        w_gate: cute.Tensor,
        w_value: cute.Tensor,
        y: cute.Tensor,
        act_id: cutlass.Constexpr,
        max_active_clusters: cutlass.Constexpr,
        stream: CUstream,
    ):
        """JIT host entry: build TMA atoms, smem layouts, SharedStorage, launch.

        Computes ``y = w_value(x) * act(w_gate(x))`` where ``act`` is selected
        by ``act_id`` (ACT_SILU / ACT_GELU) at compile time. ``x`` is A
        ([M, H] row-major), ``w_gate``/``w_value`` are the two B operands
        ([I, H] row-major), ``y`` is C ([M, I] row-major).
        """

        self.a_dtype = x.element_type
        self.b_dtype = w_gate.element_type
        self.c_dtype = y.element_type

        self.a_layout = utils.LayoutEnum.from_tensor(x)
        self.b_layout = utils.LayoutEnum.from_tensor(w_gate)
        self.c_layout = utils.LayoutEnum.from_tensor(y)

        if cutlass.const_expr(self.a_dtype not in (cutlass.Float16, cutlass.BFloat16)):
            raise TypeError("FC1GatedActSM120BF16Kernel requires FP16 or BF16 inputs")
        if cutlass.const_expr(
            self.a_dtype.width != self.b_dtype.width or self.a_dtype != w_value.element_type
        ):
            raise TypeError("Type mismatch across x / w_gate / w_value")
        if cutlass.const_expr(self.c_dtype not in (cutlass.Float16, cutlass.BFloat16)):
            raise TypeError("FC1GatedActSM120BF16Kernel requires FP16 or BF16 output")

        self._setup_attributes()

        tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(
            x,
            self.a_smem_layout_staged,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
            1,
        )

        tma_atom_b_gate, tma_tensor_b_gate = self._make_tma_atoms_and_tensors(
            w_gate,
            self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            1,
        )

        tma_atom_b_value, tma_tensor_b_value = self._make_tma_atoms_and_tensors(
            w_value,
            self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            1,
        )

        tma_atom_c, tma_tensor_c = self._make_tma_store_atoms_and_tensors(
            y,
            self.epi_smem_layout_staged,
            self.epi_tile,
        )

        tile_sched_params, grid = self._compute_grid(
            y,
            self.tile_shape_mnk,
            max_active_clusters,
        )

        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.ab_stage * 2]
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sB_gate: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sB_value: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[self.c_dtype, cute.cosize(self.epi_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        self.act_id = act_id

        self.kernel(
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b_gate,
            tma_tensor_b_gate,
            tma_atom_b_value,
            tma_tensor_b_value,
            tma_atom_c,
            tma_tensor_c,
            self.tiled_mma,
            self.cta_layout_mnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
            tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=[1, 1, 1],
            stream=stream,
        )
        return

    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b_gate: cute.CopyAtom,
        mB_gate_nkl: cute.Tensor,
        tma_atom_b_value: cute.CopyAtom,
        mB_value_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        tiled_mma: cute.TiledMma,
        cta_layout_mnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: utils.PersistentTileSchedulerParams,
    ):
        """Device kernel: fused gated FC1 + activation, two mma.sync per K-block.

        ``value_part = x @ w_value`` accumulates into ``acc_value`` and
        ``gate_part  = x @ w_gate``  accumulates into ``acc_gate``. After the
        K-loop the gated activation ``out = acc_value * act(acc_gate)`` is
        computed in registers and stored via TMA.
        """

        # ///////////////////////////////////////////////////////////////////////////////
        #  Get cta/warp/thread idx
        # ///////////////////////////////////////////////////////////////////////////////
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        # /////////////////////////////////////////////////////////////////////////////
        #  Prefetch Tma desc
        # /////////////////////////////////////////////////////////////////////////////
        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b_gate)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b_value)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_c)

        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)

        a_mcast_mask = cute.make_layout_image_mask(cta_layout_mnk, cluster_coord_mnk, mode=1)
        b_mcast_mask = cute.make_layout_image_mask(cta_layout_mnk, cluster_coord_mnk, mode=0)
        a_mcast_mask = a_mcast_mask if self.is_a_mcast else 0
        b_mcast_mask = b_mcast_mask if self.is_b_mcast else 0
        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        # One A + two B tiles live in smem each stage.
        tma_copy_bytes = cute.size_in_bytes(self.a_dtype, a_smem_layout) + 2 * cute.size_in_bytes(
            self.b_dtype, b_smem_layout
        )

        # /////////////////////////////////////////////////////////////////////////////
        #  Alloc and init pipeline
        # ///////////////////////////////////////////////////////////////////////////////
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()

        mainloop_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        mcast_size = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        consumer_arrive_cnt = mcast_size * self.num_mma_warps
        mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, consumer_arrive_cnt
        )

        cta_layout_vmnk = cute.make_layout((1, *cta_layout_mnk.shape))
        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.ab_stage,
            producer_group=mainloop_pipeline_producer_group,
            consumer_group=mainloop_pipeline_consumer_group,
            tx_count=tma_copy_bytes,
            barrier_storage=mainloop_pipeline_array_ptr,
            cta_layout_vmnk=cta_layout_vmnk,
        )

        if cute.size(self.cluster_shape_mnk) > 1:
            cute.arch.cluster_arrive_relaxed()

        # ///////////////////////////////////////////////////////////////////////////////
        #  Generate smem tensors
        # ///////////////////////////////////////////////////////////////////////////////
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sB_gate = storage.sB_gate.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sB_value = storage.sB_value.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sC = storage.sC.get_tensor(
            epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner
        )

        # ///////////////////////////////////////////////////////////////////////////////
        #  Local_tile partition global tensors
        # ///////////////////////////////////////////////////////////////////////////////
        gA_mkl = cute.local_tile(
            mA_mkl,
            cute.slice_(self.tile_shape_mnk, (None, 0, None)),
            (None, None, None),
        )
        gB_gate_nkl = cute.local_tile(
            mB_gate_nkl,
            cute.slice_(self.tile_shape_mnk, (0, None, None)),
            (None, None, None),
        )
        gB_value_nkl = cute.local_tile(
            mB_value_nkl,
            cute.slice_(self.tile_shape_mnk, (0, None, None)),
            (None, None, None),
        )
        gC_mnl = cute.local_tile(
            mC_mnl,
            cute.slice_(self.tile_shape_mnk, (None, None, 0)),
            (None, None, None),
        )

        # //////////////////////////////////////////////////////////////////////////////
        #  Partition shared tensor for TMA load A / B_gate / B_value
        # //////////////////////////////////////////////////////////////////////////////
        a_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)
        a_cta_crd = cluster_coord_mnk[1]
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a,
            a_cta_crd,
            a_cta_layout,
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA_mkl, 0, 2),
        )

        b_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)
        b_cta_crd = cluster_coord_mnk[0]
        tBsB_gate, tBgB_gate = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b_gate,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sB_gate, 0, 2),
            cute.group_modes(gB_gate_nkl, 0, 2),
        )
        tBsB_value, tBgB_value = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b_value,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sB_value, 0, 2),
            cute.group_modes(gB_value_nkl, 0, 2),
        )

        # //////////////////////////////////////////////////////////////////////////////
        #  MMA fragments
        # //////////////////////////////////////////////////////////////////////////////
        thr_mma = tiled_mma.get_slice(tidx)

        tCsA = thr_mma.partition_A(sA)
        tCsB_gate = thr_mma.partition_B(sB_gate)
        tCsB_value = thr_mma.partition_B(sB_value)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB_gate = tiled_mma.make_fragment_B(tCsB_gate[None, None, None, 0])
        tCrB_value = tiled_mma.make_fragment_B(tCsB_value[None, None, None, 0])

        tCgC = thr_mma.partition_C(gC_mnl)
        acc_shape = tCgC.shape[:3]
        acc_value = cute.make_rmem_tensor(acc_shape, self.acc_dtype)
        acc_gate = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

        # cluster wait for barrier init
        if cute.size(self.cluster_shape_mnk) > 1:
            cute.arch.cluster_wait()
        else:
            pipeline.sync(barrier_id=1)

        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        tile_sched = utils.StaticPersistentTileScheduler.create(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()

        mainloop_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.ab_stage
        )
        mainloop_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )

        # MMA warp group
        if warp_idx < self.num_mma_warps:
            cute.arch.setmaxregister_increase(self.mma_register_requirement)

            num_k_blocks = cute.size(tCrA, mode=[2])

            # Copy Atom A/B retiling (B atom reused for both gate/value)
            atom_copy_ldmatrix_A = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(self.a_layout.is_m_major_a(), 4),
                self.a_dtype,
            )
            atom_copy_ldmatrix_B = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(self.b_layout.is_n_major_b(), 4),
                self.b_dtype,
            )
            smem_tiled_copy_A = cute.make_tiled_copy_A(atom_copy_ldmatrix_A, tiled_mma)
            smem_tiled_copy_B = cute.make_tiled_copy_B(atom_copy_ldmatrix_B, tiled_mma)

            thr_copy_ldmatrix_A = smem_tiled_copy_A.get_slice(tidx)
            thr_copy_ldmatrix_B = smem_tiled_copy_B.get_slice(tidx)
            tCsA_copy_view = thr_copy_ldmatrix_A.partition_S(sA)
            tCrA_copy_view = thr_copy_ldmatrix_A.retile(tCrA)

            tCsB_gate_copy_view = thr_copy_ldmatrix_B.partition_S(sB_gate)
            tCrB_gate_copy_view = thr_copy_ldmatrix_B.retile(tCrB_gate)
            tCsB_value_copy_view = thr_copy_ldmatrix_B.partition_S(sB_value)
            tCrB_value_copy_view = thr_copy_ldmatrix_B.retile(tCrB_value)

            # The TMA store pipeline spans ALL persistent work tiles.
            # producer_acquire/commit carry no per-call state and lower to
            # cp.async.bulk.wait_group/commit_group, while the smem stage index
            # is chosen by us -- so keeping the pipeline alive across tiles
            # (instead of draining it at every tile boundary) lets the previous
            # tile's stores stay in flight while this tile's mainloop runs.
            tma_store_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.epi_stage,
                producer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    self.num_mma_warps * self.num_threads_per_warp,
                ),
            )
            # Rolling epilogue buffer slot: keeps advancing across tiles so a
            # buffer is not reused until its store has drained (epi_stage deep).
            epi_buffer_slot = cutlass.Int32(0)

            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                gC_mnl_slice = gC_mnl[(None, None, *tile_coord_mnl)]
                acc_value.fill(0.0)
                acc_gate.fill(0.0)

                # /////////////////////////////////////////////////////////////////////////////
                #  Pipelined MAINLOOP
                # /////////////////////////////////////////////////////////////////////////////
                mainloop_consumer_state.reset_count()

                peek_ab_full_status = cutlass.Boolean(1)
                if mainloop_consumer_state.count < k_tile_cnt:
                    peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                        mainloop_consumer_state
                    )

                mainloop_pipeline.consumer_wait(mainloop_consumer_state, peek_ab_full_status)
                tCsA_p = tCsA_copy_view[None, None, None, mainloop_consumer_state.index]
                tCsB_gate_p = tCsB_gate_copy_view[None, None, None, mainloop_consumer_state.index]
                tCsB_value_p = tCsB_value_copy_view[None, None, None, mainloop_consumer_state.index]
                cute.copy(
                    smem_tiled_copy_A,
                    tCsA_p[None, None, 0],
                    tCrA_copy_view[None, None, 0],
                )
                cute.copy(
                    smem_tiled_copy_B,
                    tCsB_gate_p[None, None, 0],
                    tCrB_gate_copy_view[None, None, 0],
                )
                cute.copy(
                    smem_tiled_copy_B,
                    tCsB_value_p[None, None, 0],
                    tCrB_value_copy_view[None, None, 0],
                )

                for _k_tile in range(0, k_tile_cnt - 1, 1, unroll=1):
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_block_next = 0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1

                        if k_block_idx == num_k_blocks - 1:
                            mainloop_pipeline.consumer_release(mainloop_consumer_state)
                            mainloop_consumer_state.advance()

                            peek_ab_full_status = cutlass.Boolean(1)
                            peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                                mainloop_consumer_state
                            )

                            tCsA_p = tCsA_copy_view[None, None, None, mainloop_consumer_state.index]
                            tCsB_gate_p = tCsB_gate_copy_view[
                                None, None, None, mainloop_consumer_state.index
                            ]
                            tCsB_value_p = tCsB_value_copy_view[
                                None, None, None, mainloop_consumer_state.index
                            ]
                            mainloop_pipeline.consumer_wait(
                                mainloop_consumer_state, peek_ab_full_status
                            )

                        cute.copy(
                            smem_tiled_copy_A,
                            tCsA_p[None, None, k_block_next],
                            tCrA_copy_view[None, None, k_block_next],
                        )
                        cute.copy(
                            smem_tiled_copy_B,
                            tCsB_value_p[None, None, k_block_next],
                            tCrB_value_copy_view[None, None, k_block_next],
                        )
                        cute.gemm(
                            tiled_mma,
                            acc_value,
                            tCrA[None, None, k_block_idx],
                            tCrB_value[None, None, k_block_idx],
                            acc_value,
                        )
                        cute.copy(
                            smem_tiled_copy_B,
                            tCsB_gate_p[None, None, k_block_next],
                            tCrB_gate_copy_view[None, None, k_block_next],
                        )
                        cute.gemm(
                            tiled_mma,
                            acc_gate,
                            tCrA[None, None, k_block_idx],
                            tCrB_gate[None, None, k_block_idx],
                            acc_gate,
                        )
                # end of for loop
                # Hoist out last k_tile
                for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                    k_block_next = 0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1

                    if k_block_idx == num_k_blocks - 1:
                        mainloop_pipeline.consumer_release(mainloop_consumer_state)
                        mainloop_consumer_state.advance()

                    if k_block_next > 0:
                        cute.copy(
                            smem_tiled_copy_A,
                            tCsA_p[None, None, k_block_next],
                            tCrA_copy_view[None, None, k_block_next],
                        )
                        cute.copy(
                            smem_tiled_copy_B,
                            tCsB_value_p[None, None, k_block_next],
                            tCrB_value_copy_view[None, None, k_block_next],
                        )
                    cute.gemm(
                        tiled_mma,
                        acc_value,
                        tCrA[None, None, k_block_idx],
                        tCrB_value[None, None, k_block_idx],
                        acc_value,
                    )
                    if k_block_next > 0:
                        cute.copy(
                            smem_tiled_copy_B,
                            tCsB_gate_p[None, None, k_block_next],
                            tCrB_gate_copy_view[None, None, k_block_next],
                        )
                    cute.gemm(
                        tiled_mma,
                        acc_gate,
                        tCrA[None, None, k_block_idx],
                        tCrB_gate[None, None, k_block_idx],
                        acc_gate,
                    )

                # /////////////////////////////////////////////////////////////////////////////
                #  EPILOG: gated activation + r2s + TMA store
                # /////////////////////////////////////////////////////////////////////////////
                copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
                    self.c_layout,
                    elem_ty_d=self.c_dtype,
                    elem_ty_acc=self.acc_dtype,
                )
                copy_atom_C = cute.make_copy_atom(
                    cute.nvgpu.warp.StMatrix8x8x16bOp(self.c_layout.is_m_major_c(), 4),
                    self.c_dtype,
                )
                tiled_copy_C_Atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)
                tiled_copy_r2s = cute.make_tiled_copy_S(copy_atom_r2s, tiled_copy_C_Atom)

                thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
                tRS_sD = thr_copy_r2s.partition_D(sC)
                tRS_rAcc_value = tiled_copy_r2s.retile(acc_value)
                tRS_rAcc_gate = tiled_copy_r2s.retile(acc_gate)

                rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
                tRS_rD_layout = cute.make_layout(rD_shape[:3])
                tRS_rD = cute.make_rmem_tensor(tRS_rD_layout.shape, self.acc_dtype)
                size_tRS_rD = cute.size(tRS_rD)

                sepi_for_tma_partition = cute.group_modes(sC, 0, 2)
                tcgc_for_tma_partition = cute.zipped_divide(gC_mnl_slice, self.epi_tile)

                bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_c,
                    0,
                    cute.make_layout(1),
                    sepi_for_tma_partition,
                    tcgc_for_tma_partition,
                )

                epi_tile_num = cute.size(tcgc_for_tma_partition, mode=[1])
                epi_tile_shape = tcgc_for_tma_partition.shape[1]
                epi_tile_layout = cute.make_layout(epi_tile_shape, stride=(1, epi_tile_shape[0]))

                for epi_idx in cutlass.range_constexpr(epi_tile_num):
                    # Gated activation on fp32 accumulators before the r2s copy.
                    for epi_v in cutlass.range_constexpr(size_tRS_rD):
                        acc_v = tRS_rAcc_value[epi_idx * size_tRS_rD + epi_v]
                        acc_g = tRS_rAcc_gate[epi_idx * size_tRS_rD + epi_v]
                        # act_id is restricted to ACT_SILU / ACT_GELU at the API
                        # boundary; this initial binding keeps the traced branch
                        # total without a Python-level early exit, which the DSL
                        # rejects inside a kernel body.
                        act_val = cutlass.Float32(0.0)
                        if cutlass.const_expr(self.act_id == ACT_SILU):
                            # silu(x) = x * sigmoid(x) = 0.5 * x * (1 + tanh(x/2))
                            # Exact identity, but it replaces the exp + IEEE fp32
                            # division sequence (MUFU.EX2 + MUFU.RCP + Newton
                            # refinement + rounding fixups, ~8-10 instructions)
                            # with one MUFU.TANH plus two FMUL/FMA.  This kernel
                            # issues ~8 instructions per HMMA and MUFU throughput
                            # is a quarter of the FMA rate, so the epilogue's
                            # per-element instruction count is what competes with
                            # the MMA stream.
                            act_val = (
                                cutlass.Float32(0.5)
                                * acc_g
                                * (cutlass.Float32(1.0) + cute.tanh(cutlass.Float32(0.5) * acc_g))
                            )
                        elif cutlass.const_expr(self.act_id == ACT_GELU):
                            t3 = acc_g * acc_g * acc_g
                            inner = cutlass.Float32(0.7978845608028654) * (
                                acc_g + cutlass.Float32(0.044715) * t3
                            )
                            act_val = (
                                cutlass.Float32(0.5)
                                * acc_g
                                * (cutlass.Float32(1.0) + cute.tanh(inner))
                            )
                        tRS_rD[epi_v] = acc_v * act_val

                    tRS_rD_out = cute.make_rmem_tensor(tRS_rD_layout.shape, self.c_dtype)
                    acc_vec = tRS_rD.load()
                    tRS_rD_out.store(acc_vec.to(self.c_dtype))

                    epi_buffer = epi_buffer_slot % cute.size(tRS_sD, mode=[3])
                    epi_buffer_slot = epi_buffer_slot + 1
                    cute.copy(
                        tiled_copy_r2s,
                        tRS_rD_out,
                        tRS_sD[(None, None, None, epi_buffer)],
                    )

                    cute.arch.fence_proxy("async.shared", space="cta")
                    self.epilog_sync_barrier.arrive_and_wait()

                    gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
                    if warp_idx == 0:
                        cute.copy(
                            tma_atom_c,
                            bSG_sD[(None, epi_buffer)],
                            bSG_gD[(None, gmem_coord)],
                        )
                        tma_store_pipeline.producer_commit()
                        tma_store_pipeline.producer_acquire()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            # End of while loop
            # Drain all outstanding TMA stores once, after the last tile.
            tma_store_pipeline.producer_tail()
        # End of MMA warp group
        # Start of DMA warp group
        elif warp_idx == self.num_mma_warps:
            cute.arch.setmaxregister_decrease(self.load_register_requirement)

            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                tAgA_mkl = tAgA[(None, tile_coord_mnl[0], None, tile_coord_mnl[2])]
                tBgB_gate_nkl = tBgB_gate[(None, tile_coord_mnl[1], None, tile_coord_mnl[2])]
                tBgB_value_nkl = tBgB_value[(None, tile_coord_mnl[1], None, tile_coord_mnl[2])]

                mainloop_producer_state.reset_count()

                for _k_tile in range(0, k_tile_cnt, 1, unroll=1):
                    mainloop_pipeline.producer_acquire(mainloop_producer_state)

                    tAgA_k = tAgA_mkl[(None, mainloop_producer_state.count)]
                    tAsA_pipe = tAsA[(None, mainloop_producer_state.index)]
                    tBgB_gate_k = tBgB_gate_nkl[(None, mainloop_producer_state.count)]
                    tBsB_gate_pipe = tBsB_gate[(None, mainloop_producer_state.index)]
                    tBgB_value_k = tBgB_value_nkl[(None, mainloop_producer_state.count)]
                    tBsB_value_pipe = tBsB_value[(None, mainloop_producer_state.index)]

                    cute.copy(
                        tma_atom_a,
                        tAgA_k,
                        tAsA_pipe,
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(mainloop_producer_state),
                        mcast_mask=a_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b_gate,
                        tBgB_gate_k,
                        tBsB_gate_pipe,
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(mainloop_producer_state),
                        mcast_mask=b_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b_value,
                        tBgB_value_k,
                        tBsB_value_pipe,
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(mainloop_producer_state),
                        mcast_mask=b_mcast_mask,
                    )
                    mainloop_pipeline.producer_commit(mainloop_producer_state)
                    mainloop_producer_state.advance()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            mainloop_pipeline.producer_tail(mainloop_producer_state)
        return

    @staticmethod
    def _compute_stages(
        tile_shape_mnk: tuple[int, int, int],
        a_dtype: type[cutlass.Numeric],
        b_dtype: type[cutlass.Numeric],
        epi_tile: tuple[int, int],
        c_dtype: type[cutlass.Numeric],
        smem_capacity: int,
        occupancy: int,
    ) -> tuple[int, int]:
        """Compute (ab_stage, epi_stage) for the gated kernel.

        Unlike dense_gemm, ``ab_bytes_per_stage`` counts one A tile plus two B
        tiles (gate + value), since both B operands live in smem per stage.
        """
        epi_stage = 8
        c_bytes_per_stage = cute.size(epi_tile) * c_dtype.width // 8
        epi_bytes = c_bytes_per_stage * epi_stage

        a_shape = cute.slice_(tile_shape_mnk, (None, 0, None))
        b_shape = cute.slice_(tile_shape_mnk, (0, None, None))
        ab_bytes_per_stage = (
            cute.size(a_shape) * a_dtype.width // 8 + 2 * cute.size(b_shape) * b_dtype.width // 8
        )
        mbar_helpers_bytes = 1024

        ab_stage = (
            (smem_capacity - occupancy * 1024) // occupancy - mbar_helpers_bytes - epi_bytes
        ) // ab_bytes_per_stage
        return ab_stage, epi_stage

    @staticmethod
    def _make_smem_layouts(
        tile_shape_mnk: tuple[int, int, int],
        epi_tile: tuple[int, int],
        a_dtype: type[cutlass.Numeric],
        a_layout: cute.Layout,
        b_dtype: type[cutlass.Numeric],
        b_layout: cute.Layout,
        ab_stage: int,
        c_dtype: type[cutlass.Numeric],
        c_layout: cute.Layout,
        epi_stage: int,
    ) -> tuple[cute.ComposedLayout, cute.ComposedLayout, cute.ComposedLayout]:
        """Create shared memory layouts for A, B (shared by gate/value), and C."""
        a_smem_layout_staged = sm90_utils.make_smem_layout_a(
            a_layout,
            tile_shape_mnk,
            a_dtype,
            ab_stage,
        )
        b_smem_layout_staged = sm90_utils.make_smem_layout_b(
            b_layout,
            tile_shape_mnk,
            b_dtype,
            ab_stage,
        )
        epi_smem_layout_staged = sm90_utils.make_smem_layout_epi(
            c_dtype,
            c_layout,
            epi_tile,
            epi_stage,
        )
        return a_smem_layout_staged, b_smem_layout_staged, epi_smem_layout_staged

    @staticmethod
    def _compute_grid(
        c: cute.Tensor,
        tile_shape_mnk: tuple[int, int, int],
        max_active_clusters: cutlass.Constexpr,
    ) -> tuple[int, int, int]:
        c_shape = cute.slice_(tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape
        cluster_shape_mnl = (1, 1, 1)
        tile_sched_params = utils.PersistentTileSchedulerParams(num_ctas_mnl, cluster_shape_mnl)
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )
        return tile_sched_params, grid

    @staticmethod
    def _make_tma_store_atoms_and_tensors(
        tensor_c: cute.Tensor,
        epi_smem_layout_staged: cute.ComposedLayout,
        epi_tile: tuple[int, int],
    ) -> tuple[cute.CopyAtom, cute.Tensor]:
        epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
            tensor_c,
            epi_smem_layout,
            epi_tile,
        )
        return tma_atom_c, tma_tensor_c

    @staticmethod
    def _make_tma_atoms_and_tensors(
        tensor: cute.Tensor,
        smem_layout_staged: cute.ComposedLayout,
        smem_tile: tuple[int, int],
        mcast_dim: int,
    ) -> tuple[cute.CopyAtom, cute.Tensor]:
        op = (
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
            if mcast_dim == 1
            else cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp()
        )
        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        tma_atom, tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op,
            tensor,
            smem_layout,
            smem_tile,
            num_multicast=mcast_dim,
        )
        return tma_atom, tma_tensor


class FC1GatedActSM120BF16BackwardDxKernel:
    """CuTe DSL kernel for the backward dx pass of the gated FC1 layer.

    Computes ``dx = dgate @ w_gate_T + dvalue @ w_value_T`` in a single
    kernel launch with two fused GEMMs sharing one accumulator.

    Both A and B operands must share the same K dimension (the MMA
    contraction dimension). Since A tensors (dgate/dvalue) are [M, I] with
    K=I, the B tensors must also be K=I — i.e. stored as [H, I] (transposed
    relative to the forward kernel's [I, H] weights).

    The tile shape is ``(M, H, I)`` (N=H, K=I).
    """

    def __init__(
        self,
        shape: FusedGatedMlpShape | None = None,
        *,
        tile_shape_mnk: tuple[int, int, int] = SM120_DEFAULT_TILE_SHAPE_MNK,
        atom_layout_mnk: tuple[int, int, int] = SM120_DEFAULT_ATOM_LAYOUT_MNK,
        ab_dtype: type[cutlass.Numeric] = cutlass.Float16,
        acc_dtype: type[cutlass.Numeric] = cutlass.Float32,
        output_dtype: type[cutlass.Numeric] = cutlass.Float16,
    ):
        self.shape = shape
        self.tile_shape_mnk = tuple(tile_shape_mnk)
        self.atom_layout = tuple(atom_layout_mnk)
        self.cluster_shape_mnk = (1, 1, 1)

        self.ab_dtype = ab_dtype
        self.a_dtype = ab_dtype
        self.b_dtype = ab_dtype
        self.acc_dtype = acc_dtype
        self.output_dtype = output_dtype
        self.c_dtype = output_dtype

        self.mma_inst_mnk = SM120_BF16_MMA_INST_MNK
        self.permutation_mnk = None
        self.tiled_mma = None
        self.mma = None
        self.cta_layout_mnk = None

        self.occupancy = 1
        self.num_mma_warps = self.atom_layout[0] * self.atom_layout[1] * self.atom_layout[2]
        self.num_threads_per_warp = 32
        self.threads_per_cta = (self.num_mma_warps + 1) * self.num_threads_per_warp
        self.buffer_align_bytes = 1024

        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_120")
        self.ab_stage = None
        self.epi_stage = None
        self.epi_tile = None

        self.a_smem_layout_staged = None
        self.b_smem_layout_staged = None
        self.epi_smem_layout_staged = None

        self.shared_storage = None

        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.num_mma_warps * self.num_threads_per_warp,
        )
        self.load_register_requirement = 40
        self.mma_register_requirement = 232 if self.num_mma_warps <= 4 else 168

        self._validate_static_config()

    def _validate_static_config(self) -> None:
        if self.ab_dtype not in (cutlass.Float16, cutlass.BFloat16):
            raise TypeError("FC1GatedActSM120BF16BackwardDxKernel requires FP16 or BF16 A/B inputs")
        if self.output_dtype not in (cutlass.Float16, cutlass.BFloat16):
            raise TypeError("FC1GatedActSM120BF16BackwardDxKernel requires FP16 or BF16 output")
        if self.acc_dtype != cutlass.Float32:
            raise TypeError("SM120 BF16 MMA requires Float32 accumulation")
        if self.atom_layout not in SM120_SUPPORTED_ATOM_LAYOUTS_MNK:
            raise ValueError(
                f"SM120 BF16 path supports atom_layout_mnk in "
                f"{SM120_SUPPORTED_ATOM_LAYOUTS_MNK}, got {self.atom_layout}"
            )
        if self.mma_inst_mnk != SM120_BF16_MMA_INST_MNK:
            raise ValueError("SM120 BF16 warp MMA instruction must be m16n8k16")

        if self.tile_shape_mnk not in SM120_SUPPORTED_TILE_SHAPES_MNK:
            raise ValueError(
                f"tile_shape_mnk={self.tile_shape_mnk} is not supported by the "
                f"SM120 GeForce BF16 path; expected one of "
                f"{SM120_SUPPORTED_TILE_SHAPES_MNK}"
            )

        if self.shape is None:
            return
        if self.shape.batch_tokens <= 0:
            raise ValueError("batch_tokens must be positive")
        if self.shape.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.shape.intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive")
        if self.shape.hidden_size % self.mma_inst_mnk[2] != 0:
            raise ValueError("hidden_size must be a multiple of 16 for m16n8k16 BF16 MMA")
        if self.shape.intermediate_size % self.mma_inst_mnk[2] != 0:
            raise ValueError("intermediate_size must be a multiple of 16 for m16n8k16 BF16 MMA")

    def _setup_attributes(self):
        op = MmaF16BF16Op(
            self.ab_dtype,
            self.acc_dtype,
            self.mma_inst_mnk,
        )
        tC = cute.make_layout(self.atom_layout)
        self.permutation_mnk = (
            self.atom_layout[0] * self.mma_inst_mnk[0],
            self.atom_layout[1] * self.mma_inst_mnk[1] * 2,
            self.atom_layout[2] * self.mma_inst_mnk[2],
        )
        self.tiled_mma = cute.make_tiled_mma(
            op,
            tC,
            permutation_mnk=self.permutation_mnk,
        )
        self.mma = self.tiled_mma
        self.cta_layout_mnk = cute.make_layout(self.cluster_shape_mnk)

        self.num_mcast_ctas_a = self.cluster_shape_mnk[1]
        self.num_mcast_ctas_b = self.cluster_shape_mnk[0]
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        self.epi_tile = sm90_utils.compute_tile_shape_or_override(
            self.tile_shape_mnk, self.c_dtype, is_cooperative=False
        )

        self.ab_stage, self.epi_stage = self._compute_stages(
            self.tile_shape_mnk,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.smem_capacity,
            self.occupancy,
        )
        if self.ab_stage == 0:
            raise RuntimeError(
                "ab_stage == 0: not enough shared memory for the backward dx "
                "kernel. Reduce the tile shape or the number of stages."
            )

        (
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
        ) = self._make_smem_layouts(
            self.tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.a_layout,
            self.b_dtype,
            self.b_layout,
            self.ab_stage,
            self.c_dtype,
            self.c_layout,
            self.epi_stage,
        )
        return self

    @cute.jit
    def __call__(
        self,
        dgate: cute.Tensor,
        dvalue: cute.Tensor,
        w_gate: cute.Tensor,
        w_value: cute.Tensor,
        dx: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: CUstream,
    ):
        """JIT host entry: build TMA atoms, smem layouts, SharedStorage, launch.

        Computes ``dx = dgate @ w_gate_T + dvalue @ w_value_T`` where all
        tensors are row-major with the K dimension as the leading/contiguous
        dim. ``dgate`` / ``dvalue`` are A ([M, I], K=I); ``w_gate_T`` /
        ``w_value_T`` are B ([H, I], K=I — transposed relative to the forward
        kernel's [I, H] to align the K dimension with the A operands); ``dx``
        is C ([M, H]).
        """

        self.a_dtype = dgate.element_type
        self.b_dtype = w_gate.element_type
        self.c_dtype = dx.element_type

        self.a_layout = utils.LayoutEnum.from_tensor(dgate)
        self.b_layout = utils.LayoutEnum.from_tensor(w_gate)
        self.c_layout = utils.LayoutEnum.from_tensor(dx)

        if cutlass.const_expr(self.a_dtype not in (cutlass.Float16, cutlass.BFloat16)):
            raise TypeError("FC1GatedActSM120BF16BackwardDxKernel requires FP16 or BF16 inputs")
        if cutlass.const_expr(
            self.a_dtype.width != self.b_dtype.width
            or self.a_dtype != dvalue.element_type
            or self.b_dtype != w_value.element_type
        ):
            raise TypeError("Type mismatch across dgate / dvalue / w_gate / w_value")
        if cutlass.const_expr(self.c_dtype not in (cutlass.Float16, cutlass.BFloat16)):
            raise TypeError("FC1GatedActSM120BF16BackwardDxKernel requires FP16 or BF16 output")

        self._setup_attributes()

        tma_atom_dgate, tma_tensor_dgate = self._make_tma_atoms_and_tensors(
            dgate,
            self.a_smem_layout_staged,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
            1,
        )

        tma_atom_dvalue, tma_tensor_dvalue = self._make_tma_atoms_and_tensors(
            dvalue,
            self.a_smem_layout_staged,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
            1,
        )

        tma_atom_w_gate, tma_tensor_w_gate = self._make_tma_atoms_and_tensors(
            w_gate,
            self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            1,
        )

        tma_atom_w_value, tma_tensor_w_value = self._make_tma_atoms_and_tensors(
            w_value,
            self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            1,
        )

        tma_atom_c, tma_tensor_c = self._make_tma_store_atoms_and_tensors(
            dx,
            self.epi_smem_layout_staged,
            self.epi_tile,
        )

        tile_sched_params, grid = self._compute_grid(
            dx,
            self.tile_shape_mnk,
            max_active_clusters,
        )

        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.ab_stage * 2]
            sA_dgate: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sA_dvalue: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sB_w_gate: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sB_w_value: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[self.c_dtype, cute.cosize(self.epi_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        self.kernel(
            tma_atom_dgate,
            tma_tensor_dgate,
            tma_atom_dvalue,
            tma_tensor_dvalue,
            tma_atom_w_gate,
            tma_tensor_w_gate,
            tma_atom_w_value,
            tma_tensor_w_value,
            tma_atom_c,
            tma_tensor_c,
            self.tiled_mma,
            self.cta_layout_mnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
            tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=[1, 1, 1],
            stream=stream,
        )
        return

    @cute.kernel
    def kernel(
        self,
        tma_atom_a1: cute.CopyAtom,
        mA1_mkl: cute.Tensor,
        tma_atom_a2: cute.CopyAtom,
        mA2_mkl: cute.Tensor,
        tma_atom_b1: cute.CopyAtom,
        mB1_nkl: cute.Tensor,
        tma_atom_b2: cute.CopyAtom,
        mB2_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        tiled_mma: cute.TiledMma,
        cta_layout_mnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: utils.PersistentTileSchedulerParams,
    ):
        """Device kernel: fused backward dx — double GEMM, single accumulator.

        ``acc_dx = dgate @ w_gate + dvalue @ w_value`` accumulates through the
        K-loop (K=I). Epilogue performs type conversion and TMA store — no
        activation computation.
        """

        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a1)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a2)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b1)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b2)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_c)

        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)

        a_mcast_mask = cute.make_layout_image_mask(cta_layout_mnk, cluster_coord_mnk, mode=1)
        b_mcast_mask = cute.make_layout_image_mask(cta_layout_mnk, cluster_coord_mnk, mode=0)
        a_mcast_mask = a_mcast_mask if self.is_a_mcast else 0
        b_mcast_mask = b_mcast_mask if self.is_b_mcast else 0
        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        tma_copy_bytes = 2 * cute.size_in_bytes(
            self.a_dtype, a_smem_layout
        ) + 2 * cute.size_in_bytes(self.b_dtype, b_smem_layout)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()

        mainloop_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        mcast_size = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        consumer_arrive_cnt = mcast_size * self.num_mma_warps
        mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, consumer_arrive_cnt
        )

        cta_layout_vmnk = cute.make_layout((1, *cta_layout_mnk.shape))
        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.ab_stage,
            producer_group=mainloop_pipeline_producer_group,
            consumer_group=mainloop_pipeline_consumer_group,
            tx_count=tma_copy_bytes,
            barrier_storage=mainloop_pipeline_array_ptr,
            cta_layout_vmnk=cta_layout_vmnk,
        )

        if cute.size(self.cluster_shape_mnk) > 1:
            cute.arch.cluster_arrive_relaxed()

        sA_dgate = storage.sA_dgate.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sA_dvalue = storage.sA_dvalue.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB_w_gate = storage.sB_w_gate.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sB_w_value = storage.sB_w_value.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sC = storage.sC.get_tensor(
            epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner
        )

        gA1_mkl = cute.local_tile(
            mA1_mkl,
            cute.slice_(self.tile_shape_mnk, (None, 0, None)),
            (None, None, None),
        )
        gA2_mkl = cute.local_tile(
            mA2_mkl,
            cute.slice_(self.tile_shape_mnk, (None, 0, None)),
            (None, None, None),
        )
        gB1_nkl = cute.local_tile(
            mB1_nkl,
            cute.slice_(self.tile_shape_mnk, (0, None, None)),
            (None, None, None),
        )
        gB2_nkl = cute.local_tile(
            mB2_nkl,
            cute.slice_(self.tile_shape_mnk, (0, None, None)),
            (None, None, None),
        )
        gC_mnl = cute.local_tile(
            mC_mnl,
            cute.slice_(self.tile_shape_mnk, (None, None, 0)),
            (None, None, None),
        )

        a_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)
        a_cta_crd = cluster_coord_mnk[1]
        tAsA1, tAgA1 = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a1,
            a_cta_crd,
            a_cta_layout,
            cute.group_modes(sA_dgate, 0, 2),
            cute.group_modes(gA1_mkl, 0, 2),
        )
        tAsA2, tAgA2 = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a2,
            a_cta_crd,
            a_cta_layout,
            cute.group_modes(sA_dvalue, 0, 2),
            cute.group_modes(gA2_mkl, 0, 2),
        )

        b_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)
        b_cta_crd = cluster_coord_mnk[0]
        tBsB1, tBgB1 = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b1,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sB_w_gate, 0, 2),
            cute.group_modes(gB1_nkl, 0, 2),
        )
        tBsB2, tBgB2 = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b2,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sB_w_value, 0, 2),
            cute.group_modes(gB2_nkl, 0, 2),
        )

        thr_mma = tiled_mma.get_slice(tidx)

        tCsA1 = thr_mma.partition_A(sA_dgate)
        tCsA2 = thr_mma.partition_A(sA_dvalue)
        tCsB1 = thr_mma.partition_B(sB_w_gate)
        tCsB2 = thr_mma.partition_B(sB_w_value)
        tCrA1 = tiled_mma.make_fragment_A(tCsA1[None, None, None, 0])
        tCrA2 = tiled_mma.make_fragment_A(tCsA2[None, None, None, 0])
        tCrB1 = tiled_mma.make_fragment_B(tCsB1[None, None, None, 0])
        tCrB2 = tiled_mma.make_fragment_B(tCsB2[None, None, None, 0])

        tCgC = thr_mma.partition_C(gC_mnl)
        acc_shape = tCgC.shape[:3]
        acc_dx = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

        if cute.size(self.cluster_shape_mnk) > 1:
            cute.arch.cluster_wait()
        else:
            pipeline.sync(barrier_id=1)

        k_tile_cnt = cute.size(gA1_mkl, mode=[3])

        tile_sched = utils.StaticPersistentTileScheduler.create(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()

        mainloop_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.ab_stage
        )
        mainloop_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )

        if warp_idx < self.num_mma_warps:
            cute.arch.setmaxregister_increase(self.mma_register_requirement)

            num_k_blocks = cute.size(tCrA1, mode=[2])

            atom_copy_ldmatrix_A = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(self.a_layout.is_m_major_a(), 4),
                self.a_dtype,
            )
            atom_copy_ldmatrix_B = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(self.b_layout.is_n_major_b(), 4),
                self.b_dtype,
            )
            smem_tiled_copy_A = cute.make_tiled_copy_A(atom_copy_ldmatrix_A, tiled_mma)
            smem_tiled_copy_B = cute.make_tiled_copy_B(atom_copy_ldmatrix_B, tiled_mma)

            thr_copy_ldmatrix_A = smem_tiled_copy_A.get_slice(tidx)
            thr_copy_ldmatrix_B = smem_tiled_copy_B.get_slice(tidx)
            tCsA1_copy_view = thr_copy_ldmatrix_A.partition_S(sA_dgate)
            tCrA1_copy_view = thr_copy_ldmatrix_A.retile(tCrA1)
            tCsA2_copy_view = thr_copy_ldmatrix_A.partition_S(sA_dvalue)
            tCrA2_copy_view = thr_copy_ldmatrix_A.retile(tCrA2)

            tCsB1_copy_view = thr_copy_ldmatrix_B.partition_S(sB_w_gate)
            tCrB1_copy_view = thr_copy_ldmatrix_B.retile(tCrB1)
            tCsB2_copy_view = thr_copy_ldmatrix_B.partition_S(sB_w_value)
            tCrB2_copy_view = thr_copy_ldmatrix_B.retile(tCrB2)

            # Spanning store pipeline + rolling buffer slot, see the forward
            # kernel: this lets the previous tile's TMA stores stay in flight
            # while the next tile's mainloop runs.
            tma_store_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.epi_stage,
                producer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    self.num_mma_warps * self.num_threads_per_warp,
                ),
            )
            epi_buffer_slot = cutlass.Int32(0)

            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                gC_mnl_slice = gC_mnl[(None, None, *tile_coord_mnl)]
                acc_dx.fill(0.0)

                mainloop_consumer_state.reset_count()

                peek_ab_full_status = cutlass.Boolean(1)
                if mainloop_consumer_state.count < k_tile_cnt:
                    peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                        mainloop_consumer_state
                    )

                mainloop_pipeline.consumer_wait(mainloop_consumer_state, peek_ab_full_status)
                tCsA1_p = tCsA1_copy_view[None, None, None, mainloop_consumer_state.index]
                tCsA2_p = tCsA2_copy_view[None, None, None, mainloop_consumer_state.index]
                tCsB1_p = tCsB1_copy_view[None, None, None, mainloop_consumer_state.index]
                tCsB2_p = tCsB2_copy_view[None, None, None, mainloop_consumer_state.index]
                cute.copy(
                    smem_tiled_copy_A,
                    tCsA1_p[None, None, 0],
                    tCrA1_copy_view[None, None, 0],
                )
                cute.copy(
                    smem_tiled_copy_A,
                    tCsA2_p[None, None, 0],
                    tCrA2_copy_view[None, None, 0],
                )
                cute.copy(
                    smem_tiled_copy_B,
                    tCsB1_p[None, None, 0],
                    tCrB1_copy_view[None, None, 0],
                )
                cute.copy(
                    smem_tiled_copy_B,
                    tCsB2_p[None, None, 0],
                    tCrB2_copy_view[None, None, 0],
                )

                for _k_tile in range(0, k_tile_cnt - 1, 1, unroll=1):
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_block_next = 0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1

                        if k_block_idx == num_k_blocks - 1:
                            mainloop_pipeline.consumer_release(mainloop_consumer_state)
                            mainloop_consumer_state.advance()

                            peek_ab_full_status = cutlass.Boolean(1)
                            peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                                mainloop_consumer_state
                            )

                            tCsA1_p = tCsA1_copy_view[
                                None, None, None, mainloop_consumer_state.index
                            ]
                            tCsA2_p = tCsA2_copy_view[
                                None, None, None, mainloop_consumer_state.index
                            ]
                            tCsB1_p = tCsB1_copy_view[
                                None, None, None, mainloop_consumer_state.index
                            ]
                            tCsB2_p = tCsB2_copy_view[
                                None, None, None, mainloop_consumer_state.index
                            ]
                            mainloop_pipeline.consumer_wait(
                                mainloop_consumer_state, peek_ab_full_status
                            )

                        cute.copy(
                            smem_tiled_copy_A,
                            tCsA1_p[None, None, k_block_next],
                            tCrA1_copy_view[None, None, k_block_next],
                        )
                        cute.copy(
                            smem_tiled_copy_A,
                            tCsA2_p[None, None, k_block_next],
                            tCrA2_copy_view[None, None, k_block_next],
                        )
                        cute.copy(
                            smem_tiled_copy_B,
                            tCsB1_p[None, None, k_block_next],
                            tCrB1_copy_view[None, None, k_block_next],
                        )
                        cute.copy(
                            smem_tiled_copy_B,
                            tCsB2_p[None, None, k_block_next],
                            tCrB2_copy_view[None, None, k_block_next],
                        )
                        cute.gemm(
                            tiled_mma,
                            acc_dx,
                            tCrA1[None, None, k_block_idx],
                            tCrB1[None, None, k_block_idx],
                            acc_dx,
                        )
                        cute.gemm(
                            tiled_mma,
                            acc_dx,
                            tCrA2[None, None, k_block_idx],
                            tCrB2[None, None, k_block_idx],
                            acc_dx,
                        )
                for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                    k_block_next = 0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1

                    if k_block_idx == num_k_blocks - 1:
                        mainloop_pipeline.consumer_release(mainloop_consumer_state)
                        mainloop_consumer_state.advance()

                    if k_block_next > 0:
                        cute.copy(
                            smem_tiled_copy_A,
                            tCsA1_p[None, None, k_block_next],
                            tCrA1_copy_view[None, None, k_block_next],
                        )
                        cute.copy(
                            smem_tiled_copy_A,
                            tCsA2_p[None, None, k_block_next],
                            tCrA2_copy_view[None, None, k_block_next],
                        )
                        cute.copy(
                            smem_tiled_copy_B,
                            tCsB1_p[None, None, k_block_next],
                            tCrB1_copy_view[None, None, k_block_next],
                        )
                        cute.copy(
                            smem_tiled_copy_B,
                            tCsB2_p[None, None, k_block_next],
                            tCrB2_copy_view[None, None, k_block_next],
                        )
                    cute.gemm(
                        tiled_mma,
                        acc_dx,
                        tCrA1[None, None, k_block_idx],
                        tCrB1[None, None, k_block_idx],
                        acc_dx,
                    )
                    cute.gemm(
                        tiled_mma,
                        acc_dx,
                        tCrA2[None, None, k_block_idx],
                        tCrB2[None, None, k_block_idx],
                        acc_dx,
                    )

                copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
                    self.c_layout,
                    elem_ty_d=self.c_dtype,
                    elem_ty_acc=self.acc_dtype,
                )
                copy_atom_C = cute.make_copy_atom(
                    cute.nvgpu.warp.StMatrix8x8x16bOp(self.c_layout.is_m_major_c(), 4),
                    self.c_dtype,
                )
                tiled_copy_C_Atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)
                tiled_copy_r2s = cute.make_tiled_copy_S(copy_atom_r2s, tiled_copy_C_Atom)

                thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
                tRS_sD = thr_copy_r2s.partition_D(sC)
                tRS_rAcc = tiled_copy_r2s.retile(acc_dx)

                rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
                tRS_rD_layout = cute.make_layout(rD_shape[:3])
                tRS_rD = cute.make_rmem_tensor(tRS_rD_layout.shape, self.acc_dtype)
                size_tRS_rD = cute.size(tRS_rD)

                sepi_for_tma_partition = cute.group_modes(sC, 0, 2)
                tcgc_for_tma_partition = cute.zipped_divide(gC_mnl_slice, self.epi_tile)

                bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_c,
                    0,
                    cute.make_layout(1),
                    sepi_for_tma_partition,
                    tcgc_for_tma_partition,
                )

                epi_tile_num = cute.size(tcgc_for_tma_partition, mode=[1])
                epi_tile_shape = tcgc_for_tma_partition.shape[1]
                epi_tile_layout = cute.make_layout(epi_tile_shape, stride=(1, epi_tile_shape[0]))

                for epi_idx in cutlass.range_constexpr(epi_tile_num):
                    for epi_v in cutlass.range_constexpr(size_tRS_rD):
                        tRS_rD[epi_v] = tRS_rAcc[epi_idx * size_tRS_rD + epi_v]

                    tRS_rD_out = cute.make_rmem_tensor(tRS_rD_layout.shape, self.c_dtype)
                    acc_vec = tRS_rD.load()
                    tRS_rD_out.store(acc_vec.to(self.c_dtype))

                    epi_buffer = epi_buffer_slot % cute.size(tRS_sD, mode=[3])
                    epi_buffer_slot = epi_buffer_slot + 1
                    cute.copy(
                        tiled_copy_r2s,
                        tRS_rD_out,
                        tRS_sD[(None, None, None, epi_buffer)],
                    )

                    cute.arch.fence_proxy("async.shared", space="cta")
                    self.epilog_sync_barrier.arrive_and_wait()

                    gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
                    if warp_idx == 0:
                        cute.copy(
                            tma_atom_c,
                            bSG_sD[(None, epi_buffer)],
                            bSG_gD[(None, gmem_coord)],
                        )
                        tma_store_pipeline.producer_commit()
                        tma_store_pipeline.producer_acquire()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            # Drain all outstanding TMA stores once, after the last tile.
            tma_store_pipeline.producer_tail()

        elif warp_idx == self.num_mma_warps:
            cute.arch.setmaxregister_decrease(self.load_register_requirement)

            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                tAgA1_mkl = tAgA1[(None, tile_coord_mnl[0], None, tile_coord_mnl[2])]
                tAgA2_mkl = tAgA2[(None, tile_coord_mnl[0], None, tile_coord_mnl[2])]
                tBgB1_nkl = tBgB1[(None, tile_coord_mnl[1], None, tile_coord_mnl[2])]
                tBgB2_nkl = tBgB2[(None, tile_coord_mnl[1], None, tile_coord_mnl[2])]

                mainloop_producer_state.reset_count()

                for _k_tile in range(0, k_tile_cnt, 1, unroll=1):
                    mainloop_pipeline.producer_acquire(mainloop_producer_state)

                    tAgA1_k = tAgA1_mkl[(None, mainloop_producer_state.count)]
                    tAsA1_pipe = tAsA1[(None, mainloop_producer_state.index)]
                    tAgA2_k = tAgA2_mkl[(None, mainloop_producer_state.count)]
                    tAsA2_pipe = tAsA2[(None, mainloop_producer_state.index)]
                    tBgB1_k = tBgB1_nkl[(None, mainloop_producer_state.count)]
                    tBsB1_pipe = tBsB1[(None, mainloop_producer_state.index)]
                    tBgB2_k = tBgB2_nkl[(None, mainloop_producer_state.count)]
                    tBsB2_pipe = tBsB2[(None, mainloop_producer_state.index)]

                    cute.copy(
                        tma_atom_a1,
                        tAgA1_k,
                        tAsA1_pipe,
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(mainloop_producer_state),
                        mcast_mask=a_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_a2,
                        tAgA2_k,
                        tAsA2_pipe,
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(mainloop_producer_state),
                        mcast_mask=a_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b1,
                        tBgB1_k,
                        tBsB1_pipe,
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(mainloop_producer_state),
                        mcast_mask=b_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b2,
                        tBgB2_k,
                        tBsB2_pipe,
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(mainloop_producer_state),
                        mcast_mask=b_mcast_mask,
                    )
                    mainloop_pipeline.producer_commit(mainloop_producer_state)
                    mainloop_producer_state.advance()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            mainloop_pipeline.producer_tail(mainloop_producer_state)
        return

    @staticmethod
    def _compute_stages(
        tile_shape_mnk: tuple[int, int, int],
        a_dtype: type[cutlass.Numeric],
        b_dtype: type[cutlass.Numeric],
        epi_tile: tuple[int, int],
        c_dtype: type[cutlass.Numeric],
        smem_capacity: int,
        occupancy: int,
    ) -> tuple[int, int]:
        """Compute (ab_stage, epi_stage) for the backward dx kernel.

        ``ab_bytes_per_stage`` counts two A tiles (dgate + dvalue) plus two B
        tiles (w_gate + w_value), since all four operands live in smem per stage.
        """
        epi_stage = 8
        c_bytes_per_stage = cute.size(epi_tile) * c_dtype.width // 8
        epi_bytes = c_bytes_per_stage * epi_stage

        a_shape = cute.slice_(tile_shape_mnk, (None, 0, None))
        b_shape = cute.slice_(tile_shape_mnk, (0, None, None))
        ab_bytes_per_stage = (
            2 * cute.size(a_shape) * a_dtype.width // 8
            + 2 * cute.size(b_shape) * b_dtype.width // 8
        )
        mbar_helpers_bytes = 1024

        ab_stage = (
            (smem_capacity - occupancy * 1024) // occupancy - mbar_helpers_bytes - epi_bytes
        ) // ab_bytes_per_stage
        return ab_stage, epi_stage

    @staticmethod
    def _make_smem_layouts(
        tile_shape_mnk: tuple[int, int, int],
        epi_tile: tuple[int, int],
        a_dtype: type[cutlass.Numeric],
        a_layout: cute.Layout,
        b_dtype: type[cutlass.Numeric],
        b_layout: cute.Layout,
        ab_stage: int,
        c_dtype: type[cutlass.Numeric],
        c_layout: cute.Layout,
        epi_stage: int,
    ) -> tuple[cute.ComposedLayout, cute.ComposedLayout, cute.ComposedLayout]:
        a_smem_layout_staged = sm90_utils.make_smem_layout_a(
            a_layout,
            tile_shape_mnk,
            a_dtype,
            ab_stage,
        )
        b_smem_layout_staged = sm90_utils.make_smem_layout_b(
            b_layout,
            tile_shape_mnk,
            b_dtype,
            ab_stage,
        )
        epi_smem_layout_staged = sm90_utils.make_smem_layout_epi(
            c_dtype,
            c_layout,
            epi_tile,
            epi_stage,
        )
        return a_smem_layout_staged, b_smem_layout_staged, epi_smem_layout_staged

    @staticmethod
    def _compute_grid(
        c: cute.Tensor,
        tile_shape_mnk: tuple[int, int, int],
        max_active_clusters: cutlass.Constexpr,
    ) -> tuple[int, int, int]:
        c_shape = cute.slice_(tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape
        cluster_shape_mnl = (1, 1, 1)
        tile_sched_params = utils.PersistentTileSchedulerParams(num_ctas_mnl, cluster_shape_mnl)
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )
        return tile_sched_params, grid

    @staticmethod
    def _make_tma_store_atoms_and_tensors(
        tensor_c: cute.Tensor,
        epi_smem_layout_staged: cute.ComposedLayout,
        epi_tile: tuple[int, int],
    ) -> tuple[cute.CopyAtom, cute.Tensor]:
        epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
            tensor_c,
            epi_smem_layout,
            epi_tile,
        )
        return tma_atom_c, tma_tensor_c

    @staticmethod
    def _make_tma_atoms_and_tensors(
        tensor: cute.Tensor,
        smem_layout_staged: cute.ComposedLayout,
        smem_tile: tuple[int, int],
        mcast_dim: int,
    ) -> tuple[cute.CopyAtom, cute.Tensor]:
        op = (
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
            if mcast_dim == 1
            else cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp()
        )
        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        tma_atom, tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op,
            tensor,
            smem_layout,
            smem_tile,
            num_multicast=mcast_dim,
        )
        return tma_atom, tma_tensor
