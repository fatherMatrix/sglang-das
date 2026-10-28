"""DeepSeek V4 token-to-KV pool with CP layer-sharded KV buffers."""

from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional

import torch
from sglang.srt.utils import get_bool_env_var, is_hcu

from sglang.srt.mem_cache.cp_cache_layer_split.broadcast import (
    BroadcastSlots,
    get_pynccl_broadcast_comm,
)
from sglang.srt.mem_cache.cp_cache_layer_split.deepseek_v4_layout import (
    build_cp_cache_layer_split_deepseek_v4_pool_layout,
)
from sglang.srt.mem_cache.cp_cache_layer_split.pool_base import (
    CpCacheLayerSplitPoolBase,
)
from sglang.srt.mem_cache.cp_cache_layer_split.staging import (
    StagingBufferManager,
    active_pages_for_indices,
    contiguous_request_page_capacity,
    remap_indices_to_staging,
    remap_page_table_to_staging,
)
from sglang.srt.mem_cache.deepseek_v4_compress_state import CompressStatePool
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    DSV4_TRANSFER_ATTENTION_STATE,
    DSV4_TRANSFER_C4_INDEXER_KV,
    DSV4_TRANSFER_C4_KV,
    DSV4_TRANSFER_C128_KV,
    DSV4_TRANSFER_C128_STATE,
    DSV4_TRANSFER_INDEXER_STATE,
    DSV4_TRANSFER_SWA_KV,
    DeepSeekV4LayerItem,
    DeepSeekV4TokenToKVPool,
    NopeFp8RopeBf16Pack,
)
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE

logger = logging.getLogger(__name__)

_USE_STATIC_CP_LAYER_SPLIT_PAGES = is_hcu() and get_bool_env_var(
    "SGLANG_HCU_OPT_LAYERSPLIT_STATIC_PAGES", default="true"
)

@dataclass
class _BatchActivePages:
    """Per-forward active pages plus remapped indices/page table."""

    selected_pages: Optional[torch.Tensor] = None
    remapped: Optional[torch.Tensor] = None
    static_selected_pages: bool = False


class CpCacheLayerSplitDeepSeekV4TokenToKVPool(
    CpCacheLayerSplitPoolBase, DeepSeekV4TokenToKVPool
):
    """DeepSeek V4 pool that reads non-owned layer KV through staging buffers.

    ======================================================================
    LayerSplit（LS）全貌 —— 读本文件前先看这段
    ======================================================================

    【动机】
    不开 LS 时，prefill CP 的每个 rank 都持有整段序列的完整 KV（压缩器把
    全序列 all-gather 后每个 rank 各写一份），8 卡就是 8 份全量副本。
    LS 在"层"这个维度再切一刀：每个 CP rank 只持久化一段连续层的 KV，
    于是单 rank 的持久 KV 降到约 1/8。upstream 对 V4 明确不这么做
    （它选择每 rank 复制，换取"无需层所有权协议"），所以本包是 HCU 私有。

    【拥有关系】
    get_layer_shard_range(rank, cp_size, total_layers) 给出连续区间
    [owned_start, owned_end)；只有落在区间内的层才会在本 rank 分配 buffer。
    因此池里各 family 的 buffer 数量是"owned 层数"，不是全 stage 层数。
    >>> 这是与全局代码（HiCache 层映射、容量估算、draft 打包）冲突的根源：
        凡假设"每层都有 buffer"的地方都会失效。

    【读非 owned 层 = 两轮集合通信 + 一次本地重映射】
      第 1 轮（staging.py: active_pages_for_indices）
        各 rank 把"本次 query 要读哪些页"写成定长掩码 → CP 内 all-reduce
        求并集 → 取出并集页号（静态选页决定这一步的输出长度是否已知）。
      第 2 轮（本文件 _compact_broadcast_for_read）
        owner 把并集页的 KV 行紧凑拷进 staging → pynccl 广播给所有 rank。
      重映射（remap_*_for_read）
        全局页号 → staging 下标，未选中置 -1，kernel 跳过。
    这两轮可以整段丢到旁路流（overlap），也可以贴主流串行执行。

    【PD 传输】
    requires_descriptor_matched_transfer = True 让 kv_args.cp_cache_layer_split
    为真，从而 P 侧发送时带 (family, global_layer) 描述符，D 侧按名字配对，
    而不是按"列表位置"配对（LS 下 P 每 rank 的 buffer 列表只是子集，
    位置配对必然错位）。

    【与 HiCache 的冲突面】
      ① 层集合：HiCache 的层映射默认恒等（每层都有 buffer）→ 对 LS 池必错
      ② 页集合：HiCache 回载按空闲链分配，会打散槽位；静态选页的上界依赖
                "读集合在逻辑 token 空间里是一段"，与槽位是否连号无关，
                但 staging 的读状态会引用"即将被 H2D 覆写的页"
      ③ 读状态：H2D 回载覆写 owner 持久 buffer → 上一轮的 staging 重映射 /
                pending 广播失效 → 需要 invalidate_staging_for_hicache()
      详见 mem_cache/cp_cache_layer_split/HICACHE_IMPLEMENTATION.md（devel 分支）
      以及 anno 文档中的机制拆解。
    ======================================================================
    """

    requires_descriptor_matched_transfer = True

    def __init__(
        self,
        *,
        cp_rank: int,
        cp_size: int,
        **kwargs,
    ):
        self._clear_swa_read_state()
        self._clear_extra_read_state()
        self._clear_indexer_read_state()
        self._extra_full_read_key: Optional[tuple[int, int]] = None

        # SWA, C128, and page-table-driven C4/indexer selections are constant
        # within one forward. Sparse C4 indices remain layer-specific.
        self._batch_active_pages: dict[str, _BatchActivePages] = {
            "swa": _BatchActivePages(),
            "indexer": _BatchActivePages(),
            "extra_c4_page_table": _BatchActivePages(),
            "extra_c128": _BatchActivePages(),
            "extra_c1": _BatchActivePages(),
            "extra_c2": _BatchActivePages(),
        }

        pp_start = kwargs.get("start_layer")
        if pp_start is None:
            pp_start = 0
        pp_end = kwargs.get("end_layer")
        layer_num = kwargs["layer_num"]
        if pp_end is None:
            pp_end = pp_start + layer_num

        layout = build_cp_cache_layer_split_deepseek_v4_pool_layout(
            cp_rank,
            cp_size,
            pp_start,
            pp_end,
            kwargs["compression_ratios"],
            kwargs.get("kv_source_layers", ()),
        )
        if layout.swa_layer_num == 0:
            logger.warning(
                "CpCacheLayerSplitDeepSeekV4TokenToKVPool: cp_rank=%s owns no SWA layers "
                "in PP slice [%s, %s)",
                cp_rank,
                pp_start,
                pp_end,
            )

        self._staging = StagingBufferManager()
        self._broadcast_slots = BroadcastSlots(("swa", "extra", "indexer"))
        self._require_prefetched_reads = False

        self._init_cp_cache_layer_split(
            cp_rank=cp_rank,
            cp_size=cp_size,
            layer_shard_start_layer=pp_start,
            layer_shard_layer_num=pp_end - pp_start,
        )
        super().__init__(
            **kwargs,
            cp_cache_layer_split_layout=layout,
        )
        self._swa_global_to_local = self._build_owned_layer_local_index_map()
        self._allocate_staging_buffers()
        self._log_layer_shard_plan()
        logger.info(
            "DSV4 Cache LayerSplit buffers: SWA=%s, C4=%s, C128=%s, "
            "C4_INDEXER=%s, C4_STATE=%s, C128_STATE=%s, C4_INDEXER_STATE=%s",
            layout.swa_layer_num,
            layout.c4_layer_num,
            layout.c128_layer_num,
            layout.c4_indexer_layer_num,
            layout.c4_state_layer_num,
            layout.c128_state_layer_num,
            layout.c4_indexer_state_layer_num,
        )
        self._rebuild_compressed_layer_mapping_for_cp()

    def _init_paged_compress_states(self, enable_memory_saver: bool):
        if set(self.compression_ratios) <= {0, 1, 2}:
            # The ratio-2 pending-pair state is needed on every CP rank because
            # each rank builds the replicated FP4 indexer cache from global rows.
            return super()._init_paged_compress_states(enable_memory_saver)
        total_L = len(self.compression_ratios)
        self.compress_state_pools: list[Optional[CompressStatePool]] = [None] * total_L
        self.indexer_compress_state_pools: list[Optional[CompressStatePool]] = [
            None
        ] * total_L

        for idx in range(self._stage_start, self._stage_end):
            ratio = self.compression_ratios[idx]
            if ratio == 0:
                continue

            if self._owns_attention_state_layer_id(idx):
                self.compress_state_pools[idx] = self._make_attn_state_pool(
                    ratio, enable_memory_saver
                )

            if ratio == 4 and self._owns_indexer_state_layer_id(idx):
                self.indexer_compress_state_pools[idx] = self._make_indexer_state_pool(
                    ratio, enable_memory_saver
                )

    def get_kv_layer_ids(self) -> list[int]:
        return [layer_id for _, layer_id in self.get_kv_transfer_layout()]

    def get_index_k_int8_packed_buffer(self, layer_id: int):
        # HCU consumes the same raw packed pages as the FP8 accessor. On a
        # non-owner these must come from the prefetched, compact staging pool.
        return self.get_index_k_with_scale_buffer(layer_id)

    def get_kv_transfer_layout(self) -> list:
        """Descriptors parallel to this rank's sharded V4 KV buffer list."""
        layout: list = []
        stage_layers = range(self._stage_start, self._stage_end)

        layout.extend(
            (DSV4_TRANSFER_C4_KV, layer_id)
            for layer_id in stage_layers
            if self.compression_ratios[layer_id] == 4
            and self._owns_c4_kv_layer_id(layer_id)
        )
        layout.extend(
            (DSV4_TRANSFER_C4_INDEXER_KV, layer_id)
            for layer_id in stage_layers
            if self.compression_ratios[layer_id] == 4
            and self._owns_indexer_kv_layer_id(layer_id)
        )
        layout.extend(
            (DSV4_TRANSFER_C128_KV, layer_id)
            for layer_id in stage_layers
            if self.compression_ratios[layer_id] == 128
            and self._owns_c128_kv_layer_id(layer_id)
        )
        for ratio in (1, 2):
            sources = self.sources_by_ratio.get(ratio, ())
            layout.extend(
                (f"c{ratio}_kv", source)
                for source in sources
                if self._is_layer_owned(source)
            )
            # FP4 indexer keys stay replicated so each CP rank can run local
            # TopK. The PD sender may transfer identical keys from each rank.
            indexer_pool = self.index_pools.get(ratio)
            if indexer_pool is not None:
                layout.extend((f"c{ratio}_indexer", source) for source in sources)
                if indexer_pool.index_k_with_scale_buffer is None:
                    layout.extend(
                        (f"c{ratio}_indexer_scale", source) for source in sources
                    )
        return layout

    def get_state_transfer_layout(self) -> list:
        """Descriptors parallel to this rank's sharded V4 state buffer list."""
        layout: list = []
        swa_layers = sorted(
            self._swa_global_to_local,
            key=lambda layer_id: self._swa_global_to_local[layer_id],
        )
        swa_layers = swa_layers[: len(self.swa_kv_pool.kv_buffer)]
        layout.extend((DSV4_TRANSFER_SWA_KV, layer_id) for layer_id in swa_layers)

        layout.extend(
            (DSV4_TRANSFER_ATTENTION_STATE, layer_id)
            for layer_id in range(self._stage_start, self._stage_end)
            if self.compression_ratios[layer_id] == 4
            and self._owns_attention_state_layer_id(layer_id)
        )
        layout.extend(
            (DSV4_TRANSFER_INDEXER_STATE, layer_id)
            for layer_id in range(self._stage_start, self._stage_end)
            if self.compression_ratios[layer_id] == 4
            and self._owns_indexer_state_layer_id(layer_id)
        )
        return layout

    def get_c128_state_transfer_layout(self) -> list:
        """Descriptors parallel to this rank's C128 state buffer list."""
        layout = [
            (DSV4_TRANSFER_C128_STATE, layer_id)
            for layer_id in range(self._stage_start, self._stage_end)
            if self.compression_ratios[layer_id] == 128
            and self._owns_attention_state_layer_id(layer_id)
        ]
        layout.extend(
            ("c2_pair_state", source)
            for source in self.sources_by_ratio.get(2, ())
        )
        return layout

    def get_hicache_host_layer_mapping(self) -> dict[str, dict[int, int]]:
        """Per-family ``{pp_local_layer_id -> local_buffer_index}`` for HiCache."""
        pp_start = self._stage_start

        swa_layers_sorted = sorted(
            self._swa_global_to_local.items(), key=lambda kv: kv[1]
        )
        swa_layers_kept = swa_layers_sorted[: len(self.swa_kv_pool.kv_buffer)]
        swa_map: dict[int, int] = {
            gid - pp_start: local_id for gid, local_id in swa_layers_kept
        }

        c4_kv_map: dict[int, int] = {}
        c128_kv_map: dict[int, int] = {}
        for gidx in range(pp_start, self._stage_end):
            item = self.layer_mapping[gidx]
            if item is None or item.compress_layer_id is None:
                continue
            pp_local = gidx - pp_start
            if item.compress_ratio == 4 and self._owns_c4_kv_layer_id(gidx):
                c4_kv_map[pp_local] = item.compress_layer_id
            elif item.compress_ratio == 128 and self._owns_c128_kv_layer_id(gidx):
                c128_kv_map[pp_local] = item.compress_layer_id

        c4_indexer_map: dict[int, int] = {}
        c4_indexer_local = 0
        for gidx in range(pp_start, self._stage_end):
            if self.compression_ratios[gidx] != 4:
                continue
            if not self._owns_indexer_kv_layer_id(gidx):
                continue
            c4_indexer_map[gidx - pp_start] = c4_indexer_local
            c4_indexer_local += 1

        c4_state_map: dict[int, int] = {}
        c128_state_map: dict[int, int] = {}
        c4_indexer_state_map: dict[int, int] = {}
        c4s_local = c128s_local = c4is_local = 0
        for gidx in range(pp_start, self._stage_end):
            ratio = self.compression_ratios[gidx]
            pp_local = gidx - pp_start
            if ratio == 4 and self._owns_attention_state_layer_id(gidx):
                c4_state_map[pp_local] = c4s_local
                c4s_local += 1
            elif ratio == 128 and self._owns_attention_state_layer_id(gidx):
                c128_state_map[pp_local] = c128s_local
                c128s_local += 1
            if ratio == 4 and self._owns_indexer_state_layer_id(gidx):
                c4_indexer_state_map[pp_local] = c4is_local
                c4is_local += 1

        return {
            "swa": swa_map,
            "c4_kv": c4_kv_map,
            "c128_kv": c128_kv_map,
            "c4_indexer": c4_indexer_map,
            "c4_state": c4_state_map,
            "c128_state": c128_state_map,
            "c4_indexer_state": c4_indexer_state_map,
        }

    def _owns_swa_layer_id(self, layer_id: int) -> bool:
        return self._is_layer_owned(layer_id)

    def should_skip_swa_write(self, layer_id: int) -> bool:
        return not self._owns_swa_layer_id(layer_id)

    def _owns_c4_kv_layer_id(self, layer_id: int) -> bool:
        return self._is_layer_owned(layer_id)

    def _owns_c128_kv_layer_id(self, layer_id: int) -> bool:
        return self._is_layer_owned(layer_id)

    def _owns_core_layer_id(self, layer_id: int) -> bool:
        ratio = self.compression_ratios[layer_id]
        if ratio in (1, 2):
            return self._is_layer_owned(self.source_layer_of(layer_id))
        return ratio in (4, 128) and self._is_layer_owned(layer_id)

    def _owns_extra_key_layer_id(self, layer_id: int) -> bool:
        return self._owns_core_layer_id(layer_id)

    def _owns_indexer_kv_layer_id(self, layer_id: int) -> bool:
        if self.compression_ratios[layer_id] in (1, 2):
            return True
        return self._is_layer_owned(layer_id)

    def _owns_attention_state_layer_id(self, layer_id: int) -> bool:
        return self._owns_core_layer_id(layer_id)

    def _owns_indexer_state_layer_id(self, layer_id: int) -> bool:
        return self._owns_indexer_kv_layer_id(layer_id)

    def should_skip_core_compressor_write(self, layer_id: int) -> bool:
        if self.compression_ratios[layer_id] in (1, 2):
            return False
        return not self._is_layer_owned(layer_id)

    def should_skip_indexer_compressor_write(self, layer_id: int) -> bool:
        if self.compression_ratios[layer_id] in (1, 2):
            return False
        return not self._is_layer_owned(layer_id)

    def should_prefetch_extra_from_page_table(self, layer_id: int) -> bool:
        item = self.layer_mapping[layer_id]
        return item is not None and item.compress_ratio == 4

    @staticmethod
    def _pool_num_pages(pool) -> int:
        if hasattr(pool, "global_page_size"):
            return (
                pool.size + pool.global_page_size + pool.page_size - 1
            ) // pool.page_size
        return (pool.size + pool.page_size + 1) // pool.page_size

    def _active_pages_for_indices(
        self,
        indices: torch.Tensor,
        page_size: int,
        max_pages: int,
        static_size: Optional[int] = None,
    ) -> torch.Tensor:
        return active_pages_for_indices(
            indices, page_size, max_pages, get_pynccl_broadcast_comm(), static_size
        )

    def _compact_broadcast_for_read(
        self,
        layer_id: int,
        indices: torch.Tensor,
        source: Optional[torch.Tensor],
        staging: torch.Tensor,
        selected_pages: torch.Tensor,
        page_size: int,
        max_pages: int,
        broadcast_kind: str,
        remap_indices: bool = True,
        selected_pages_static: bool = False,
        in_broadcast_slot: bool = False,
    ) -> torch.Tensor:
        """Broadcast owner's compact active pages and optionally remap indices.

        第 2 轮集合通信：owner 紧凑拷贝 + 全员广播。

        参数语义：
          source          —— owner 的 KV buffer；非 owner 传 None
          staging         —— 复用的临时缓冲（按整池最坏情况预分配）
          selected_pages  —— 第 1 轮产出的页号；静态选页时尾部含 -1 填充槽
          in_broadcast_slot —— True 表示调用方已持有旁路流槽位（异步路径），
                              本函数就不再自己 start 一条流

        关于下面的容量检查：staging 是按"整池页数"分配的，而并集掩码最多
        产生 max_pages 个非零项，所以该检查实际上到不了；真正保证不越界的
        是第 1 轮的容量上界（staging.py: contiguous_request_page_capacity）。
        """
        if selected_pages.numel() > staging.shape[0]:
            raise RuntimeError(
                "CP Cache LayerSplit staging buffer is smaller than active page set: "
                f"active_pages={selected_pages.numel()}, capacity={staging.shape[0]}"
            )

        if broadcast_kind == "extra":
            # All Extra ratios alias the same staging storage.
            self._extra_full_read_key = None
        if broadcast_kind == "extra" and not in_broadcast_slot:
            # A zero-query rank can skip attention and leave the previous extra
            # broadcast active. Drain it before the owner's compact copy writes
            # the shared C4/C128 storage, not only when start() launches NCCL.
            self._broadcast_slots.clear("extra")

        active_pages = selected_pages.numel()
        broadcast_pages = max(1, active_pages)
        owner_layer_id = (
            self.source_layer_of(layer_id)
            if broadcast_kind == "extra"
            and self.compression_ratios[layer_id] in (1, 2)
            else layer_id
        )
        owner_cp = self._get_layer_owner_rank(owner_layer_id)
        # 只有 owner 有真实数据（source 非 None）；其余 rank 跳过拷贝直接参与广播。
        if self.cp_rank == owner_cp:
            assert source is not None
            if active_pages > 0:
                copy_pages = selected_pages
                if selected_pages_static:
                    # 静态选页的填充槽是 -1，clamp 成 0 即重复拷 page 0。
                    # 这些槽位不会被任何 query 索引到（remap 已把它们指向
                    # 越界哨兵），所以拷什么内容都无所谓。
                    copy_pages = selected_pages.clamp_min(0)
                # Pack directly into staging: advanced indexing would allocate
                # a full payload and then copy it again before the broadcast.
                torch.index_select(
                    source, 0, copy_pages.to(torch.long), out=staging[:active_pages]
                )
            else:
                staging[:1].copy_(source[:1])

        pynccl_comm = get_pynccl_broadcast_comm()
        if in_broadcast_slot:
            # The caller owns the slot, including page selection and remapping.
            with pynccl_comm.change_state(enable=True):
                pynccl_comm.broadcast(staging[:broadcast_pages], owner_cp)
        else:
            self._broadcast_slots.start(
                broadcast_kind,
                layer_id,
                staging[:broadcast_pages],
                owner_cp,
                pynccl_comm,
            )
        if not remap_indices:
            return indices
        return remap_indices_to_staging(indices, selected_pages, page_size, max_pages)

    def _extra_family_name(self, compress_ratio: int) -> str:
        if compress_ratio in (1, 2):
            return f"extra_c{compress_ratio}"
        if compress_ratio == 4:
            return "extra_c4"
        if compress_ratio == 128:
            return "extra_c128"
        raise ValueError(f"unsupported extra KV ratio {compress_ratio}")

    def _indexer_page_bytes(self) -> int:
        pool = self.c4_indexer_kv_pool
        return pool.page_size * pool.get_bytes_per_token()

    def _get_staging_buffer(self, family: str) -> torch.Tensor:
        buffer = self._staging.get_existing(family)
        if buffer is None:
            raise RuntimeError(
                f"CP Cache LayerSplit staging buffer is not allocated: {family}"
            )
        return buffer

    def _allocate_staging_buffers(self) -> None:
        """Allocate the budgeted worst-case read scratch before serving."""
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ):
                self._staging.allocate(
                    "swa",
                    self._pool_num_pages(self.swa_kv_pool),
                    lambda n: self.swa_kv_pool.create_buffer(num_pages=n),
                )
                # C4 and C128 are mutually exclusive within a layer. Both use
                # the same "extra" broadcast slot; the eager layer loop consumes
                # its attention input before the next layer reuses this storage.
                # Keep their different padded page layouts as separate views.
                stage_ratios = self.compression_ratios[
                    self._stage_start : self._stage_end
                ]
                self._staging.allocate_shared(
                    {
                        self._extra_family_name(ratio): (
                            self._pool_num_pages(pool),
                            lambda n, pool=pool: pool.create_buffer(num_pages=n),
                        )
                        for ratio, pool in self.kv_pools.items()
                        if ratio in stage_ratios and pool is not None
                    }
                )
                indexer_pool = self.c4_indexer_kv_pool
                if indexer_pool is not None:
                    self._staging.allocate(
                        "indexer",
                        self._pool_num_pages(indexer_pool),
                        lambda n: torch.empty(
                            (n, self._indexer_page_bytes()),
                            dtype=indexer_pool.index_k_with_scale_buffer_dtype,
                            device=indexer_pool.device,
                        ),
                    )

    def reset_batch_active_pages(self) -> None:
        """Reset query plans; a full-history payload also covers replay tails."""
        for family, cache in self._batch_active_pages.items():
            if (
                self._extra_full_read_key is not None
                and family == self._extra_family_name(self._extra_full_read_key[0])
            ):
                cache.remapped = None
                continue
            cache.selected_pages = None
            cache.remapped = None
            cache.static_selected_pages = False

    def _clear_swa_read_state(self) -> None:
        self._swa_remapped_indices: Optional[torch.Tensor] = None
        self._swa_remapped_layer_id: Optional[int] = None

    def _clear_extra_read_state(self) -> None:
        self._extra_staging_layer_id: Optional[int] = None
        self._extra_page_table_selected_pages: Optional[torch.Tensor] = None
        self._extra_page_table_remap_layer_id: Optional[int] = None
        self._extra_page_table_page_size: Optional[int] = None
        self._extra_page_table_max_pages: Optional[int] = None
        self._extra_remapped_indices: Optional[torch.Tensor] = None
        self._extra_remapped_layer_id: Optional[int] = None

    def _clear_indexer_read_state(self) -> None:
        self._indexer_staging_layer_id: Optional[int] = None
        self._indexer_remapped_page_table: Optional[torch.Tensor] = None
        self._indexer_remapped_layer_id: Optional[int] = None

    def _build_owner_local_layer_map(self, compress_ratio: int) -> dict[int, int]:
        """Map each family layer to its local index in the owning rank's pool."""
        owner_offsets = [0] * self.cp_size
        result = {}
        for layer_id in range(self._stage_start, self._stage_end):
            if self.compression_ratios[layer_id] != compress_ratio:
                continue
            owner_cp = self._get_layer_owner_rank(layer_id)
            result[layer_id] = owner_offsets[owner_cp]
            owner_offsets[owner_cp] += 1
        return result

    def _owner_compress_layer_item(self, layer_id: int) -> DeepSeekV4LayerItem:
        item = self.layer_mapping[layer_id]
        assert item is not None
        assert item.compress_ratio in (1, 2, 4, 128), (
            f"layer {layer_id} has no compressed KV (ratio={item.compress_ratio})"
        )
        if not (self._stage_start <= layer_id < self._stage_end):
            raise RuntimeError(f"layer {layer_id} is outside the local PP stage")
        if item.compress_ratio in (1, 2):
            source = self.source_layer_of(layer_id)
            owner_local = self._low_owner_local_by_source[source]
            compress_kv_pool = self.kv_pools[item.compress_ratio]
        else:
            owner_local = (
                self._c4_owner_local_by_layer[layer_id]
                if item.compress_ratio == 4
                else self._c128_owner_local_by_layer[layer_id]
            )
            compress_kv_pool = (
                self.c4_kv_pool if item.compress_ratio == 4 else self.c128_kv_pool
            )
        return DeepSeekV4LayerItem(
            compress_ratio=item.compress_ratio,
            compress_layer_id=owner_local,
            compress_kv_pool=compress_kv_pool,
        )

    def _c4_indexer_local_layer_id(self, layer_id: int) -> int:
        try:
            return self._c4_indexer_owner_local_by_layer[layer_id]
        except KeyError:
            raise RuntimeError(f"layer {layer_id} not in C4 indexer KV set") from None

    def prefetch_swa_layer(
        self,
        layer_id: int,
        indices: torch.Tensor,
        read_indices: Optional[torch.Tensor] = None,
        num_reqs: Optional[int] = None,
        read_token_capacity: Optional[int] = None,
    ) -> None:
        """Start current-layer SWA page broadcast; caller waits before attention.

        本层 SWA 的"借用"入口。执行顺序（与 _compact_broadcast_for_read 合起来）：

          1) wait_layer_transfer(layer_id)
             —— 若本层正好有 HiCache 回载在途，先等它落地（流级事件），
                否则会读到恢复前的旧内容。
          2) _clear_extra_read_state()
             —— 清掉上一层的 extra 读状态（SWA / extra 两个 family 交替使用）。
          3) 选页（每 batch 只做一次，结果缓存在 _batch_active_pages["swa"]）
             —— 先算 static_size（静态选页的上界，可为 None），
                再调 _active_pages_for_indices 做第 1 轮集合通信 + 取页号，
                最后 remap_indices_to_staging 得到本 rank 的读下标。
                ★ 静态 vs 动态只影响第 1 轮里"取页号"的输出长度是否已知，
                  从而决定第 2 轮能否异步化。
          4) owner 侧定位自己的 buffer
             —— 非 owner 时 owner_buf 为 None（数据只在 owner 显存里）。
          5) _compact_broadcast_for_read 完成第 2 轮：拷贝 + 广播。

        注意 3) 之后同一次 forward 内后续层会复用选页结果，
        因为同一批请求的读集合在一层之内是确定的。
        """
        # 【步骤 1】等本层的 HiCache 回载落地（若有）。
        #   回载会覆写 owner 的持久 buffer，而下面第 2 轮的广播正是从那份 buffer 读，
        #   不等就会读到恢复前的内容。用流级事件等待，不阻塞主机。
        self.wait_layer_transfer(layer_id)
        self._clear_extra_read_state()
        # 【步骤 2】取本 family 的池几何：页数上限与页大小。
        #   掩码长度 = max_pages，选页结果的下标也就天然落在 [0, max_pages)。
        max_pages = self._pool_num_pages(self.swa_kv_pool)
        page_size = self.swa_kv_pool.page_size
        swa_cache = self._batch_active_pages["swa"]
        # 【步骤 3】第 1 轮集合通信：算"本批所有 rank 合起来要读哪些页"。
        #   每个 batch 只算一次，结果缓存在 _batch_active_pages["swa"] 里，
        #   同一次 forward 内后续层复用（页集合在一层之内是确定的）。
        #   其中的 static_size 就是静态选页的上界：
        #     有值 → nonzero_static，输出长度主机已知
        #     None → nonzero，长度要读回设备（一次 D2H）
        if swa_cache.selected_pages is None:
            static_size = None
            if (
                _USE_STATIC_CP_LAYER_SPLIT_PAGES
                and (read_indices is not None or read_token_capacity is not None)
                and num_reqs is not None
                and getattr(self, "request_window", None) is None
            ):
                static_size = contiguous_request_page_capacity(
                    read_indices.numel() if read_indices is not None else read_token_capacity,
                    num_reqs,
                    page_size,
                    max_pages,
                )
            swa_cache.static_selected_pages = static_size is not None
            swa_cache.selected_pages = self._active_pages_for_indices(
                read_indices if read_indices is not None else indices,
                page_size,
                max_pages,
                static_size,
            )
            swa_cache.remapped = remap_indices_to_staging(
                indices,
                swa_cache.selected_pages,
                page_size,
                max_pages,
                static_selected_pages=swa_cache.static_selected_pages,
            )
        # 【步骤 4】第 2 轮集合通信的准备：定位 owner 的源 buffer。
        #   非 owner 的 owner_buf 为 None —— 数据只存在于 owner 的显存里，
        #   所以必须靠广播（而不是各读各的）。
        selected_pages = swa_cache.selected_pages
        staging = self._get_staging_buffer("swa")
        owner_cp = self._get_layer_owner_rank(layer_id)
        owner_buf = None
        if self.cp_rank == owner_cp:
            owner_buf = self.swa_kv_pool.kv_buffer[self._swa_local_layer_id(layer_id)]
        self._compact_broadcast_for_read(
            layer_id,
            indices,
            owner_buf,
            staging,
            selected_pages,
            page_size,
            max_pages,
            broadcast_kind="swa",
            remap_indices=False,
            selected_pages_static=swa_cache.static_selected_pages,
        )
        self._swa_remapped_indices = swa_cache.remapped
        self._swa_remapped_layer_id = layer_id

    def wait_swa_prefetch(self, layer_id: int) -> None:
        """Wait for the pending SWA prefetch before attention consumes it.

        汇聚点：finish() 会让当前流等待该 layer 对应的旁路流，把 第1轮选页 +
        第2轮拷贝广播 的结果变成可见。模型层在 attention 之前显式调用它。
        （extra / indexer 的等待不是在这里，而是在 get_*_buffer 内部自动 finish。）
        """
        self._broadcast_slots.finish("swa", layer_id)

    def _prefetch_c128_key_pages(self, layer_id: int, indices: torch.Tensor) -> None:
        """Broadcast compact owner C128 KV pages and remap indices."""
        self.wait_layer_transfer(layer_id)
        item = self.layer_mapping[layer_id]
        assert item is not None
        assert item.compress_ratio == 128
        owner_item = self._owner_compress_layer_item(layer_id)
        assert owner_item.compress_kv_pool is not None
        pool = owner_item.compress_kv_pool
        max_pages = self._pool_num_pages(pool)
        page_size = pool.page_size
        c128_cache = self._batch_active_pages["extra_c128"]
        if c128_cache.selected_pages is None:
            c128_cache.selected_pages = self._active_pages_for_indices(
                indices, page_size, max_pages
            )
            c128_cache.remapped = remap_indices_to_staging(
                indices, c128_cache.selected_pages, page_size, max_pages
            )
        selected_pages = c128_cache.selected_pages
        staging = self._get_staging_buffer("extra_c128")
        owner_cp = self._get_layer_owner_rank(layer_id)
        owner_buf = None
        if self.cp_rank == owner_cp:
            owner_buf = pool.kv_buffer[owner_item.compress_layer_id]
        self._compact_broadcast_for_read(
            layer_id,
            indices,
            owner_buf,
            staging,
            selected_pages,
            page_size,
            max_pages,
            broadcast_kind="extra",
            remap_indices=False,
        )
        self._extra_remapped_indices = c128_cache.remapped
        self._extra_remapped_layer_id = layer_id
        self._extra_staging_layer_id = layer_id

    def prefetch_extra_key_layer_from_page_table(
        self, layer_id: int, page_table: torch.Tensor
    ) -> None:
        """Prefetch C4 extra KV pages selected by the batch page table."""
        self.wait_layer_transfer(layer_id)
        item = self.layer_mapping[layer_id]
        assert item is not None
        if item.compress_ratio != 4:
            return
        self._clear_extra_read_state()
        owner_item = self._owner_compress_layer_item(layer_id)
        assert owner_item.compress_kv_pool is not None
        pool = owner_item.compress_kv_pool
        max_pages = self._pool_num_pages(pool)
        page_size = pool.page_size
        c4_pt_cache = self._batch_active_pages["extra_c4_page_table"]
        if c4_pt_cache.selected_pages is None:
            c4_pt_cache.selected_pages = self._active_pages_for_indices(
                page_table, 1, max_pages
            )
        selected_pages = c4_pt_cache.selected_pages
        staging = self._get_staging_buffer("extra_c4")
        owner_cp = self._get_layer_owner_rank(layer_id)
        owner_buf = None
        if self.cp_rank == owner_cp:
            owner_buf = pool.kv_buffer[owner_item.compress_layer_id]
        self._compact_broadcast_for_read(
            layer_id,
            page_table,
            owner_buf,
            staging,
            selected_pages,
            1,
            max_pages,
            broadcast_kind="extra",
            remap_indices=False,
        )
        self._extra_staging_layer_id = layer_id
        self._extra_page_table_selected_pages = selected_pages
        self._extra_page_table_remap_layer_id = layer_id
        self._extra_page_table_page_size = page_size
        self._extra_page_table_max_pages = max_pages

    def _prefetch_index_k_pages(self, layer_id: int, page_table: torch.Tensor) -> None:
        """Broadcast compact owner C4 indexer pages for ``layer_id``."""
        self.wait_layer_transfer(layer_id)
        compress_ratio, _, _ = self.layer_mapping[layer_id]
        assert compress_ratio == 4, f"only c4 has indexer, got {compress_ratio = }"
        self._clear_indexer_read_state()
        owner_cp = self._get_layer_owner_rank(layer_id)
        owner_local = self._c4_indexer_local_layer_id(layer_id)
        pool = self.c4_indexer_kv_pool
        max_pages = self._pool_num_pages(pool)
        indexer_cache = self._batch_active_pages["indexer"]
        if indexer_cache.selected_pages is None:
            indexer_cache.selected_pages = self._active_pages_for_indices(
                page_table, 1, max_pages
            )
            indexer_cache.remapped = remap_page_table_to_staging(
                page_table, indexer_cache.selected_pages, max_pages
            )
        selected_pages = indexer_cache.selected_pages
        staging = self._get_staging_buffer("indexer")
        if self.cp_rank == owner_cp:
            owner_buf = pool.get_index_k_with_scale_buffer(owner_local)
        else:
            owner_buf = None
        self._compact_broadcast_for_read(
            layer_id,
            page_table,
            owner_buf,
            staging,
            selected_pages,
            1,
            max_pages,
            broadcast_kind="indexer",
            remap_indices=False,
        )
        self._indexer_remapped_page_table = indexer_cache.remapped
        self._indexer_remapped_layer_id = layer_id
        self._indexer_staging_layer_id = layer_id

    def prefetch_c128_key_layer(self, layer_id: int, core_metadata) -> None:
        """Prefetch current-layer C128 KV after the owner-side write."""
        item = self.layer_mapping[layer_id]
        if item is None:
            return
        if item.compress_ratio != 128:
            return
        indices = core_metadata.c128_page_indices
        if indices is None:
            raise RuntimeError(
                f"CP Cache LayerSplit missing C128 indices for layer {layer_id}"
            )
        self._prefetch_c128_key_pages(layer_id, indices)

    def prefetch_index_k_layer(self, layer_id: int, page_table: torch.Tensor) -> None:
        """Prefetch current-layer C4 indexer KV after the owner-side write."""
        item = self.layer_mapping[layer_id]
        if item is None:
            return
        if item.compress_ratio == 4:
            self._prefetch_index_k_pages(layer_id, page_table)

    # 【prefetch_low_ratio_extra_layer】C1/C2 压缩 KV 的"借用"入口。
    #   与 prefetch_swa_layer 同构（同样是两轮通信），差别有三处：
    #     1. owner 是"source 层"所属的 rank，不是本层层号的 owner ——
    #        晚层共享 L20 的记忆，所以同一份 C1/C2 会被多个 rank 借。
    #     2. 选页的输入是压缩后的 token 号（按 ratio 折算），页大小也是折算后的。
    #     3. 只有拿到静态 size 且不在图捕获中时，整段才走 async_pages
    #        提交到旁路流（可见 "only fixed-size selection can be submitted
    #        without a device-to-host shape sync" 那句注释）。
    def prefetch_low_ratio_extra_layer(
        self,
        layer_id: int,
        indices: torch.Tensor,
        read_indices: Optional[torch.Tensor] = None,
        num_reqs: Optional[int] = None,
        read_token_capacity: Optional[int] = None,
        cache_full_read: bool = False,
    ) -> None:
        """Fetch all pages read by attention or sparse-prefill dequantization."""
        ratio = self.compression_ratios[layer_id]
        assert ratio in (1, 2)
        self.wait_layer_transfer(layer_id)
        self._clear_extra_read_state()
        pool = self.kv_pools[ratio]
        assert pool is not None
        max_pages = self._pool_num_pages(pool)
        source = self.source_layer_of(layer_id)
        selected_cache = self._batch_active_pages[self._extra_family_name(ratio)]
        reuse_full_read = (
            self._extra_full_read_key == (ratio, source)
            and layer_id > source
            and not torch.cuda.is_current_stream_capturing()
        )
        if reuse_full_read:
            # A rank with no queries may not have consumed the source's fetch.
            self._broadcast_slots.clear("extra")
        static_size = None
        if (
            _USE_STATIC_CP_LAYER_SPLIT_PAGES
            and (read_indices is not None or read_token_capacity is not None)
            and num_reqs is not None
            and getattr(self, "request_window", None) is None
        ):
            static_size = contiguous_request_page_capacity(
                read_indices.numel() if read_indices is not None else read_token_capacity,
                num_reqs,
                pool.page_size,
                max_pages,
            )
        async_pages = (
            not reuse_full_read
            and static_size is not None
            and not torch.cuda.is_current_stream_capturing()
        )
        # Page-mask all-reduce shares the KV communicator. Keep it off the
        # main stream too, otherwise it serializes Q behind the SWA broadcast.
        # Dynamic nonzero still runs on the original path; only fixed-size
        # selection can be submitted without a device-to-host shape sync.
        with (
            self._broadcast_slots.launch("extra", layer_id, indices.device)
            if async_pages
            else nullcontext()
        ):
            if not reuse_full_read:
                selected_cache.selected_pages = self._active_pages_for_indices(
                    read_indices if read_indices is not None else indices,
                    pool.page_size,
                    max_pages,
                    static_size,
                )
                selected_cache.static_selected_pages = static_size is not None
                staging = self._get_staging_buffer(self._extra_family_name(ratio))
                owner_buf = None
                if self._is_layer_owned(source):
                    owner_buf = pool.kv_buffer[self._low_owner_local_by_source[source]]
                self._compact_broadcast_for_read(
                    layer_id,
                    indices,
                    owner_buf,
                    staging,
                    selected_cache.selected_pages,
                    pool.page_size,
                    max_pages,
                    broadcast_kind="extra",
                    remap_indices=False,
                    selected_pages_static=selected_cache.static_selected_pages,
                    in_broadcast_slot=async_pages,
                )
                if (
                    cache_full_read
                    and read_indices is not None
                    and ratio == 1
                    and layer_id == source
                    and async_pages
                ):
                    self._extra_full_read_key = (ratio, source)
            self._extra_staging_layer_id = layer_id
            self._extra_remapped_indices = remap_indices_to_staging(
                indices,
                selected_cache.selected_pages,
                pool.page_size,
                max_pages,
                static_selected_pages=selected_cache.static_selected_pages,
            )
            self._extra_remapped_layer_id = layer_id

    def _rebuild_compressed_layer_mapping_for_cp(self) -> None:
        """``compress_layer_id`` counts only layers owned by this CP rank."""
        if set(self.compression_ratios) <= {0, 1, 2}:
            # Base mappings remain global source indices for the replicated
            # indexer K pool. Extra KV uses this owner-local map instead.
            self._low_owner_local_by_source = {}
            for ratio in (1, 2):
                offsets = [0] * self.cp_size
                for source in self.sources_by_ratio.get(ratio, ()):
                    owner = self._get_layer_owner_rank(source)
                    self._low_owner_local_by_source[source] = offsets[owner]
                    offsets[owner] += 1
            return
        self._c4_owner_local_by_layer = self._build_owner_local_layer_map(4)
        self._c128_owner_local_by_layer = self._build_owner_local_layer_map(128)
        self._c4_indexer_owner_local_by_layer = self._build_owner_local_layer_map(4)

        c1_cnt = 0
        self.layer_mapping = [None] * len(self.compression_ratios)

        for idx in range(self._stage_start, self._stage_end):
            ratio = self.compression_ratios[idx]
            if ratio == 4:
                owns_extra = self._owns_c4_kv_layer_id(idx)
            elif ratio == 128:
                owns_extra = self._owns_c128_kv_layer_id(idx)
            else:
                owns_extra = self._owns_swa_layer_id(idx)

            if not owns_extra:
                self.layer_mapping[idx] = DeepSeekV4LayerItem(
                    compress_ratio=ratio,
                    compress_layer_id=None,
                    compress_kv_pool=None,
                )
                continue

            if ratio == 0:
                self.layer_mapping[idx] = DeepSeekV4LayerItem(
                    compress_ratio=0,
                    compress_layer_id=c1_cnt,
                )
                c1_cnt += 1
            elif ratio == 4:
                self.layer_mapping[idx] = DeepSeekV4LayerItem(
                    compress_ratio=4,
                    compress_layer_id=self._c4_owner_local_by_layer[idx],
                    compress_kv_pool=self.c4_kv_pool,
                )
            elif ratio == 128:
                self.layer_mapping[idx] = DeepSeekV4LayerItem(
                    compress_ratio=128,
                    compress_layer_id=self._c128_owner_local_by_layer[idx],
                    compress_kv_pool=self.c128_kv_pool,
                )
            else:
                raise ValueError(f"Unsupported compression ratio: {ratio}")

    def _swa_local_layer_id(self, layer_id: int) -> int:
        if layer_id not in self._swa_global_to_local:
            raise RuntimeError(
                f"cp_rank={self.cp_rank} does not own SWA for global layer {layer_id}"
            )
        return self._swa_global_to_local[layer_id]

    def clear_staging_remap_for_read(self) -> None:
        for kind in self._broadcast_slots.kinds():
            self._broadcast_slots.clear(kind)
        self._extra_full_read_key = None
        self._clear_swa_read_state()
        self._clear_extra_read_state()
        self._clear_indexer_read_state()
        self.reset_batch_active_pages()

    def prepare_forward(self, *, require_prefetched_reads: bool) -> None:
        """Reset read state and select strict CP-prefill read semantics."""
        self.clear_staging_remap_for_read()
        self._require_prefetched_reads = require_prefetched_reads

    def _raise_for_missing_non_owned_prefetch(self, family: str, layer_id: int) -> None:
        if self._require_prefetched_reads:
            raise RuntimeError(
                "CP Cache LayerSplit attempted to read a non-owned "
                f"{family} layer without prefetch: layer_id={layer_id}, "
                f"cp_rank={self.cp_rank}"
            )

    def get_swa_key_buffer(self, layer_id: int) -> torch.Tensor:
        if self._swa_remapped_layer_id == layer_id:
            return self._get_staging_buffer("swa")
        if self._owns_swa_layer_id(layer_id):
            return super().get_swa_key_buffer(layer_id)
        self._raise_for_missing_non_owned_prefetch("SWA", layer_id)
        return self._get_staging_buffer("swa")

    def get_swa_key_buffer_radix(self, layer_id: int) -> torch.Tensor:
        # LS 的"读路径分叉点"，attention 取 K buffer 时进这里：
        #   · 本层已 prefetch 过（_swa_remapped_layer_id 命中）→ 读 staging
        #     （内容是本 rank 拥有该层的那个 peer 广播过来的）
        #   · 本 rank 就是 owner            → 读自己的持久 buffer
        #   · 既不是 owner 又没 prefetch     → 直接抛错。
        #     ★ 这条 assert 说明 第1轮+第2轮 对 non-owned 层是强制的，
        #       静态选页只是决定"怎么取页号"，不是可选优化。
        if self._swa_remapped_layer_id == layer_id:
            return self._get_staging_buffer("swa")
        if self._owns_swa_layer_id(layer_id):
            return super().get_swa_key_buffer_radix(layer_id)
        self._raise_for_missing_non_owned_prefetch("SWA", layer_id)
        return self._get_staging_buffer("swa")

    def remap_swa_indices_for_read(
        self, layer_id: int, indices: torch.Tensor
    ) -> torch.Tensor:
        if self._swa_remapped_layer_id == layer_id:
            assert self._swa_remapped_indices is not None
            return self._swa_remapped_indices
        return indices

    def remap_flat_token_ids_for_read(self, layer_id: int, indices, *, family: str):
        """Map HCU sparse-prefill dequantization IDs into compact read pages.

        These IDs differ in shape from the per-query attention indices cached
        above, so they must be remapped independently without replacing them.
        """
        if family == "swa":
            assert self._swa_remapped_layer_id == layer_id
            pool = self.swa_kv_pool
            selected_cache = self._batch_active_pages["swa"]
        else:
            assert self._extra_staging_layer_id == layer_id
            ratio = self.compression_ratios[layer_id]
            pool = self.kv_pools[ratio]
            key = (
                "extra_c4_page_table" if ratio == 4 else self._extra_family_name(ratio)
            )
            selected_cache = self._batch_active_pages[key]
        selected = selected_cache.selected_pages
        assert selected is not None
        return remap_indices_to_staging(
            indices,
            selected,
            pool.page_size,
            self._pool_num_pages(pool),
            static_selected_pages=selected_cache.static_selected_pages,
        )

    def set_swa_key_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_nope_fp8_rope_bf16_pack: NopeFp8RopeBf16Pack,
    ) -> None:
        if not self._owns_swa_layer_id(layer_id):
            return
        super().set_swa_key_buffer(layer_id, loc, cache_nope_fp8_rope_bf16_pack)

    def set_swa_key_buffer_radix(
        self,
        layer_id: int,
        swa_loc: torch.Tensor,
        cache_nope_fp8_rope_bf16_pack: NopeFp8RopeBf16Pack,
    ) -> None:
        if not self._owns_swa_layer_id(layer_id):
            return
        super().set_swa_key_buffer_radix(
            layer_id, swa_loc, cache_nope_fp8_rope_bf16_pack
        )

    def set_swa_key_buffer_radix_fused(
        self,
        layer_id: int,
        swa_loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        if not self._owns_swa_layer_id(layer_id):
            return
        return super().set_swa_key_buffer_radix_fused(layer_id, swa_loc, cache_k)

    def set_swa_key_buffer_radix_lightop_fused(
        self,
        layer_id: int,
        swa_loc: torch.Tensor,
        cache_k: torch.Tensor,
        eps: float = 1e-8,
    ) -> None:
        if not self._owns_swa_layer_id(layer_id):
            return
        return super().set_swa_key_buffer_radix_lightop_fused(
            layer_id, swa_loc, cache_k, eps
        )

    def set_swa_key_buffer_radix_fused_norm_rope(
        self,
        layer_id: int,
        swa_loc: torch.Tensor,
        kv: torch.Tensor,
        kv_weight: torch.Tensor,
        eps: float,
        freqs_cis: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        if not self._owns_swa_layer_id(layer_id):
            return
        return super().set_swa_key_buffer_radix_fused_norm_rope(
            layer_id,
            swa_loc,
            kv,
            kv_weight,
            eps,
            freqs_cis,
            positions,
        )

    def get_extra_key_buffer(self, layer_id: int) -> torch.Tensor | None:
        item = self.layer_mapping[layer_id]
        assert item is not None
        if item.compress_ratio not in (1, 2, 4, 128):
            return None
        if self._extra_staging_layer_id == layer_id:
            family = self._extra_family_name(item.compress_ratio)
            self._broadcast_slots.finish("extra", layer_id)
            return self._get_staging_buffer(family)
        if self._owns_extra_key_layer_id(layer_id):
            return super().get_extra_key_buffer(layer_id)
        self._raise_for_missing_non_owned_prefetch(
            self._extra_family_name(item.compress_ratio), layer_id
        )
        return self._get_staging_buffer(self._extra_family_name(item.compress_ratio))

    def get_extra_key_buffer_for_write(self, layer_id: int) -> torch.Tensor | None:
        self._extra_full_read_key = None
        # Read staging is compact and cannot be addressed with global write
        # slots. Non-owners still compute latents and replicated index-K/state.
        if not self._owns_extra_key_layer_id(layer_id):
            return None
        self.wait_layer_transfer(layer_id)
        if self.compression_ratios[layer_id] in (1, 2):
            item = self._owner_compress_layer_item(layer_id)
            return item.compress_kv_pool.get_key_buffer(item.compress_layer_id)
        return super().get_extra_key_buffer(layer_id)

    def get_extra_key_page_size(self, layer_id: int) -> int:
        # HCU sparse prefill asks for geometry on every rank, including ranks
        # with no persistent buffer for this layer. Staging uses the same ABI.
        ratio = self.compression_ratios[layer_id]
        if ratio in (1, 2):
            return self.kv_pools[ratio].page_size
        if ratio == 4:
            return self.c4_kv_pool.page_size
        if ratio == 128:
            return self.c128_kv_pool.page_size
        raise ValueError(f"Layer {layer_id} has no compressed KV cache")

    def remap_extra_indices_for_read(
        self, layer_id: int, indices: torch.Tensor
    ) -> torch.Tensor:
        if self._extra_remapped_layer_id == layer_id:
            assert self._extra_remapped_indices is not None
            return self._extra_remapped_indices
        if self._extra_page_table_remap_layer_id == layer_id:
            assert self._extra_page_table_selected_pages is not None
            assert self._extra_page_table_page_size is not None
            assert self._extra_page_table_max_pages is not None
            return remap_indices_to_staging(
                indices,
                self._extra_page_table_selected_pages,
                self._extra_page_table_page_size,
                self._extra_page_table_max_pages,
            )
        return indices

    def set_extra_key_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_nope_fp8_rope_bf16_pack: NopeFp8RopeBf16Pack,
    ) -> None:
        self._extra_full_read_key = None
        if not self._owns_extra_key_layer_id(layer_id):
            return
        if self.compression_ratios[layer_id] in (1, 2):
            item = self._owner_compress_layer_item(layer_id)
            return item.compress_kv_pool.set_key_buffer(
                item.compress_layer_id, loc, cache_nope_fp8_rope_bf16_pack
            )
        super().set_extra_key_buffer(layer_id, loc, cache_nope_fp8_rope_bf16_pack)

    def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        if self.compression_ratios[layer_id] in (1, 2):
            return super().get_index_k_with_scale_buffer(layer_id)
        if self._indexer_staging_layer_id == layer_id:
            self._broadcast_slots.finish("indexer", layer_id)
            return self._get_staging_buffer("indexer")
        if self._owns_indexer_kv_layer_id(layer_id):
            return self.c4_indexer_kv_pool.get_index_k_with_scale_buffer(
                self._c4_indexer_local_layer_id(layer_id)
            )
        self._raise_for_missing_non_owned_prefetch("C4 indexer", layer_id)
        return self._get_staging_buffer("indexer")

    def remap_indexer_page_table_for_read(
        self, layer_id: int, page_table: torch.Tensor
    ) -> torch.Tensor:
        if self._indexer_remapped_layer_id == layer_id:
            assert self._indexer_remapped_page_table is not None
            return self._indexer_remapped_page_table
        return page_table

    def set_index_k_scale_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
        index_k_scale: torch.Tensor,
    ) -> None:
        if self.compression_ratios[layer_id] in (1, 2):
            return super().set_index_k_scale_buffer(
                layer_id, loc, index_k, index_k_scale
            )
        if not self._owns_indexer_kv_layer_id(layer_id):
            return
        self.c4_indexer_kv_pool.set_index_k_scale_buffer(
            self._c4_indexer_local_layer_id(layer_id), loc, index_k, index_k_scale
        )

    def set_extra_key_buffer_fused(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        freqs_cis: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> None:
        self._extra_full_read_key = None
        if not self._owns_extra_key_layer_id(layer_id):
            return
        if self.compression_ratios[layer_id] in (1, 2):
            item = self._owner_compress_layer_item(layer_id)
            return item.compress_kv_pool.set_key_buffer_fused(
                item.compress_layer_id, loc, cache_k, valid_mask, freqs_cis
            )
        super().set_extra_key_buffer_fused(
            layer_id, loc, cache_k, freqs_cis=freqs_cis, valid_mask=valid_mask
        )

    def set_index_k_fused(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        if self.compression_ratios[layer_id] in (1, 2):
            return super().set_index_k_fused(layer_id, loc, cache_k)
        if not self._owns_indexer_kv_layer_id(layer_id):
            return
        self.c4_indexer_kv_pool.set_index_fused(
            self._c4_indexer_local_layer_id(layer_id), loc, cache_k
        )
