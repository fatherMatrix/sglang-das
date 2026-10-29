"""DeepSeek V4 pool layouts for CP Cache LayerSplit."""

from __future__ import annotations

from dataclasses import dataclass

from sglang.srt.mem_cache.cp_cache_layer_split.utils import (
    get_global_layer_shard_range,
)


@dataclass(frozen=True)
class CpCacheLayerSplitDeepSeekV4PoolLayout:
    """Per-rank layer-buffer counts for each V4 KV sub-pool."""

    swa_layer_num: int
    c4_layer_num: int
    c128_layer_num: int
    c4_indexer_layer_num: int
    c4_state_layer_num: int
    c128_state_layer_num: int
    c4_indexer_state_layer_num: int
    c1_layer_num: int = 0
    c2_layer_num: int = 0
    c1_indexer_layer_num: int = 0
    c2_indexer_layer_num: int = 0
    c2_state_layer_num: int = 0


def build_cp_cache_layer_split_deepseek_v4_pool_layout(
    cp_rank: int,
    cp_size: int,
    start_layer: int,
    end_layer_exclusive: int,
    compression_ratios: list[int],
    kv_source_layers: list[int] | tuple[int, ...] = (),
) -> CpCacheLayerSplitDeepSeekV4PoolLayout:
    """Buffer counts for one attention-CP rank.

    ==================================================================
    LayerSplit 的分片规则：每个 CP rank 只拿【stage 内一段连续层】
    ==================================================================
    输入：本 rank 的 cp_rank / cp_size、PP stage 的 [start, end)、
          逐层压缩比、以及 kv_source 层列表。
    输出：本 rank 各家族要分配多少个 buffer（swa_layer_num / c1_layer_num /
          c2_layer_num / c4_layer_num / c128_layer_num / indexer 数等）。

    两步：
      ① get_global_layer_shard_range(cp_rank, cp_size, start, len)
           → [owned_start, owned_end)：把 stage 的 N 层切成 cp_size 段连续区间
             （cp_size=8、N=40 时每 rank 5 层，形如 rank3 = [24, 29)）
      ② 在 owned 区间内按压缩比【计数】
           → 池子按"数量"分配 buffer，**不给非 owned 层留占位**

    ★ 注意"连续"这两个字：
        和 CP 切 token 的 round-robin（t % cp_size）完全不同 —— 那边是"隔 8 取 1"，
        这边是"连续一段"。因为层是按设备切分的，连续段能让 PD 传输按
        (family, layer) 描述符顺序配对，也让 HiCache 的层映射更简单。
        代价是：读别的 rank 的层必须走 staging（广播 + 下标重映射），
        这也正是在 attention 里那两处 prefetch/wait 存在的原因。

    ★ 每个家族单独计数，是因为它们的池宽度不同（见 KV 池的"家族"注释）：
        SWA     每层一块滑窗 KV
        c1/c2   V4.1 的低压缩比家族（V4.1 只有这两个）
        c4/c128 V4 Pro 的家族
        indexer 按 rank 复制，不参与层分片（这里只算 owned 层的数量对齐 buffer）
    """
    if not 0 <= start_layer < end_layer_exclusive <= len(compression_ratios):
        raise ValueError(
            "Invalid DSV4 Cache LayerSplit stage: "
            f"start_layer={start_layer}, end_layer={end_layer_exclusive}, "
            f"compression_ratios={len(compression_ratios)}"
        )
    owned_start, owned_end = get_global_layer_shard_range(
        cp_rank,
        cp_size,
        start_layer,
        end_layer_exclusive - start_layer,
    )

    def _count(compress_ratio: int) -> int:
        return sum(
            1
            for layer_id in range(owned_start, owned_end)
            if compression_ratios[layer_id] == compress_ratio
        )

    c4_layer_num = _count(4)
    c128_layer_num = _count(128)
    source_counts = {
        ratio: sum(
            1
            for layer_id in kv_source_layers
            if owned_start <= layer_id < owned_end
            and compression_ratios[layer_id] == ratio
        )
        for ratio in (1, 2)
    }
    return CpCacheLayerSplitDeepSeekV4PoolLayout(
        swa_layer_num=owned_end - owned_start,
        c4_layer_num=c4_layer_num,
        c128_layer_num=c128_layer_num,
        c4_indexer_layer_num=c4_layer_num,
        c4_state_layer_num=c4_layer_num,
        c128_state_layer_num=c128_layer_num,
        c4_indexer_state_layer_num=c4_layer_num,
        c1_layer_num=source_counts[1],
        c2_layer_num=source_counts[2],
        c1_indexer_layer_num=source_counts[1],
        c2_indexer_layer_num=source_counts[2],
        c2_state_layer_num=source_counts[2],
    )


def build_cp_cache_layer_split_deepseek_v4_worst_case_pool_layout(
    cp_size: int,
    start_layer: int,
    end_layer_exclusive: int,
    compression_ratios: list[int],
    kv_source_layers: list[int] | tuple[int, ...] = (),
) -> CpCacheLayerSplitDeepSeekV4PoolLayout:
    """Max per-pool layer counts across all CP ranks."""
    layouts = [
        build_cp_cache_layer_split_deepseek_v4_pool_layout(
            cp_rank,
            cp_size,
            start_layer,
            end_layer_exclusive,
            compression_ratios,
            kv_source_layers,
        )
        for cp_rank in range(cp_size)
    ]
    return CpCacheLayerSplitDeepSeekV4PoolLayout(
        swa_layer_num=max(layout.swa_layer_num for layout in layouts),
        c4_layer_num=max(layout.c4_layer_num for layout in layouts),
        c128_layer_num=max(layout.c128_layer_num for layout in layouts),
        c4_indexer_layer_num=max(layout.c4_indexer_layer_num for layout in layouts),
        c4_state_layer_num=max(layout.c4_state_layer_num for layout in layouts),
        c128_state_layer_num=max(layout.c128_state_layer_num for layout in layouts),
        c4_indexer_state_layer_num=max(
            layout.c4_indexer_state_layer_num for layout in layouts
        ),
        c1_layer_num=max(layout.c1_layer_num for layout in layouts),
        c2_layer_num=max(layout.c2_layer_num for layout in layouts),
        c1_indexer_layer_num=max(layout.c1_indexer_layer_num for layout in layouts),
        c2_indexer_layer_num=max(layout.c2_indexer_layer_num for layout in layouts),
        c2_state_layer_num=max(layout.c2_state_layer_num for layout in layouts),
    )
