"""DeepSeek V4.1 wrappers around the LayerSplit pool API."""

from sglang.srt.environ import envs
from sglang.srt.mem_cache.cp_cache_layer_split.deepseek_v4_pool import (
    CpCacheLayerSplitDeepSeekV4TokenToKVPool,
)
from sglang.srt.model_executor.forward_context import get_attn_backend


def is_cp_cache_layer_split_deepseek_v4_pool(pool) -> bool:
    return isinstance(pool, CpCacheLayerSplitDeepSeekV4TokenToKVPool)


def _should_sync_cp_cache_layer_split(pool, forward_batch) -> bool:
    if not is_cp_cache_layer_split_deepseek_v4_pool(pool):
        return False
    # Even short extend/warmup batches without attn_cp_metadata still read the
    # layer-sharded cache. All CP ranks participate in the same extend step.
    return (
        forward_batch is not None
        and forward_batch.forward_mode.is_context_parallel_extend()
    )


def maybe_prepare_cp_cache_layer_split_forward(pool, forward_batch) -> None:
    if is_cp_cache_layer_split_deepseek_v4_pool(pool):
        pool.prepare_forward(
            require_prefetched_reads=forward_batch.forward_mode.is_context_parallel_extend()
        )


def _get_core_attn_metadata():
    attn_backend = get_attn_backend()
    metadata = getattr(attn_backend, "forward_metadata", None)
    core_metadata = getattr(metadata, "core_attn_metadata", None)
    if core_metadata is None:
        raise RuntimeError("CP Cache LayerSplit requires DSV4 core attention metadata")
    return core_metadata


def maybe_prefetch_cp_kv_swa(pool, layer_id: int, forward_batch=None) -> None:
    """发起「借本层 SWA 窗口 KV」的预取（LayerSplit 的第 1 + 第 2 轮）。

    ★ 两重门，任一不成立就是 no-op —— 所以这个函数可以在所有路径上无脑调用：
        ① pool 必须是 LS 池（isinstance 检查）
        ② 本批必须是 CP extend
      注意 ② 故意不看 attn_cp_metadata：即使"短 extend / warmup 批"没有那份
      元数据，它仍在读层分片后的 cache，所有 CP rank 也必须同序参与。
    """
    if _should_sync_cp_cache_layer_split(pool, forward_batch):
        metadata = get_attn_backend().forward_metadata
        # sparse_prefill_cache：本 forward 的"稀疏 prefill 工作区"，
        # 里面缓存了
        #   · swa_token_ids            本批要读的 SWA 行号（精确）
        #   · ensure_compressed(ratio)  每个压缩家族的 flat_token_ids
        #
        # ★ 它是【per-forward 懒建】的，所以确实存在"还没有"的层：
        #     1) 生命周期：挂在 DSV4Metadata 上，每个 forward 由
        #        init_forward_metadata 重新造一个（__post_init__ 置 None，
        #        deepseek_v4_backend.py:1515/1549）→ 跨 forward 不保留。
        #     2) 两个可能的创建时机：
        #        a. forward 之前：prepare_prefill_shared_read_snapshot()
        #           （model_runner.py:1730 / eager_runner.py:328）
        #           —— 但只在 SGLANG_ENABLE_PREFILL_WAR_READ_DONE（默认 False）
        #              + EXTEND + dflash family + 非策略CP 时才真造，
        #              其余情况直接 return。
        #        b. 【默认路径】第一层真正跑稀疏 attention 时，在
        #           _forward_extend_sparse 内部造好并挂上（:6996-7001）。
        #     3) 而本函数的调用点在 _forward_prepare（attention 之前），
        #        所以那一层的 prefetch 必然看到 None —— 顺序决定的，不是 bug：
        #          prepare(prefetch) → attention(建 cache)
        #        在走稀疏 prefill 的平台上（NVIDIA），第一个稀疏层建的 cache
        #        只对【它之后】的层可用。
        #
        # ★★ 但 HCU 上要特别注意：b 这条路走不到。
        #    _forward_attention 里那条稀疏分支被 `if not _is_hcu:` 包着
        #    （deepseek_v4_backend.py:6765），所以 Hygon 上永远不会调
        #    _forward_prefill_sparse → 也就永远不在 attention 里建这个 cache。
        #    ⇒ 除非 SGLANG_ENABLE_PREFILL_WAR_READ_DONE=1（默认 False，
        #      此时由 prepare_prefill_shared_read_snapshot 提前建），
        #      HCU 上【整个 forward 都是 None】。
        #
        #    由此推出两条实际覆盖差异（HCU + 默认配置 + 非尾段）：
        #      · swa  ：cache=None → 走下面的 else 分支算出
        #               Σmin(seq, extend+window-1)，num_reqs 也有
        #               ⇒ static_size 仍可算，静态选页【生效】（估得保守）
        #      · extra：cache=None 且 tail=None → read_indices / read_token_capacity
        #               / num_reqs 【全部为 None】→ 池里 static_size=None
        #               ⇒ 退化成动态 torch.nonzero，静态选页【不生效】
        #      只有 bounded replay 的尾段（late_layer_tail 非 None）会给
        #      extra 补上 Σceil(seq/ratio)，那时静态选页才回来。
        cache = getattr(metadata, "sparse_prefill_cache", None)
        read_token_capacity = None      # ← 给 静态选页的上界（行数）
        num_reqs = cache.num_reqs if cache is not None else None
        # ══════════════════════════════════════════════════════════════════
        # 这一段在算什么：「本批一共有多少个 token 的 KV 可能被读到」
        #   （这些 token 可能是别的 rank 算的，所以要借；只需要"行数"，
        #     不需要"是哪几行"——后者要等 attention 建出 cache 才有）
        #
        # 先分清两个长度（以"请求共 1000 token、本次新来 200"为例）：
        #   seq_lens_cpu         该请求【总长】（历史+本次）       = 1000
        #   extend_seq_lens_cpu  本次 forward【新增】多少          =  200
        #   sliding_window       窗口大小（V4.1 = 128）            =  128
        #
        # 窗口语义：每个 query 行看它前面 W 个 token（含自己）。
        # 把一批行的窗口【取并集】就是读集合，下面两支各算这个并集的长度。
        # ══════════════════════════════════════════════════════════════════
        if cache is None:
            # ── 还没有工作区：只能按长度算 ─────────────────────────────
            tail = metadata.late_layer_tail
            if tail is not None:
                # Replay floors exclude SWA history before each request's tail.
                # ── 尾段（bounded replay）：Σ尾长，【精确】───────────────
                # 尾段的规则是：每行的窗口下界被 floor 锁在尾段起点，
                #   floor = seq_len - tail_len        （backend: swa_replay_start）
                #
                #   位置:  … 871 │ 872 ………………………………… 999
                #                 ↑ tail 起点 = floor = seq_len - tail_len = 1000-128
                #   第 1 行(872)： 窗口 = [872, 872]     ← 回不到 871
                #   最后一行(999)：窗口 = [872, 999]
                #   并集 = [872, 999] → 长度 = 128 = tail_len      ✅ 精确
                #
                # 因为所有行的窗口都落在同一段尾段里，并集就【恰好等于尾段本身】，
                # 不多不少 → Σtail_len 就是真实行数，不是上界。
                # （上面的例子：seq=1000、extend=200 → tail_len=min(128,200)=128）
                #
                # ★ tail.extend_seq_lens_cpu 是什么：
                #     类型 = Python list[int]（不是 tensor！），长度 = 请求数 bs，
                #     每项 = 该请求的尾长 = min(SWA_WINDOW, extend_len)。
                #     例 extend = 1000/200/50 → [128, 128, 50]，Σ = 306。
                #   名字里之所以叫 "extend"：尾段是"用 tail_lens 当 extend 长度
                #   重新跑一遍 init_forward_metadata_prefill"，所以沿用了
                #   extend 那套字段名（同一个语义，不同的行集合）。
                #   因为是 list，下面的 sum() 是纯 Python 整数相加 ——
                #   不碰设备，零同步。（GPU 版是 tail.extend_seq_lens）
                extend_lens = tail.extend_seq_lens_cpu
                read_token_capacity = sum(extend_lens)
            else:
                # Before the first sparse attention builds its cache, bound the
                # full CP union, including the history before this extend chunk.
                # ── 非尾段：chunk + 前面 W−1 个历史，是【保守上界】──────
                # 没有 floor，本次第一行会回看 chunk 前面的历史：
                #
                #   位置: … 673 ……………………… 799 │ 800 ………………………… 999
                #         └─── 已有历史 800 ───┘ └─── 本次 extend 200 ───┘
                #                                  ↑ chunk 起点 S=800
                #   第一行(800) 窗口 = [673, 800]     ← 前看 W−1=127 个
                #   最后一行(999)窗口 = [872, 999]
                #   并集 = [673, 999] → 长度 = 327 = 200 + 127 = extend + W − 1
                #
                # ★ 为什么是 W−1 而不是 W：第一行自己已算在 extend 里，
                #   它往前只能再回看 W−1 个。
                # ★ 为什么外层还要 min(seq, …)：新请求历史不足时，下界被 0
                #   截断。例：seq=50、extend=50 → min(50, 50+127) = 50。
                # ★ 为什么说它是"上界"：它假设每请求都能往前看满 W−1 个；
                #   实际窗口可能因跨 rank / 尚未本地而略少。估大了只是多搬页
                #   （安全方向）；估小了才会被 nonzero_static 静默截断。
                extend_lens = forward_batch.extend_seq_lens_cpu
                read_token_capacity = sum(
                    min(seq_len, extend_len + pool.sliding_window - 1)
                    for seq_len, extend_len in zip(
                        forward_batch.seq_lens_cpu.tolist(), extend_lens, strict=True
                    )
                )
            # 静态上界公式 ceil(tokens/ps) + 2*num_reqs 需要【请求数】，
            # 这个数是精确的（CPU 侧列表长度），与"行数是否精确"无关。
            # 注意 cache 不为 None 时 num_reqs 来自 cache.num_reqs，不走这里。
            num_reqs = len(extend_lens)
        # Sparse prefill dequantizes the full gathered SWA span, not only the
        # per-query attention indices.
        # ↑ 这句解释 read_indices 为什么用【整段 span】而不是 per-query 索引：
        #   反量化是按"gather 后的整段 SWA"做的，只搬 query 索引覆盖到的页
        #   会让 kernel 读到未反量化的数据。
        pool.prefetch_swa_layer(
            layer_id,
            # 读集合：本层要读哪些页（来自 core 元数据，非 per-query）
            _get_core_attn_metadata().swa_page_indices,
            # 精确读行（None 表示让池按 num_reqs / read_token_capacity 估上界）
            read_indices=cache.swa_token_ids if cache is not None else None,
            num_reqs=num_reqs,                    # 仅静态选页用
            read_token_capacity=read_token_capacity,   # 仅静态选页用
        )


def maybe_wait_cp_kv_swa_prefetch(pool, layer_id: int, forward_batch=None) -> None:
    if _should_sync_cp_cache_layer_split(pool, forward_batch):
        pool.wait_swa_prefetch(layer_id)


def maybe_prefetch_cp_kv_extra(pool, layer_id: int, forward_batch=None) -> None:
    """发起「借 C1/C2 压缩 KV」的预取。与 swa 版的三点不同：

      ① 只有 ratio ∈ {1,2} 的层需要（ratio 0 层没有压缩家族，直接跳过）；
      ② 读集合来自 **indexer 选中的块**（sparse_page_indices(ratio)），
         不是 core 的逐行页表；
      ③ 多一个 cache_full_read 标志：表示"这次要按整段历史读"，
         因为它影响 staging 的容量估算（不是安全开关）。
    """
    if _should_sync_cp_cache_layer_split(pool, forward_batch):
        # 本层的压缩比；V4.1 是 1/2，V4 Pro 是 4/128（后者不进这个函数）
        ratio = pool.compression_ratios[layer_id]
        if ratio in (1, 2):
            metadata = get_attn_backend().forward_metadata
            cache = getattr(metadata, "sparse_prefill_cache", None)
            core = _get_core_attn_metadata()
            read_indices = None            # 精确读行（有 cache 时才有）
            read_token_capacity = None     # 上界（无 cache 时用）
            cache_full_read = False        # 是否声明"整段历史"
            num_reqs = cache.num_reqs if cache is not None else None
            if cache is not None:
                # The workspace dequantizes every compressed position before
                # TopK selects rows; broadcasting only TopK pages is unsafe.
                # ↑ 关键：workspace 在 TopK【之前】就把所有压缩位置反量化了，
                #   所以只广播 TopK 选中的页是不够的（会缺数据）→ 必须按
                #   ensure_compressed() 给出的"全量压缩位置"取行。
                #   flat_token_ids = 该家族压缩域里的行号（摊平后）。
                read_indices = cache.ensure_compressed(
                    ratio, core.page_table, pool.kv_pools[ratio].page_size
                ).flat_token_ids
                # With explicit sparse prefill and nonempty original CP shards,
                # every rank builds the full-history gather before this source.
                # Tail metadata may later drop that cache, but not the KV data.
                # ↑ 三个条件全真才声明"整段读"：
                #   ① ratio == 1        —— 只有 C1 家族走这条 full-history 路径
                #   ② 开了显式 sparse prefill（FLASHMLA_SPARSE_PREFILL）
                #   ③ min(extend) >= cp_size —— 原 CP 分片非空；若某 rank 本地
                #      一行都没有，它不会建 full-history gather，声明整段就是浪费
                #   注意 tail 元数据之后可能把 cache 丢掉，但 **KV 数据不会丢**，
                #   所以这里宁可按整段声明（容量偏大是安全方向）。
                cache_full_read = (
                    ratio == 1
                    and envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.get()
                    and min(forward_batch.extend_seq_lens_cpu, default=0) >= pool.cp_size
                )
            elif metadata.late_layer_tail is not None:
                # Tail queries can select keys anywhere in each full history.
                # Bound their page union without materializing a chunk cache.
                # ↑ 没有 cache 但有尾段元数据：尾段 query 可能选中任一整段历史
                #   里的键 → 只能按"每请求压缩域全长"估并集上界。
                #   ceil(seq/ratio) 就是该请求压缩后的位置数。
                seq_lens = forward_batch.seq_lens_cpu.tolist()
                read_token_capacity = sum(
                    (seq_len + ratio - 1) // ratio for seq_len in seq_lens
                )
                num_reqs = len(seq_lens)
            pool.prefetch_low_ratio_extra_layer(
                layer_id,
                # 读集合：indexer 为每个 query 选中的压缩块（页号）
                core.sparse_page_indices(ratio),
                read_indices=read_indices,                 # 有 cache → 精确行
                num_reqs=num_reqs,                         # 静态上界用
                read_token_capacity=read_token_capacity,   # 静态上界用
                cache_full_read=cache_full_read,           # 容量估算用
            )
