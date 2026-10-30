"""Staging buffers and index remap kernels for CP Cache LayerSplit."""

from __future__ import annotations

from math import prod
from typing import Callable, Optional

import torch

from sglang.srt.utils import is_hcu

_is_hcu = is_hcu()


@torch.compile(dynamic=True)
def build_active_pages_mask(
    indices: torch.Tensor,
    page_size: int,
    max_pages: int,
) -> torch.Tensor:
    """把"本 rank 要读的 token 下标"折成一张【定长页掩码】。

    背景：LayerSplit 下本 rank 只持有 part 的层，读别的层的 KV 前要先知道
    "要读哪些页"，再让 owner 把这些页搬过来。掩码就是这一步的中间表示。

    为什么是"定长"：它要参与 all-reduce（集合通信），所有 rank 的缓冲形状
    必须一致 → 长度恒为 max_pages（整池页数），与真实活跃页数无关。

    细节：
      · indices 里的 -1（无效行/padding）由 valid 掩掉；
        clamp(min=0) 既避免负下标，又把它们统一指到 page 0（不会置位，因为
        index_put_ 的值是 valid，-1 那格贡献 0）。
      · accumulate=True：同一页被多个 token 命中会累加，所以值可能 >1。
        后续只判"非零"（nonzero / all_reduce 求和），所以 >1 无害。
    """
    local_mask = torch.zeros(max_pages, dtype=torch.int32, device=indices.device)
    valid = indices >= 0
    safe_indices = torch.clamp(indices, min=0)
    page_ids = torch.div(safe_indices, page_size, rounding_mode="floor")
    local_mask.index_put_(
        (page_ids.flatten().to(torch.long),),
        valid.flatten().to(torch.int32),
        accumulate=True,
    )
    return local_mask


def all_reduce_active_pages_mask(local_mask: torch.Tensor, pynccl_comm) -> torch.Tensor:
    """第 1 轮集合通信：跨 attention-CP 组求【并集】。

    用 sum 而不是 max 也能得到并集——因为后续只关心"是否非零"：
      某页只要有任一 rank 命中，求和后就 >0；全都没命中才是 0。
    ★ 这一步是所有 rank 都必须参与的集合通信（顺序固定、不可跳过），
      也是"空批/空 rank 也要调用 prefetch"的原因。
    """
    with pynccl_comm.change_state(enable=True):
        pynccl_comm.all_reduce(local_mask)
    return local_mask


def contiguous_request_page_capacity(
    total_tokens: int, num_reqs: int, page_size: int, max_pages: int
) -> int:
    """Bound pages touched by per-request contiguous token spans.

    A span can touch at most ceil(length / page_size) + 1 pages. Summing across
    requests is bounded by ceil(total_tokens / page_size) + 2 * num_reqs; the
    extra per-request allowance also covers padded entries mapped to page 0.

    ★ 这个函数只算"上界"，静态选页靠它把输出长度变成主机已知的常量。
      它的成立前提与"页在池里是否连续"无关——页数由 token 数决定：

        · 池按整页分配，一个请求的 N 个 token 就占 ceil(N/page_size) 个页，
          这些页在池里是连号还是散号，页数都不变；
        · 逻辑上连续的一段 token 最多跨 ceil(L/page_size) + 1 个页
          （最坏起点偏移多跨一页）；
        · 滑窗历史（SWA_WINDOW-1 个 token）紧邻 extend 起点之前，
          最多额外占 1 个页（窗口宽度 < 页大小）；
        · 每请求 +2 就分别对应上面两项：一次边界偏移 + 一次滑窗溢出。
          （page 0 的 padding 项也由这 2 个余量吸收。）

      所以真正被这个上界依赖的假设是"每个请求的读集合在逻辑 token 空间里
      是一段"（extend + 其前滑窗），而不是"物理页连号"。
      调用方传进来的 total_tokens 已经按 min(seq_len, extend + window - 1)
      折算过，正是为了维持这个前提。

    ── 三步推导 ────────────────────────────────────────────────
      ① 单请求：读段 [S, S+L)，S 未必对齐页边界
             跨页数 ≤ ⌈L / ps⌉ + 1
      ② 求和  ：Σ⌈L_r / ps⌉ ≤ ⌈ΣL_r / ps⌉ + (n − 1)
             （⌈a⌉ + ⌈b⌉ ≤ ⌈a+b⌉ + 1，n 项累积最多多 n−1）
      ③ 合并  ：Σ(⌈L_r/ps⌉ + 1) ≤ ⌈total/ps⌉ + (n−1) + n
                              = ⌈total/ps⌉ + 2n − 1 ≤ ⌈total/ps⌉ + 2n

      例（ps = 256，3 个请求读段长 327 / 177 / 80）：
            total = 584 → ⌈584/256⌉ = 3
            capacity = 3 + 2×3 = 9
            真实占页（各请求独立算）= 2 + 2 + 1 = 5   ⇒ 9 ≥ 5，余量 4

    ── 空批为什么返回 1 ────────────────────────────────────────
      total_tokens == 0 时上界可能算出 0，而 0 长度的张量会让下游
      staging[:0] / pynccl broadcast(0 字节) 变得尴尬，所以至少给 1
      （拷贝侧另有 broadcast_pages = max(1, active_pages) 兜底）。

    ★★ 这个前提会被 HiCache 的 load_back 打破。骨架图：

      ★★★ 一句话记住（理解整件事的关键）：
              页数 ≈ ⌈总长 / page_size⌉ + 段数
            正常每请求 1~2 段 → 2n 够用；
            HiCache 回载后段数 D_r 变大 → 2n 不够 → 上界失效。
            注意"段数"指【物理槽被切成几段】，不是逻辑 token 连续与否。

      页数由什么决定？不是"逻辑连续"，而是【物理槽有没有被切段】——
      掩码里算的是 page = 物理slot // ps（见 build_active_pages_mask），
      所以同一请求的槽一旦被切成多段，每个段边界都可能多占一个"半满页"。

      【正常】一个请求的 KV 在物理槽上是一段
        物理槽:  |============ 一段 L ============|
        页    :     |页a |页b |页c |          ← ⌈L/ps⌉ 个整页，最坏 +1（起点不对齐）

      【HiCache 回载后】同一请求被拆成 D_r 段（段与段之间夹着别的请求的页）
        物理槽:  |-- 段1 --|      |--- 段2 ---|   |- 段3 -|
        页    :   |页a|页b|      |页c|页d|页e|     |页f|
                   ↑ 每个段边界都多出一个半满页
        ⇒ 真实页数 ≈ ⌈L/ps⌉ + (D_r − 1)

      于是：
        static_size（本函数）= ⌈total/ps⌉ + 2n      ← 只兜住"每请求 ≤2 段"
        真实页数           ≈ ⌈total/ps⌉ + Σ D_r    ← HiCache 下 ΣD_r 可以 ≫ 2n
                                    └──── 2n 不够 ⇒ 上界失效

      失效后果（静默：无断言、无日志）：
        真实活跃页数 > static_size
          → torch.nonzero_static 只保留前 static_size 个（多出的页被无声丢弃）
          → remap 把那些页映射成 -1
          → attention kernel 跳过它们
          → 读到"缺键"的记忆：结果错，但一切正常返回

      ⇒ 使上界失效的不是"页号变散"（页号一直是散的；整页分配下无害），
        而是【段数变多】。恢复时 _allocate_external_slots 从空闲链整块取页
        （paged.py: alloc → free_pages[:n]），与已有前缀之间没有续页语义
        （对比 alloc_extend 的 last_loc+1 续页），于是
          "L1 前缀一段 + 恢复块一段 + 后续 extend 一段 + 部分释放的空洞"
        这种形态就出现了。

      修法（"段计数"）：把每个请求的 D_r 记账（alloc_extend / load_back /
      free / evict 时更新），把上界改成 ⌈total/ps⌉ + Σ D_r。
      在那之前，只能让这条组合强制退回动态 nonzero ——
      即 HICACHE_IMPLEMENTATION.md §C.4 / deepseek_v4_pool.py 的
      _LS_HICACHE_BOUNDED_REPLAY 门控。
    """
    if total_tokens == 0:
        return min(max_pages, 1)
    return min(
        max_pages,
        (total_tokens + page_size - 1) // page_size + 2 * num_reqs,
    )


@torch.compile(dynamic=True, disable=_is_hcu)
def remap_indices_to_staging(
    indices: torch.Tensor,
    selected_pages: torch.Tensor,
    page_size: int,
    max_pages: int,
    static_selected_pages: bool = False,
) -> torch.Tensor:
    """把"全局池里的 token 下标"翻译成"staging 里的紧凑下标"。

    ── 它解决的是"地址换了"的问题 ────────────────────────────────
      第 2 轮里 owner 只把【选中的那几页】紧凑拷进 staging：
          staging 第 0 页 = 池里 selected_pages[0] 那一页
          staging 第 1 页 = 池里 selected_pages[1] 那一页  ...
      于是"池页号"与"staging 页号"不是同一个数，读者手里的 indices
      （池内 token 下标）必须整体改写一遍，否则会读错行。

    ── 输入 / 输出 ─────────────────────────────────────────────
      indices                [N]  要读的 token 在【池】里的下标（槽号）
      selected_pages         [K]  第 1 轮选出的页号（顺序 = staging 的页序）
      page_size                   一页多少 token
      max_pages                   池的总页数（= 掩码长度）
      static_selected_pages       选页是否用了 nonzero_static（尾部含 -1 填充槽）
      → 返回                  [N]  同一批 token 在【staging】里的下标；
                                  所在页没被选中的 token 得到【负值】（kernel 跳过）

    ── 算法（三步）─────────────────────────────────────────────
      ① 建逆映射  page_map[池页号] = staging 页序号
      ② 拆下标    page = idx // ps，off = idx % ps
      ③ 重拼      new = page_map[page] * ps + off

    ── 例子（ps=4, max_pages=8, selected_pages=[2,5]）─────────────
      page_map = [-1, -1,  0, -1, -1,  1, -1, -1]   ← 池页2→staging页0，池页5→页1
      indices  = [ 9, 10, 11, 21, -1, 13]
      remapped = [ 1,  2,  3,  5, -1, -3]
                   └─ 池页2 的 token 9..11 → staging 页0 的第 1..3 行
                                 ↑        ↑   └ 池页3 未被选中 ⇒ 负值
                          池页5 的 21   indices 的 -1 原样保留

      ★ 未选中的页给出的是【负值】而不是恰好 -1：
        page_map[page] = -1 ⇒ new = -ps + off ∈ [-ps, -1]，恒负 ⇒ kernel 一律跳过。
    """
    # ── HCU：整段交给 LightOp 融合核（commit 0c8baedbfa）──────────
    #   语义完全等价，只是把下面三步合成一个 HIP kernel；
    #   torch.compile 在 HCU 上也是关掉的（disable=_is_hcu）。
    if _is_hcu:
        from lightop.attention import dsv41_remap_indices_to_staging

        return dsv41_remap_indices_to_staging(
            indices, selected_pages, page_size, max_pages, static_selected_pages
        )

    # ── ① 建逆映射 page_map：池页号 → staging 页序号 ───────────────
    if static_selected_pages:
        # 静态选页：selected_pages 尾部有 -1 填充槽。
        # 这些 -1 绝不能直接当下标 —— 会写到 page_map[-1]（= 最后一个真实槽），
        # 把某个真实页的映射覆盖掉（静默读错页）。
        # 做法：把 -1 引到 [max_pages, max_pages + n) 这段【额外槽】，
        #      并把 page_map 相应加长；真实页仍写 [0, max_pages)。
        selected_valid = selected_pages >= 0
        selected_slots = torch.where(
            selected_valid,
            selected_pages.to(torch.long),
            max_pages
            + torch.arange(
                selected_pages.numel(), dtype=torch.long, device=indices.device
            ),
        )
        page_map = torch.full(
            (max_pages + selected_pages.numel(),),
            -1,
            dtype=torch.int32,
            device=indices.device,
        )
    else:
        # 动态选页：selected_pages 全是真实页号，直接当 [0, max_pages) 的下标
        selected_slots = selected_pages.to(torch.long)
        page_map = torch.full((max_pages,), -1, dtype=torch.int32, device=indices.device)
    # 第 i 个被选中的页 → staging 第 i 页；没被选中的页保持 -1
    page_map[selected_slots] = torch.arange(
        selected_pages.numel(), dtype=torch.int32, device=indices.device
    )

    # ── ②③ 拆下标 → 换页 → 重拼 ──────────────────────────────────
    valid = indices >= 0                    # indices 里的 -1（无效行 / CP padding）要原样保留
    safe_indices = torch.clamp(indices, min=0)   # 防负下标；-1 最终由下面的 where 还原
    page_ids = torch.div(safe_indices, page_size, rounding_mode="floor")  # 池页号
    offsets = safe_indices - page_ids * page_size                        # 页内偏移
    new_pages = page_map[page_ids.to(torch.long)].to(indices.dtype)      # 池页号 → staging 页序号（未选中 = -1）
    remapped = new_pages * page_size + offsets   # 重拼成 staging 下标；new_pages=-1 时结果恒负
    return torch.where(valid, remapped, indices)  # 无效行还原成原来的 -1


@torch.compile(dynamic=True)
def remap_page_table_to_staging(
    page_table: torch.Tensor,
    selected_pages: torch.Tensor,
    max_pages: int,
) -> torch.Tensor:
    page_map = torch.full((max_pages,), -1, dtype=torch.int32, device=page_table.device)
    page_map[selected_pages.to(torch.long)] = torch.arange(
        selected_pages.numel(), dtype=torch.int32, device=page_table.device
    )

    valid = page_table >= 0
    safe_pages = torch.clamp(page_table, min=0)
    remapped = page_map[safe_pages.to(torch.long)].to(page_table.dtype)
    return torch.where(valid, remapped, page_table)


def active_pages_for_indices(
    indices: torch.Tensor,
    page_size: int,
    max_pages: int,
    pynccl_comm,
    static_size: Optional[int] = None,
) -> torch.Tensor:
    """Select pages touched by any CP rank; all ranks must call in the same order.

    ★ LayerSplit 读非 owned 层的第 1 轮集合通信，分三步：

      1) build_active_pages_mask：把本 rank 要读的 token 下标转成页号，
         在长度固定为 max_pages 的 int32 掩码上置位（掩码本身大小恒定）。
      2) all_reduce_active_pages_mask：CP 组内求和 → "任意 rank 命中"的并集。
         所有 rank 必须同序调用（集合通信）。
      3) 从并集掩码里取出被选中的页号：
           static_size 不为 None → torch.nonzero_static(size=..., fill_value=-1)
                                   输出长度是常量，主机不需要读设备
           static_size 为 None   → torch.nonzero(...)
                                   输出长度设备才知道 → 主机必须等一次 D2H

      第 3 步的差别不在"结果对不对"，而在"主机能不能连续提交后续指令"：
      第 2 轮（owner 拷贝 + 广播）的字节数等于页数 × item_len，页数要作为
      主机侧参数提交；动态长度会把 主机 卡在同步点上，既让设备空转，
      也让旁路流排不进指令，于是整段只能留在主流上串行执行。

      填充槽（-1）由 remap_indices_to_staging 映射成越界哨兵，
      owner 侧拷贝时 clamp 成 0（重复拷 page 0，内容不会被读）。

      ── 两个函数的输出对照（mask = [0,1,0,1,1,0]，真实非零 3 个）──────
        torch.nonzero(mask)                      → [1, 3, 4]        形状 (3,)
        torch.nonzero_static(mask, size=5,
                             fill_value=-1)      → [1, 3, 4, -1, -1] 形状 (5,)
        ★ 前 k 项【完全相同】，差别只在长度：
            · nonzero         长度 = 真实个数 k —— 只有设备知道
            · nonzero_static  长度 = 调用方传入的整数 —— 主机一开始就知道
        ★ nonzero_static 不会多丢页，但它【不检查】size 是否够：
          若 size=2 → [1, 3]，第 3 个页被静默丢掉（无断言、无日志）
          ⇒ size 必须 ≥ 真实页数，这正是 static_size 上界要成立的原因
    """
    local_mask = build_active_pages_mask(indices, page_size, max_pages)
    local_mask = all_reduce_active_pages_mask(local_mask, pynccl_comm)
    if static_size is not None:
        return torch.nonzero_static(
            local_mask, size=static_size, fill_value=-1
        ).flatten()
    # Dynamic output size causes a device-to-host synchronization on HCU.
    return torch.nonzero(local_mask, as_tuple=False).flatten()


class StagingBufferManager:
    """Family-keyed staging buffers allocated before serving."""

    def __init__(self) -> None:
        self._buffers: dict[str, Optional[torch.Tensor]] = {}

    def allocate(
        self,
        family: str,
        num_pages: int,
        allocate_fn: Callable[[int], torch.Tensor],
    ) -> torch.Tensor:
        if family in self._buffers:
            raise RuntimeError(f"Staging buffer is already allocated: {family}")
        buffer = allocate_fn(num_pages)
        self._buffers[family] = buffer
        return buffer

    def get_existing(self, family: str) -> Optional[torch.Tensor]:
        return self._buffers.get(family)

    def allocate_shared(
        self, families: dict[str, tuple[int, Callable[[int], torch.Tensor]]]
    ) -> None:
        """Allocate aliased views for families with disjoint read/write lifetimes.

        Zero-page templates preserve each pool's padded page ABI without first
        allocating separate full-sized buffers. Callers must serialize reuse.
        """
        if not families:
            return
        for family, (num_pages, _) in families.items():
            if family in self._buffers:
                raise RuntimeError(f"Staging buffer is already allocated: {family}")
            if num_pages < 0:
                raise ValueError(f"Negative staging page count: {family}={num_pages}")

        templates = {
            family: allocate_fn(0)
            for family, (_, allocate_fn) in families.items()
        }
        first = next(iter(templates.values()))
        shapes = {}
        for family, template in templates.items():
            if template.ndim < 1 or template.shape[0] != 0:
                raise ValueError("Shared staging requires zero-page templates")
            if template.dtype != first.dtype or template.device != first.device:
                raise ValueError("Shared staging requires matching dtype and device")
            shapes[family] = (families[family][0], *template.shape[1:])

        storage = first.new_zeros(max(prod(shape) for shape in shapes.values()))
        for family, shape in shapes.items():
            self._buffers[family] = storage[: prod(shape)].view(shape)
