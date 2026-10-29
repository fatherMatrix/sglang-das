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
    """Sum the per-rank active-page mask across the attention-CP group."""
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

    ★★ 这个前提会被 HiCache 的 load_back 打破（2026-10-26 起，devel 分支）：

      恢复时 `_allocate_external_slots` 从空闲链整块取页（paged.py 的
      alloc → free_pages[:n]），与请求已有前缀之间**没有续页语义**
      （对比 alloc_extend 的 last_loc+1 续页）。于是同一个请求的 KV 变成
      "L1 前缀一段 + 恢复块一段 + 后续 extend 一段 + 部分释放留下的空洞"。

        页数仍然 = ceil(总 token 数/ps) + 半满页数；
        而半满页数 ≈ 段数 D_r —— 不再受"每请求 ≤2"约束。

      也就是说：使上界失效的不是"页号变散"（页号一直是散的，整页分配下无害），
      而是**段数变多**。当真实活跃页数 > static_size 时，nonzero_static
      只会保留前 static_size 个（静默截断、无断言），remap 把溢出页映射成 -1，
      attention kernel 直接跳过 → 读到缺键的记忆。

      这正是 C.4 在组合期强制退动态 nonzero 的原因
      （deepseek_v4_pool.py 的 _LS_HICACHE_BOUNDED_REPLAY 门控），
      也是"段计数方案"要补的洞：把 D_r 记账并加进 extra_segments。
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
    if _is_hcu:
        from lightop.attention import dsv41_remap_indices_to_staging

        return dsv41_remap_indices_to_staging(
            indices, selected_pages, page_size, max_pages, static_selected_pages
        )

    if static_selected_pages:
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
        selected_slots = selected_pages.to(torch.long)
        page_map = torch.full((max_pages,), -1, dtype=torch.int32, device=indices.device)
    page_map[selected_slots] = torch.arange(
        selected_pages.numel(), dtype=torch.int32, device=indices.device
    )

    valid = indices >= 0
    safe_indices = torch.clamp(indices, min=0)
    page_ids = torch.div(safe_indices, page_size, rounding_mode="floor")
    offsets = safe_indices - page_ids * page_size
    new_pages = page_map[page_ids.to(torch.long)].to(indices.dtype)
    remapped = new_pages * page_size + offsets
    return torch.where(valid, remapped, indices)


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
