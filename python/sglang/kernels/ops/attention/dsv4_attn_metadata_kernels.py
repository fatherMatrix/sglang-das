from __future__ import annotations

from typing import Optional

import msgspec
import torch
import triton
import triton.language as tl


def _inputs_on_cuda(*args, **kwargs) -> bool:
    """Route kernel dispatch by input placement: the first tensor argument
    decides. CUDA inputs take the fused triton kernel; CPU inputs take the
    torch reference implementation (triton is CUDA-only, and CPU-side callers
    such as unit tests exercise the reference path)."""
    for value in (*args, *kwargs.values()):
        if isinstance(value, torch.Tensor):
            return value.is_cuda
    raise AssertionError("kernel dispatch requires at least one tensor argument")


class ExpandPrefillCausallyResult(msgspec.Struct):
    seq_lens_casual: torch.Tensor
    req_pool_indices_repeated: torch.Tensor


class ExpandPrefillCausally:
    @classmethod
    def execute(cls, *args, **kwargs) -> ExpandPrefillCausallyResult:
        if _inputs_on_cuda(*args, **kwargs):
            return cls.triton(*args, **kwargs)
        return cls.torch(*args, **kwargs)

    @classmethod
    def torch(
        cls,
        *,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        extend_start_loc: Optional[torch.Tensor],
        seq_lens_cpu: Optional[list[int]],
        extend_seq_lens_cpu: Optional[list[int]],
        num_tokens: int,
        padded_num_tokens: Optional[int],
    ) -> ExpandPrefillCausallyResult:
        return expand_prefill_causally(
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            extend_seq_lens=extend_seq_lens,
            extend_start_loc=extend_start_loc,
            seq_lens_cpu=seq_lens_cpu,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            num_tokens=num_tokens,
            padded_num_tokens=padded_num_tokens,
        )

    @classmethod
    def triton(
        cls,
        *,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        extend_start_loc: Optional[torch.Tensor],
        seq_lens_cpu: Optional[list[int]],
        extend_seq_lens_cpu: Optional[list[int]],
        num_tokens: int,
        padded_num_tokens: Optional[int],
    ) -> ExpandPrefillCausallyResult:
        return expand_prefill_causally_triton(
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            extend_seq_lens=extend_seq_lens,
            num_tokens=num_tokens,
            padded_num_tokens=padded_num_tokens,
        )


def expand_prefill_causally(
    *,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    extend_start_loc: Optional[torch.Tensor],
    seq_lens_cpu: Optional[list[int]],
    extend_seq_lens_cpu: Optional[list[int]],
    num_tokens: int,
    padded_num_tokens: Optional[int],
) -> ExpandPrefillCausallyResult:
    device = req_pool_indices.device
    cuda_int32_kwargs = {"dtype": torch.int32, "device": device}

    if extend_start_loc is not None:
        repeats = extend_seq_lens.to(torch.int64)
        req_pool_indices_repeated = torch.repeat_interleave(
            req_pool_indices, repeats, output_size=num_tokens
        )
        start_positions = seq_lens.to(torch.int32) - extend_seq_lens.to(torch.int32) + 1
        start_positions_repeated = torch.repeat_interleave(
            start_positions, repeats, output_size=num_tokens
        )
        start_locs_repeated = torch.repeat_interleave(
            extend_start_loc.to(torch.int32), repeats, output_size=num_tokens
        )
        token_offsets = (
            torch.arange(num_tokens, **cuda_int32_kwargs) - start_locs_repeated
        )
        seq_lens_casual = start_positions_repeated + token_offsets

        if padded_num_tokens is not None and padded_num_tokens > num_tokens:
            pad_size = padded_num_tokens - num_tokens
            seq_lens_casual = torch.nn.functional.pad(
                seq_lens_casual, (0, pad_size), value=1
            )
            req_pool_indices_repeated = torch.cat(
                (
                    req_pool_indices_repeated,
                    req_pool_indices_repeated[-1:].expand(pad_size),
                )
            )
        return ExpandPrefillCausallyResult(
            seq_lens_casual=seq_lens_casual,
            req_pool_indices_repeated=req_pool_indices_repeated,
        )

    assert seq_lens_cpu is not None and extend_seq_lens_cpu is not None
    seq_lens_casual = torch.empty(num_tokens, **cuda_int32_kwargs)
    idx_to_req_repeated = torch.empty(num_tokens, **cuda_int32_kwargs)
    offset = 0
    for i, (kv_len, qo_len) in enumerate(zip(seq_lens_cpu, extend_seq_lens_cpu)):
        out = seq_lens_casual[offset : offset + qo_len]
        offset += qo_len
        torch.arange(kv_len - qo_len + 1, kv_len + 1, out=out)
        idx_to_req_repeated[offset - qo_len : offset].fill_(i)

    assert offset == num_tokens
    req_pool_indices_repeated = req_pool_indices[idx_to_req_repeated]

    if padded_num_tokens is not None and padded_num_tokens > num_tokens:
        pad_size = padded_num_tokens - num_tokens
        seq_lens_casual = torch.nn.functional.pad(
            seq_lens_casual, (0, pad_size), value=1
        )
        req_pool_indices_repeated = torch.cat(
            (
                req_pool_indices_repeated,
                req_pool_indices_repeated[-1:].expand(pad_size),
            )
        )
    return ExpandPrefillCausallyResult(
        seq_lens_casual=seq_lens_casual,
        req_pool_indices_repeated=req_pool_indices_repeated,
    )


@triton.jit
def _expand_prefill_causally_kernel(
    req_pool_ptr,
    seq_lens_ptr,
    extend_seq_lens_ptr,
    seq_lens_casual_ptr,
    req_pool_repeated_ptr,
    bs,
    num_tokens,
    total_tokens,
    BLOCK: tl.constexpr,
    BS_P2: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total_tokens

    b = tl.arange(0, BS_P2)
    bmask = b < bs
    extend = tl.load(extend_seq_lens_ptr + b, mask=bmask, other=0).to(tl.int32)
    start_locs = tl.cumsum(extend, axis=0) - extend

    is_real = offs < num_tokens
    t = tl.where(is_real, offs, 0).to(tl.int32)
    started = (start_locs[None, :] <= t[:, None]) & bmask[None, :]
    r = tl.sum(started.to(tl.int32), axis=1) - 1
    r = tl.where(is_real, r, bs - 1).to(tl.int64)

    seq_len = tl.load(seq_lens_ptr + r, mask=mask, other=0).to(tl.int32)
    ext = tl.load(extend_seq_lens_ptr + r, mask=mask, other=0).to(tl.int32)
    start_loc = tl.sum(tl.where(started, extend[None, :], 0).to(tl.int32), axis=1) - ext
    causal = (seq_len - ext + 1) + (t - start_loc)
    causal = tl.where(is_real, causal, 1)

    rp = tl.load(req_pool_ptr + r, mask=mask, other=0)
    tl.store(seq_lens_casual_ptr + offs, causal, mask=mask)
    tl.store(req_pool_repeated_ptr + offs, rp, mask=mask)


def expand_prefill_causally_triton(
    *,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    num_tokens: int,
    padded_num_tokens: Optional[int],
) -> ExpandPrefillCausallyResult:
    bs = req_pool_indices.shape[0]
    device = req_pool_indices.device
    total_tokens = (
        padded_num_tokens
        if padded_num_tokens is not None and padded_num_tokens > num_tokens
        else num_tokens
    )

    seq_lens_casual = torch.empty(total_tokens, dtype=torch.int32, device=device)
    req_pool_indices_repeated = torch.empty(
        total_tokens, dtype=req_pool_indices.dtype, device=device
    )
    BLOCK = 256
    _expand_prefill_causally_kernel[(triton.cdiv(total_tokens, BLOCK),)](
        req_pool_indices,
        seq_lens,
        extend_seq_lens,
        seq_lens_casual,
        req_pool_indices_repeated,
        bs,
        num_tokens,
        total_tokens,
        BLOCK=BLOCK,
        BS_P2=triton.next_power_of_2(max(bs, 1)),
    )
    return ExpandPrefillCausallyResult(
        seq_lens_casual=seq_lens_casual,
        req_pool_indices_repeated=req_pool_indices_repeated,
    )


class PageTablePositionsResult(msgspec.Struct):
    seq_lens_casual: torch.Tensor
    positions_casual: torch.Tensor
    page_table: torch.Tensor
    swa_topk_lengths: torch.Tensor


class BuildPageTablePositions:
    @classmethod
    def execute(cls, *args, **kwargs) -> PageTablePositionsResult:
        if _inputs_on_cuda(*args, **kwargs):
            return cls.triton(*args, **kwargs)
        return cls.torch(*args, **kwargs)

    @classmethod
    def torch(
        cls,
        *,
        req_to_token: torch.Tensor,
        req_pool_indices_repeated: torch.Tensor,
        seq_lens_casual: torch.Tensor,
        max_seq_len: int,
        page_size: int,
        swa_window: int,
    ) -> PageTablePositionsResult:
        return build_page_table_positions(
            req_to_token=req_to_token,
            req_pool_indices_repeated=req_pool_indices_repeated,
            seq_lens_casual=seq_lens_casual,
            max_seq_len=max_seq_len,
            page_size=page_size,
            swa_window=swa_window,
        )

    @classmethod
    def triton(
        cls,
        *,
        req_to_token: torch.Tensor,
        req_pool_indices_repeated: torch.Tensor,
        seq_lens_casual: torch.Tensor,
        max_seq_len: int,
        page_size: int,
        swa_window: int,
    ) -> PageTablePositionsResult:
        return build_page_table_positions_triton(
            req_to_token=req_to_token,
            req_pool_indices_repeated=req_pool_indices_repeated,
            seq_lens_casual=seq_lens_casual,
            max_seq_len=max_seq_len,
            page_size=page_size,
            swa_window=swa_window,
        )


def build_page_table_positions(
    *,
    req_to_token: torch.Tensor,
    req_pool_indices_repeated: torch.Tensor,
    seq_lens_casual: torch.Tensor,
    max_seq_len: int,
    page_size: int,
    swa_window: int,
) -> PageTablePositionsResult:
    seq_lens_casual = seq_lens_casual.to(torch.int32)
    positions_casual = seq_lens_casual - 1
    page_table = req_to_token[
        req_pool_indices_repeated.to(torch.int64), :max_seq_len:page_size
    ]
    page_table = (page_table // page_size).to(torch.int32)
    swa_topk_lengths = torch.clamp(seq_lens_casual, max=swa_window)
    return PageTablePositionsResult(
        seq_lens_casual=seq_lens_casual,
        positions_casual=positions_casual,
        page_table=page_table,
        swa_topk_lengths=swa_topk_lengths,
    )


@triton.jit
def _page_table_positions_kernel(
    req_to_token_ptr,
    req_pool_ptr,
    seq_lens_ptr,
    seq_lens_out_ptr,
    positions_out_ptr,
    page_table_ptr,
    topk_out_ptr,
    rt_stride,
    num_pages,
    page_size,
    swa_window,
    BLOCK_P: tl.constexpr,
):
    row = tl.program_id(0)
    seq_len = tl.load(seq_lens_ptr + row).to(tl.int32)
    tl.store(seq_lens_out_ptr + row, seq_len)
    tl.store(positions_out_ptr + row, seq_len - 1)
    tl.store(topk_out_ptr + row, tl.minimum(seq_len, swa_window))

    rp = tl.load(req_pool_ptr + row).to(tl.int64)
    base = req_to_token_ptr + rp * rt_stride
    out_base = page_table_ptr + row.to(tl.int64) * num_pages
    for p0 in range(0, num_pages, BLOCK_P):
        p = p0 + tl.arange(0, BLOCK_P)
        pmask = p < num_pages
        tok = tl.load(base + p.to(tl.int64) * page_size, mask=pmask, other=0).to(
            tl.int32
        )
        tl.store(out_base + p, tok // page_size, mask=pmask)


def build_page_table_positions_triton(
    *,
    req_to_token: torch.Tensor,
    req_pool_indices_repeated: torch.Tensor,
    seq_lens_casual: torch.Tensor,
    max_seq_len: int,
    page_size: int,
    swa_window: int,
) -> PageTablePositionsResult:
    num_q = seq_lens_casual.shape[0]
    num_pages = (max_seq_len + page_size - 1) // page_size
    device = seq_lens_casual.device

    seq_lens_out = torch.empty(num_q, dtype=torch.int32, device=device)
    positions_out = torch.empty(num_q, dtype=torch.int32, device=device)
    page_table = torch.empty((num_q, num_pages), dtype=torch.int32, device=device)
    topk_out = torch.empty(num_q, dtype=torch.int32, device=device)
    BLOCK_P = 256
    _page_table_positions_kernel[(num_q,)](
        req_to_token,
        req_pool_indices_repeated,
        seq_lens_casual,
        seq_lens_out,
        positions_out,
        page_table,
        topk_out,
        req_to_token.stride(0),
        num_pages,
        page_size,
        swa_window,
        BLOCK_P=BLOCK_P,
    )
    return PageTablePositionsResult(
        seq_lens_casual=seq_lens_out,
        positions_casual=positions_out,
        page_table=page_table,
        swa_topk_lengths=topk_out,
    )


class BuildCausalSwaPageIndices:
    @classmethod
    def execute(cls, *args, **kwargs) -> torch.Tensor:
        if _inputs_on_cuda(*args, **kwargs):
            return cls.triton(*args, **kwargs)
        return cls.torch(*args, **kwargs)

    @classmethod
    def torch(
        cls,
        *,
        req_to_token: torch.Tensor,
        full_to_swa_mapping: torch.Tensor,
        req_pool_indices_repeated: torch.Tensor,
        seq_lens_casual: torch.Tensor,
        swa_window: int,
        page_index_aligned_size: int,
        swa_replay_start: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return build_causal_swa_page_indices(
            req_to_token=req_to_token,
            full_to_swa_mapping=full_to_swa_mapping,
            req_pool_indices_repeated=req_pool_indices_repeated,
            seq_lens_casual=seq_lens_casual,
            swa_window=swa_window,
            page_index_aligned_size=page_index_aligned_size,
            swa_replay_start=swa_replay_start,
        )

    @classmethod
    def triton(
        cls,
        *,
        req_to_token: torch.Tensor,
        full_to_swa_mapping: torch.Tensor,
        req_pool_indices_repeated: torch.Tensor,
        seq_lens_casual: torch.Tensor,
        swa_window: int,
        page_index_aligned_size: int,
        swa_replay_start: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return build_causal_swa_page_indices_triton(
            req_to_token=req_to_token,
            full_to_swa_mapping=full_to_swa_mapping,
            req_pool_indices_repeated=req_pool_indices_repeated,
            seq_lens_casual=seq_lens_casual,
            swa_window=swa_window,
            page_index_aligned_size=page_index_aligned_size,
            swa_replay_start=swa_replay_start,
        )


def late_layer_tail_layout(
    *,
    extend_lens_cpu: list[int],
    seq_lens_cpu: list[int],
    tail_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[int], torch.Tensor]:
    """Tail rows of each prefill extend: its last min(tail_len, extend_len) tokens.
    Returns (token indices into the extend, per-request tail lengths, per-row
    absolute window floor).

    ==================================================================
    在流程里的位置：decoder SWA bounded replay 的"第一步"
    ==================================================================
    调用者是 DeepseekV4AttnBackend._build_late_layer_tail_metadata（每个
    prefill forward 一次）。它只做纯数学：把"每请求 extend 多长、历史多长"
    两个 CPU 列表，换成"晚层要算的那批行"的描述。产出的三样东西：

      token_indices : 每行（= 每个尾 token，见下）在【原 extend】里的行号
      tail_lens_cpu : 每请求的尾长 = min(tail_len, extend_len)
      floor         : 每行的 SWA 窗口下界（绝对位置）

    ★ 两种粒度（同 LateLayerTail 的约定）：
        每请求（长度 = bs）    tail_lens_cpu
        每行  （长度 = Σ尾长） token_indices / floor
        "行 = 一个 token"，按请求分组、组内保持原序。

    ★ 算例（bs=2, tail_len=128, prefix=[5000, 0]）
        extend_lens = [200, 50]  seq_lens = [5200, 50]
        tail_lens   = [128, 50]  Σ = 178
        行 0..127   : 请求0，token_indices = 72..199，  floor = 5072
        行 128..177 : 请求1，token_indices = 200..249， floor = 0
      注意 token_indices 是【批内 extend 行号】，不是请求内序号：
        请求1 的 extend 排在请求0 的 200 行之后，所以它从 200 开始而不是 0。
      而 floor 是【绝对位置】（请求 0 是 5072，请求 1 是 0）——
      两者相差一个 prefix_len。floor 恰好等于"该请求尾段第一行的绝对位置"：
        seq_len - tail_len
      （请求0：5200-128 = 5072；请求1：50-50 = 0。）

    ★ 两个分支与代价
        单请求：arange + full，零 H2D
        多请求：把三组长度打成一个 (3, bs) 张量做【一次】H2D，
                之后全部在设备上算 —— 启动次数不随 bs 增长
                （注释原话："launch count does not grow with bs"）
    ==================================================================
    """
    # 每请求尾长：不足一个窗口就取整个 extend（短 extend 不补零）。
    # 这一步纯 CPU，因为输入本来就是 CPU 列表。
    tail_lens_cpu = [min(tail_len, n) for n in extend_lens_cpu]
    if len(extend_lens_cpu) == 1:
        # ---- 单请求快路径：尾段就是 extend 的最后 t 行，可以零拷贝构造 ----
        # 先把三个长度取出来（都是 Python int，不产生任何设备操作）：
        #   n = 本 chunk 的 extend 行数（这个请求要新算多少 token）
        #   t = 尾长 = min(tail_len, n)（不足一个窗口就全取）
        #   s = seq_len = prefix_len + n（含历史的绝对总长）
        #
        # 只有一个请求时，"批内行号"就等于"它自己的行号"：0 .. n-1
        # 对应的绝对位置是 [s-n, s-1]（extend 紧接在 prefix 之后）。
        #   ★ 尾段 = 最后 t 行  →  行号 [n-t, n-1]
        #                      →  位置 [s-t, s-1]
        #   所以尾段第一行的【绝对位置】= s - t，这就是窗口下界 floor。
        n, t, s = extend_lens_cpu[0], tail_lens_cpu[0], seq_lens_cpu[0]

        # floor = 长度 t、每个元素都是 (s - t) 的 int32 张量。
        #   为什么是"每行一个"而不是一个标量？
        #     下游的 swa_replay_start 是按行消费的张量（每行给自己的窗口下界）；
        #     而同一请求内所有行的下界相同，所以就是同一个常数复制 t 份。
        #   int32 是为了对齐 kernel 的入参 ABI。
        #   语义：该行能回看的最早绝对位置，保证窗口不越过尾段起点
        #        （尾段之前的 SWA 行这一次根本没写过，读了就是脏数据）。
        floor = torch.full((t,), s - t, dtype=torch.int32, device=device)

        # arange(n-t, n) = [n-t, n-t+1, ..., n-1]：跳过前 n-t 行，只取最后 t 行。
        #   · n=200, t=128 → 72..199（跳过前 72 行）
        #   · n=50,  t=50  → 0..49   （短 extend：整段都是尾段，不跳过任何行）
        # 交付顺序与多请求分支一致：(token_indices, tail_lens_cpu, floor)
        return torch.arange(n - t, n, device=device), tail_lens_cpu, floor
    # ---- 多请求：一次 H2D 拿三组长度，其余全在设备上推导 ----
    #
    # ★ H2D 就发生在【下一行】：
    #     torch.tensor([...], device=device)
    #   入参是 3 个 Python list（在主机内存），目标 device 是加速器，
    #   所以这一次构造 = 一次 主机→设备 拷贝。
    #   数据量 = 3 × bs × 8 字节（int64），与"尾行总数"无关。
    #
    # ★ "其余在设备上推导"指它之后的每一步都是设备端算子：
    #     lens 切片 / arange / repeat_interleave / cumsum / [req] 索引 /
    #     逐元素加减 / .to(int32)
    #   全过程中没有任何 .item() / .tolist() / 依赖张量取值的 Python if，
    #   因此不会触发 设备→主机 的同步（那才是真正会让流水线停顿的操作）。
    #
    # ★ 为什么不在主机上把三个结果都算好再一次性拷上来？
    #   那样要构造长度 Σ尾长 的 Python list 再整体拷贝（bs=64、Σ=8192 时
    #   约 64KB，外加 8192 次 Python 循环）；现在只拷 3×bs 个标量（约 1.5KB），
    #   由设备自己展开。H2D 的量从 O(Σ尾长) 降到 O(bs)，且 kernel 启动次数
    #   与 bs 无关（原注释："launch count does not grow with bs"）。
    #
    # 对照单请求快路径：那里的 n/t/s 都是 Python 标量，torch.full 的填充值和
    # arange 的起止都作为【kernel 参数】传入，不需要任何张量参与拷贝 → 0 次 H2D。
    # One H2D copy for the three length vectors; launch count does not grow with bs.
    lens = torch.tensor([extend_lens_cpu, tail_lens_cpu, seq_lens_cpu], device=device)
    extend_lens, tail_lens, seq_lens = lens[0], lens[1], lens[2]
    total = sum(tail_lens_cpu)
    # req：每个行属于哪个请求（长度 total）。展开后天然按请求分组、组内保序。
    req = torch.repeat_interleave(
        torch.arange(len(tail_lens_cpu), device=device), tail_lens, output_size=total
    )
    # offs：该行在【本请求组内】的偏移（0..tail_len-1）。
    #   arange(total) - 本请求组的起始行号
    #   起始行号 = cumsum(tail_lens) - tail_lens
    offs = (
        torch.arange(total, device=device)
        - (torch.cumsum(tail_lens, 0) - tail_lens)[req]
    )
    # token_indices：本请求尾段第一行的批内行号 + 组内偏移。
    #   请求 i 的 extend 占批内行 [cumsum[i-1], cumsum[i])，
    #   其尾段第一行 = cumsum[i] - tail_len[i]。
    token_indices = (torch.cumsum(extend_lens, 0) - tail_lens)[req] + offs
    # floor：本请求的绝对窗口下界 = seq_len - tail_len（同请求内所有行相同，
    #   所以用 [req] 广播成每行）。转 int32 是因为下游 kernel 吃 int32。
    floor = (seq_lens - tail_lens)[req].to(torch.int32)
    return token_indices, tail_lens_cpu, floor


def build_causal_swa_page_indices(
    *,
    req_to_token: torch.Tensor,
    full_to_swa_mapping: torch.Tensor,
    req_pool_indices_repeated: torch.Tensor,
    seq_lens_casual: torch.Tensor,
    swa_window: int,
    page_index_aligned_size: int,
    swa_replay_start: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Window slots each query attends to, -1 where empty. swa_replay_start floors
    each row's window at that absolute position; None is the plain causal window."""
    device = seq_lens_casual.device
    pos_causal = seq_lens_casual - 1
    num_qo_tokens = seq_lens_casual.size(0)
    offsets = pos_causal.unsqueeze(1) - torch.arange(
        swa_window, dtype=torch.int32, device=device
    ).unsqueeze(0)
    if swa_replay_start is None:
        invalid_offset_mask = offsets < 0
    else:
        invalid_offset_mask = offsets < swa_replay_start.to(offsets.dtype).unsqueeze(1)
    offsets.masked_fill_(invalid_offset_mask, 0)
    raw_indices = req_to_token[req_pool_indices_repeated[:, None], offsets]
    assert raw_indices.shape == (num_qo_tokens, swa_window)
    raw_indices.masked_fill_(invalid_offset_mask, -1)
    swa_indices = full_to_swa_mapping[raw_indices]
    swa_indices = swa_indices.to(torch.int32)

    padded_width = (
        (swa_window + page_index_aligned_size - 1) // page_index_aligned_size
    ) * page_index_aligned_size
    if padded_width == swa_window:
        return swa_indices
    return torch.nn.functional.pad(
        swa_indices, (0, padded_width - swa_window), value=-1
    )


@triton.jit
def _causal_swa_page_indices_kernel(
    req_to_token_ptr,
    full_to_swa_ptr,
    req_pool_ptr,
    seq_lens_ptr,
    swa_replay_start_ptr,
    out_ptr,
    rt_stride,
    swa_window,
    padded_width,
    HAS_SWA_REPLAY_START: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    pos = tl.load(seq_lens_ptr + row).to(tl.int64) - 1
    rp = tl.load(req_pool_ptr + row).to(tl.int64)
    base = req_to_token_ptr + rp * rt_stride
    out_base = out_ptr + row.to(tl.int64) * padded_width
    if HAS_SWA_REPLAY_START:
        floor = tl.load(swa_replay_start_ptr + row).to(tl.int64)
    else:
        floor = tl.zeros((), dtype=tl.int64)

    for k0 in range(0, padded_width, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        kmask = k < padded_width
        off = pos - k.to(tl.int64)
        valid = (k < swa_window) & (off >= floor) & kmask
        full_loc = tl.load(base + tl.where(valid, off, 0), mask=valid, other=-1).to(
            tl.int64
        )
        swa = tl.load(full_to_swa_ptr + full_loc, mask=valid, other=-1).to(tl.int32)
        tl.store(out_base + k, tl.where(valid, swa, -1), mask=kmask)


def build_causal_swa_page_indices_triton(
    *,
    req_to_token: torch.Tensor,
    full_to_swa_mapping: torch.Tensor,
    req_pool_indices_repeated: torch.Tensor,
    seq_lens_casual: torch.Tensor,
    swa_window: int,
    page_index_aligned_size: int,
    swa_replay_start: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    num_qo_tokens = seq_lens_casual.size(0)
    padded_width = (
        (swa_window + page_index_aligned_size - 1) // page_index_aligned_size
    ) * page_index_aligned_size
    out = torch.empty(
        (num_qo_tokens, padded_width),
        dtype=torch.int32,
        device=seq_lens_casual.device,
    )
    BLOCK_K = 256
    has_swa_replay_start = swa_replay_start is not None
    _causal_swa_page_indices_kernel[(num_qo_tokens,)](
        req_to_token,
        full_to_swa_mapping,
        req_pool_indices_repeated,
        seq_lens_casual,
        # Unused when HAS_SWA_REPLAY_START is False; any valid pointer will do.
        swa_replay_start if has_swa_replay_start else seq_lens_casual,
        out,
        req_to_token.stride(0),
        swa_window,
        padded_width,
        HAS_SWA_REPLAY_START=has_swa_replay_start,
        BLOCK_K=BLOCK_K,
    )
    return out


@triton.jit
def _small_page_table(
    REQ_TO_TOKEN,
    REQS,
    LENS,
    OUT_LENS,
    POS,
    PAGES,
    SWA,
    STRIDE: tl.constexpr,
    NUM_PAGES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    WINDOW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, tile = tl.program_id(0), tl.program_id(1)
    if tile == 0:
        length = tl.load(LENS + row).to(tl.int32)
        tl.store(OUT_LENS + row, length)
        tl.store(POS + row, length - 1)
        tl.store(SWA + row, tl.minimum(length, WINDOW))
    req = tl.load(REQS + row).to(tl.int64)
    p = tile * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(
        REQ_TO_TOKEN + req * STRIDE + p.to(tl.int64) * PAGE_SIZE,
        mask=p < NUM_PAGES,
        other=0,
    ).to(tl.int32)
    tl.store(PAGES + row * NUM_PAGES + p, slot // PAGE_SIZE, mask=p < NUM_PAGES)


def build_page_table_positions_small(
    *,
    req_to_token,
    req_pool_indices_repeated,
    seq_lens_casual,
    max_seq_len,
    page_size,
    swa_window,
):
    assert page_size > 0 and page_size & (page_size - 1) == 0
    rows = seq_lens_casual.numel()
    pages = triton.cdiv(max_seq_len, page_size)
    kw = dict(device=seq_lens_casual.device, dtype=torch.int32)
    lengths, positions, swa = [torch.empty(rows, **kw) for _ in range(3)]
    table = torch.empty((rows, pages), **kw)
    _small_page_table[(rows, triton.cdiv(pages, 256))](
        req_to_token,
        req_pool_indices_repeated,
        seq_lens_casual,
        lengths,
        positions,
        table,
        swa,
        req_to_token.stride(0),
        pages,
        page_size,
        swa_window,
        256,
    )
    return PageTablePositionsResult(
        seq_lens_casual=lengths,
        positions_casual=positions,
        page_table=table,
        swa_topk_lengths=swa,
    )
