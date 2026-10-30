"""DeepSeek-V4.1 manifold-constrained hyper-connections (mHC).

The residual is ``R [T, hc_mult, hidden]`` -- hc_mult parallel streams per token --
and each sublayer is wrapped in a hyper-connection (DeepSeek's ``attn_hc`` and
``ffn_hc``) carrying a ``(pre, post, comb)`` triplet. Per hyper-connection:

    1. mix stats  predict ``(pre, post, comb)`` from ``R``
    2. combine    collapse the streams with the PREVIOUS triplet's ``pre``, then norm
    3. sublayer   attention or MoE, including its tensor-parallel reduction
    4. post       ``R'[i] = post[i] * y + sum_j comb[j][i] * R[j]``

Step 1's ``pre`` is consumed by step 2 of the *next* hyper-connection -- the lag
that lets step 1 run beside steps 2-3 and makes V4's fused post+pre inexpressible.
``attn_hc -> ffn_hc`` stays inside a layer; ``ffn_hc -> attn_hc`` crosses layers,
where an Engram or a row selection can replace ``R`` (`with_residual`/`take_rows`).

The layer owns parameters, parallelism and token-range policy; this module owns
the kernel dispatch.

======================================================================
符号表（后面所有公式都用这些名字）
======================================================================
  T              本批 token 行数
  d / hidden     5120
  n / hc_mult    残差流的条数 = 4
  R              [T, n, d]   残差：每 token n 条并行流
  H  (fn)        [n(2+n), n·d] = [24, 20480]   mixing 矩阵（fp32）
  s  (scale)     [3]          三段输出各自的缩放
  b  (base)      [n(2+n)] = [24]              三段输出各自的偏置
  m              scalar       post 的倍数（hc_post_mult_value）
  ε_pre          标量         pre 的下限（+eps，避免 0 把流彻底掐死）
  ε_rms / ε_sink 标量         数值保护
  I              20           Sinkhorn 迭代次数

  pre            [T, n]       压流系数：子层输入取哪几条流、各占多少
  post           [T, n]       写回系数：子层输出按比例加进每条流
  comb           [T, n, n]    流间混合矩阵：残差自身如何重组
  y              [T, d]       子层输出（attn 或 MoE）

======================================================================
一层的精确数据流（注意 pre 与 R 的"错位一个 block"）
======================================================================
  进来时 state 带着两样：(R_k, pre_{k-1})

    ① mix stats（在 R_k 上算，可与子层并行）
         x_flat = R_k.flatten(1)                          [T, n·d]
         rms    = 1/sqrt(mean(x_flat²) + ε_rms)            ← 均值跨 n·d 求
         M      = (x_flat @ Hᵀ) · rms                      [T, 24]
         pre_k  = σ(M[:, 0:n]·s0 + b0) + ε_pre             [T, n]
         post_k = 2σ(M[:, n:2n]·s1 + b1) · m               [T, n]
         comb_k = Sinkhorn(M[:, 2n:].view(T,n,n)·s2 + b2, I)  [T, n, n]

    ② combine（用的是【上一个】子层算出的 pre_{k-1}，不是 pre_k）
         x_k = norm( Σ_h pre_{k-1}[h] · R_k[h] )           [T, d]

    ③ 子层
         y_k = f_k(x_k)                                    [T, d]

    ④ post（用 ① 刚算出的 post_k / comb_k）
         R_{k+1}[i] = post_k[i]·y_k + Σ_j comb_k[i][j]·R_k[j]     [T, n, d]

    ⑤ 传给下一个子层：(R_{k+1}, pre_k)

  ★ 与论文/文档（A.4.3）的边界编号对照，避免看代码时对不上：
        文档把"子层 l-1 的 post"到"子层 l 的 combine"之间叫【边界 l】，
        边界 l 上有三件事： (a) 写回 X_l  (b) H(X_l)  (c) A·X_l 混合
        代码里 (a)(b) 挤在同一个函数（run_*_post）里，因为两者都在
        "y_l 已经出来"之后才能做，所以下标会差一拍：

          文档                       代码
          (a)@边界 l  写回 X_l   ↔   run_post(l-1) 里的残差写回
          (b)@边界 l  算 H(X_l)  ↔   run_post(l) 里的 _compute_triplet(R_l)
          (c)@边界 l  混合 X_l   ↔   combine(l)，用 pre_{l-1}

        注意 (b)@边界 l 在代码里落在 run_post(l)（即边界 l+1）—— 这不改变
        语义（它只依赖 X_l），正是"H 离开关键路径"的实现：先让 (c) 用旧 A 跑，
        新的 triplet 晚一拍再算。

  ★ 这个错位（pre_k 由 R_k 算出、却作用在 R_{k+1} 上）就是 single-pass mHC。
    它带来的收益只有一条，但很关键：② 不再依赖【本层】的 ①。
      没有错位（V4 Pro）：② 必须先拿到 pre_k，① 卡在关键路径上，
                         形状是 ①(R_k) → ② → ③，无法重叠。
      有错位（V4.1）    ：② 只用上层留下的 pre_{k-1}，于是 ① 可以被推迟发行，
                         形状是 ②③ 走主流、① 走 stats 流，两者并行。
    实现手法（见 fork_stats_stream / _compute_triplet）：
      · fork 点在 combine【之前】：stats_stream.wait_stream(main)
        —— 侧流只需要"combine 之前那一刻的残差"，不需要子层结果；
      · 发行点在 post（子层之后）：把 kernel 排进侧流队列，
        GPU 侧与主流上仍在执行的子层 kernel 并行；
      · join 点在 post 读 triplet 之前：current_stream.wait_stream(stats_stream)。
      即"依赖早、发行晚"——所以能重叠。
    代码证据：use_stats_stream 第一句就是 `if not cfg.pre_from_prev: return False`。

  ★ Sinkhorn 为什么必须做（comb 的约束）：
      先 exp(· - 行max) 做行归一，再列归一，交替 I 次 → 近似双随机矩阵。
      它保证各流的总质量守恒：没有这个约束时，40 层连乘会让某些流被指数级
      放大、另一些被压到 0（mixing 矩阵的谱半径失控）。

======================================================================
为什么 mHC 处处在融合：残差的带宽账（读这一节就能把后面串起来）
======================================================================
  一条残差 R = [T, 4, 5120] bf16 = 每 token 40 KB。
  一个子层要对它做三件事，天然是三次访问：

      M (mix stats) 读 R  → (pre, post, comb)
      C (combine)   读 R  → 压成 x
      P (post)      读 R，写 R'

  于是"融合"只有一种含义：**让这三件事共享同一次 R 的读写**。阶梯是

      无融合  : 读 R 三次 + 写 R' 一次       = 4 趟 × 40 KB/token
      P+C+norm: 写 R' 时顺手算 Σ pre·R' + norm = 3 趟   ← 主线（HcNormed）
      P+C+M   : 连 H·R' 也在同一 kernel 里   = 2 趟   ← gfx95（HcPending）

  每一级都要一个前提，这就是后面两个"机制"存在的原因：
      · 想做 P+C+norm  → 需要知道"下游那个 norm 折不折得动"
                          ⇒ HcNextBoundary（能力描述符）
      · 想做 P+C+M     → 需要"把 post 推迟"，否则 M 的输入 R' 还没落地
                          ⇒ HcPending（推迟句柄）

======================================================================
流程图：V4.1 的融合到底发生在哪
======================================================================
  记号（比 pre/attn/post 更细，因为"用 pre"和"算 pre"是两步）：
    M  mix stats : R → (pre, post, comb)      H·R + RMS + σ/Sinkhorn(20)
    C  combine   : norm(Σ pre·R) → 子层输入
    S  sublayer  : attn 或 ffn
    P  post      : R' = post·y + comb·R

  ── V4 Pro（无错位）：每步都串行，M 卡在关键路径 ──────────────────
       M_k ─▶ C_k ─▶ S_k ─▶ P_k ─▶ M_{k+1} ─▶ C_{k+1} ─▶ …
       （C_k 必须等 M_k 的 pre_k）

  ── V4.1（错位）：M 走侧流，P_k 顺带做 C_{k+1} ────────────────────
       main : C_k ─────▶ S_k ─────▶ P_k ─▶ C_{k+1} ─▶ S_{k+1} ─▶ …
       side :        M_k(R_k)              M_{k+1}(R_{k+1})
                     └ 与 S_k 并行         └ 与 S_{k+1} 并行
       ↑ C_k 用的是 pre_{k-1}（上一层留下的），所以不必等 M_k

  ── 一个层内的实际形状（融合边界用 ═ 标出）──────────────────────
     ┌─ layer i ─────────────────────────────────────────────────┐
     │  C_attn(用 pre_prev) ─▶ MQALayer ─▶ P_attn ═▶ C_ffn+norm   │
     │        ▲                    MQALayer 内部含               │
     │        │                    KV 写 + 稀疏注意力 + wo_a       │
     │        │                                                   │
     │   M_attn(侧流, 与 MQALayer 并行)                            │
     │                                                            │
     │  ── 下面是 ffn 段 ──                                        │
     │  C_ffn 已在上一行的 ═ 里做完，这里直接拿去用 → S_ffn(MoE)    │
     │  ─▶ P_ffn ═▶ C_{下一层 attn}+norm                           │
     │        │                                                   │
     │   M_ffn(侧流, 与 MoE 并行)                                  │
     └────────────────────────────────────────────────────────────┘

  ── 六个融合点，逐一标注在这张图上 ──────────────────────────────
     ① M 内部三合一      H·R(GEMM) + RMS + σ门控 + Sinkhorn(20 轮)
                          → mhc_pre / (HCU: rmsnorm_with_sinkhorn)
     ② C 内部二合一      加权求和 + RMSNorm
                          → hc_combine_norm
     ③ P 内部二合一      外积(post·y) + 流间混合(comb·R)
                          → mhc_post
     ④ P_k ═ C_{k+1} + norm     ★ V4.1 特有的跨子层融合
                          写完 R_{k+1} 立刻在同一 kernel 里做
                          norm(Σ pre_k·R_{k+1})，返回 HcNormed；
                          下一子层的 C 直接短路，一次 R_{k+1} 读写都不重复。
                          → mhc_post_combine_norm_prefill（blackwell）
                          → all_reduce_mhc_post_combine_norm（与 all-reduce 合并）
     ⑤ P + 集合通信      注意力侧：all-reduce 的 epilogue 里做 P
                          MoE 侧  ：finalize + shared add + all-reduce + P + C + norm
                          （+量化）→ moe_finalize_all_reduce_mhc_post_combine_norm_quant
     ⑥ 跨层 deferred     本层 P 推迟进下一层边界 kernel（gfx95）
                          → hc_boundary_fused_deferred

     注：④ 的 seam 分两种，对应 _init_boundaries 建的两个 boundary
         local_boundary : ATTN[i] → FFN[i]      （层内）
         next_boundary  : FFN[i]  → ATTN[i+1]   （跨层；Engram 层/尾段收窄时关闭）

  ★ 为什么"post + 下一层 combine + norm"能融合、而"post + 下一层 pre"不能：
      · 能：post 写完 R_{k+1} 之后，下一层的 combine 只需要
            norm(Σ pre_k·R_{k+1}) —— pre_k 此刻已经知道，R_{k+1} 就在手里，
            于是"写 R + 压流 + norm"可以一次做完（HcNormed / mhc_post_combine_norm_prefill）。
      · 不能：pre_{k+1} 要用 R_{k+1} 做一次 24×20480 的 GEMM + Sinkhorn，
            那是下一轮 mix stats 的活；硬塞进 post 的 kernel 里既拿不到
            H 的分块，也破坏了"① 与 ③ 并行"这个设计前提。

======================================================================
四步各自的融合清单（V4.1 到底有哪些融合，按生效范围排）
======================================================================
  ① mix stats（步骤 1）—— H·R + RMS + Sinkhorn(20 轮) + 门控 → (pre, post, comb)
       通用  : mhc_pre（kernels/ops/layernorm/mhc.py:1243）
       TileLang 变体 : mhc_pre_big_fuse_tilelang / mhc_pre_big_fuse_with_norm_tilelang
                       （连下一层 norm 一起做）
       HCU   : rmsnorm_with_sinkhorn / hc_mix_reduce_sinkhorn_vec
               （mhc_boundary_hip.py；条件 _use_hcu_hc_mix_stats =
                 is_hcu and SGLANG_USE_LIGHTOP and SGLANG_HCU_HC_MIX_STATS_OPT，
                 且权重已 split 成 tf32/bf16 两半，shape (24, 20480)）
       ★ 另有 stats_stream：把步骤 1 放独立流与 2~3 重叠（use_stats_stream）。
         ★ 但侧流本身只在 CUDA 且 blackwell/sm90 时创建
           （deepseek_v4.py: hc_stats_stream = Stream() if _is_cuda and
             (is_blackwell or is_sm90) and SGLANG_OPT_USE_MULTI_STREAM_OVERLAP
             and hc_pre_from_prev else None）。
           **HCU 上 hc_stats_stream 是 None** —— use_stats_stream 即使返回 True，
           下面把 stats_stream 赋成 None，所有 wait/record 都退化成 no-op，
           M 仍在主流程里内联执行（串在 S 之后、P 之前）。
           HCU 的补偿是 ① 本身走 lightop 融合 kernel（Mega mHC），
           而不是靠并行掩盖。

  ② combine（步骤 2）—— Σ_m pre[m]·R[m] + RMSNorm（+ 可选量化/打散）
       hc_combine（elementwise/hc_combine.py）/ hc_combine_norm
       （layernorm/hc_combine_norm.py）
       ★ 若上一个 post 已返回 HcNormed/HcQuantized，combine 直接短路，什么都不算

  ③ post（步骤 4）—— R'[i] = post[i]·y + Σ_j comb[j][i]·R[j]
       mhc_post / mhc_post_tilelang（mhc.py）
       mhc_post_split_h（gfx95，HC==4，小批按 hidden 切 CTA）

  ④ post + 下一层 combine + norm
       mhc_post_combine_norm_prefill（layernorm/...py）
       条件：can_fuse_post → blackwell + pre_from_prev + mult==4 + hidden==5120
             + attn_dp_size==1（另加行数 4096..65536）

  ⑤ post + combine + norm + TP all-reduce（注意力侧）
       all_reduce_mhc_post_combine_norm（communication/all_reduce_mhc.py）
       条件：_can_fuse_attn_mhc → tp_size == attn_tp_size == 4 + can_fuse_post

  ⑥ MoE finalize + all-reduce + post + combine + norm（+ 量化）
       moe_finalize_all_reduce_mhc_post_combine_norm[_quant]
       条件：_can_fuse_ffn_mhc → mlp.tp_size == 4 + 无 A2A + can_fuse_post
             （quant 版还要求 expert_weights.shape[0] <= 128）

  ⑦ 跨层 deferred post（把本层 post 推迟进下一层边界 kernel）
       hc_boundary_fused_deferred（mhc_boundary_hip.py；gfx95）
       条件：_is_gfx95_supported + hc_pre_from_prev + hc_mult == 4
             （见 deepseek_v4.py 块 11；defer_post 遇到 Engram/捕获层会放弃）

  ⑧ 出口 hc_head —— RMSNorm + F.linear(hc_fn) + 混合 + 最终 norm
       mhc_head.py（layernorm/mhc_head.py）

  ⑨ 入口 hc_broadcast（[T,H] → [T,hc_mult,H]）；
     HCU 的 CP 推迟路径可与切片融合（lightop mhc_repeat_cp_sglang）

  ⑩ 跨层"post + 下一层 pre"（V4 Pro 的 mhc_fused_post_pre）
       ★ 对 V4.1【结构性不可表达】：本模块 docstring 第 12-13 行说的 lag 就是它
         （pre 由上一个 post 产出，边界处已无 pre 可算）。所以
         use_fused_mhc_post_pre 被置 False，这不是平台缺失，是语义冲突。

TODO: move dsv4 MHC into the same abstraction
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple, Optional, Tuple, TypeAlias, Union

import msgspec
import torch
import torch.nn.functional as F

from sglang.kernels.ops.layernorm.mhc_post_split_h import mhc_post_split_h
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.utils import is_npu_arch35
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.quantization.mxfp8_input import Mxfp8SwizzledInput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.runner import compile_in_capture_mode
from sglang.srt.models.deepseek_v2 import MoEOutput, _is_hip, _is_npu, _is_xpu
from sglang.srt.runtime_context import get_parallel, get_platform
from sglang.srt.utils import get_bool_env_var, is_gfx95_supported, is_hcu

_is_hcu = is_hcu()
_use_lightop = get_bool_env_var("SGLANG_USE_LIGHTOP")
_hc_mix_stats_opt = envs.SGLANG_HCU_HC_MIX_STATS_OPT.get()
_use_hcu_hc_mix_stats = _is_hcu and _use_lightop and _hc_mix_stats_opt
_use_aiter_tilelang_mhc = get_bool_env_var(
    "SGLANG_ROCM_USE_AITER_TILELANG_MHC", default="false"
)

_is_gfx95_supported = is_gfx95_supported()

# ---------------------------------------------------------------------------
# What a layer hands this module
# ---------------------------------------------------------------------------

HcTriplet = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class HcConfig(NamedTuple):
    """Everything about a layer's mHC that is fixed for the model's lifetime."""

    mult: int
    sinkhorn_iters: int
    eps: float
    rms_eps: float
    hidden: int
    pre_from_prev: bool
    cp_prefill: bool


class HcSubLayer(NamedTuple):
    """One hyper-connection's fixed material. Holds the layer's Parameters by
    reference; a plain tuple registers nothing, so checkpoint keys stay put."""

    cfg: HcConfig
    fn: torch.Tensor
    scale: torch.Tensor
    base: torch.Tensor
    norm: RMSNorm
    tf32_parts: Optional[Any] = None
    bf16_parts: Optional[Any] = None


# ---------------------------------------------------------------------------
# How far the previous hyper-connection got toward the next sublayer's input
# ---------------------------------------------------------------------------


class HcNormed(NamedTuple):
    """Summed and normed; the sublayer can take these rows as they are."""

    rows: torch.Tensor


class HcQuantized(NamedTuple):
    """Normed, and already swizzled for the sublayer's first projection."""

    rows: torch.Tensor
    swizzled: Mxfp8SwizzledInput


HcPreOutput: TypeAlias = Union[HcNormed, HcQuantized]
"""The next sublayer's input, built by the previous post; a receiver skips step 2
(and ``pre``) entirely. Always normed -- combine and norm travel together."""


class AttnOutput(NamedTuple):
    """Attention's wo_b rows with the TP reduction left to the fused post."""

    partial: torch.Tensor


# ---------------------------------------------------------------------------
# The real parameters: what crosses from one hyper-connection to the next
# ---------------------------------------------------------------------------


class HcPending(NamedTuple):
    """A post not run yet; the residual it would produce is implied. The backward
    counterpart of `HcPreOutput`: the HIP boundary kernel takes both and runs the
    previous post, this triplet and this collapse in one launch."""

    rows: torch.Tensor
    residual: torch.Tensor
    post: torch.Tensor
    comb: torch.Tensor


class HcState(msgspec.Struct):
    """The mHC state of the stream between two hyper-connections.

    ``streams`` is the residual ``R``, or -- out of the HIP fused boundary -- the
    post that would rebuild it, still outstanding. Mutable for one reason:
    `release` must empty a consumed state in every frame that still binds it.
    """

    streams: Union[torch.Tensor, HcPending, None]
    pre: Optional[torch.Tensor] = None
    input: Optional[HcPreOutput] = None

    def release(self) -> None:
        """Empty the state at its last reader, the combine; the residual the post
        still needs lives on through the caller's alias. Stale holders -- above
        all the model loop's binding across the layer call -- then stop pinning
        the dead residual through the next sublayer."""
        self.streams = None
        self.pre = None
        self.input = None

    @property
    def residual(self) -> torch.Tensor:
        """The materialized streams; run an outstanding post first (`materialized`)."""
        assert isinstance(self.streams, torch.Tensor)
        return self.streams

    def materialized(self, cfg: HcConfig) -> HcState:
        """Run the outstanding post, if any; residual readers call this first."""
        if not isinstance(self.streams, HcPending):
            return self
        p = self.streams
        return HcState(post(cfg, p.rows, p.residual, p.post, p.comb), self.pre)

    def with_residual(self, residual: torch.Tensor) -> HcState:
        """The residual was rewritten: drop ``input``; ``pre`` weights streams, not
        rows, so it survives."""
        return HcState(residual, self.pre)

    def take_rows(self, rows: Callable[[torch.Tensor], torch.Tensor]) -> HcState:
        """Keep only the rows ``rows`` selects -- the late-layer tail's narrowing."""
        pre = None if self.pre is None else rows(self.pre)
        assert self.streams is not None
        if isinstance(self.streams, HcPending):
            return HcState(HcPending(*(rows(t) for t in self.streams)), pre)
        else:
            return HcState(rows(self.streams), pre)


# ---------------------------------------------------------------------------
# The plan: how a hyper-connection intends to run its post
# ---------------------------------------------------------------------------


class HcNextBoundary(NamedTuple):
    """What the next hyper-connection can take off this one's post; resolved after
    weights load. None where the seam is closed (last layer, Engram).

    ==================================================================
    "boundary"（接缝）到底是什么
    ==================================================================
    它不是一条线，而是**两个子层之间的交接点**：上一个子层的 post
    产出 R'，下一个子层需要 norm(Σ pre·R')。这个交接点上有三件事可以
    被上游顺带做掉，能不能做取决于下面三个字段。

    一次 forward 里每层有两种 boundary（都在 _init_boundaries 里建）：
        local_boundary : ATTN[i] → FFN[i]       norm = post_attention_layernorm
        next_boundary  : FFN[i]  → ATTN[i+1]    norm = 下一层的 input_layernorm
    命名里的 next 是"下一个子层"，不是"下一层"——后者是 next_boundary 的特例。

    三个字段各自的用途：
        norm          上游 kernel 要把 norm 折进去，需要它的
                      weight 与 variance_epsilon（RMSNorm 的两个输入）。
        norm_fusable  这个 norm 折不折得动（见 make_boundary 的五项检查）。
                      False 时上游只做纯 post，下游自己算 combine+norm。
        accepts_mxfp8 下游是否接受 HcQuantized（normed + swizzled）。
                      True 时上游连"量化+打散"一起做，省一次量化 kernel；
                      False 时只返回到 HcNormed 为止。

    三个消费者（传进去的 next 就是这里的对象）：
        _post_fusion                → mhc_post_combine_norm_prefill
        run_attn_post(AttnOutput)   → all_reduce_mhc_post_combine_norm
        run_moe_post(deferred)      → moe_finalize_all_reduce_mhc_post_combine_norm[_quant]
                                      （_quant 版由 accepts_mxfp8 决定）

    seam 关闭的四级（严重程度递增）：
        ① 下一层挂 Engram / 末层 → bind_next 不绑 → next_boundary is None
        ② 尾段要把行收窄     → run_ffn_hc 里 seam_open=False → 传 None
        ③ norm 不可融合       → norm_fusable=False（seam 在，只是退化成纯 post）
        ④ 下游不吃打散输入     → accepts_mxfp8=False（只到 HcNormed）

    生命周期：
        bind_next(layers[i+1])          模型 build 完成后调用一次（记 _next_layer）
        _init_boundaries()              首次 forward（accepts_mxfp8 要等权重转换）
        refresh_mhc_norm_weight_cache() 权重变更后清空两者，下次 forward 重建

    ★ 与 HcPending（gfx95）的区别：
        本类 + HcNormed 是**数据流融合**：post 顺手把 combine+norm 算出来。
        HcPending 是**控制流融合**：把 post 整个推迟，让下一个边界 kernel
        一次做"上个 post + 本次 combine + 本次 mixing stats"，再把
        reduce/sinkhorn 挂在紧随的 RMSNorm launch 上（HcCoefficients）。
    """

    norm: RMSNorm
    accepts_mxfp8: bool
    norm_fusable: bool


def is_deferred_finalize(routed) -> bool:
    """Whether a `MoEOutput` really carries an unfinalized handle for the post."""
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        FlashInferTrtllmDeferredFinalizeOutput,
    )

    return isinstance(routed, FlashInferTrtllmDeferredFinalizeOutput)


def make_boundary(norm: RMSNorm, *, accepts_mxfp8: bool) -> HcNextBoundary:
    """构造"接缝能力描述符"：告诉上一个 post，它能在多大程度上顺手把下一层也做了。

    ★ 解决什么问题
      两个子层之间的接缝上，本来要串行做：post 写回残差 → 下一个子层 combine
      （加权求和）+ norm +（可选）量化。这些都能合进 post 的那个 kernel，
      但**不是总成立** —— 取决于下一层 norm 的形态、以及它的第一个投影
      能不能吃"已打散"的 MXFP8 输入。把判断结果一次性算好、存成三个字段，
      上游 post 就不用每层每批重新判断。

    ★ 三个字段各管一段融合
      norm         下一个子层的 RMSNorm 对象 —— 上游要它的 weight 与
                   variance_epsilon 才能把 norm 折进自己的 kernel。
      norm_fusable 这个 norm 能不能折（见下面五项检查）。False 时 post 只做
                   自己那步，下一个 combine 自己去算 norm。
      accepts_mxfp8 下一个子层是否接受 HcQuantized（normed + swizzled）。
                   True 时 post 的 kernel 连"量化+打散"一起做，
                   返回 HcQuantized，下一个 combine 直接拿去喂第一个 GEMM；
                   False 时只返回到 HcNormed 为止。

    ★ 为什么 norm_fusable 要查这五项（每一项都对应一个 kernel 内联不了的形态）
        cast_x_before_out_mul   要在 out_mul 之前 cast，融合 kernel 里没这条路径
        variance_size_override  方差维度被改写，kernel 只支持标准 (…, 5120)
        weight.dtype == bf16    融合 kernel 的 ABI 是 bf16
        shape == (5120,)        当前实现只做了这个宽度（见 TODO）
        is_contiguous           要按连续地址读 weight

    ★ 何时调用（两次时机都很关键）
      构造后首次 forward 的 _init_boundaries 里：因为 accepts_mxfp8 依赖
      process_weights_after_loading 的结果（量化转换会改变"能否吃打散输入"）。
    """
    return HcNextBoundary(
        norm=norm,
        accepts_mxfp8=accepts_mxfp8,
        norm_fusable=(
            not norm.cast_x_before_out_mul
            and norm.variance_size_override is None
            and norm.weight.dtype == torch.bfloat16
            # TODO: support other norm shape
            and norm.weight.shape == (5120,)
            and norm.weight.is_contiguous()
        ),
    )


# ---------------------------------------------------------------------------
# Step 1 -- the mix GEMM and the triplet it splits into
# ---------------------------------------------------------------------------


def use_stats_stream(cfg: HcConfig, forward_batch: ForwardBatch, x: torch.Tensor):
    if not cfg.pre_from_prev:
        return False
    decode_like = forward_batch.forward_mode.is_decode() or (
        forward_batch.forward_mode.is_target_verify() and x.shape[0] > 0
    )
    return decode_like and (
        not get_platform().is_sm90
        or x.shape[0] == 1
        or (forward_batch.forward_mode.is_decode() and 1 < x.shape[0] <= 64)
    )


def mix_stats(
    hc: HcSubLayer, x: torch.Tensor, stats_stream: Optional[torch.cuda.Stream] = None
) -> HcTriplet:
    """Predict the triplet on the caller's stream or the stats stream."""
    if stats_stream is None:
        return _mix_stats_impl(hc, x)
    main_stream = torch.cuda.current_stream()
    x.record_stream(stats_stream)
    with torch.cuda.stream(stats_stream):
        coefficients = _mix_stats_impl(hc, x)
    for coefficient in coefficients:
        coefficient.record_stream(main_stream)
    return coefficients


def _mix_stats_impl(hc: HcSubLayer, x: torch.Tensor) -> HcTriplet:
    from sglang.kernels.ops.layernorm.mhc import (
        hc_mix_stats,
        hc_mix_stats_sinkhorn,
        hc_split_sinkhorn,
    )

    cfg = hc.cfg

    x_flat = x.flatten(1)

    from sglang.srt.batch_invariant_ops import (
        is_batch_invariant_mode_enabled,
    )

    parts = bf16_parts = None
    hopper_medium = get_platform().is_sm90 and 32 <= x_flat.shape[0] < 4096
    if (
        x.is_cuda
        and (x_flat.shape[0] >= 128 or hopper_medium)
        and x_flat.is_contiguous()
        and (get_platform().is_sm100 or get_platform().is_sm90)
        and envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.get()
        and not is_batch_invariant_mode_enabled()
    ):
        parts, bf16_parts = hc.tf32_parts, hc.bf16_parts

    use_bf16_projection = bf16_parts is not None and (
        hopper_medium or 4096 <= x_flat.shape[0] <= 65536
    )
    hopper_fused_stats = get_platform().is_sm90 and (
        x.shape[0] == 1 or (bf16_parts is not None and 32 <= x.shape[0] <= 65536)
    )
    # gfx950 uses the fused Triton port at every row count (MI350X).
    use_fused_stats = _is_gfx95_supported or (
        torch.version.cuda is not None
        and (get_platform().is_blackwell or hopper_fused_stats)
    )
    if x.is_cuda and use_fused_stats and x.dtype == torch.bfloat16:
        # The default split-K/Sinkhorn fusion preserves batch invariance;
        # compensated projections above are disabled in batch-invariant mode.
        if use_bf16_projection:
            from sglang.kernels.ops.layernorm.mhc import (
                hc_mix_stats_sinkhorn_bf16x3,
            )

            pre, post, comb = hc_mix_stats_sinkhorn_bf16x3(
                x_flat,
                bf16_parts,
                hc.scale,
                hc.base,
                cfg.sinkhorn_iters,
                cfg.rms_eps,
                cfg.eps,
            )
        elif parts is not None:
            from sglang.kernels.ops.layernorm.mhc import (
                hc_mix_stats_sinkhorn_deepgemm,
            )

            pre, post, comb = hc_mix_stats_sinkhorn_deepgemm(
                x_flat,
                parts,
                hc.scale,
                hc.base,
                cfg.sinkhorn_iters,
                cfg.rms_eps,
                cfg.eps,
            )
        else:
            pre, post, comb = hc_mix_stats_sinkhorn(
                x_flat,
                hc.fn,
                hc.scale,
                hc.base,
                cfg.mult,
                cfg.sinkhorn_iters,
                cfg.rms_eps,
                cfg.eps,
            )
        return pre, post, comb
    if (
        _use_hcu_hc_mix_stats
        and x_flat.dtype == torch.bfloat16
        and x_flat.is_contiguous()
        and not is_batch_invariant_mode_enabled()
    ):
        parts, bf16_parts = hc.tf32_parts, hc.bf16_parts
        use_bf16x3 = 4096 <= x_flat.shape[0] <= 65536
        if parts is not None and (not use_bf16x3 or bf16_parts is not None):
            from lightop import (
                hc_mix_stats_sinkhorn_bf16x3,
                hc_mix_stats_sinkhorn_deepgemm,
            )

            if use_bf16x3:
                return hc_mix_stats_sinkhorn_bf16x3(
                    x_flat,
                    bf16_parts,
                    hc.scale,
                    hc.base,
                    cfg.sinkhorn_iters,
                    cfg.rms_eps,
                    cfg.eps,
                )
            return hc_mix_stats_sinkhorn_deepgemm(
                x_flat,
                parts,
                hc.scale,
                hc.base,
                cfg.sinkhorn_iters,
                cfg.rms_eps,
                cfg.eps,
            )
    if x.is_cuda and torch.version.cuda is not None:
        # cuBLAS/torch reductions can change order with num_tokens; this kernel
        # keeps the mixing and RMS reductions batch-invariant.
        mixes = hc_mix_stats(x_flat, hc.fn, cfg.rms_eps).unsqueeze(1)
    else:
        x_flat = x_flat.float()
        rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + cfg.rms_eps)
        mixes = (F.linear(x_flat, hc.fn) * rsqrt).unsqueeze(1)
    if _is_xpu:
        from sglang.srt.models.deepseek_v4 import get_mhc_ops

        split_sinkhorn = get_mhc_ops().hc_split_sinkhorn
    else:
        split_sinkhorn = hc_split_sinkhorn
    pre, post, comb = split_sinkhorn(
        mixes,
        hc.scale,
        hc.base,
        cfg.mult,
        cfg.sinkhorn_iters,
        cfg.eps,
    )
    return pre.squeeze(1), post.squeeze(1), comb.squeeze(1)


# ---------------------------------------------------------------------------
# Step 2 -- collapse the streams and normalize
# ---------------------------------------------------------------------------


def fork_stats_stream(stats_stream: Optional[torch.cuda.Stream]) -> None:
    """Let the side stream pick up from here; the layers call it right before
    each combine (fork position measured indistinguishable across batch sizes)."""
    if stats_stream is not None:
        stats_stream.wait_stream(torch.cuda.current_stream())


def combine(
    hc: HcSubLayer,
    state: HcState,
    quantized: Optional[list] = None,
) -> torch.Tensor:
    """The sublayer's input: ``norm(sum_k pre[k] * R[k])``, short-circuited by
    whatever the previous post left in ``state.input``."""
    shortcut = state.input
    if isinstance(shortcut, HcQuantized):
        assert quantized is not None
        quantized.append(shortcut.swizzled)
        return shortcut.rows
    if isinstance(shortcut, HcNormed):
        return shortcut.rows
    return _combine(hc, state, quantized)


def _combine(
    hc: HcSubLayer,
    state: HcState,
    quantized: Optional[list],
) -> torch.Tensor:
    """The combine computed from the residual's rows."""
    from sglang.kernels.ops.layernorm.mhc import hc_combine

    cfg, norm = hc.cfg, hc.norm
    x, apply_pre = state.residual, state.pre
    quantize = quantized is not None
    x_flat = x.flatten(1)

    if apply_pre is None:
        # 只有【第一个子层】会走到这里：错位设计下"上一个 A"还不存在。
        #   X_0 = hc_broadcast(e) 的 4 行【完全相同】，所以
        #       A·X_0 = (Σ_k a_k)·e   —— 只是一个标量缩放
        #   而 RMSNorm 对输入整体缩放不变，于是
        #       norm(A·X_0) ≡ norm(e) = norm(X_0[:, 0, :])
        #   完全等价，不是近似。省掉一次 4 行读 + 一次加权求和。
        return norm(x[:, 0, :])
    from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled

    hopper_fused = (
        get_platform().is_sm90
        and not quantize
        and (0 < x.shape[0] <= 96 or 4096 <= x.shape[0] <= 65536)
        and not is_batch_invariant_mode_enabled()
    )
    if (
        x.is_cuda
        and (get_platform().is_blackwell or hopper_fused)
        and x.dtype == torch.bfloat16
        and apply_pre.stride(1) == 1
        and cfg.mult == 4
        and cfg.hidden == 5120
        and norm.weight.dtype == torch.bfloat16
        and not norm.cast_x_before_out_mul
        and norm.variance_size_override is None
    ):
        # One fused form for every row count. Each row's reduction order is
        # fixed regardless of the grid split, so this holds under
        # batch-invariant mode too.
        if quantize and x.shape[0] <= 128:
            from sglang.kernels.ops.layernorm.hc_combine_norm import (
                hc_combine_norm_mxfp8,
            )

            y, y_q, y_sf = hc_combine_norm_mxfp8(
                x_flat, apply_pre, norm.weight, norm.variance_epsilon
            )
            quantized.append(Mxfp8SwizzledInput(y_q, y_sf))
            return y
        from sglang.kernels.ops.layernorm.hc_combine_norm import hc_combine_norm

        return hc_combine_norm(x_flat, apply_pre, norm.weight, norm.variance_epsilon)
    return norm(hc_combine(x_flat, apply_pre, cfg.mult, x.dtype))


# ---------------------------------------------------------------------------
# Step 4 -- rebuild the streams, optionally collapsing the next input too
# ---------------------------------------------------------------------------


def post(
    cfg: HcConfig,
    x: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    """``R'[i] = post[i] * x + sum_j comb[j][i] * R[j]``, by whichever kernel fits."""
    if x.shape[0] == 0:
        return torch.empty((0, cfg.mult, x.shape[-1]), dtype=x.dtype, device=x.device)

    if _is_npu:
        if not is_npu_arch35():
            return torch.ops.custom.npu_hc_post(x, residual, post_mix, comb)
        # The A5 build of npu_hc_post is batched -- it requires a leading batch axis
        # on every operand.
        return torch.ops.custom.npu_hc_post(
            x.unsqueeze(0),
            residual.unsqueeze(0),
            post_mix.unsqueeze(0),
            comb.unsqueeze(0),
        ).squeeze(0)

    if _is_xpu:
        from sglang.srt.models.deepseek_v4 import get_mhc_ops

        return get_mhc_ops().mhc_post(x, residual, post_mix, comb)

    if (
        get_platform().is_blackwell
        and cfg.pre_from_prev
        and cfg.mult == 4
        and cfg.hidden == 5120
        # This 384 is the kernel's own tile limit, unrelated to the collective's
        # push slots; it covers every narrow batch, not one particular regime.
        and x.shape[0] <= 384
    ):
        return mhc_post_split_h(x, residual, post_mix, comb)

    if (
        _is_hcu
        and _use_aiter_tilelang_mhc
        and envs.SGLANG_OPT_USE_TILELANG_MHC_POST.get()
    ):
        from aiter.ops.tilelang import mhc_post_fwd

        return mhc_post_fwd(x, residual, post_mix, comb)

    if _is_hip:
        from sglang.srt.models.deepseek_common.amd import deepseek_v4_hip as _hip

        y = _hip.hc_post(cfg, x, residual, post_mix, comb)
        if y is not None:
            return y

    if envs.SGLANG_OPT_USE_FLASHINFER_MHC.get():
        from flashinfer.mhc import mhc_post

        return mhc_post(x, residual, post_mix, comb)

    if envs.SGLANG_OPT_USE_TILELANG_MHC_POST.get():
        if (
            cfg.pre_from_prev
            and get_platform().is_sm90
            and cfg.mult == 4
            and cfg.hidden == 5120
            and 1 <= x.shape[0] <= 64
        ):
            return mhc_post_split_h(x, residual, post_mix, comb)

        from sglang.kernels.ops.layernorm.mhc import mhc_post

        return mhc_post(x, residual, post_mix, comb)

    if _is_hip:
        from aiter.ops.mhc import mhc_post

        result = torch.empty_like(residual)
        mhc_post(result, x, residual, post_mix, comb)
        return result

    assert residual.shape == (x.shape[0], cfg.mult, x.shape[-1])
    assert post_mix.shape == (x.shape[0], cfg.mult)
    assert comb.shape == (x.shape[0], cfg.mult, cfg.mult)

    @compile_in_capture_mode
    def post_torch_impl(x, residual, post_mix, comb):
        return (
            post_mix.unsqueeze(-1) * x.unsqueeze(1)
            + (comb.unsqueeze(-1) * residual.unsqueeze(2)).sum(dim=1)
        ).type_as(x)

    return post_torch_impl(x, residual, post_mix, comb)


# ---------------------------------------------------------------------------
# Planning the post
# ---------------------------------------------------------------------------


def can_fuse_post(cfg: HcConfig) -> bool:
    """Whether a fused post is expressible at all for this model, device and
    parallel layout.

    The capability half of the decision; which token ranges should take it, and what
    batch-invariant mode should do about it, are the caller's policy.
    """
    return (
        get_platform().is_blackwell
        and cfg.pre_from_prev
        and cfg.mult == 4
        and cfg.hidden == 5120
        and get_parallel().attn_dp_size == 1
        and not cfg.cp_prefill
    )


# ---------------------------------------------------------------------------
# Running the post
# ---------------------------------------------------------------------------


def _compute_triplet(
    hc: HcSubLayer,
    residual: torch.Tensor,
    stats_stream: Optional[torch.cuda.Stream],
) -> HcTriplet:
    """Issue the triplet (on the side stream, beside the sublayer) and join before
    the post reads it."""
    coefficients = mix_stats(hc, residual, stats_stream)
    if stats_stream is not None:
        torch.cuda.current_stream().wait_stream(stats_stream)
    return coefficients


def _post_fusion(
    hc: HcSubLayer,
    y: torch.Tensor,
    residual: torch.Tensor,
    coefficients: HcTriplet,
    next: Optional[HcNextBoundary],
) -> HcState:
    """Step 4 without a collective: the wide-tile kernel folds the next combine +
    norm in where it serves this seam, otherwise the pure post runs and the next
    combine computes itself."""
    cfg = hc.cfg
    pre, post_mix, comb = coefficients
    if (
        next is not None
        and next.norm_fusable
        and cfg.pre_from_prev
        and get_platform().is_blackwell
        and 4096 <= y.shape[0] <= 65536
        and cfg.hidden == 5120
        and cfg.mult == 4
        and get_parallel().attn_dp_size == 1
        and not cfg.cp_prefill
    ):
        from sglang.kernels.ops.layernorm.mhc_post_combine_norm_prefill import (
            mhc_post_combine_norm_prefill,
        )

        updated, normalized = mhc_post_combine_norm_prefill(
            y,
            residual,
            post_mix,
            comb,
            pre,
            next.norm.weight,
            next.norm.variance_epsilon,
        )
        return HcState(updated, pre, HcNormed(normalized))
    return HcState(post(cfg, y, residual, post_mix, comb), pre)


def run_attn_post(
    hc: HcSubLayer,
    out: Union[torch.Tensor, AttnOutput],
    residual: torch.Tensor,
    *,
    stats_stream: Optional[torch.cuda.Stream],
    next: Optional[HcNextBoundary],
    world_size: int,
) -> HcState:
    """The attention post. An `AttnOutput` rides the collective kernel (which also
    folds ``next``'s norm); the attention may decline the handover even when asked,
    so the type is the ground truth."""
    if isinstance(out, AttnOutput):
        from sglang.kernels.ops.communication.all_reduce_mhc import (
            all_reduce_mhc_post_combine_norm,
        )

        assert next is not None
        pre, post_mix, comb = _compute_triplet(hc, residual, stats_stream)
        _, updated, normalized = all_reduce_mhc_post_combine_norm(
            out.partial,
            residual,
            post_mix,
            comb,
            pre,
            next.norm.weight,
            next.norm.variance_epsilon,
            world_size=world_size,
        )
        return HcState(updated, pre, HcNormed(normalized))
    coefficients = _compute_triplet(hc, residual, stats_stream)
    return _post_fusion(hc, out, residual, coefficients, next)


def run_moe_post(
    hc: HcSubLayer,
    out: Union[torch.Tensor, MoEOutput],
    residual: torch.Tensor,
    *,
    stats_stream: Optional[torch.cuda.Stream],
    next: Optional[HcNextBoundary],
    world_size: int,
) -> HcState:
    """The MoE post. A deferred finalize the push plane can carry rides the
    collective kernel (finalize + shared add + all-reduce, quantizing ``next``'s
    input when a boundary is given); anything else is finalized here and takes
    the plain post."""
    from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
        can_fuse_all_reduce,
    )

    if (
        isinstance(out, MoEOutput)
        and is_deferred_finalize(out.routed)
        # expert_weights rows are the true token count; gemm2_out is the expanded
        # [T x top_k (padded)] view and overstates the plane load by ~6x.
        and can_fuse_all_reduce(out.routed.expert_weights.shape[0], hc.cfg.hidden)
    ):
        pre, post_mix, comb = _compute_triplet(hc, residual, stats_stream)
        args = (
            out.routed.gemm2_out,
            out.routed.expanded_idx_to_permuted_idx,
            out.routed.expert_weights,
            out.routed.top_k,
            out.shared,
            residual,
            post_mix,
            comb,
        )
        if (
            next is not None
            and next.accepts_mxfp8
            and next.norm_fusable
            # The quant epilogue supports at most 128 batch size
            and out.routed.expert_weights.shape[0] <= 128
        ):
            from sglang.kernels.ops.communication.all_reduce_mhc import (
                moe_finalize_all_reduce_mhc_post_combine_norm_quant,
            )

            _, updated, normalized, q, sf = (
                moe_finalize_all_reduce_mhc_post_combine_norm_quant(
                    *args,
                    pre,
                    next.norm.weight,
                    next.norm.variance_epsilon,
                    world_size=world_size,
                )
            )
            return HcState(
                updated, pre, HcQuantized(normalized, Mxfp8SwizzledInput(q, sf))
            )
        from sglang.kernels.ops.communication.all_reduce_mhc import (
            moe_finalize_all_reduce_mhc_post,
        )

        _, updated = moe_finalize_all_reduce_mhc_post(*args, world_size=world_size)
        return HcState(updated, pre)
    if isinstance(out, MoEOutput):
        from sglang.srt.layers.moe import post_experts_all_reduce
        from sglang.srt.layers.moe.utils import should_add_replicated_moe_output

        # The post boundary owns the sum; a replicated shared expert is added
        # after the reduction so it is counted only once across TP ranks.
        pieces = out
        out = post_experts_all_reduce(pieces.get_merged())
        if (
            pieces.shared is not None
            and pieces.shared_is_replicated
            and should_add_replicated_moe_output()
        ):
            out += pieces.shared
    coefficients = _compute_triplet(hc, residual, stats_stream)
    return _post_fusion(hc, out, residual, coefficients, next)
