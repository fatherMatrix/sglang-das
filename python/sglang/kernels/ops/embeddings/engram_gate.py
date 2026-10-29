"""Fused FP32 Engram gate with a single final cast to the activation dtype."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _engram_gate_kernel(
    X,
    KV,
    QW,
    KW,
    O,
    D: tl.constexpr,
    HC: tl.constexpr,
    EPS: tl.constexpr,
    CLAMP: tl.constexpr,
    B: tl.constexpr,
):
    """一行 = 一个 (token, hc 通道) 对；公式（全是 fp32）：

        x   = X[t, c, :]                  ← 残差 state.residual 的第 c 个通道
        key = KV[t, c, :]                 ← 该通道专属的 key
        val = KV[t, HC, :]                ← 【所有通道共享】的 value
        w   = q_weight[c] * k_weight[c]   ← 逐维门控权重（可学的静态参数）

        rstd = 1/sqrt(mean(x^2)+eps) * 1/sqrt(mean(key^2)+eps)
        dot  = Σ(x*w*key) * rstd / sqrt(D)      ← 双向归一化内积 ≈ 余弦相似度
        gate = sigmoid(copysign(sqrt(max(|dot|, CLAMP)), dot))   ∈ (0,1)

        out[t, c, :] = x + gate * val

    ⇒ **用相似度当门的加性记忆**：
         · 记忆与当前残差"像" → dot 大 → gate→1 → 把 val 加进去；
         · 不像 → gate→0 → 几乎不加（残差原样通过）。
       val 是所有通道共用的一份，但每个通道的 key 与 q/k 权重不同，
       gate 也就不同 —— 所以同一个记忆在不同通道上被"有选择地"吸收。
       copysign(sqrt(|dot|)) 把符号带回来，gate<0.5 表示负相关（抑制）。
    """
    row = tl.program_id(0)
    token = row // HC
    hc = row % HC
    col = tl.arange(0, B)
    mask = col < D
    x = tl.load(X + row * D + col, mask, 0).to(tl.float32)
    key = tl.load(KV + token * (HC + 1) * D + hc * D + col, mask, 0).to(tl.float32)
    q_weight = tl.load(QW + hc * D + col, mask, 0).to(tl.float32)
    k_weight = tl.load(KW + hc * D + col, mask, 0).to(tl.float32)
    weight = q_weight * k_weight
    rstd = tl.rsqrt(tl.sum(x * x, 0) / D + EPS) * tl.rsqrt(
        tl.sum(key * key, 0) / D + EPS
    )
    dot = tl.sum((x * weight) * key, 0) * rstd * (D**-0.5)
    gate = tl.sigmoid(libdevice.copysign(tl.sqrt(tl.maximum(tl.abs(dot), CLAMP)), dot))
    value = tl.load(KV + token * (HC + 1) * D + HC * D + col, mask, 0).to(tl.float32)
    tl.store(O + row * D + col, x + gate * value, mask)


def fused_engram_gate(
    x: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    clamp_value: float,
) -> torch.Tensor:
    assert x.ndim == 3 and kv.ndim == 2
    t, hc, d = x.shape
    assert kv.shape == (t, (hc + 1) * d)
    assert q_weight.shape == k_weight.shape == (hc, d)
    assert all(
        a.is_cuda and a.is_contiguous() and a.device == x.device
        for a in (x, kv, q_weight, k_weight)
    )
    assert all(
        a.dtype in (torch.bfloat16, torch.float32) for a in (x, kv, q_weight, k_weight)
    )
    out = torch.empty_like(x)
    if t:
        _engram_gate_kernel[(t * hc,)](
            x,
            kv,
            q_weight,
            k_weight,
            out,
            d,
            hc,
            eps,
            clamp_value,
            triton.next_power_of_2(d),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out
