"""Slot-keyed async broadcast plumbing for CP Cache LayerSplit."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

import torch

from sglang.srt.distributed import get_attn_cp_group


@dataclass
class PendingBroadcast:
    """State for one in-flight owner->peers broadcast."""

    stream: Optional[torch.cuda.Stream] = None
    layer_id: Optional[int] = None
    active: bool = False


def get_pynccl_broadcast_comm():
    """Return the attention-CP PyNCCL communicator used for KV broadcasts."""
    group = get_attn_cp_group()
    pynccl_comm = getattr(group, "pynccl_comm", None)
    if pynccl_comm is None or not getattr(pynccl_comm, "available", False):
        raise RuntimeError(
            "CP Cache LayerSplit requires an available PyNCCL communicator "
            "on the attention-CP group for KV broadcasts."
        )
    return pynccl_comm


# 【BroadcastSlots】把"owner 广播"从主流程里摘出来的旁路流机制。
#
#   为什么需要它：LayerSplit 的每一层都要把"该层读集合涉及的页"从 owner rank
#   广播给其它 CP rank。如果这条广播排在主流程（当前 stream）上，它就和
#   本层的 Q/attention 严格串行；排到旁路流（side stream）上，则可以与
#   上一层/本层前半段的计算重叠，只要保证"真正读 staging 之前等它完成"。
#
#   三个动作构成一条有序的依赖链（每个 kind 一个槽，互不干扰）：
#
#     launch(kind, layer)  ── ① clear(kind)：先等上一次用同一槽的广播做完
#                             ② pending.stream.wait_stream(main stream)
#                                （旁路流等主流程把数据准备好，例如 owner 拷贝）
#                             ③ 切到 pending.stream 执行 with 块（广播本身）
#     finish(kind, layer)  ── 校验 layer 一致（防止错层消费），再 clear(kind)
#     clear(kind)          ── main stream.wait_stream(pending.stream)
#                             （主流程等广播完成，之后才允许读 staging）
#
#   两个使用要点：
#     · launch 内部会先 clear；所以"同槽复用"天然安全，但每层只能用一次槽。
#     · clear/finish 只对"当前 stream"建依赖。这正是 HiCache 恢复能复用它
#       的原因：恢复在 H2D stream 上下文里调用池的
#       invalidate_staging_for_hicache()，于是 current_stream() 就是 H2D 流，
#       "H2D 等所有在飞行的广播" 被写成了一条普通的 stream 依赖，
#       逐层 restore 才安全落盘（见 l2_transfer.py 的 submit_host_to_device）。
#
#   与静态选页的关系：只有 static_size 已知（nonzero_static）时，
#   async_pages 才能打开（deepseek_v4_pool.py:979-983）；
#   动态 nonzero 已经同步过主机，再排旁路流没有意义，只会多一层簿记开销。
class BroadcastSlots:
    """Per-pool slot registry for in-flight LayerSplit broadcasts."""

    def __init__(self, slot_kinds: Iterable[str]) -> None:
        self._slots: dict[str, PendingBroadcast] = {
            kind: PendingBroadcast() for kind in slot_kinds
        }

    def kinds(self) -> tuple[str, ...]:
        return tuple(self._slots.keys())

    def pending(self, kind: str) -> PendingBroadcast:
        try:
            return self._slots[kind]
        except KeyError:
            raise ValueError(f"unknown broadcast kind: {kind}")

    def start(
        self,
        kind: str,
        layer_id: int,
        tensor: torch.Tensor,
        owner_cp: int,
        pynccl_comm,
    ) -> None:
        """Launch an async broadcast on this slot's side stream."""
        with self.launch(kind, layer_id, tensor.device):
            with pynccl_comm.change_state(enable=True):
                pynccl_comm.broadcast(tensor, owner_cp)

    @contextmanager
    def launch(self, kind: str, layer_id: int, device: torch.device) -> Iterator[None]:
        """Run work on a slot's side stream and track it until consumption."""
        self.clear(kind)
        pending = self.pending(kind)
        device_module = torch.get_device_module(device)
        if pending.stream is None:
            pending.stream = device_module.Stream(device=device)
        pending.stream.wait_stream(device_module.current_stream(device))
        pending.layer_id = layer_id
        pending.active = True
        try:
            with device_module.stream(pending.stream):
                yield
        except Exception:
            self.clear(kind)
            raise

    def clear(self, kind: str) -> None:
        """Wait on the slot's side stream so the current stream can proceed."""
        pending = self.pending(kind)
        if not pending.active:
            return
        if pending.stream is not None:
            device_module = torch.get_device_module(pending.stream.device)
            device_module.current_stream(pending.stream.device).wait_stream(
                pending.stream
            )
        pending.layer_id = None
        pending.active = False

    def finish(self, kind: str, layer_id: int) -> None:
        """Wait for the pending broadcast; ``layer_id`` must match."""
        pending = self.pending(kind)
        if not pending.active:
            return
        if pending.layer_id != layer_id:
            raise RuntimeError(
                "CP Cache LayerSplit tried to finish an unexpected "
                f"{kind} broadcast: pending_layer={pending.layer_id}, "
                f"layer={layer_id}"
            )
        self.clear(kind)
