"""Constraints for the V4 CUDA/HCU prefill LayerSplit implementation."""


def validate_cp_cache_layer_split(args, hf_config):
    from sglang.srt.environ import envs
    from sglang.srt.utils import is_cuda, is_hcu

    def require(condition, message):
        if not condition:
            raise ValueError("--enable-cp-cache-layer-split " + message)

    # The V4.1 config normalizer maps this architecture to the V4 runtime
    # class; model_type remains the reliable V4.1 discriminator.
    require(getattr(hf_config, "model_type", None) == "deepseek_v41", "requires DeepSeek V4.1")
    text_config = getattr(hf_config, "text_config", hf_config)
    ratios = set(text_config.compress_ratios)
    require(ratios <= {0, 1, 2}, f"supports compression ratios 0/1/2, got {ratios}")
    require(
        not args.enable_dsa_cache_layer_split,
        "must not be combined with --enable-dsa-cache-layer-split",
    )
    require(
        args.disaggregation_mode == "prefill", "requires --disaggregation-mode prefill"
    )
    require(
        args.enable_prefill_cp
        and args.cp_strategy == "interleave"
        and args.attn_cp_size > 1,
        "requires --enable-prefill-cp --cp-strategy interleave --attn-cp-size > 1",
    )
    require(args.pp_size == 1, "requires --pp-size 1")
    require(
        not args.enable_two_batch_overlap,
        "does not support two-batch overlap",
    )
    require(is_cuda() or is_hcu(), "requires CUDA or HCU")
    require(
        args.disaggregation_transfer_backend == "mooncake",
        "requires the Mooncake transfer backend",
    )
    # ★ 这条限制是"死亡三角"的第三道门（第一道是 V4.1 DSpark PD 的拓扑白名单，
    #   第二道是 conn.py 里"CP>1 必须走描述符匹配的 LayerSplit"）。
    #
    #   三者叠起来的效果：门1 要求开 prefill CP → 门2 要求 CP>1 就必须开 LS
    #   → 门3 禁止 LS 挂 HiCache，于是"CP + DSpark + PD + HiCache"在 HCU 上
    #   不存在合法配置。
    #
    #   为什么 LS 目前挂不了 HiCache（三处，均为实现完成度问题）：
    #     ① 层集合：HiCache 的层映射默认恒等（每层都有 buffer），
    #        而 LS 池只给 owned 连续层分配 buffer
    #        → 见 _resolve_deepseek_v4_layer_mappings；
    #     ② 页集合：owner 的持久 buffer 会被 HiCache 的 H2D 回载覆写，
    #        而 staging 重映射 / pending 广播缓存的是"覆写前"的读状态；
    #     ③ 静态选页：选页结果缓存在 _batch_active_pages 里跨层复用，
    #        回载改变槽位布局后必须作废。
    #
    #   upstream 没有 V4 LS，所以不存在这条互斥；upstream 的 DSA layer split
    #   与 HiCache 是打通的（mem_cache/pool_host/* 有 owned-only host 映射），
    #   说明这不是结构性问题。
    require(
        not args.enable_hisparse and not args.enable_hierarchical_cache,
        "does not yet support HiSparse or HiCache in this HCU port",
    )
    require(not args.prefill_only_disable_kv_cache, "requires the prefill KV cache")
    require(
        not args.enable_encoder_swa_bounded_replay,
        "does not support encoder SWA bounded replay",
    )
    require(envs.SGLANG_OPT_USE_COMPRESSOR_V2.get(), "requires Compressor V2")
    require(
        not envs.SGLANG_DISAGG_STAGING_BUFFER.get(),
        "does not support the separate disaggregation staging transport",
    )
    require(
        args.speculative_algorithm in (None, "DSPARK"),
        "supports DSPARK speculation only",
    )
    if args.speculative_algorithm == "DSPARK":
        require(
            bool(getattr(text_config, "dspark_target_layer_ids", None)),
            "requires a DSPARK model configuration",
        )
    from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate import (
        is_unified_kv_triton,
    )

    require(not is_unified_kv_triton(), "does not support unified_kv_triton")
    from sglang.srt.layers.attention.dsv4.sparse_prefill_utils import (
        use_dsv4_q8kv8_sparse_prefill,
    )

    require(
        not use_dsv4_q8kv8_sparse_prefill(args.dsv4_prefill_backend),
        "does not yet support the Q8 sparse-prefill backend in this port",
    )
    # Compacted pages have dynamic shapes and the staging state belongs to one
    # forward at a time. The normal eager loop preserves CP collective order.
    from sglang.srt.model_executor.cuda_graph_config import Backend

    require(
        args.cuda_graph_config.prefill.backend == Backend.DISABLED,
        "requires --cuda-graph-backend-prefill disabled",
    )
