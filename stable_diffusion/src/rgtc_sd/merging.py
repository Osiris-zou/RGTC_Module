"""Reliability-guided 2-D matching for Stable Diffusion."""
from typing import Callable, Tuple, Dict, Any
import math
import torch
from .tomesd.merge import bipartite_soft_matching_random2d, do_nothing, mps_gather_workaround
from .tomesd.utils import init_generator

# ============================================================
# 2. Ours-SD: 在 ToMeSD 的 2D matching 上加入 top1/top2 margin
# ============================================================

def bipartite_soft_matching_random2d_reliability_guided(
    metric: torch.Tensor,
    w: int,
    h: int,
    sx: int,
    sy: int,
    r: int,
    beta: float = 0.015,
    no_rand: bool = False,
    generator: torch.Generator = None,
) -> Tuple[Callable, Callable]:
    """
    Stable Diffusion 版本的 reliability-guided token merging。

    与 ToMeSD 保持一致：
    1. 使用 2D random bipartite partition；
    2. 使用 merge / unmerge 机制；
    3. 使用相同 ratio、sx、sy、max_downsample。

    唯一区别：
        ToMe score = top1
        Ours score = top1 + beta * (top1 - top2)
    """
    B, N, _ = metric.shape

    if r <= 0:
        return do_nothing, do_nothing

    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather

    with torch.no_grad():
        hsy, wsx = h // sy, w // sx

        if hsy <= 0 or wsx <= 0:
            return do_nothing, do_nothing

        # ------------------------------------------------------------
        # 2D dst/src 划分：保持 ToMeSD 原逻辑
        # ------------------------------------------------------------
        if no_rand:
            rand_idx = torch.zeros(
                hsy,
                wsx,
                1,
                device=metric.device,
                dtype=torch.int64,
            )
        else:
            if generator is None:
                generator = init_generator(metric.device)

            rand_idx = torch.randint(
                sy * sx,
                size=(hsy, wsx, 1),
                device=generator.device,
                generator=generator,
            ).to(metric.device)

        idx_buffer_view = torch.zeros(
            hsy,
            wsx,
            sy * sx,
            device=metric.device,
            dtype=torch.int64,
        )

        idx_buffer_view.scatter_(
            dim=2,
            index=rand_idx,
            src=-torch.ones_like(rand_idx, dtype=rand_idx.dtype),
        )

        idx_buffer_view = (
            idx_buffer_view
            .view(hsy, wsx, sy, sx)
            .transpose(1, 2)
            .reshape(hsy * sy, wsx * sx)
        )

        if (hsy * sy) < h or (wsx * sx) < w:
            idx_buffer = torch.zeros(h, w, device=metric.device, dtype=torch.int64)
            idx_buffer[: hsy * sy, : wsx * sx] = idx_buffer_view
        else:
            idx_buffer = idx_buffer_view

        rand_idx = idx_buffer.reshape(1, -1, 1).argsort(dim=1)

        del idx_buffer, idx_buffer_view

        num_dst = hsy * wsx

        a_idx = rand_idx[:, num_dst:, :]   # source tokens
        b_idx = rand_idx[:, :num_dst, :]   # target tokens

        def split(x: torch.Tensor):
            C = x.shape[-1]
            src = gather(
                x,
                dim=1,
                index=a_idx.expand(B, N - num_dst, C),
            )
            dst = gather(
                x,
                dim=1,
                index=b_idx.expand(B, num_dst, C),
            )
            return src, dst

        # ------------------------------------------------------------
        # 相似度计算：保持 ToMeSD 的 cosine similarity
        # ------------------------------------------------------------
        metric = metric / metric.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        a, b = split(metric)

        scores = a @ b.transpose(-1, -2)

        r = min(a.shape[1], r)

        if r <= 0:
            return do_nothing, do_nothing

        # ------------------------------------------------------------
        # ToMe 原始方式：
        #   node_max = top1
        #
        # Ours:
        #   calibrated = top1 + beta * (top1 - top2)
        # ------------------------------------------------------------
        if scores.shape[-1] >= 2:
            top2_vals, _ = scores.topk(k=2, dim=-1)

            top1 = top2_vals[..., 0]
            top2 = top2_vals[..., 1]

            node_max, node_idx = scores.max(dim=-1)

            margin = top1 - top2
            margin = torch.nan_to_num(
                margin,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            calibrated_score = node_max + beta * margin
            calibrated_score = torch.nan_to_num(
                calibrated_score,
                nan=-torch.inf,
                posinf=-torch.inf,
                neginf=-torch.inf,
            )
        else:
            node_max, node_idx = scores.max(dim=-1)
            calibrated_score = node_max

        edge_idx = calibrated_score.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]
        src_idx = edge_idx[..., :r, :]

        dst_idx = gather(
            node_idx[..., None],
            dim=-2,
            index=src_idx,
        )

    # ------------------------------------------------------------
    # merge / unmerge 逻辑：保持 ToMeSD 原样
    # ------------------------------------------------------------

    def merge(x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
        src, dst = split(x)
        n, t1, c = src.shape

        unm = gather(
            src,
            dim=-2,
            index=unm_idx.expand(n, t1 - r, c),
        )

        src = gather(
            src,
            dim=-2,
            index=src_idx.expand(n, r, c),
        )

        dst = dst.scatter_reduce(
            -2,
            dst_idx.expand(n, r, c),
            src,
            reduce=mode,
        )

        return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]

        unm = x[..., :unm_len, :]
        dst = x[..., unm_len:, :]

        _, _, c = unm.shape

        src = gather(
            dst,
            dim=-2,
            index=dst_idx.expand(B, r, c),
        )

        out = torch.zeros(
            B,
            N,
            c,
            device=x.device,
            dtype=x.dtype,
        )

        out.scatter_(
            dim=-2,
            index=b_idx.expand(B, num_dst, c),
            src=dst,
        )

        out.scatter_(
            dim=-2,
            index=gather(
                a_idx.expand(B, a_idx.shape[1], 1),
                dim=1,
                index=unm_idx,
            ).expand(B, unm_len, c),
            src=unm,
        )

        out.scatter_(
            dim=-2,
            index=gather(
                a_idx.expand(B, a_idx.shape[1], 1),
                dim=1,
                index=src_idx,
            ).expand(B, r, c),
            src=src,
        )

        return out

    return merge, unmerge


# ============================================================
# 3. 根据 full / tome / reliability_guided 计算 merge / unmerge
# ============================================================

def compute_sd_merge(
    x: torch.Tensor,
    h: int,
    w: int,
    tome_info: Dict[str, Any],
):
    """
    x: [B, H*W, C]
    h, w: 当前 attention block 的 latent feature map 尺寸
    tome_info: configure_sd_token_merging 写入的配置
    """
    if tome_info is None:
        return (
            do_nothing,
            do_nothing,
            do_nothing,
            do_nothing,
            do_nothing,
            do_nothing,
        )

    method = tome_info.get("method", "full")

    if method == "full":
        return (
            do_nothing,
            do_nothing,
            do_nothing,
            do_nothing,
            do_nothing,
            do_nothing,
        )

    args = tome_info["args"]

    original_h, original_w = tome_info["size"]
    original_tokens = original_h * original_w

    current_tokens = x.shape[1]
    downsample = int(
        math.ceil(
            math.sqrt(
                max(1, original_tokens // current_tokens)
            )
        )
    )

    if downsample > args["max_downsample"]:
        m, u = do_nothing, do_nothing
    else:
        ratio = args["ratio"]
        r = int(current_tokens * ratio)

        if args["generator"] is None:
            args["generator"] = init_generator(x.device)
        elif args["generator"].device != x.device:
            args["generator"] = init_generator(
                x.device,
                fallback=args["generator"],
            )

        use_rand = False if x.shape[0] % 2 == 1 else args["use_rand"]

        if method == "tome":
            m, u = bipartite_soft_matching_random2d(
                metric=x,
                w=w,
                h=h,
                sx=args["sx"],
                sy=args["sy"],
                r=r,
                no_rand=not use_rand,
                generator=args["generator"],
            )

        elif method == "reliability_guided":
            m, u = bipartite_soft_matching_random2d_reliability_guided(
                metric=x,
                w=w,
                h=h,
                sx=args["sx"],
                sy=args["sy"],
                r=r,
                beta=args["beta"],
                no_rand=not use_rand,
                generator=args["generator"],
            )

        else:
            raise ValueError(
                f"Unknown merge method: {method}. "
                f"Expected full / tome / reliability_guided."
            )

    m_a, u_a = (m, u) if args["merge_attn"] else (do_nothing, do_nothing)
    m_c, u_c = (m, u) if args["merge_crossattn"] else (do_nothing, do_nothing)
    m_m, u_m = (m, u) if args["merge_mlp"] else (do_nothing, do_nothing)

    return m_a, m_c, m_m, u_a, u_c, u_m


# ============================================================
# 4. 给 diffusion 模型写入配置
# ============================================================

def configure_sd_token_merging(
    diffusion_model: torch.nn.Module,
    method: str = "full",
    ratio: float = 0.0,
    beta: float = 0.015,
    max_downsample: int = 1,
    sx: int = 2,
    sy: int = 2,
    use_rand: bool = False,
    merge_attn: bool = True,
    merge_crossattn: bool = False,
    merge_mlp: bool = False,
    original_size=(64, 64),
):
    """
    给当前 SD diffusion 模型配置 token merging。

    method:
        full: 不做 token merging
        tome: ToMeSD 原始方法
        reliability_guided: confidence-aware edge calibration

    original_size:
        对 512x512 SD v1.5 来说 latent 是 64x64。
    """
    assert method in ["full", "tome", "reliability_guided"]

    info = {
        "method": method,
        "size": original_size,
        "args": {
            "ratio": float(ratio),
            "beta": float(beta),
            "max_downsample": int(max_downsample),
            "sx": int(sx),
            "sy": int(sy),
            "use_rand": bool(use_rand),
            "generator": None,
            "merge_attn": bool(merge_attn),
            "merge_crossattn": bool(merge_crossattn),
            "merge_mlp": bool(merge_mlp),
        },
    }

    count = 0

    for module in diffusion_model.modules():
        if module.__class__.__name__ == "UNET_AttentionBlock":
            module._sd_tome_info = info
            count += 1

    print(
        f"[SD-ToMe] method={method}, ratio={ratio}, beta={beta}, "
        f"max_downsample={max_downsample}, patched_blocks={count}"
    )

    return diffusion_model