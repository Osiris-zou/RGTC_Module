"""Baseline ToMe matching and reliability-guided edge ranking."""
import math
from typing import Callable, Tuple
import torch

MergePair = Tuple[Callable[[torch.Tensor, str], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]

def _identity(x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    return x

def _normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)

def _build(metric_tokens, r, unm_idx, src_idx, dst_idx, distill_token):
    def merge(x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(-2, unm_idx.expand(n, t1-r, c))
        selected = src.gather(-2, src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), selected, reduce=mode)
        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        return torch.cat([unm, dst], dim=1)
    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape
        src = dst.gather(-2, dst_idx.expand(n, r, c))
        out = torch.zeros(n, metric_tokens, c, device=x.device, dtype=x.dtype)
        out[..., 1::2, :] = dst
        out.scatter_(-2, (2*unm_idx).expand(n, unm_len, c), unm)
        out.scatter_(-2, (2*src_idx).expand(n, r, c), src)
        return out
    return merge, unmerge

def bipartite_soft_matching(metric: torch.Tensor, r: int, class_token: bool=False, distill_token: bool=False, beta: float=0.0) -> MergePair:
    """Original similarity-only ToMe bipartite soft matching."""
    return _matching(metric, r, class_token, distill_token, beta=0.0)

def reliability_guided_matching(metric: torch.Tensor, r: int, class_token: bool=False, distill_token: bool=False, beta: float=0.015) -> MergePair:
    """Rank source edges by top1 + beta*(top1-top2), retaining top1 target assignment."""
    return _matching(metric, r, class_token, distill_token, beta=beta)

def _matching(metric, r, class_token, distill_token, beta):
    protected = int(class_token) + int(distill_token)
    t = metric.shape[1]
    r = min(int(r), (t-protected)//2)
    if r <= 0:
        return _identity, _identity
    with torch.no_grad():
        metric = _normalize(metric)
        src, dst = metric[..., ::2, :], metric[..., 1::2, :]
        scores = src @ dst.transpose(-1, -2)
        if class_token:
            scores[..., 0, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf
        top1, target_idx = scores.max(dim=-1)
        if beta != 0.0 and scores.shape[-1] >= 2:
            top2 = scores.topk(k=2, dim=-1, largest=True, sorted=True).values
            margin = torch.nan_to_num(top2[...,0]-top2[...,1], nan=0.0, posinf=0.0, neginf=0.0)
            rank_score = top1 + float(beta)*margin
        else:
            rank_score = top1
        if class_token:
            rank_score[...,0] = -math.inf
        rank_score = torch.nan_to_num(rank_score, nan=-math.inf, posinf=math.inf, neginf=-math.inf)
        order = rank_score.argsort(dim=-1, descending=True)[...,None]
        unm_idx, src_idx = order[...,r:,:], order[...,:r,:]
        dst_idx = target_idx[...,None].gather(-2, src_idx)
        if class_token:
            unm_idx = unm_idx.sort(dim=1)[0]
    return _build(t, r, unm_idx, src_idx, dst_idx, distill_token)

def merge_weighted_average(merge, x, size=None):
    if size is None:
        size = torch.ones_like(x[...,0,None])
    x = merge(x*size, mode="sum")
    size = merge(size, mode="sum")
    return x/size.clamp_min(1e-6), size

def merge_source(merge, x, source=None):
    if source is None:
        n,t,_=x.shape
        source=torch.eye(t,device=x.device,dtype=x.dtype)[None].expand(n,t,t)
    return merge(source, mode="amax")
