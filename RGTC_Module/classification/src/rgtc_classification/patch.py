"""Runtime patch for timm VisionTransformer models."""
from typing import Tuple
import torch
from timm.models.vision_transformer import Attention, Block, VisionTransformer
from .merge import bipartite_soft_matching, reliability_guided_matching, merge_source, merge_weighted_average
from .schedule import parse_r

class PatchedAttention(Attention):
    def forward(self, x: torch.Tensor, size: torch.Tensor=None) -> Tuple[torch.Tensor, torch.Tensor]:
        b,n,c=x.shape
        qkv=self.qkv(x).reshape(b,n,3,self.num_heads,c//self.num_heads).permute(2,0,3,1,4)
        q,k,v=qkv[0],qkv[1],qkv[2]
        attn=(q@k.transpose(-2,-1))*self.scale
        if size is not None: attn=attn+size.log()[:,None,None,:,0]
        attn=self.attn_drop(attn.softmax(dim=-1))
        out=(attn@v).transpose(1,2).reshape(b,n,c)
        return self.proj_drop(self.proj(out)), k.mean(1)

class PatchedBlock(Block):
    def _dp1(self,x): return self.drop_path1(x) if hasattr(self,'drop_path1') else self.drop_path(x)
    def _dp2(self,x): return self.drop_path2(x) if hasattr(self,'drop_path2') else self.drop_path(x)
    def forward(self,x):
        info=self._rgtc_info
        attn_size=info['size'] if info['prop_attn'] else None
        attn_out,metric=self.attn(self.norm1(x),attn_size)
        if info.get('diagnostic_layer') == getattr(self,'_rgtc_layer_idx',-1): info['diagnostic_metric']=metric.detach().cpu()
        x=x+self._dp1(attn_out)
        r=info['r'].pop(0)
        if r>0:
            if info['method']=='tome': fn=bipartite_soft_matching
            elif info['method']=='reliability_guided': fn=reliability_guided_matching
            else: raise ValueError(f"Unsupported method: {info['method']}")
            merge,_=fn(metric,r,info['class_token'],info['distill_token'],beta=info['beta'])
            if info['trace_source']: info['source']=merge_source(merge,x,info['source'])
            x,info['size']=merge_weighted_average(merge,x,info['size'])
        return x+self._dp2(self.mlp(self.norm2(x)))

def _patched_model_class(cls):
    class PatchedVisionTransformer(cls):
        def forward(self,*args,**kwargs):
            self._rgtc_info['r']=parse_r(len(self.blocks),self.r)
            self._rgtc_info['size']=None; self._rgtc_info['source']=None
            return super().forward(*args,**kwargs)
    return PatchedVisionTransformer

def patch_timm(model: VisionTransformer, method: str='reliability_guided', beta: float=0.015, trace_source: bool=False, prop_attn: bool=True):
    if method not in {'tome','reliability_guided'}: raise ValueError('method must be tome or reliability_guided')
    if hasattr(model,'_rgtc_info'):
        model._rgtc_info.update(method=method,beta=float(beta),trace_source=trace_source,prop_attn=prop_attn); return model
    model.__class__=_patched_model_class(model.__class__); model.r=0
    model._rgtc_info={'r':0,'size':None,'source':None,'trace_source':trace_source,'prop_attn':prop_attn,'class_token':getattr(model,'cls_token',None) is not None,'distill_token':getattr(model,'dist_token',None) is not None,'method':method,'beta':float(beta)}
    idx=0
    for module in model.modules():
        if isinstance(module,Block): module.__class__=PatchedBlock; module._rgtc_info=model._rgtc_info; module._rgtc_layer_idx=idx; idx+=1
        elif isinstance(module,Attention): module.__class__=PatchedAttention
    return model
