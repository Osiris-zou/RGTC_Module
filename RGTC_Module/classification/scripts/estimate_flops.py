import argparse, torch
from fvcore.nn import FlopCountAnalysis
from common import load_model
p=argparse.ArgumentParser(); p.add_argument('--model',required=True); p.add_argument('--method',choices=['full','tome','reliability_guided'],default='reliability_guided'); p.add_argument('--r',type=int,default=0); p.add_argument('--beta',type=float,default=.015); p.add_argument('--checkpoint'); p.add_argument('--pretrained',action='store_true'); p.add_argument('--device',default='cuda'); a=p.parse_args(); d=torch.device(a.device); m=load_model(a.model,a.method,a.r,a.beta,a.checkpoint,a.pretrained,d); print(f'{FlopCountAnalysis(m,(torch.randn(1,3,224,224,device=d),)).total()/1e9:.6f}')
