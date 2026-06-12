import argparse
from rgtc_classification import parse_r
p=argparse.ArgumentParser(); p.add_argument('--layers',type=int,required=True); p.add_argument('--tokens',type=int,default=197); p.add_argument('--r',type=int,required=True); a=p.parse_args(); t=a.tokens
for i,r in enumerate(parse_r(a.layers,a.r),1): t-=min(r,t//2); print(i,r,t)
