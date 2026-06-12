import argparse, json, statistics, torch
from common import load_model

def main():
 p=argparse.ArgumentParser(); p.add_argument('--model',required=True); p.add_argument('--method',choices=['full','tome','reliability_guided'],default='reliability_guided'); p.add_argument('--r',type=int,default=0); p.add_argument('--beta',type=float,default=0.015); p.add_argument('--checkpoint'); p.add_argument('--pretrained',action='store_true'); p.add_argument('--batch-size',type=int,default=64); p.add_argument('--warmup',type=int,default=50); p.add_argument('--runs',type=int,default=200); p.add_argument('--repeats',type=int,default=5); p.add_argument('--device',default='cuda'); p.add_argument('--output'); a=p.parse_args(); dev=torch.device(a.device)
 model=load_model(a.model,a.method,a.r,a.beta,a.checkpoint,a.pretrained,dev); x=torch.randn(a.batch_size,3,224,224,device=dev); vals=[]
 with torch.inference_mode():
  for _ in range(a.repeats):
   for _ in range(a.warmup): model(x)
   torch.cuda.synchronize(); start=torch.cuda.Event(True); end=torch.cuda.Event(True); start.record()
   for _ in range(a.runs): model(x)
   end.record(); torch.cuda.synchronize(); vals.append(a.batch_size*a.runs/(start.elapsed_time(end)/1000))
 result={'throughput_mean':statistics.mean(vals),'throughput_median':statistics.median(vals),'throughput_all':vals,'batch_size':a.batch_size,'warmup':a.warmup,'runs':a.runs,'repeats':a.repeats}
 print(json.dumps(result,indent=2));
 if a.output: open(a.output,'w').write(json.dumps(result,indent=2))
if __name__=='__main__': main()
