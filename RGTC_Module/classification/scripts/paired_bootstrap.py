import argparse, json, numpy as np
p=argparse.ArgumentParser(); p.add_argument('--baseline',required=True); p.add_argument('--proposed',required=True); p.add_argument('--samples',type=int,default=10000); p.add_argument('--seed',type=int,default=0); p.add_argument('--output'); a=p.parse_args(); b=np.load(a.baseline).astype(float); o=np.load(a.proposed).astype(float); assert b.shape==o.shape; diff=o-b; rng=np.random.default_rng(a.seed); means=np.empty(a.samples)
for i in range(a.samples): means[i]=diff[rng.integers(0,len(diff),len(diff))].mean()*100
res={'num_images':len(diff),'delta_top1_pp':diff.mean()*100,'ci95_low_pp':float(np.percentile(means,2.5)),'ci95_high_pp':float(np.percentile(means,97.5)),'bootstrap_samples':a.samples,'seed':a.seed}; print(json.dumps(res,indent=2));
if a.output: open(a.output,'w').write(json.dumps(res,indent=2))
