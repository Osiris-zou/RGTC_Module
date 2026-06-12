import argparse, csv, time
from pathlib import Path
import torch
from transformers import CLIPTokenizer
from rgtc_sd import generate, preload_models_from_standard_weights

def main():
 p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--tokenizer-dir',required=True); p.add_argument('--prompt-file',required=True); p.add_argument('--output-dir',required=True); p.add_argument('--method',choices=['full','tome','reliability_guided'],default='reliability_guided'); p.add_argument('--ratio',type=float,default=.5); p.add_argument('--beta',type=float,default=.005); p.add_argument('--seeds',type=int,nargs='+',default=[0,1]); p.add_argument('--steps',type=int,default=50); p.add_argument('--cfg-scale',type=float,default=7.5); p.add_argument('--device',default='cuda'); a=p.parse_args()
 out=Path(a.output_dir)/a.method; out.mkdir(parents=True,exist_ok=True); tokenizer=CLIPTokenizer(vocab_file=str(Path(a.tokenizer_dir)/'vocab.json'),merges_file=str(Path(a.tokenizer_dir)/'merges.txt')); models=preload_models_from_standard_weights(a.checkpoint,a.device); prompts=[x.strip() for x in open(a.prompt_file,encoding='utf-8') if x.strip()]; rows=[]
 for class_id,prompt in enumerate(prompts):
  for seed in a.seeds:
   torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t=time.perf_counter(); image=generate(prompt=prompt,uncond_prompt='',do_cfg=True,cfg_scale=a.cfg_scale,sampler_name='ddpm',n_inference_steps=a.steps,models=models,seed=seed,device=a.device,idle_device='cpu',tokenizer=tokenizer,merge_method=a.method,merge_ratio=0 if a.method=='full' else a.ratio,merge_beta=a.beta,merge_max_downsample=1,merge_sx=2,merge_sy=2,merge_use_rand=True); torch.cuda.synchronize(); elapsed=time.perf_counter()-t; mem=torch.cuda.max_memory_allocated()/1024**3; name=f'class_{class_id:04d}_seed{seed}.png'; image.save(out/name); rows.append({'method':a.method,'class_id':class_id,'prompt':prompt,'seed':seed,'ratio':a.ratio,'beta':a.beta,'image_path':str(Path('generated')/a.method/name),'time_sec':elapsed,'peak_mem_gb':mem})
 with open(Path(a.output_dir)/f'{a.method}_generation_log.csv','w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
if __name__=='__main__': main()
