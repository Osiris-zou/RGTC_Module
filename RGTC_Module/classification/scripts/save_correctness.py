import argparse, json, torch, numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from common import transform, load_model

def main():
 p=argparse.ArgumentParser(); p.add_argument('--data-path',required=True); p.add_argument('--model',required=True); p.add_argument('--method',choices=['full','tome','reliability_guided'],default='reliability_guided'); p.add_argument('--r',type=int,default=0); p.add_argument('--beta',type=float,default=0.015); p.add_argument('--checkpoint'); p.add_argument('--pretrained',action='store_true'); p.add_argument('--preprocess',choices=['inception','imagenet'],default='inception'); p.add_argument('--batch-size',type=int,default=64); p.add_argument('--workers',type=int,default=6); p.add_argument('--device',default='cuda'); p.add_argument('--output',required=True); a=p.parse_args()
 ds=ImageFolder(a.data_path,transform=transform(a.preprocess)); dl=DataLoader(ds,batch_size=a.batch_size,shuffle=False,num_workers=a.workers,pin_memory=True)
 model=load_model(a.model,a.method,a.r,a.beta,a.checkpoint,a.pretrained,torch.device(a.device)); n=c1=c5=0; correct=[]
 with torch.inference_mode():
  for x,y in tqdm(dl):
   out=model(x.to(a.device)); y=y.to(a.device); n+=y.numel(); pred=out.topk(5,1).indices; batch=(pred[:,0]==y); correct.extend(batch.cpu().numpy().astype(np.uint8).tolist()); c1+=batch.sum().item(); c5+=(pred==y[:,None]).any(1).sum().item()
 result={'model':a.model,'method':a.method,'r':a.r,'beta':a.beta,'images':n,'top1':100*c1/n,'top5':100*c5/n}
 print(json.dumps(result,indent=2));
 np.save(a.output,np.asarray(correct,dtype=np.uint8)); open(a.output+'.json','w').write(json.dumps(result,indent=2))
if __name__=='__main__': main()
