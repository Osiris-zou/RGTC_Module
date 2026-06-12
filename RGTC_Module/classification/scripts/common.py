from pathlib import Path
import torch, timm
from torchvision import transforms
from rgtc_classification import patch_timm

def transform(name):
    stats=([0.5]*3,[0.5]*3) if name=='inception' else ([0.485,0.456,0.406],[0.229,0.224,0.225])
    return transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize(*stats)])

def load_model(model_name, method, r, beta, checkpoint, pretrained, device, prop_attn=True):
    model=timm.create_model(model_name,pretrained=pretrained and not checkpoint,num_classes=1000)
    if method!='full': patch_timm(model,method='tome' if method=='tome' else 'reliability_guided',beta=beta,prop_attn=prop_attn); model.r=r
    if checkpoint:
        state=torch.load(checkpoint,map_location='cpu'); state=state.get('state_dict',state.get('model',state)) if isinstance(state,dict) else state
        state={k.removeprefix('module.'):v for k,v in state.items()}; model.load_state_dict(state,strict=False)
    return model.eval().to(device)
