"""Check the selected accelerator with real tensor operations."""
import argparse,json,os
import torch
from runtime_control import select_platform,environment

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--platform',choices=['auto','cuda','mps'],default='auto');p.add_argument('--gpu',default='0');a=p.parse_args()
    accelerator=select_platform(a.platform);device='mps' if accelerator=='mps' else 'cuda:'+a.gpu
    value=(torch.ones(4,device=device)*2).sum()
    if accelerator=='mps':torch.mps.synchronize()
    else:torch.cuda.synchronize(int(a.gpu))
    if value.item()!=8:raise RuntimeError('Accelerator operation check failed')
    report=environment(accelerator)|dict(device=device,accelerator_smoke_test='passed',
        mps_environment={k:os.environ.get(k) for k in ['PYTORCH_ENABLE_MPS_FALLBACK','PYTORCH_MPS_FAST_MATH','PYTORCH_MPS_HIGH_WATERMARK_RATIO','PYTORCH_MPS_PREFER_METAL']})
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
