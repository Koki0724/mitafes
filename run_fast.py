"""Validated presets for native-resolution, full-cadence GPU counting."""
import argparse,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--camera',choices=['glass','overhead'],required=True)
    p.add_argument('--source',nargs='+',required=True);p.add_argument('--output',required=True)
    p.add_argument('--mode',choices=['live','offline'],default='offline')
    p.add_argument('--gpus',nargs='+',default=['0'])
    p.add_argument('--start-seconds',type=float,default=0);p.add_argument('--max-seconds',type=float)
    for flag in ['preview','resume','record-tracks','record-detections']:p.add_argument('--'+flag,action='store_true')
    a=p.parse_args()
    if a.mode=='live' and len(a.gpus)!=1:p.error('Live preset uses one GPU; offline mode supports multiple GPUs')
    batch=1 if a.mode=='live' else (16 if len(a.gpus)>1 and a.camera=='glass' else 8)
    threads=2 if a.mode=='live' else (12 if len(a.gpus)>1 else 8)
    command=[sys.executable,str(ROOT/'run.py'),'--camera',a.camera,'--source',*a.source,'--output',a.output,
             '--backend','graph','--batch',str(batch),'--decode-threads',str(threads),'--device',a.gpus[0],
             '--start-seconds',str(a.start_seconds)]
    if len(a.gpus)>1:command+=['--devices',*a.gpus]
    if a.mode=='live':command+=['--pace']
    if a.max_seconds is not None:command+=['--max-seconds',str(a.max_seconds)]
    for flag in ['preview','resume','record_tracks','record_detections']:
        if getattr(a,flag):command+=['--'+flag.replace('_','-')]
    raise SystemExit(subprocess.run(command).returncode)


if __name__=='__main__':main()
