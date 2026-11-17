"""Native-source head inference. Saves observations, never reads count references."""
import argparse,json,queue,threading,time,copy
from pathlib import Path
import cv2,numpy as np,torch
from gpu_backend import GpuModel
from inference_pool import inferred_batches
from run import produce,capture


class HeadDetector:
    def __init__(self,a,device):
        self.a=a;self.device=device
        self.model=GpuModel(a.model,device,tuple(a.shape),'graph',a.batch,False)
    def infer(self,batch):
        ims=[r['image'] for r in batch];h,w=ims[0].shape[:2]
        x1,y1,x2,y2=[round(v*s) for v,s in zip(self.a.crop,[w,h,w,h])]
        result=self.model.predict([im[y1:y2,x1:x2] for im in ims],max(self.a.shape),.1,.7,[0])
        output=[]
        for r,p in zip(batch,result):
            boxes=p.boxes.data.numpy().copy()
            if len(boxes):boxes[:,[0,2]]+=x1;boxes[:,[1,3]]+=y1
            output.append(dict(index=r['index'],head_detections=boxes.tolist()))
        return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',default='models/11m-head.pt');p.add_argument('--devices',nargs='+',default=['0'])
    p.add_argument('--shape',type=int,nargs=2,default=[896,640]);p.add_argument('--crop',type=float,nargs=4,default=[.4,0,.8,1])
    p.add_argument('--start-seconds',type=float,default=0);p.add_argument('--max-seconds',type=float)
    p.add_argument('--stride',type=int,default=2);p.add_argument('--batch',type=int,default=8)
    p.add_argument('--decode-threads',type=int,default=8)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    a.decoder='cv2';a.pace=False;torch.set_num_threads(2);cv2.setNumThreads(1);torch.backends.cudnn.benchmark=True
    cap=capture(a.source[0],a.decode_threads);w,h,fps=int(cap.get(3)),int(cap.get(4)),cap.get(5);cap.release()
    start=time.perf_counter();workers=[]
    for device in a.devices:
        d=HeadDetector(a,device);d.infer([dict(index=0,image=np.zeros((h,w,3),np.uint8)) for _ in range(a.batch)])
        torch.cuda.empty_cache();workers.append(d)
    startup=time.perf_counter()-start;start=time.perf_counter()
    q=queue.Queue(2);stop=threading.Event()
    producer=threading.Thread(target=produce,args=(a,q,stop,dict(width=w,height=h,fps=fps),start,0),daemon=True);producer.start()
    n=0
    try:
        with (a.output/'heads.jsonl').open('w') as out:
            for batch,items,seconds in inferred_batches(q,workers):
                for item in items:out.write(json.dumps(item)+'\n');n+=1
                if n%200<len(items):out.flush();print(json.dumps(dict(observations=n,seconds=time.perf_counter()-start)),flush=True)
    finally:stop.set()
    result=dict(observations=n,processing_seconds=time.perf_counter()-start,startup_seconds=startup,model=a.model,shape=a.shape,crop=a.crop,stride=a.stride,source=a.source,start_seconds=a.start_seconds,max_seconds=a.max_seconds,gpus=a.devices)
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
