"""Sequential model timings; full-minute figures require separate end-to-end runs."""
import gc,json,time
from pathlib import Path
import cv2,torch,numpy as np
from gpu_backend import GpuModel

ROOT=Path(__file__).resolve().parent;MODELS=ROOT.parent/'cpu_counter/models'


def main():
    torch.set_num_threads(2);cv2.setNumThreads(1);torch.backends.cudnn.benchmark=True
    cap=cv2.VideoCapture(str(ROOT.parent/'data/1219/ch1_20261106_121900-124148_vscode.mp4'),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,2])
    images=[]
    for frame in [606,1238,1360]:cap.set(1,frame);ok,im=cap.read();assert ok;images.append(im)
    cap.release();results=[]
    cases=[('eager',1,False),('graph',1,False),('graph',2,False),('graph',4,False),('graph',8,False),('graph',1,True)]
    for backend,batch,half in cases:
        for stem,shape,crop,classes in [('yolo11m-pose',(736,512),(.4,0,.8,1),[0]),('yolo11n',(384,640),(0,0,1,1),[25])]:
            before=time.perf_counter();model=GpuModel(MODELS/f'{stem}.pt','0',shape,backend,batch,half)
            startup=time.perf_counter()-before;samples=[];counts=[]
            x1,y1,x2,y2=[round(v*s) for v,s in zip(crop,[2880,1620,2880,1620])]
            for repeat in range(8):
                im=images[repeat%len(images)][y1:y2,x1:x2];t=time.perf_counter()
                out=model.predict([im]*batch,1280,.1,.7,classes,half=half)
                if repeat>=2:samples.append((time.perf_counter()-t)/batch)
                counts.append([len(o.boxes) for o in out])
            result=dict(model=stem,backend=backend,batch=batch,half=half,startup_seconds=startup,
                        median_ms_per_image=float(np.median(samples))*1000,p95_ms_per_image=float(np.percentile(samples,95))*1000,
                        detections=counts,model_pre_post_only=True)
            results.append(result);print(json.dumps(result),flush=True)
            (ROOT/'runs/microbenchmark.json').write_text(json.dumps(results,indent=2))
            del model;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
