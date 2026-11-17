"""Apple Silicon MPS inference with bounded buffering and persistent counting.

No manual reference or expected event count is read by this program.
"""
import argparse,csv,json,pickle,queue,sqlite3,threading,time,hashlib,math
from pathlib import Path
import cv2
import numpy as np
import torch
import ultralytics.utils.torch_utils as torch_utils
from ultralytics import YOLO
from engine import StreamingCounter
from mac_runtime import require_mps,synchronize,environment,peak_rss_mb

ROOT=Path(__file__).resolve().parent
MODELS=ROOT/'models'


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',nargs='+',required=True)
    p.add_argument('--camera',choices=['glass','overhead'],required=True)
    p.add_argument('--model',default=None)
    p.add_argument('--umbrella-model',default=str(MODELS/'yolo11s.pt'))
    p.add_argument('--imgsz',type=int,default=1280)
    p.add_argument('--umbrella-imgsz',type=int,default=1280)
    p.add_argument('--crop',type=float,nargs=4,default=[0,0,1,1])
    p.add_argument('--stride',type=int,default=2)
    p.add_argument('--umbrella-every',type=int,default=1)
    p.add_argument('--batch',type=int,default=1)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--pace',action='store_true',help='Replay at source timestamps to measure live-like latency')
    p.add_argument('--start-seconds',type=float,default=0,help='Skip this many source seconds; output time starts at zero')
    p.add_argument('--max-seconds',type=float,default=None)
    p.add_argument('--record-tracks',action='store_true')
    p.add_argument('--record-detections',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    return p


def capture(source):
    # File replay measures feasibility without network or frame-drop ambiguity.
    if not Path(source).is_file():raise ValueError(f'Local video file required: {source}')
    params=[cv2.CAP_PROP_N_THREADS,2] if hasattr(cv2,'CAP_PROP_N_THREADS') else []
    cap=cv2.VideoCapture(str(source),cv2.CAP_FFMPEG,params)
    if not cap.isOpened():
        cap.release()
        cap=cv2.VideoCapture(str(source),cv2.CAP_FFMPEG)
    if not cap.isOpened():raise RuntimeError(f'Cannot open source: {source}')
    return cap


class Detector:
    def __init__(self,a,w,h):
        self.a=a;self.w=w;self.h=h
        self.crop=[round(v*s) for v,s in zip(a.crop,[w,h,w,h])]
        x1,y1,x2,y2=self.crop
        if not (0<=x1<x2<=w and 0<=y1<y2<=h):raise ValueError('Invalid crop')
        self.person=YOLO(a.model).to('mps')
        if self.person.task!=('pose' if a.camera=='glass' else 'detect'):
            raise ValueError('Glass requires a pose model; overhead requires a detection model')
        if a.camera=='glass':
            self.umbrella=YOLO(a.umbrella_model).to('mps')
        else:self.umbrella=None

    def predict_person(self,images):
        x1,y1,x2,y2=self.crop
        return self.person.predict([im[y1:y2,x1:x2] for im in images],imgsz=self.a.imgsz,
                                   conf=.1,iou=.7,classes=[0] if self.umbrella else [0,25],
                                   device='mps',half=False,verbose=False)

    def predict_umbrella(self,images):
        return self.umbrella.predict(images,imgsz=self.a.umbrella_imgsz,conf=.1,iou=.7,classes=[25],
                                     device='mps',half=False,verbose=False)

    def infer(self,batch):
        images=[r['image'] for r in batch]
        u_indices=[i for i,r in enumerate(batch) if (r['index']//self.a.stride)%self.a.umbrella_every==0]
        people=self.predict_person(images)
        umbrellas=dict(zip(u_indices,self.predict_umbrella([images[i] for i in u_indices]))) if self.umbrella and u_indices else {}
        x1,y1,_,_=self.crop
        result=[]
        for i,(record,p) in enumerate(zip(batch,people)):
            det=p.boxes.data.cpu().numpy().copy()
            if len(det):det[:,[0,2]]+=x1;det[:,[1,3]]+=y1
            points=p.keypoints.data.cpu().numpy().copy() if p.keypoints is not None else None
            if points is not None and len(points):points[:,:,0]+=x1;points[:,:,1]+=y1
            det=det.tolist();points=points.tolist() if points is not None else None
            if i in umbrellas:
                udets=umbrellas[i].boxes.data.cpu().tolist();det.extend(udets)
                points.extend([[[0,0,0] for _ in range(17)] for _ in udets])
            result.append(dict(index=record['index'],detections=det,**({'keypoints':points} if points is not None else {})))
        return result

    def warmup(self):
        sample=[dict(image=np.zeros((self.h,self.w,3),np.uint8),index=i*self.a.stride) for i in range(self.a.batch)]
        for _ in range(2):self.infer(sample)
        synchronize()
        for model in [self.person,self.umbrella]:
            if model is not None and model.predictor.model.device.type!='mps':
                raise RuntimeError('YOLO did not initialize on MPS; stopping benchmark')


def put_buffer(buffer, item, stop):
    while not stop.is_set():
        try:
            buffer.put(item,timeout=.1)
            return True
        except queue.Full:pass
    return False


def produce(a,buffer,stop,meta,wall_start,start_index):
    cap=None
    try:
        offset=-round(getattr(a,'start_seconds',0)*meta['fps']);batch=[]
        for source in a.source:
            if stop.is_set():return
            cap=capture(source)
            w,h,fps=int(cap.get(3)),int(cap.get(4)),cap.get(5)
            if (w,h)!=(meta['width'],meta['height']) or abs(fps-meta['fps'])>.001:
                raise ValueError('Segment dimensions and FPS must agree; tracking is continuous across segments')
            total=int(cap.get(7));first=max(0,start_index-offset)
            if total>0 and first>=total:offset+=total;cap.release();cap=None;continue
            if first and not cap.set(cv2.CAP_PROP_POS_FRAMES,first):
                raise RuntimeError(f'Cannot seek to frame {first}: {source}')
            local=first
            while not stop.is_set():
                index=offset+local
                if a.max_seconds is not None and index/fps>=a.max_seconds:break
                ok,im=cap.read()
                if not ok:
                    if local<total:raise RuntimeError(f'Decode stopped early at frame {local}/{total}: {source}')
                    break
                local+=1
                if index%a.stride:continue
                if a.pace:
                    delay=wall_start+(index-start_index)/fps-time.perf_counter()
                    if delay>0 and stop.wait(delay):return
                batch.append(dict(index=index,image=im,arrival=time.perf_counter(),
                                  scheduled=wall_start+(index-start_index)/fps))
                if len(batch)==a.batch:
                    if not put_buffer(buffer,batch,stop):return
                    batch=[]
            cap.release();cap=None;offset+=local
            if stop.is_set() or (a.max_seconds is not None and offset/fps>=a.max_seconds):break
        if batch and not put_buffer(buffer,batch,stop):return
        put_buffer(buffer,None,stop)
    except BaseException as exc:put_buffer(buffer,exc,stop)
    finally:
        if cap is not None:cap.release()


def main():
    process_start=time.perf_counter();a=parser().parse_args()
    require_mps()
    if a.model is None:a.model=str(MODELS/('yolo11m-pose.pt' if a.camera=='glass' else 'yolo11s.pt'))
    if a.threads<1 or a.imgsz<1 or a.umbrella_imgsz<1:raise ValueError('Threads and image sizes must be positive')
    if not math.isfinite(a.start_seconds) or a.start_seconds<0:raise ValueError('Start seconds must be finite and nonnegative')
    if a.max_seconds is not None and (not math.isfinite(a.max_seconds) or a.max_seconds<=0):raise ValueError('Max seconds must be finite and positive')
    for path in [a.model]+([a.umbrella_model] if a.camera=='glass' else []):
        if not Path(path).is_file() or Path(path).suffix!='.pt':raise ValueError(f'Local PyTorch .pt weights required: {path}')
    if a.batch<1 or a.stride<1 or a.umbrella_every<1:raise ValueError('Batch and strides must be positive')
    torch_utils.NUM_THREADS=a.threads;torch.set_num_threads(a.threads);cv2.setNumThreads(1)
    profile_name='glass_1219_heads_including_bottom.json' if a.camera=='glass' else 'overhead_1300_v2.json'
    profile=json.loads((ROOT/'reference/profiles'/profile_name).read_text())
    metadata=[]
    for source in a.source:
        cap=capture(source)
        metadata.append((int(cap.get(3)),int(cap.get(4)),cap.get(5),int(cap.get(7))))
        cap.release()
    w,h,fps,_=metadata[0]
    if w<=0 or h<=0 or not math.isfinite(fps) or fps<=0 or any(m[3]<=0 for m in metadata):
        raise ValueError('Video dimensions, FPS and frame count must be valid')
    if any(m[:2]!=(w,h) or abs(m[2]-fps)>.001 for m in metadata):
        raise ValueError('Segment dimensions and FPS must agree')
    available_frames=sum(m[3] for m in metadata)-round(a.start_seconds*fps)
    if available_frames<=0:raise ValueError('Start offset is past the end of the input')
    end_frame=min(available_frames,math.ceil(a.max_seconds*fps)) if a.max_seconds is not None else available_frames
    detector=Detector(a,w,h);detector.warmup()
    a.output.mkdir(parents=True,exist_ok=True)
    signature={k:v for k,v in vars(a).items() if k not in ['output','resume','pace','record_tracks','record_detections','max_seconds']}
    signature['profile']=profile
    signature['device']='mps'
    signature['model_sha256']={str(Path(path).resolve()):hashlib.sha256(Path(path).read_bytes()).hexdigest()
                               for path in [a.model]+([a.umbrella_model] if a.camera=='glass' else [])}
    signature['sources']=[dict(path=str(Path(path).resolve()),size=Path(path).stat().st_size,
                              mtime_ns=Path(path).stat().st_mtime_ns) for path in a.source]
    digest=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
    db=sqlite3.connect(a.output/'events.sqlite');db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS events(event_id INTEGER PRIMARY KEY,direction TEXT,frame INTEGER,seconds REAL,track_id INTEGER,x REAL,y REAL)')
    db.execute('CREATE TABLE IF NOT EXISTS audits(event_id INTEGER PRIMARY KEY,payload TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS checkpoint(id INTEGER PRIMARY KEY,signature TEXT,state BLOB)')
    saved=db.execute('SELECT signature,state FROM checkpoint WHERE id=1').fetchone()
    if saved:
        if not a.resume:raise ValueError('Output already contains a checkpoint; choose a new output or --resume')
        if saved[0]!=digest:raise ValueError('Resume configuration does not match checkpoint')
        # Only load checkpoints generated locally by this program.
        engine=StreamingCounter.restore(pickle.loads(saved[1]));start_index=engine.last_frame+a.stride
    else:engine=StreamingCounter(profile,w,h,fps,a.stride);start_index=0
    if a.resume:
        # Debug logs are not the canonical store. Discard rows newer than the
        # committed state so an interrupted batch cannot duplicate them on replay.
        for name in ['tracks.jsonl','detections.jsonl','timings.jsonl']:
            path=a.output/name
            if not path.exists():continue
            temp=path.with_suffix('.truncate.tmp')
            with path.open() as source,temp.open('w') as destination:
                for line in source:
                    try:row=json.loads(line)
                    except json.JSONDecodeError:break
                    if row.get('frame',row.get('index'))<=engine.last_frame:destination.write(line)
            temp.replace(path)
    startup=time.perf_counter()-process_start
    (a.output/'config.json').write_text(json.dumps(dict(args=vars(a)|{'output':str(a.output)},profile=profile,
                                                    environment=environment(),model_sha256=signature['model_sha256']),indent=2))
    buffer=queue.Queue(maxsize=2);stop=threading.Event();wall_start=time.perf_counter()
    producer=threading.Thread(target=produce,args=(a,buffer,stop,dict(width=w,height=h,fps=fps),wall_start,start_index),daemon=True);producer.start()
    records=(a.output/'tracks.jsonl').open('a' if a.resume else 'w') if a.record_tracks else None
    detections=(a.output/'detections.jsonl').open('a' if a.resume else 'w') if a.record_detections else None
    metrics=(a.output/'timings.jsonl').open('a' if a.resume else 'w')
    observations=0;inference_seconds=tracking_seconds=0;last_commit=time.perf_counter();max_batch_seconds=0
    first_index=None;last_index=None
    try:
        while True:
            batch=buffer.get()
            if isinstance(batch,BaseException):raise batch
            if batch is None:break
            t=time.perf_counter();items=detector.infer(batch);synchronize();inference=time.perf_counter()-t
            inference_seconds+=inference;max_batch_seconds=max(max_batch_seconds,inference)
            for source,item in zip(batch,items):
                t=time.perf_counter();record,events,audits=engine.update(item);tracking_seconds+=time.perf_counter()-t
                for e in events:db.execute('INSERT INTO events VALUES(?,?,?,?,?,?,?)',(e['event_id'],e['direction'],e['frame'],e['seconds'],e['track_id'],e['x'],e['y']))
                for audit in audits:db.execute('INSERT INTO audits VALUES(?,?)',(audit['umbrella_event_id'],json.dumps(audit)))
                if records:records.write(json.dumps(record)+'\n')
                if detections:detections.write(json.dumps(item)+'\n')
                latency=time.perf_counter()-(source['scheduled'] if a.pace else source['arrival'])
                metrics.write(json.dumps(dict(frame=item['index'],latency_seconds=latency,inference_batch_seconds=inference,batch_size=len(batch)))+'\n')
                observations+=1;last_index=item['index']
                if first_index is None:first_index=last_index
            if time.perf_counter()-last_commit>=1:
                for handle in [records,detections,metrics]:
                    if handle:handle.flush()
                db.execute('INSERT OR REPLACE INTO checkpoint VALUES(1,?,?)',(digest,pickle.dumps(engine.snapshot())))
                db.commit();last_commit=time.perf_counter()
            if observations%100<len(batch):print(json.dumps(dict(observations=observations,seconds=round(time.perf_counter()-wall_start,2),counts=engine.summary())),flush=True)
        if a.max_seconds is None:
            for audit in engine.audit(engine.last_frame/fps,final=True):db.execute('INSERT INTO audits VALUES(?,?)',(audit['umbrella_event_id'],json.dumps(audit)))
        for handle in [records,detections,metrics]:
            if handle:handle.flush()
        db.execute('INSERT OR REPLACE INTO checkpoint VALUES(1,?,?)',(digest,pickle.dumps(engine.snapshot())));db.commit()
    except BaseException:
        db.close()
        raise
    finally:
        stop.set()
        producer.join(timeout=5)
        if records:records.close()
        if detections:detections.close()
        metrics.close()
    wall=time.perf_counter()-wall_start
    with (a.output/'events.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['event_id','direction','frame','seconds','track_id','x','y'])
        writer.writerows(db.execute('SELECT * FROM events ORDER BY event_id'))
    with (a.output/'umbrella_audit.json').open('w') as f:
        f.write('[');separator=''
        for row in db.execute('SELECT payload FROM audits ORDER BY event_id'):
            f.write(separator+row[0]);separator=',\n'
        f.write(']')
    db.close()
    # Metrics kept on disk during processing; only aggregate after a finite benchmark.
    # A fixed-size histogram keeps post-run reporting bounded for long recordings.
    latency_histogram=np.zeros(60001,np.int64);latency_count=0;latency_max=0
    with (a.output/'timings.jsonl').open() as f:
        for line in f:
            row=json.loads(line)
            if row['frame']<start_index:continue
            latency=max(0,row['latency_seconds']);latency_max=max(latency_max,latency)
            latency_histogram[min(round(latency*1000),60000)]+=1;latency_count+=1
    def percentile(fraction):
        if not latency_count:return None
        return float(np.searchsorted(np.cumsum(latency_histogram),latency_count*fraction)/1000)
    video_seconds=(min(end_frame,last_index+a.stride)-first_index)/fps if observations else 0
    expected_observations=max(0,math.ceil((end_frame-start_index)/a.stride))
    summary=engine.summary()|dict(source=a.source,source_fps=fps,analysis_fps=fps/a.stride,
                 width=w,height=h,start_seconds=a.start_seconds,stride=a.stride,
                 observations=observations,video_seconds=video_seconds,processing_seconds=wall,
                 startup_seconds=startup,total_seconds=time.perf_counter()-process_start,
                 throughput_observations_fps=observations/wall,realtime_factor=video_seconds/wall,
                 expected_observations=expected_observations,all_observations_processed=observations==expected_observations,
                 inference_seconds=inference_seconds,tracking_seconds=tracking_seconds,
                 latency_p50_seconds=percentile(.5),latency_p95_seconds=percentile(.95),
                 latency_max_seconds=latency_max,max_inference_batch_seconds=max_batch_seconds,
                 paced=a.pace,queue_capacity_batches=2,queue_peak_capacity_frames=2*a.batch,
                 model=a.model,imgsz=a.imgsz,crop=a.crop,device='mps',batch=a.batch,half=False,
                 backend='torch',environment=environment(),
                 peak_process_rss_mb=peak_rss_mb(),
                 mps_current_allocated_mb=torch.mps.current_allocated_memory()/1024**2,
                 mps_driver_allocated_mb=torch.mps.driver_allocated_memory()/1024**2,
                 latency_histogram_resolution_seconds=.001,
                 umbrella_model=a.umbrella_model if a.camera=='glass' else 'shared person detector',
                 umbrella_device='mps',umbrella_every=a.umbrella_every,
                 note='Throughput includes decode, inference, ordered tracking, event persistence and requested JSONL output; excludes overlay/video encoding. File replay only; RTSP/network loss and eight-hour stability unverified.')
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
