"""Optimized GPU inference with bounded buffering and persistent counting.

No manual reference or expected event count is read by this program.
"""
import argparse,csv,json,os,pickle,queue,sqlite3,threading,time,hashlib,resource,copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2
import numpy as np
import torch
import ultralytics.utils.torch_utils as torch_utils
from ultralytics import YOLO
from engine import StreamingCounter
from runtime_control import (select_platform,capture_file,put_buffer,validate_sources,
                             observations_between,recording_identity,model_identities,
                             trim_uncommitted_logs,latency_windows,environment,peak_rss_mb,file_sha256,dependency_versions)

ROOT=Path(__file__).resolve().parent
POC=ROOT.parent/'counting_poc'


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',nargs='+',required=True)
    p.add_argument('--camera',choices=['glass','overhead'],required=True)
    p.add_argument('--platform',choices=['auto','cuda','mps'],default='auto')
    p.add_argument('--backend',choices=['eager','graph','onnx','trt','trt_graph'],default='graph')
    p.add_argument('--model',default=None)
    p.add_argument('--umbrella-model',default=None)
    p.add_argument('--imgsz',type=int,default=None)
    p.add_argument('--umbrella-imgsz',type=int,default=640)
    p.add_argument('--person-shape',type=int,nargs=2,default=None)
    p.add_argument('--umbrella-shape',type=int,nargs=2,default=[384,640])
    p.add_argument('--crop',type=float,nargs=4,default=None)
    p.add_argument('--rect-letterbox',action='store_true',default=True)
    p.add_argument('--parallel-models',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--cudnn-benchmark',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--decoder',choices=['cv2','ffmpeg','nvdec'],default='cv2')
    p.add_argument('--decode-width',type=int,default=1280)
    p.add_argument('--decode-threads',type=int,default=2)
    p.add_argument('--preview',action='store_true',help='Time online annotation and H.264 output too')
    p.add_argument('--encoder',choices=['nvenc','x264'],default='nvenc')
    p.add_argument('--start-barrier',default=None,help='Optional local file barrier for synchronized validation')
    p.add_argument('--device',default='0')
    p.add_argument('--devices',nargs='+',default=None,help='Optional offline neural worker GPUs; counting remains ordered')
    p.add_argument('--worker-mode',choices=['thread','process'],default='thread')
    p.add_argument('--umbrella-device',default=None)
    p.add_argument('--stride',type=int,default=2)
    p.add_argument('--person-iou',type=float,default=.7)
    p.add_argument('--profile-overrides',default='{}',help='JSON camera/tracker parameter overrides')
    p.add_argument('--umbrella-every',type=int,default=1)
    p.add_argument('--flow-every',type=int,choices=[1],default=1,help=argparse.SUPPRESS)
    p.add_argument('--batch',type=int,default=1)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--scalar-tracker',action='store_true',help='Use frozen pairwise Python loops for comparison')
    p.add_argument('--half',action='store_true')
    p.add_argument('--pace',action='store_true',help='Replay at source timestamps to measure live-like latency')
    p.add_argument('--start-seconds',type=float,default=0,help='Source offset; event times start from this point')
    p.add_argument('--max-seconds',type=float,default=None)
    p.add_argument('--record-tracks',action='store_true')
    p.add_argument('--record-detections',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    return p


def capture(source,threads=2):
    return capture_file(source,threads)


class Detector:
    def __init__(self,a,w,h):
        self.a=a;self.w=w;self.h=h
        self.crop=[round(v*s) for v,s in zip(a.crop,[w,h,w,h])]
        x1,y1,x2,y2=self.crop
        if not (0<=x1<x2<=w and 0<=y1<y2<=h):raise ValueError('Invalid crop')
        if a.platform=='mps':
            from mps_backend import MpsModel
            self.person=MpsModel(a.model,a.person_shape)
        elif a.backend=='onnx':
            from onnx_backend import OnnxModel
            self.person=OnnxModel(a.model,a.device)
        elif a.backend.startswith('trt'):
            from trt_backend import TrtModel
            self.person=TrtModel(a.model,a.device,tuple(a.person_shape),a.backend,a.batch,a.half)
        else:
            from gpu_backend import GpuModel
            self.person=GpuModel(a.model,a.device,tuple(a.person_shape),a.backend,a.batch,a.half)
        if self.person.task!=('pose' if a.camera=='glass' else 'detect'):
            raise ValueError('Glass requires a pose model; overhead requires a detection model')
        self.umbrella_device=a.umbrella_device or a.device
        if a.camera=='glass':
            if a.platform=='mps':self.umbrella=MpsModel(a.umbrella_model,a.umbrella_shape)
            elif a.backend=='onnx':self.umbrella=OnnxModel(a.umbrella_model,self.umbrella_device)
            elif a.backend.startswith('trt'):self.umbrella=TrtModel(a.umbrella_model,self.umbrella_device,tuple(a.umbrella_shape),a.backend,a.batch,a.half)
            else:self.umbrella=GpuModel(a.umbrella_model,self.umbrella_device,tuple(a.umbrella_shape),a.backend,a.batch,a.half)
        else:self.umbrella=None
        self.executor=ThreadPoolExecutor(2) if self.umbrella and self.umbrella_device!=a.device else None
        self.flow=None
        if a.flow_every!=1:raise ValueError('GPU presets keep every observation neural; optical flow is disabled')

    def predict_person(self,images):
        x1,y1,x2,y2=self.runtime_crop
        cropped=[im[y1:y2,x1:x2] for im in images]
        self.person_observation_kind='network'
        gray=None;fallback=False
        if self.flow:
            gray=self.flow.gray(cropped[0])
            if (self.current_index//self.a.stride)%self.a.flow_every:
                proposal,fallback=self.flow.propose(cropped[0],gray,self.person.names)
                if not fallback:
                    self.flow.stats['flow_calls']+=1;self.person_observation_kind='optical_flow'
                    self.flow.commit(gray,proposal);return [proposal]
        result=self.person.predict(cropped,imgsz=self.a.imgsz,
                                   conf=.1,iou=self.a.person_iou,classes=[0] if self.umbrella else [0,25],
                                   device=self.a.device,half=self.a.half,verbose=False)
        if self.flow:
            self.flow.stats['network_calls']+=1
            if fallback:self.flow.stats['fallback_calls']+=1;self.person_observation_kind='network_fallback'
            self.flow.commit(gray,result[0])
        return result

    def predict_umbrella(self,images):
        return self.umbrella.predict(images,imgsz=self.a.umbrella_imgsz,conf=.1,iou=.7,classes=[25],
                                     device=self.umbrella_device,half=self.a.half,verbose=False)

    def infer(self,batch):
        self.current_index=batch[0]['index']
        images=[r['image'] for r in batch]
        ih,iw=images[0].shape[:2]
        self.runtime_crop=[round(v*s) for v,s in zip(self.a.crop,[iw,ih,iw,ih])]
        u_indices=[i for i,r in enumerate(batch) if (r['index']//self.a.stride)%self.a.umbrella_every==0]
        if self.a.platform=='cuda' and self.a.parallel_models and self.a.backend!='onnx' and self.umbrella and u_indices:
            x1,y1,x2,y2=self.runtime_crop
            self.person.enqueue([im[y1:y2,x1:x2] for im in images])
            self.umbrella.enqueue([images[i] for i in u_indices])
            people=self.person.collect(.1,self.a.person_iou,[0])
            umbrellas=dict(zip(u_indices,self.umbrella.collect(.1,.7,[25])))
        elif self.executor and u_indices:
            ufuture=self.executor.submit(self.predict_umbrella,[images[i] for i in u_indices])
            people=self.predict_person(images);umbrellas=dict(zip(u_indices,ufuture.result()))
        else:
            people=self.predict_person(images)
            umbrellas=dict(zip(u_indices,self.predict_umbrella([images[i] for i in u_indices]))) if self.umbrella and u_indices else {}
        x1,y1,_,_=self.runtime_crop
        result=[]
        for i,(record,p) in enumerate(zip(batch,people)):
            det=p.boxes.data.cpu().numpy().copy()
            if len(det):det[:,[0,2]]+=x1;det[:,[1,3]]+=y1
            points=p.keypoints.data.cpu().numpy().copy() if p.keypoints is not None else None
            if points is not None and len(points):points[:,:,0]+=x1;points[:,:,1]+=y1
            if len(det):det[:,[0,2]]*=self.w/iw;det[:,[1,3]]*=self.h/ih
            if points is not None and len(points):points[:,:,0]*=self.w/iw;points[:,:,1]*=self.h/ih
            det=det.tolist();points=points.tolist() if points is not None else None
            if i in umbrellas:
                udets=umbrellas[i].boxes.data.cpu().numpy().copy()
                if len(udets):udets[:,[0,2]]*=self.w/iw;udets[:,[1,3]]*=self.h/ih
                udets=udets.tolist();det.extend(udets)
                points.extend([[[0,0,0] for _ in range(17)] for _ in udets])
            result.append(dict(index=record['index'],detections=det,
                               umbrella_observed=self.umbrella is None or i in umbrellas,
                               **({'person_observation_kind':self.person_observation_kind} if self.flow else {}),
                               **({'keypoints':points} if points is not None else {})))
        return result

    def warmup(self):
        sample=[dict(image=np.zeros((self.h,self.w,3),np.uint8),index=i*self.a.stride) for i in range(self.a.batch)]
        for _ in range(2):self.infer(sample)
        if self.flow:
            self.flow.previous_gray=None;self.flow.boxes=None;self.flow.keypoints=None
            self.flow.stats={key:0 for key in self.flow.stats}

    def snapshot(self):return self.flow.snapshot() if self.flow else None

    def restore(self,state):
        if state is not None:
            if self.flow is None:raise ValueError('Checkpoint requires optical flow')
            self.flow.restore(state)


def produce(a,buffer,stop,meta,wall_start,start_index):
    if getattr(a,'decoder','cv2')!='cv2':
        from video_io import produce_ffmpeg
        return produce_ffmpeg(a,buffer,stop,meta,wall_start,start_index)
    cap=None
    try:
        offset=-round(getattr(a,'start_seconds',0)*meta['fps']);batch=[]
        for source in a.source:
            if stop.is_set():return
            cap=capture(source,getattr(a,'decode_threads',2))
            w,h,fps=int(cap.get(3)),int(cap.get(4)),cap.get(5)
            if (w,h)!=(meta['width'],meta['height']) or abs(fps-meta['fps'])>.001:
                raise ValueError('Segment dimensions and FPS must agree; tracking is continuous across segments')
            total=int(cap.get(7));first=max(0,start_index-offset)
            if total>0 and first>=total:offset+=total;cap.release();continue
            if first and not cap.set(cv2.CAP_PROP_POS_FRAMES,first):raise RuntimeError(f'Cannot seek to frame {first}: {source}')
            local=first
            while not stop.is_set():
                index=offset+local
                if a.max_seconds is not None and index/fps>=a.max_seconds:break
                if index%a.stride:
                    ok=cap.grab()
                    if not ok:
                        if local<total:raise RuntimeError(f'Decode stopped early at frame {local}/{total}: {source}')
                        break
                    local+=1
                    continue
                ok,im=cap.read()
                if not ok:
                    if local<total:raise RuntimeError(f'Decode stopped early at frame {local}/{total}: {source}')
                    break
                local+=1
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
    a.platform=select_platform(a.platform)
    if a.platform=='mps':
        if a.half or a.worker_mode!='thread' or a.backend not in ['graph','eager'] or a.decoder=='nvdec':
            raise ValueError('MPS uses FP32 eager inference, thread workers, and software video decoding')
        if a.devices and a.devices!=['mps']:raise ValueError('MPS uses a single Apple GPU; omit --devices')
        a.device='mps';a.devices=None;a.umbrella_device=None;a.backend='eager'
        if a.preview:a.encoder='x264'
    if a.devices:a.device=a.devices[0]
    models=ROOT/'models'
    if not (models/'yolo11m-pose.pt').exists() and (ROOT.parent/'cpu_counter/models').is_dir():models=ROOT.parent/'cpu_counter/models'
    if a.person_shape is None:a.person_shape=[736,512] if a.camera=='glass' else [736,1280]
    if a.imgsz is None:a.imgsz=max(a.person_shape)
    if a.imgsz!=max(a.person_shape):raise ValueError('Change --person-shape to select a different static input')
    if any(v%32 for v in a.person_shape+a.umbrella_shape):raise ValueError('Static input dimensions must be multiples of 32')
    if a.crop is None:a.crop=[.4,0,.8,1] if a.camera=='glass' else [0,0,1,1]
    if a.model is None:
        if a.backend.startswith('trt'):a.model=str(ROOT/'models'/f'{a.camera}_fp32_b{a.batch}.engine')
        else:
            stem=('yolo11m-pose-exact512' if a.camera=='glass' else 'yolo11n-1280') if a.backend=='onnx' else ('yolo11m-pose' if a.camera=='glass' else 'yolo11n')
            a.model=str(models/(stem+('.onnx' if a.backend=='onnx' else '.pt')))
    if a.umbrella_model is None:a.umbrella_model=str(ROOT/'models'/f'umbrella_fp32_b{a.batch}.engine') if a.backend.startswith('trt') else str(models/('yolo11n-640.onnx' if a.backend=='onnx' else 'yolo11n.pt'))
    if a.batch<1 or a.stride<1 or a.umbrella_every<1 or a.flow_every<1 or a.threads<1 or a.decode_threads<1:raise ValueError('Batch, strides, and threads must be positive')
    meta=validate_sources(a.source,a.start_seconds,a.max_seconds,a.decode_threads)
    w,h,fps=meta['width'],meta['height'],meta['fps']
    if any(not np.isfinite(v) for v in a.crop) or not (0<=a.crop[0]<a.crop[2]<=1 and 0<=a.crop[1]<a.crop[3]<=1):raise ValueError('Invalid crop')
    torch_threads=a.threads
    torch_utils.NUM_THREADS=torch_threads;torch.set_num_threads(torch_threads);cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark=a.cudnn_benchmark
    profile_name='glass_1219_heads_including_bottom.json' if a.camera=='glass' else 'overhead_1300_v2.json'
    profile=json.loads((ROOT/'reference/profiles'/profile_name).read_text())
    overrides=json.loads(a.profile_overrides)
    if not isinstance(overrides,dict):raise ValueError('Profile overrides must be an object')
    profile.update(overrides)
    profile['vectorized_tracking']=not a.scalar_tracker
    if a.umbrella_every!=1:profile['umbrella_every']=a.umbrella_every
    a.output.mkdir(parents=True,exist_ok=True)
    signature={k:v for k,v in vars(a).items() if k not in ['output','resume','pace','record_tracks','record_detections','max_seconds','start_barrier']}
    if a.person_iou==.7:signature.pop('person_iou') # Preserve pre-option checkpoints.
    if not overrides:signature.pop('profile_overrides')
    if a.flow_every==1:signature.pop('flow_every') # Preserve compatibility with pre-flow checkpoints.
    else:
        from flow_inference import HeadOpticalFlow
        signature['flow_policy']=HeadOpticalFlow.POLICY
    if not a.rect_letterbox:signature.pop('rect_letterbox')
    signature['profile']=profile
    signature['signature_version']=2
    signature['dependency_versions']=dependency_versions()
    if a.platform=='mps':signature['mps_environment']=environment('mps')['mps_environment']
    signature['sources']=recording_identity(a.source)
    signature['model_sha256']=model_identities([a.model]+([a.umbrella_model] if a.camera=='glass' else []))
    signature['implementation_sha256']={name:file_sha256(ROOT/name) for name in
        ['run.py','engine.py','robust_tracker.py','occlusion_counter.py','vectorized_tracker.py','vectorized_association.py',
         'gpu_backend.py','mps_backend.py','runtime_control.py','reference/pose_tracker.py','reference/camera_counter.py']}
    digest=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
    db=sqlite3.connect(a.output/'events.sqlite');db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS events(event_id INTEGER PRIMARY KEY,direction TEXT,frame INTEGER,seconds REAL,track_id INTEGER,x REAL,y REAL)')
    db.execute('CREATE TABLE IF NOT EXISTS audits(event_id INTEGER PRIMARY KEY,payload TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS checkpoint(id INTEGER PRIMARY KEY,signature TEXT,state BLOB)')
    saved=db.execute('SELECT signature,state FROM checkpoint WHERE id=1').fetchone()
    if saved:
        if not a.resume:db.close();raise ValueError('Output already contains a checkpoint; choose a new output or --resume')
        if saved[0]!=digest:db.close();raise ValueError('Resume configuration does not match checkpoint, model, input, or counting implementation')
        # Only load checkpoints generated locally by this program.
        restored=pickle.loads(saved[1]);engine=StreamingCounter.restore(restored);start_index=engine.last_frame+a.stride
        if engine.last_frame>=meta['end_frame']:db.close();raise ValueError('Requested end is earlier than the committed recording position')
    else:engine=StreamingCounter(profile,w,h,fps,a.stride);start_index=0
    if a.resume:
        # Debug logs are not the canonical store. Discard rows newer than the
        # committed state so an interrupted batch cannot duplicate them on replay.
        trim_uncommitted_logs(a.output,engine.last_frame)
    devices=a.devices or [a.device]
    if len(set(devices))!=len(devices):raise ValueError('Worker GPUs must be distinct')
    if a.devices and a.umbrella_device is not None:raise ValueError('Use one worker GPU per model pair')
    detectors=[]
    for device in devices:
        worker=copy.copy(a);worker.device=device
        if a.worker_mode=='process':
            from process_workers import RemoteDetector
            current=RemoteDetector(worker,w,h)
        else:current=Detector(worker,w,h);current.warmup()
        detectors.append(current)
    detector=detectors[0]
    if saved:detector.restore(restored.get('detector_state'))
    startup_gpu_peak_mb=torch.cuda.max_memory_allocated(int(a.device))/1048576 if a.platform=='cuda' and a.worker_mode=='thread' else None
    if a.platform=='cuda' and a.worker_mode=='thread':
        for device in devices:
            with torch.cuda.device(int(device)):torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(int(device))
    def checkpoint_snapshot():return engine.snapshot()|{'detector_state':detector.snapshot()}
    startup=time.perf_counter()-process_start
    (a.output/'config.json').write_text(json.dumps(dict(args=vars(a)|{'output':str(a.output)},profile=profile,
                                                    environment=environment(a.platform),identity=signature),indent=2))
    barrier_start=time.perf_counter()
    if a.start_barrier:
        (a.output/'ready.json').write_text(json.dumps(dict(pid=os.getpid(),device=a.device)))
        while not Path(a.start_barrier).exists():
            if time.perf_counter()-barrier_start>120:raise TimeoutError('Start barrier was not released')
            time.sleep(.02)
    barrier_wait=time.perf_counter()-barrier_start
    buffer=queue.Queue(maxsize=2);stop=threading.Event();wall_start=time.perf_counter()
    producer=threading.Thread(target=produce,args=(a,buffer,stop,dict(width=w,height=h,fps=fps),wall_start,start_index),daemon=True);producer.start()
    records=(a.output/'tracks.jsonl').open('a' if a.resume else 'w') if a.record_tracks else None
    detections=(a.output/'detections.jsonl').open('a' if a.resume else 'w') if a.record_detections else None
    metrics=(a.output/'timings.jsonl').open('a' if a.resume else 'w')
    observations=0;inference_seconds=tracking_seconds=0;last_commit=time.perf_counter();max_batch_seconds=0
    first_index=None;last_index=None
    preview=None;preview_info=None
    if a.preview:
        from video_io import PreviewSink
        preview=PreviewSink(a.output/('counting_preview.mp4' if not saved else f'preview_from_{start_index}.mp4'),w,h,fps/a.stride,profile,a.encoder,a.device)
    try:
        from inference_pool import inferred_batches
        for batch,items,inference in inferred_batches(buffer,detectors):
            if len(batch)!=len(items) or any(b['index']!=i['index'] for b,i in zip(batch,items)):
                raise RuntimeError('Inference observations do not match the source batch')
            inference_seconds+=inference;max_batch_seconds=max(max_batch_seconds,inference)
            for source,item in zip(batch,items):
                t=time.perf_counter();record,events,audits=engine.update(item);tracking_seconds+=time.perf_counter()-t
                for e in events:db.execute('INSERT INTO events VALUES(?,?,?,?,?,?,?)',(e['event_id'],e['direction'],e['frame'],e['seconds'],e['track_id'],e['x'],e['y']))
                for audit in audits:db.execute('INSERT INTO audits VALUES(?,?)',(audit['umbrella_event_id'],json.dumps(audit)))
                if records:records.write(json.dumps(record)+'\n')
                if detections:detections.write(json.dumps(item)+'\n')
                if preview:preview.put(source['image'],record,engine.summary(),fps)
                latency=time.perf_counter()-(source['scheduled'] if a.pace else source['arrival'])
                metrics.write(json.dumps(dict(frame=item['index'],latency_seconds=latency,inference_batch_seconds=inference,batch_size=len(batch)))+'\n')
                observations+=1;last_index=item['index']
                if first_index is None:first_index=last_index
            if time.perf_counter()-last_commit>=1:
                for handle in [records,detections,metrics]:
                    if handle:handle.flush()
                db.execute('INSERT OR REPLACE INTO checkpoint VALUES(1,?,?)',(digest,pickle.dumps(checkpoint_snapshot())))
                db.commit();last_commit=time.perf_counter()
            if observations%100<len(batch):print(json.dumps(dict(observations=observations,seconds=round(time.perf_counter()-wall_start,2),counts=engine.summary())),flush=True)
        if observations!=observations_between(start_index,meta['end_frame'],a.stride):
            raise RuntimeError('The recording ended without processing all planned observations')
        if a.max_seconds is None:
            for audit in engine.audit(engine.last_frame/fps,final=True):db.execute('INSERT INTO audits VALUES(?,?)',(audit['umbrella_event_id'],json.dumps(audit)))
        for handle in [records,detections,metrics]:
            if handle:handle.flush()
        db.execute('INSERT OR REPLACE INTO checkpoint VALUES(1,?,?)',(digest,pickle.dumps(checkpoint_snapshot())));db.commit()
    except BaseException:
        db.rollback();db.close()
        raise
    finally:
        stop.set()
        producer.join(timeout=5)
        if records:records.close()
        if detections:detections.close()
        metrics.close()
        if preview:preview_info=preview.close()
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
    video_seconds=min(last_index+a.stride,meta['end_frame'])/fps-first_index/fps if observations else 0
    expected=observations_between(start_index,meta['end_frame'],a.stride)
    summary=engine.summary()|dict(source=a.source,source_fps=fps,analysis_fps=fps/a.stride,
                 resumed=bool(saved),metrics_start_frame=start_index,
                 timing_scope='This invocation only; event counts include restored history when resumed.',
                 observations=observations,video_seconds=video_seconds,processing_seconds=wall,
                 expected_observations=expected,all_observations_processed=observations==expected,
                 requested_duration_available=meta['requested_duration_available'],platform=a.platform,
                 startup_seconds=startup,total_seconds=time.perf_counter()-process_start,
                 barrier_wait_seconds=barrier_wait,tensorrt_version=getattr(detector.person,'tensorrt_version',None),
                 throughput_observations_fps=observations/wall,realtime_factor=video_seconds/wall,
                 inference_seconds=inference_seconds,tracking_seconds=tracking_seconds,
                 latency_p50_seconds=percentile(.5),latency_p95_seconds=percentile(.95),
                 latency_max_seconds=latency_max,max_inference_batch_seconds=max_batch_seconds,
                 paced=a.pace,queue_capacity_batches=2,queue_peak_capacity_frames=2*a.batch,
                 model=a.model,imgsz=a.imgsz,crop=a.crop,device=a.device,batch=a.batch,half=a.half,
                 rect_letterbox=a.rect_letterbox,
                 backend=a.backend,
                 flow_every=a.flow_every,flow_statistics=detector.flow.stats if detector.flow else None,
                 flow_policy=detector.flow.POLICY if detector.flow else None,
                 person_graph_precision=getattr(detector.person,'graph_precision','FP32'),
                 vectorized_tracking=not a.scalar_tracker,
                 cpu_only=False,torch_cuda_build=torch.version.cuda,threads=a.threads,
                 gpu_name=torch.cuda.get_device_name(int(a.device)) if a.platform=='cuda' else 'Apple Silicon GPU',gpu_count_used=len(set(devices+[a.umbrella_device or a.device])),
                 neural_worker_devices=devices,inference_seconds_scope='Sum of worker batch times; overlaps when multiple GPUs are used.',
                 worker_mode=a.worker_mode,worker_gpu_stats=[d.stats() for d in detectors] if a.worker_mode=='process' else None,
                 process_transport=getattr(detector,'transport',None),
                 cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                 person_shape=a.person_shape,umbrella_shape=a.umbrella_shape,parallel_models=a.parallel_models,
                 decoder=a.decoder,decode_width=a.decode_width if a.decoder!='cv2' else w,
                 decode_threads=a.decode_threads,
                 preview=preview_info,encoder=a.encoder if a.preview else None,
                 peak_gpu_allocated_mb=torch.cuda.max_memory_allocated(int(a.device))/1048576 if a.platform=='cuda' else None,
                 gpu_memory_scope='Torch allocator only; TensorRT, CUDA context, and separate codec processes are excluded. Live validation includes a total-device NVML snapshot.',
                 startup_gpu_peak_mb=startup_gpu_peak_mb,
                 final_gpu_allocated_mb=torch.cuda.memory_allocated(int(a.device))/1048576 if a.platform=='cuda' else None,
                 final_gpu_reserved_mb=torch.cuda.memory_reserved(int(a.device))/1048576 if a.platform=='cuda' else None,
                 mps_current_allocated_mb=torch.mps.current_allocated_memory()/1048576 if a.platform=='mps' else None,
                 mps_driver_allocated_mb=torch.mps.driver_allocated_memory()/1048576 if a.platform=='mps' else None,
                 peak_process_rss_mb=peak_rss_mb(),
                 latency_windows=latency_windows(a.output/'timings.jsonl',fps,a.pace,start_index),
                 latency_histogram_resolution_seconds=.001,
                 umbrella_model=a.umbrella_model if a.camera=='glass' else 'shared person detector',
                 umbrella_device=a.umbrella_device or a.device,umbrella_every=a.umbrella_every,
                 note='Includes decode, inference, ordered tracking, persistence, logs, and optional preview drawing/encoding/flush. Model initialization reported separately. RTSP loss and full eight-hour stability remain unverified.')
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
    if a.worker_mode=='process':
        for current in detectors:current.close()


if __name__=='__main__':main()
