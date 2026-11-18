"""Live RTSP version of run_fused.py: camera -> decode -> 3 networks -> 2 counters -> streaming merge.

Purpose: verify the data flow for real-time operation (keeping up with 25 fps input,
bounded latency, no unbounded buffering, reconnection). Counting parameters are the
calibrated glass preset, so counts are meaningless for a camera with another view.

The RTSP URL is read from ~/.mitafes_rtsp (mode 600) unless --url is given; it is never
printed or written to outputs.
"""
import argparse,json,os,queue,subprocess,threading,time,sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2,numpy as np,torch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from run_fused import Branch,BRANCHES,UMBRELLA_SHAPE,letterbox,to_items
from gpu_backend import GpuModel
from stream_merge import StreamMerger
from runtime_control import peak_rss_mb

W,H,FPS,STRIDE=2880,1620,25.0,2


def redact(text):
    text=str(text)
    return text.split('@',1)[-1] if '@' in text else text


class LiveReader(threading.Thread):
    """Reads every frame (HEVC needs it), keeps every STRIDE-th, never blocks on the consumer.

    A full queue means processing fell behind: that batch is dropped and counted,
    instead of letting the RTSP/TCP backlog and latency grow without bound.
    """
    def __init__(self,url,out,batch,stop,disconnect_at=None):
        super().__init__(daemon=True)
        self.url=url;self.out=out;self.batch=batch;self.stop=stop;self.disconnect_at=disconnect_at
        self.stats=dict(received=0,kept=0,dropped_frames=0,reconnects=0,read_failures=0,open_failures=0)
        self.t0=None;self.pts0=None;self.last_drift=None;self.error=None

    def open(self):
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS']='rtsp_transport;tcp|stimeout;5000000'
        cap=cv2.VideoCapture(self.url,cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,8])
        if not cap.isOpened():return None
        size=(int(cap.get(3)),int(cap.get(4)))
        if size!=(W,H):cap.release();raise ValueError(f'Stream is {size}, preset needs {(W,H)}')
        return cap

    def run(self):
        if isinstance(self.url,Path):return self.run_file()
        index=0;batch=[];simulated=False
        try:
            while not self.stop.is_set():
                cap=self.open()
                if cap is None:
                    self.stats['open_failures']+=1;self.stop.wait(2);continue
                if self.t0 is None:self.t0=time.perf_counter()
                else:
                    # Continue the time axis across the outage so the counters see a gap.
                    if batch:
                        try:self.out.put_nowait(batch)
                        except queue.Full:self.stats['dropped_frames']+=len(batch)
                        batch=[]
                    index=max(index,round((time.perf_counter()-self.t0)*FPS));index+=(-index)%STRIDE;self.pts0=None
                while not self.stop.is_set():
                    if self.disconnect_at and not simulated and time.perf_counter()-self.t0>=self.disconnect_at:
                        simulated=True;self.stats['reconnects']+=1;print(json.dumps(dict(event='simulated_disconnect')),flush=True);break
                    keep=index%STRIDE==0
                    if keep:ok,im=cap.read()
                    else:ok=cap.grab()
                    if not ok:self.stats['read_failures']+=1;self.stats['reconnects']+=1;break
                    now=time.perf_counter();self.stats['received']+=1
                    pts=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
                    if self.pts0 is None:self.pts0=(pts,now)
                    # Wall time minus stream time since connect: grows if decode falls behind the camera.
                    self.last_drift=(now-self.pts0[1])-(pts-self.pts0[0])
                    if keep:
                        self.stats['kept']+=1;batch.append(dict(index=index,image=im,arrival=now))
                        if len(batch)==self.batch:
                            try:self.out.put_nowait(batch)
                            except queue.Full:self.stats['dropped_frames']+=len(batch)
                            batch=[]
                    index+=1
                cap.release()
        except BaseException as exc:self.error=exc
        finally:
            if batch:self.out.put(batch)
            self.out.put(None)

    def run_file(self):
        """Virtual camera: frame i of a recording 'arrives' at t0 + i/25 s, like a live stream.

        Latency is measured from that scheduled arrival, so a reader that falls behind
        real time shows up as growing latency / drift, exactly as with a camera backlog.
        """
        batch=[]
        try:
            cap=cv2.VideoCapture(str(self.url),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,8])
            if not cap.isOpened():raise RuntimeError('Cannot open recording')
            size=(int(cap.get(3)),int(cap.get(4)))
            if size!=(W,H):raise ValueError(f'Recording is {size}, preset needs {(W,H)}')
            self.t0=time.perf_counter();index=0
            while not self.stop.is_set():
                scheduled=self.t0+index/FPS;delay=scheduled-time.perf_counter()
                if delay>0 and self.stop.wait(delay):break
                keep=index%STRIDE==0
                if keep:ok,im=cap.read()
                else:ok=cap.grab()
                if not ok:break
                now=time.perf_counter();self.stats['received']+=1;self.last_drift=now-scheduled
                if keep:
                    self.stats['kept']+=1;batch.append(dict(index=index,image=im,arrival=scheduled))
                    if len(batch)==self.batch:
                        try:self.out.put_nowait(batch)
                        except queue.Full:self.stats['dropped_frames']+=len(batch)
                        batch=[]
                index+=1
            cap.release()
        except BaseException as exc:self.error=exc
        finally:
            if batch:self.out.put(batch)
            self.out.put(None)


def gpu_power():
    try:
        out=subprocess.run(['nvidia-smi','--query-gpu=power.draw,temperature.gpu,utilization.gpu','--format=csv,noheader,nounits'],
                           capture_output=True,text=True,timeout=2).stdout.strip().split(',')
        return dict(power_w=float(out[0]),temp_c=int(out[1]),util=int(out[2]))
    except Exception:return None


def main():
    started=time.perf_counter()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',help='RTSP URL (default: contents of ~/.mitafes_rtsp)')
    p.add_argument('--file',type=Path,help='Replay a recording at real-time pace instead of a camera')
    p.add_argument('--duration',type=float,default=600,help='Seconds to run')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='0')
    p.add_argument('--models-dir',type=Path,default=ROOT/'models')
    p.add_argument('--batch',type=int,default=8);p.add_argument('--prep-threads',type=int,default=6)
    p.add_argument('--queue-batches',type=int,default=4)
    p.add_argument('--simulate-disconnect-at',type=float,help='Drop and reopen the stream after N seconds')
    a=p.parse_args()
    url=a.file.resolve() if a.file else (a.url or (Path.home()/'.mitafes_rtsp').read_text().strip())
    a.output=a.output.resolve()
    if a.output.exists() and any(a.output.iterdir()):raise FileExistsError('Choose an unused output directory')
    a.output.mkdir(parents=True)
    torch.set_num_threads(2);cv2.setNumThreads(1);torch.backends.cudnn.benchmark=True
    meta=dict(width=W,height=H,fps=FPS)
    branches=[Branch(n,s,c,o,meta,a.output,a.models_dir/'yolo11m-pose.pt',a.device,a.batch,False,STRIDE) for n,s,c,o in BRANCHES]
    umbrella=GpuModel(a.models_dir/'yolo11n.pt',a.device,UMBRELLA_SHAPE,'graph',a.batch,False)
    nets=[b.net for b in branches]+[umbrella]
    jobs=[(b.shape,b.crop) for b in branches]+[(UMBRELLA_SHAPE,[0,0,W,H])]
    merger=StreamMerger();startup=time.perf_counter()-started
    print(json.dumps(dict(event='ready',startup_seconds=round(startup,1),source=redact(url))),flush=True)

    frames=queue.Queue(maxsize=a.queue_batches);stop=threading.Event()
    reader=LiveReader(url,frames,a.batch,stop,a.simulate_disconnect_at);reader.start()
    pool=ThreadPoolExecutor(a.prep_threads)
    log=(a.output/'live_status.jsonl').open('w');events_log=(a.output/'live_events.jsonl').open('w')
    window=deque();all_latency=[];processed=0;last_report=time.perf_counter();last_index=-1;gaps=0
    deadline=time.perf_counter()+a.duration
    def prep(image,shape,crop):
        x1,y1,x2,y2=crop;return letterbox(image[y1:y2,x1:x2],shape)
    try:
        while True:
            if time.perf_counter()>=deadline:stop.set()
            try:batch=frames.get(timeout=1)
            except queue.Empty:
                if stop.is_set() and not reader.is_alive():break
                continue
            if batch is None:break
            arrays=[np.stack([f.result() for f in [pool.submit(prep,r['image'],s,c) for r in batch]]) for s,c in jobs]
            shapes=[[(c[3]-c[1],c[2]-c[0],3)]*len(batch) for _,c in jobs]
            for r in batch:r.pop('image')
            for net,array,shape in zip(nets,arrays,shapes):
                if len(array)<a.batch:array=np.concatenate([array,np.zeros((a.batch-len(array),*array.shape[1:]),np.uint8)])
                net.enqueue_prepared(array,shape)
            people=[net.collect(.1,.7,[0]) for net in nets[:2]];umbrellas=umbrella.collect(.1,.7,[25])
            per_branch=[to_items(batch,ps,umbrellas,b.crop,W,H) for b,ps in zip(branches,people)]
            for i,source in enumerate(batch):
                if last_index>=0 and source['index']!=last_index+STRIDE:gaps+=1
                last_index=source['index']
                (rf,ef),(rr,er)=[b.update(items[i]) for b,items in zip(branches,per_branch)]
                added=merger.update(source['index'],rf,rr,ef,er)
                now=time.perf_counter()
                for e in ef:events_log.write(json.dumps(dict(e,source='full',wall=now))+'\n')
                for e in added:events_log.write(json.dumps(dict(e,source='roi_added',wall=now))+'\n')
                latency=now-source['arrival'];window.append(latency);all_latency.append(latency);processed+=1
            if time.perf_counter()-last_report>=10:
                lat=sorted(window);window.clear();last_report=time.perf_counter()
                elapsed=last_report-reader.t0 if reader.t0 else 0
                row=dict(t=round(elapsed,1),received_fps=round(reader.stats['received']/elapsed,2) if elapsed else None,
                         processed_obs=processed,queue_batches=frames.qsize(),dropped_frames=reader.stats['dropped_frames'],
                         reconnects=reader.stats['reconnects'],gaps=gaps,
                         decode_drift_s=round(reader.last_drift,3) if reader.last_drift is not None else None,
                         latency_p50=round(lat[len(lat)//2],3) if lat else None,latency_p95=round(lat[int(len(lat)*.95)],3) if lat else None,
                         counts=merger.totals(),gpu=gpu_power(),rss_mb=round(peak_rss_mb()))
                log.write(json.dumps(row)+'\n');log.flush();events_log.flush();print(json.dumps(row),flush=True)
    finally:
        stop.set();reader.join(timeout=10);pool.shutdown(cancel_futures=True);log.close()
    for e in merger.finish():events_log.write(json.dumps(dict(e,source='roi_added',wall=time.perf_counter()))+'\n')
    events_log.close()
    for b in branches:b.close(dict(mode='live'))
    if reader.error:raise reader.error
    lat=sorted(all_latency)
    from merge_indexed import merge
    offline=merge(a.output/'full',a.output/'roi',a.output/'result_offline_check')
    summary=dict(duration_seconds=a.duration,startup_seconds=startup,reader=reader.stats,processed_observations=processed,
                 expected_observations_at_12_5fps=round(a.duration*FPS/STRIDE),index_gaps=gaps,
                 latency_p50=lat[len(lat)//2] if lat else None,latency_p95=lat[int(len(lat)*.95)] if lat else None,
                 latency_max=lat[-1] if lat else None,streaming_counts=merger.totals(),
                 offline_merge_counts=dict(ltr=offline['ltr'],rtl=offline['rtl'],events=offline['events']),
                 streaming_equals_offline=merger.totals()['events']==offline['events'],peak_process_rss_mb=peak_rss_mb(),
                 note='Counts use the calibrated glass preset; with a different camera view they are not accuracy measurements.')
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
