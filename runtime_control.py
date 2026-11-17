"""Portable recording validation and cancellation, adapted from the Mac runner.

Upstream: Koki0724/mitafes, 84f080f4dbb01c9247ea6dd7bfe8492a5af5aa7f.
Backend selection does not change the camera counting policy.
"""
import hashlib,json,math,os,platform,queue,resource,sys
from importlib.metadata import version
from pathlib import Path
import cv2


def select_platform(requested='auto'):
    import torch
    if requested=='auto':
        requested='cuda' if torch.cuda.is_available() else 'mps'
    if requested=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable; inference was not switched to CPU')
    elif requested=='mps':
        if platform.system()!='Darwin' or platform.machine()!='arm64':
            raise RuntimeError('MPS requires an Apple Silicon Mac and arm64 Python; CPU fallback is disabled')
        if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable; CPU fallback is disabled')
        value=(torch.ones(4,device='mps')*2).sum();torch.mps.synchronize()
        if value.item()!=8:raise RuntimeError('MPS allocation/operation check failed')
    else:raise ValueError('Unknown platform')
    return requested


def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024)


def file_sha256(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def put_buffer(buffer,item,stop):
    while not stop.is_set():
        try:buffer.put(item,timeout=.1);return True
        except queue.Full:pass
    return False


def capture_file(source,threads=2):
    if not Path(source).is_file():raise ValueError(f'Local video file required: {source}')
    params=[cv2.CAP_PROP_N_THREADS,threads] if hasattr(cv2,'CAP_PROP_N_THREADS') else []
    cap=cv2.VideoCapture(str(source),cv2.CAP_FFMPEG,params)
    if not cap.isOpened():
        cap.release();cap=cv2.VideoCapture(str(source),cv2.CAP_FFMPEG)
    if not cap.isOpened():cap.release();raise RuntimeError(f'Cannot open source: {source}')
    return cap


def validate_sources(sources,start_seconds=0,max_seconds=None,threads=2):
    if not sources:raise ValueError('At least one source is required')
    if not math.isfinite(start_seconds) or start_seconds<0:raise ValueError('Start seconds must be finite and nonnegative')
    if max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds<=0):
        raise ValueError('Max seconds must be finite and positive')
    metadata=[]
    for source in sources:
        cap=capture_file(source,threads)
        try:m=dict(width=int(cap.get(3)),height=int(cap.get(4)),fps=cap.get(5),frames=int(cap.get(7)))
        finally:cap.release()
        if m['width']<=0 or m['height']<=0 or not math.isfinite(m['fps']) or m['fps']<=0 or m['frames']<=0:
            raise ValueError(f'Invalid recording metadata: {source}')
        metadata.append(m)
    first=metadata[0]
    if any(m['width']!=first['width'] or m['height']!=first['height'] or abs(m['fps']-first['fps'])>.001 for m in metadata):
        raise ValueError('Segment dimensions and FPS must agree')
    available=sum(m['frames'] for m in metadata)-round(start_seconds*first['fps'])
    if available<=0:raise ValueError('Start offset is past the end of the input')
    end=min(available,math.ceil(max_seconds*first['fps'])) if max_seconds is not None else available
    return first|dict(available_frames=available,end_frame=end,segments=metadata,
                     requested_duration_available=max_seconds is None or available>=math.ceil(max_seconds*first['fps']))


def observations_between(start,end,stride):
    first=start+(-start)%stride
    return max(0,(end-1-first)//stride+1)


def recording_identity(sources):
    return [dict(path=str(Path(p).resolve()),size=Path(p).stat().st_size,mtime_ns=Path(p).stat().st_mtime_ns) for p in sources]


def model_identities(paths):
    return {str(Path(p).resolve()):file_sha256(p) for p in dict.fromkeys(paths)}


def trim_uncommitted_logs(folder,last_frame):
    for name in ['tracks.jsonl','detections.jsonl','timings.jsonl']:
        path=Path(folder)/name
        if not path.exists():continue
        temp=path.with_suffix('.truncate.tmp')
        with path.open() as src,temp.open('w') as dst:
            for line in src:
                try:row=json.loads(line)
                except json.JSONDecodeError:break
                frame=row.get('frame',row.get('index'))
                if frame is not None and frame<=last_frame:dst.write(line)
        temp.replace(path)


def latency_windows(path,fps,paced,start_frame=0):
    import numpy as np
    windows=[];bucket=None;latencies=[]
    def append():
        if latencies:windows.append(dict(source_start_seconds=bucket*10,observations=len(latencies),
                                         latency_p50_seconds=float(np.median(latencies)),latency_p95_seconds=float(np.percentile(latencies,95))))
    with Path(path).open() as f:
        for line in f:
            row=json.loads(line)
            if row['frame']<start_frame:continue
            current=int(row['frame']/fps//10)
            if current!=bucket:append();bucket=current;latencies=[]
            latencies.append(row['latency_seconds'])
    append()
    return dict(latency_basis='source schedule' if paced else 'decoded frame arrival',windows=windows,
                last_minus_first_window_p95_seconds=windows[-1]['latency_p95_seconds']-windows[0]['latency_p95_seconds'] if len(windows)>=2 and paced else None)


def environment(accelerator):
    import torch
    return dict(platform=platform.platform(),machine=platform.machine(),python=sys.version,
                torch=torch.__version__,ultralytics=__import__('ultralytics').__version__,accelerator=accelerator,
                cuda_build=torch.version.cuda,mps_available=torch.backends.mps.is_available(),packages=dependency_versions(),
                mps_environment={k:os.environ.get(k) for k in ['PYTORCH_ENABLE_MPS_FALLBACK','PYTORCH_MPS_FAST_MATH','PYTORCH_MPS_HIGH_WATERMARK_RATIO','PYTORCH_MPS_PREFER_METAL']})


def dependency_versions():
    return {k:version(k) for k in ['torch','torchvision','ultralytics','opencv-python','numpy','scipy','lap']}
