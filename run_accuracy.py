"""Two image contexts, independent tracking, and spatial passage reconciliation.

Glass-camera experimental accuracy preset. All runtime inputs are video/model
observations; audit annotations and reference counts are never loaded here.
"""
import argparse,json,subprocess,sys,time,hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parent
FULL_PROFILE=dict(robust_tracking=True,pose_new_track_thresh=.2,occlusion_crossings=True)
ROI_PROFILE=dict(robust_tracking=True)

def call(command,log,resume=False):
    status=log.parent/'pipeline_status.json'
    status.write_text(json.dumps(dict(state='running',stage=log.stem,log=str(log)))+'\n')
    try:
        with log.open('a' if resume else 'w') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
    except BaseException as e:
        status.write_text(json.dumps(dict(state='failed',stage=log.stem,log=str(log),error=str(e)))+'\n')
        raise

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpus',nargs='+');p.add_argument('--max-seconds',type=float)
    p.add_argument('--platform',choices=['auto','cuda','mps'],default='auto')
    p.add_argument('--models-dir',type=Path);p.add_argument('--resume',action='store_true')
    p.add_argument('--cached-full',type=Path);p.add_argument('--cached-roi',type=Path)
    a=p.parse_args();a.source=[s.resolve() for s in a.source];a.output=a.output.resolve()
    if bool(a.cached_full)!=bool(a.cached_roi):p.error('Both detection logs are required for cached replay')
    if a.resume and a.cached_full:p.error('Cached replay is deterministic and does not need --resume; use a new output')
    from runtime_control import validate_sources,select_platform,recording_identity,file_sha256,observations_between
    meta=validate_sources(a.source,max_seconds=a.max_seconds,threads=12)
    w,h,fps=meta['width'],meta['height'],meta['fps']
    if (w,h)!=(2880,1620) or abs(fps-25)>.001:raise ValueError('This calibrated preset requires the 2880x1620, 25 fps glass camera')
    # Explicit backends are validated by the inference child. Keep this
    # orchestration process light; it does not load tensors or run a model.
    selected='replay' if a.cached_full else (a.platform if a.platform!='auto' else ('cuda' if a.gpus else select_platform('auto')))
    if selected=='cuda' and a.gpus is None:
        import torch
        a.gpus=[str(i) for i in range(min(2,torch.cuda.device_count()))]
    if selected=='mps' and a.gpus:p.error('Omit --gpus on Apple Silicon')
    identity=dict(source=recording_identity(a.source),platform=selected,gpus=a.gpus,
                  models_dir=str(a.models_dir.resolve()) if a.models_dir else None,
                  profiles=dict(full=FULL_PROFILE,roi=ROI_PROFILE),runner_sha256=file_sha256(__file__),
                  reconciliation_sha256={p:file_sha256(ROOT/p) for p in ['merge_indexed.py','merge_passages.py','fusion.py']})
    identity['digest']=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    if a.output.exists() and any(a.output.iterdir()):
        if not a.resume:raise FileExistsError('Choose an unused output directory or --resume')
        old=json.loads((a.output/'pipeline_config.json').read_text())
        if old['digest']!=identity['digest']:raise ValueError('Pipeline input/configuration does not match the saved run')
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'pipeline_config.json').write_text(json.dumps(identity,indent=2)+'\n')
    start=time.perf_counter();timings={}
    for name,shape,crop,profile,cache in [('full',[736,1280],[0,0,1,1],FULL_PROFILE,a.cached_full),('roi',[736,512],[.4,0,.8,1],ROI_PROFILE,a.cached_roi)]:
        folder=a.output/name;t=time.perf_counter()
        if cache:
            command=[sys.executable,str(ROOT/'replay.py'),'--detections',str(cache.resolve()),'--output',str(folder),'--profile',json.dumps(profile)]
        else:
            command=[sys.executable,str(ROOT/'run.py'),'--camera','glass','--source',*map(str,a.source),'--output',str(folder),
                     '--platform',selected,'--backend','graph' if selected=='cuda' else 'eager',
                     '--batch','8' if selected=='cuda' else '1','--decode-threads','12','--record-tracks','--record-detections',
                     '--person-shape',*map(str,shape),'--crop',*map(str,crop),'--profile-overrides',json.dumps(profile)]
            if selected=='cuda':command+=['--devices',*a.gpus]
            if a.models_dir:
                command+=['--model',str(a.models_dir.resolve()/'yolo11m-pose.pt'),'--umbrella-model',str(a.models_dir.resolve()/'yolo11n.pt')]
            if a.resume and (folder/'events.sqlite').exists():command+=['--resume']
        if a.max_seconds is not None:command+=['--max-seconds',str(a.max_seconds)]
        call(command,a.output/(name+'.log'),a.resume);timings[name]=time.perf_counter()-t
    stages={name:json.loads((a.output/name/'summary.json').read_text()) for name in ['full','roi']}
    complete=all(d.get('all_observations_processed',d['observations']==observations_between(0,meta['end_frame'],2)) for d in stages.values())
    if not complete:
        message='A counting branch did not process all planned observations'
        (a.output/'pipeline_status.json').write_text(json.dumps(dict(state='failed',stage='completion_check',error=message))+'\n')
        raise RuntimeError(message)
    call([sys.executable,str(ROOT/'merge_indexed.py'),'--primary',str(a.output/'full'),'--secondary',str(a.output/'roi'),'--output',str(a.output/'result')],a.output/'merge.log')
    result=json.loads((a.output/'result/summary.json').read_text())
    result.update(source=list(map(str,a.source)),mode='cached replay' if a.cached_full else 'video inference',elapsed_seconds=time.perf_counter()-start,stage_wall_seconds=timings,
                  resumed=a.resume,platform=selected,timing_scope='This invocation only; counts include saved history',
                  all_observations_processed=complete,requested_duration_available=meta['requested_duration_available'],
                  input_available_seconds=meta['available_frames']/fps,requested_max_seconds=a.max_seconds,
                  accuracy_status='Accuracy must be evaluated separately for each source; see VALIDATION.md',profiles=dict(full=FULL_PROFILE,roi=ROI_PROFILE))
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    (a.output/'pipeline_status.json').write_text(json.dumps(dict(state='complete',events=result['events']))+'\n')

if __name__=='__main__':main()
