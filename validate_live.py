"""Synchronized one-GPU live replay or one-GPU-per-camera offline validation."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
from evaluate import evaluate

ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--backend',default='graph');p.add_argument('--tag',default='live_graph')
    p.add_argument('--device',default='0');p.add_argument('--no-preview',action='store_true')
    p.add_argument('--glass-model',default=None)
    p.add_argument('--offline-split',action='store_true',help='Use one GPU per camera without pacing, offline batch=8')
    a=p.parse_args()
    parent=ROOT/'runs'/a.tag;parent.mkdir(parents=True,exist_ok=True);barrier=parent/'start.signal'
    if barrier.exists():raise ValueError('Choose a new validation tag')
    children=[];logs=[];started=time.perf_counter()
    try:
        for camera,source,start in [('glass','1219/ch1_20261106_121900-124148_vscode.mp4',0),
                                    ('overhead','1300/ch1_20261106_130000-133423_vscode.mp4',670)]:
            output=parent/camera;log=(parent/f'{camera}.log').open('w');logs.append(log)
            command=[sys.executable,str(ROOT/'run.py'),'--camera',camera,'--source',str(ROOT.parent/'data'/source),
                     '--start-seconds',str(start),'--max-seconds','60','--device',a.device,'--backend',a.backend,
                     '--record-tracks','--record-detections','--start-barrier',str(barrier),'--output',str(output)]
            if a.offline_split:
                command[command.index('--device')+1]='0' if camera=='glass' else '1'
                command+=['--batch','8','--decode-threads','8']
            else:command+=['--pace']
            if not a.no_preview:command+=['--preview']
            if camera=='glass' and a.glass_model:command+=['--model',str(Path(a.glass_model).resolve())]
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT);children.append(process)
            deadline=time.perf_counter()+100
            while not (output/'ready.json').exists():
                if process.poll() is not None:raise RuntimeError(f'{camera} exited before readiness; see {log.name}')
                if time.perf_counter()>deadline:raise TimeoutError(f'{camera} warmup timed out')
                time.sleep(.05)
            print(camera,'ready',flush=True)
        barrier.touch();release=time.perf_counter();snapshot=None
        while any(process.poll() is None for process in children):
            if any(process.poll() not in [None,0] for process in children):raise RuntimeError('Camera process failed')
            if time.perf_counter()-release>100:raise TimeoutError('Replay failed to keep up')
            if snapshot is None and time.perf_counter()-release>10:
                snapshot=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout
                print('steady GPU snapshot:',snapshot.strip(),flush=True)
            time.sleep(.2)
        report=dict(gpus_used=2 if a.offline_split else 1,device='one per camera' if a.offline_split else a.device,
                    paced=not a.offline_split,backend=a.backend,source_seconds_per_camera=60,
                    two_camera_wall_seconds=time.perf_counter()-release,initialization_wall_seconds=release-started,
                    online_preview=not a.no_preview,steady_gpu_snapshot=snapshot,cameras={})
        for camera in ['glass','overhead']:
            output=parent/camera;s=json.loads((output/'summary.json').read_text());accuracy=evaluate(camera,output)
            timings=[json.loads(line) for line in (output/'timings.jsonl').read_text().splitlines()]
            steady=[r['latency_seconds'] for r in timings if r['frame']>=125]
            report['cameras'][camera]=dict(summary=s,accuracy=accuracy,
                  after_first_5_seconds_latency_p50=float(np.median(steady)),
                  after_first_5_seconds_latency_p95=float(np.percentile(steady,95)),
                  after_first_5_seconds_latency_max=float(max(steady)))
            assert accuracy['matched_trajectories']==accuracy['expected']==accuracy['automatic']
            assert s['observations']==750
            if s['preview']:assert s['preview']['frames']==750
        (parent/'validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
    finally:
        for process in children:
            if process.poll() is None:process.terminate()
        for process in children:
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait()
        for log in logs:log.close()


if __name__=='__main__':main()
