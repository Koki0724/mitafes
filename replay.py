"""Replay actual inference logs for counter development, with no reference input."""
import argparse,csv,json,time
from pathlib import Path
from engine import StreamingCounter


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--detections',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tracker',choices=['original','robust'],default='robust')
    p.add_argument('--profile',default='{}')
    p.add_argument('--stride',type=int,default=2)
    p.add_argument('--max-seconds',type=float)
    p.add_argument('--heads',type=Path)
    p.add_argument('--head-conf',type=float,default=.35)
    a=p.parse_args();root=Path(__file__).resolve().parent
    profile=json.loads((root/'reference/profiles/glass_1219_heads_including_bottom.json').read_text())
    profile.update(json.loads(a.profile));profile['robust_tracking']=a.tracker=='robust'
    engine=StreamingCounter(profile,2880,1620,25,a.stride)
    a.output.mkdir(parents=True,exist_ok=True)
    events=[];audits=[];start=time.perf_counter();n=0
    heads=a.heads.open() if a.heads else None
    with a.detections.open() as f,(a.output/'tracks.jsonl').open('w') as out:
        for line in f:
            try:item=json.loads(line)
            except json.JSONDecodeError:break # Snapshot of a currently written log.
            if a.max_seconds is not None and item['index']/25>=a.max_seconds:break
            if heads:
                from head_fusion import add_heads
                item=add_heads(item,json.loads(next(heads)),a.head_conf)
            record,e,u=engine.update(item);events+=e;audits+=u
            out.write(json.dumps(record)+'\n');n+=1
    audits+=engine.audit(engine.last_frame/25,final=True)
    if heads:heads.close()
    with (a.output/'events.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['event_id','direction','frame','seconds','track_id','x','y']);w.writeheader();w.writerows(events)
    (a.output/'umbrella_audit.json').write_text(json.dumps(audits,indent=2)+'\n')
    result=engine.summary()|dict(observations=n,replay_seconds=time.perf_counter()-start,tracker=a.tracker,profile=profile)
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
