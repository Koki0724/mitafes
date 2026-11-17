"""Track the head detector independently, without disturbing body identities."""
import argparse,csv,json,time
from pathlib import Path
import numpy as np
from engine import StreamingCounter

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--heads',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--conf',type=float,default=.2);p.add_argument('--gap',type=float,default=.6)
    a=p.parse_args();root=Path(__file__).resolve().parent
    profile=json.loads((root/'reference/profiles/glass_1219_heads_including_bottom.json').read_text())
    profile.update(robust_tracking=True,pose_new_track_thresh=a.conf,counter_max_gap_seconds=a.gap)
    engine=StreamingCounter(profile,2880,1620,25,2);a.output.mkdir(parents=True,exist_ok=True)
    events=[];start=time.perf_counter();n=0
    with a.heads.open() as src,(a.output/'tracks.jsonl').open('w') as dst:
        for line in src:
            item=json.loads(line);heads=[h for h in item['head_detections'] if h[4]>=a.conf]
            points=[]
            for h in heads:
                k=np.zeros((17,3));k[:5,:2]=(np.asarray(h[:2])+h[2:4])/2;k[:5,2]=max(.3,h[4]);points.append(k.tolist())
            item=dict(index=item['index'],detections=heads,keypoints=points,head_only_indices=list(range(len(heads))))
            record,e,_=engine.update(item);events+=e;dst.write(json.dumps(record)+'\n');n+=1
    with (a.output/'events.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['event_id','direction','frame','seconds','track_id','x','y']);w.writeheader();w.writerows(events)
    result=engine.summary()|dict(observations=n,seconds=time.perf_counter()-start,configuration=vars(a));result['configuration']={k:str(v) for k,v in vars(a).items()}
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
