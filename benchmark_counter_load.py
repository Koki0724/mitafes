"""Synthetic 1x/2x/4x post-detection load, including umbrellas and counting.

The image tensor and neural network are not duplicated. This isolates the
parts whose work grows with object count; it cannot establish occlusion accuracy.
"""
import json,math,time
from pathlib import Path
import numpy as np
from engine import StreamingCounter

ROOT=Path(__file__).resolve().parent
PROFILE=json.loads((ROOT/'reference/profiles/glass_1219_heads_including_bottom.json').read_text())


def sample(count,frame):
    columns=math.ceil(math.sqrt(count*1.5));rows=math.ceil(count/columns)
    detections=[];points=[]
    for i in range(count):
        x=600+(i%columns)*1200/columns+frame*2;y=350+(i//columns)*750/rows
        detections.append([x-35,y-25,x+35,y+300,.9,0])
        k=np.zeros((17,3));k[:5]=[[x,y,.9],[x-4,y-2,.9],[x+4,y-2,.9],[x-12,y,.9],[x+12,y,.9]]
        points.append(k.tolist())
    for i in range(0,count,4):
        x=(detections[i][0]+detections[i][2])/2;y=detections[i][3]-144
        detections.append([x-100,y-120,x+100,y,.9,25]);points.append(np.zeros((17,3)).tolist())
    return dict(index=frame,detections=detections,keypoints=points)


def run(count,vector):
    engine=StreamingCounter(PROFILE|{'vectorized_tracking':vector},2880,1620,25,2)
    timings=[];events=[];audits=[]
    for frame in range(0,160,2):
        item=sample(count,frame)
        t=time.perf_counter();record,e,a=engine.update(item);elapsed=time.perf_counter()-t
        if frame>=20:timings.append(elapsed)
        events.extend(e)
        audits.extend(a)
    assert len(record['people'])==count
    assert len(record['umbrellas'])==math.ceil(count/4)
    assert events
    return dict(median_ms=float(np.median(timings))*1000,p95_ms=float(np.percentile(timings,95))*1000,
                observations=len(timings),simultaneous_people=len(record['people']),simultaneous_umbrellas=len(record['umbrellas'])),(events,audits)


def main():
    results=[]
    for count in [15,30,60]:
        scalar,a=run(count,False);vector,b=run(count,True)
        assert a==b
        result=dict(people=count,umbrellas=math.ceil(count/4),scalar=scalar,vectorized=vector,
                    median_speedup=scalar['median_ms']/vector['median_ms'],events_identical=True)
        results.append(result);print(json.dumps(result),flush=True)
    report=dict(results=results,artificial_load_only=True,neural_inference_included=False,
                real_dense_occlusion_accuracy_verified=False,source_fps=25,analysis_fps=12.5)
    (ROOT/'runs/crowd_counter_benchmark.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
