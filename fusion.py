"""Merge pose observations from different image contexts, without event inputs."""
import numpy as np


def box_iou(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    area=np.prod(np.maximum(0,np.minimum(a[2:4],b[2:4])-np.maximum(a[:2],b[:2])))
    union=np.prod(a[2:4]-a[:2])+np.prod(b[2:4]-b[:2])-area
    return float(area/max(union,1))


def same_pose(a,ka,b,kb):
    overlap=box_iou(a,b)
    if overlap<.4:return False
    ka,kb=np.asarray(ka),np.asarray(kb)
    shared=(ka[:,2]>=.3)&(kb[:,2]>=.3)
    if shared.sum()>=3:
        distances=np.linalg.norm(ka[shared,:2]-kb[shared,:2],axis=1)
        tolerance=np.clip(min(a[3]-a[1],b[3]-b[1])*.06,12,36)
        if np.median(distances)<tolerance:return True
    # Near-identical boxes with no common joints are duplicate fallback boxes.
    return overlap>.9 and shared.sum()<3


def fuse_items(primary,secondary):
    pairs=[]
    for item in [primary,secondary]:
        for d,k in zip(item['detections'],item['keypoints']):
            if int(d[5])!=0:continue
            duplicate=next((i for i,(a,ka) in enumerate(pairs) if same_pose(a,ka,d,k)),None)
            if duplicate is None:pairs.append((d,k))
            # Keep primary context to avoid selecting a head solely by confidence.
    # Only primary owns auxiliary observations, so cadence and identities agree.
    for d,k in zip(primary['detections'],primary['keypoints']):
        if int(d[5])!=0:pairs.append((d,k))
    return primary|dict(detections=[d for d,k in pairs],keypoints=[k for d,k in pairs])


if __name__=='__main__':
    import argparse,json
    from pathlib import Path
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--primary',type=Path,required=True)
    p.add_argument('--secondary',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();n=0
    with a.primary.open() as f,a.secondary.open() as g,a.output.open('w') as out:
        for x,y in zip(f,g,strict=True):
            x,y=json.loads(x),json.loads(y)
            assert x['index']==y['index']
            out.write(json.dumps(fuse_items(x,y))+'\n');n+=1
    print(n)
