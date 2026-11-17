"""Fuse observed head boxes with pose observations; no event/count inputs."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from fusion import box_iou


def add_heads(item,head_item,min_conf=.35):
    assert item['index']==head_item['index']
    detections=[list(d) for d in item['detections']]
    points=[np.asarray(k,float).copy() for k in item['keypoints']]
    heads=[d for d in head_item['head_detections'] if d[4]>=min_conf]
    people=[i for i,d in enumerate(detections) if int(d[5])==0]
    cost=np.full((len(people),len(heads)),1e6)
    from pose_tracker import head_anchor
    for row,i in enumerate(people):
        b=detections[i];anchor,scale,method=head_anchor(b[:4],points[i])
        for col,head in enumerate(heads):
            center=(np.asarray(head[:2])+head[2:4])/2
            size=max(head[2]-head[0],head[3]-head[1],20)
            distance=np.linalg.norm(anchor-center)
            # Torso boxes can begin below an occluded head. Allow a bounded
            # anatomical extension, then solve one-to-one across all heads.
            near_upper_body=(b[0]-.15*(b[2]-b[0])<=center[0]<=b[2]+.15*(b[2]-b[0])
                             and b[1]-.4*size<=center[1]<=b[1]+.4*(b[3]-b[1]))
            if distance<=size*.8 or (near_upper_body and distance<=size*1.7):
                cost[row,col]=distance/size
    ii,jj=linear_sum_assignment(cost)
    used=set();matched=0
    for row,col in zip(ii,jj):
        if cost[row,col]>2:continue
        i=people[row];head=heads[col]
        # Keep the original, already calibrated pose anchor. The specialist
        # supplies missing people, not a different coordinate for every person.
        used.add(int(col));matched+=1
    head_only=[]
    for col,head in enumerate(heads):
        if col in used:continue
        center=(np.asarray(head[:2])+head[2:4])/2
        k=np.zeros((17,3));k[:5,:2]=center;k[:5,2]=max(.3,head[4])
        head_only.append(len(detections));detections.append(list(head));points.append(k)
    return item|dict(detections=detections,keypoints=[k.tolist() for k in points],
                     head_only_indices=head_only,head_detections=heads,
                     head_fusion=dict(matched=matched,added=len(head_only)))
