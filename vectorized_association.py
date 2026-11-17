"""Vectorize umbrella/person distances; retain all original ambiguity thresholds."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def associate_umbrellas(umbrellas,people,profile):
    if not umbrellas or not people:return []
    u=np.asarray([x['xyxy'] for x in umbrellas]);p=np.asarray([x['xyxy'] for x in people])
    ux=(u[:,0]+u[:,2])/2
    uy=u[:,3]+profile['umbrella_foot_offset']*(u[:,3]-u[:,1])
    uw=np.maximum(u[:,2]-u[:,0],1);uh=np.maximum(u[:,3]-u[:,1],1)
    px=(p[:,0]+p[:,2])/2;py=p[:,3]
    costs=np.hypot((px[None,:]-ux[:,None])/uw[:,None],(py[None,:]-uy[:,None])/uh[:,None])
    rows,cols=linear_sum_assignment(costs);result=[]
    for i,j in zip(rows,cols):
        alternatives=np.r_[costs[i,:j],costs[i,j+1:],costs[:i,j],costs[i+1:,j]]
        gap=alternatives.min()-costs[i,j] if len(alternatives) else float('inf')
        if costs[i,j]<=profile['umbrella_match_max_cost'] and gap>=profile['umbrella_match_margin']:
            result.append((umbrellas[i]['id'],people[j]['id'],float(costs[i,j])))
    return result
