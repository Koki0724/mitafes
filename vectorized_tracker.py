"""Same head tracker decisions, with vectorized pairwise distance calculations."""
from collections import deque
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import linear_sum_assignment
sys.path.insert(0,str(Path(__file__).resolve().parent/'reference'))
from pose_tracker import PoseAnchorTracker,head_anchor


def batch_anchors(detections,keypoints):
    boxes=np.asarray(detections,float)[:,:4];points=np.asarray(keypoints,float)
    visible=points[:,:5,2]>=.25;counts=visible.sum(axis=1);safe=np.maximum(counts,1)
    values=np.where(visible[:,:,None],points[:,:5,:2],np.inf)
    ordered=np.sort(values,axis=1);row=np.arange(len(boxes))
    anchors=(ordered[row,(safe-1)//2]+ordered[row,safe//2])/2
    maximum=np.where(visible[:,:,None],points[:,:5,:2],-np.inf).max(axis=1)
    minimum=values.min(axis=1)
    scales=np.maximum(30.,np.linalg.norm(maximum-minimum,axis=1)*1.5)
    heights=boxes[:,3]-boxes[:,1]
    scales=np.maximum(scales,np.minimum(heights*.1,90.))
    methods=np.full(len(boxes),'head_keypoints',dtype=object)
    for i in np.flatnonzero(counts==0):
        # Invisible-head fallbacks are uncommon; preserve the original arithmetic.
        anchors[i],scales[i],methods[i]=head_anchor(boxes[i],points[i])
    return anchors,scales,methods


def batch_predictions(tracks,time):
    predictions=np.array([t['anchor'] for t in tracks])
    groups={}
    for i,t in enumerate(tracks):groups.setdefault(len(t['history']),[]).append(i)
    for size,indices in groups.items():
        if size<3:continue
        histories=[list(tracks[i]['history']) for i in indices]
        times=np.array([[x[0] for x in hist] for hist in histories])
        values=np.array([[x[1] for x in hist] for hist in histories])
        eligible=times[:,-1]-times[:,0]>=.15
        centered=times-times.mean(axis=1)[:,None]
        velocity=(centered[:,:,None]*(values-values.mean(axis=1)[:,None,:])).sum(axis=1)/(centered**2).sum(axis=1)[:,None]
        velocity=np.clip(velocity,-900,900);velocity[~eligible]=0
        gaps=np.array([time-tracks[i]['last'] for i in indices])
        predictions[indices]+=velocity*gaps[:,None]
    return predictions


class VectorizedPoseAnchorTracker(PoseAnchorTracker):
    def update(self,detections,keypoints,frame):
        time=frame/self.fps
        self.tracks={i:t for i,t in self.tracks.items() if time-t['last']<=self.profile['tracker_lost_seconds']}
        observations=[]
        if len(detections):
            anchors,scales,methods=batch_anchors(detections,keypoints)
            for d,anchor,scale,method in zip(detections,anchors,scales,methods):
                observations.append(dict(xyxy=list(d[:4]),confidence=float(d[4]),anchor=anchor,scale=scale,anchor_method=method))
        unique=[];accepted_anchors=[];accepted_scales=[];accepted_heads=[]
        for obs in sorted(observations,key=lambda x:-x['confidence']):
            duplicate=False
            if accepted_anchors and obs['anchor_method']=='head_keypoints':
                distance=np.linalg.norm(np.asarray(accepted_anchors)-obs['anchor'],axis=1)
                threshold=.3*np.minimum(obs['scale'],accepted_scales)
                duplicate=bool(np.any((distance<threshold)&np.asarray(accepted_heads)))
            if not duplicate:
                unique.append(obs);accepted_anchors.append(obs['anchor']);accepted_scales.append(obs['scale'])
                accepted_heads.append(obs['anchor_method']=='head_keypoints')
        observations=unique;ids=list(self.tracks)
        cost=np.full((len(ids),len(observations)),1e6)
        if ids and observations:
            tracks=[self.tracks[tid] for tid in ids]
            predictions=batch_predictions(tracks,time)
            anchors=np.asarray([o['anchor'] for o in observations])
            gaps=np.asarray([time-t['last'] for t in tracks])
            radii=np.maximum([t['scale'] for t in tracks],30.)*self.profile.get('anchor_match_radius',1.6)+20*gaps
            distances=np.linalg.norm(predictions[:,None,:]-anchors[None,:,:],axis=2)
            raw=distances/radii[:,None]+.08*gaps[:,None]+.05*(1-np.asarray([o['confidence'] for o in observations]))[None,:]
            cost=np.where(distances<=radii[:,None],raw,1e6)
        used=set();matches={}
        if cost.size:
            rows,cols=linear_sum_assignment(cost)
            for row,col in zip(rows,cols):
                if cost[row,col]<1:matches[col]=ids[row];used.add(col)
        for col,obs in enumerate(observations):
            if col not in used and obs['confidence']>=self.profile.get('pose_new_track_thresh',.3):
                tid=self.next_id;self.next_id+=1
                self.tracks[tid]=dict(last=time,anchor=obs['anchor'],scale=obs['scale'],history=deque(maxlen=8))
                matches[col]=tid
        result=[]
        for col,tid in matches.items():
            obs=observations[col];track=self.tracks[tid]
            if time-track['last']>.4:track['history'].clear()
            track['history'].append((time,obs['anchor']))
            track.update(last=time,anchor=obs['anchor'],scale=.8*track['scale']+.2*obs['scale'])
            result.append(dict(id=tid,xyxy=obs['xyxy'],confidence=obs['confidence'],anchor=obs['anchor'].tolist(),anchor_method=obs['anchor_method']))
        return result
