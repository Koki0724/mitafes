"""Anonymous head-anchor tracking for the oblique, frequently occluded camera.

Only observed detections are returned: coasting predictions never vote on counts.
Coordinates and thresholds are spatial/temporal; there are no clip-specific times.
"""
from collections import deque
import numpy as np
from scipy.optimize import linear_sum_assignment


def head_anchor(box, points):
    points=np.asarray(points,float)
    visible=points[:5,2]>=.25
    if visible.any():
        anchor=np.median(points[:5,:2][visible],axis=0)
        spread=np.ptp(points[:5,:2][visible],axis=0)
        scale=max(30.,float(np.linalg.norm(spread))*1.5)
        # A single visible ear still needs a plausible spatial uncertainty.
        scale=max(scale,min((box[3]-box[1])*.10,90.))
        return anchor,scale,'head_keypoints'
    shoulders=points[5:7]
    if np.all(shoulders[:,2]>=.3):
        span=np.linalg.norm(shoulders[0,:2]-shoulders[1,:2])
        return shoulders[:,:2].mean(axis=0)-[0,span*.65],max(30.,span*.7),'shoulder_fallback'
    return np.array([(box[0]+box[2])/2,box[1]+min((box[3]-box[1])*.1,(box[2]-box[0])*.3)]),max(30.,min(90.,(box[3]-box[1])*.1)),'box_fallback'


class PoseAnchorTracker:
    def __init__(self,fps,profile):
        self.fps=fps
        self.profile=profile
        self.tracks={}
        self.next_id=1

    @staticmethod
    def predict(track,time):
        history=list(track['history'])
        velocity=np.zeros(2)
        if len(history)>=3 and history[-1][0]-history[0][0]>=.15:
            times=np.array([x[0] for x in history])
            anchors=np.array([x[1] for x in history])
            centered=times-times.mean()
            velocity=(centered[:,None]*(anchors-anchors.mean(axis=0))).sum(axis=0)/(centered**2).sum()
            velocity=np.clip(velocity,-900,900)
        return track['anchor']+velocity*(time-track['last']),velocity

    def update(self,detections,keypoints,frame):
        time=frame/self.fps
        self.tracks={i:t for i,t in self.tracks.items() if time-t['last']<=self.profile['tracker_lost_seconds']}
        observations=[]
        for d,k in zip(detections,keypoints):
            anchor,scale,method=head_anchor(d[:4],k)
            observations.append(dict(xyxy=list(d[:4]),confidence=float(d[4]),anchor=anchor,scale=scale,anchor_method=method))
        # The pose detector can emit a head/torso box and a full-body box for
        # the same head. Ordinary box IoU NMS misses this nested-box duplicate.
        unique=[]
        for obs in sorted(observations,key=lambda x:-x['confidence']):
            duplicate=any(obs['anchor_method']==other['anchor_method']=='head_keypoints'
                          and np.linalg.norm(obs['anchor']-other['anchor'])<.3*min(obs['scale'],other['scale'])
                          for other in unique)
            if not duplicate:unique.append(obs)
        observations=unique
        ids=list(self.tracks)
        cost=np.full((len(ids),len(observations)),1e6)
        for row,tid in enumerate(ids):
            track=self.tracks[tid];pred,velocity=self.predict(track,time)
            gap=time-track['last']
            radius=max(track['scale'],30.)*self.profile.get('anchor_match_radius',1.6)+20*gap
            for col,obs in enumerate(observations):
                error=np.linalg.norm(obs['anchor']-pred)
                if error>radius:continue
                # Prefer continuity when multiple similar heads overlap.
                cost[row,col]=error/radius+.08*gap+.05*(1-obs['confidence'])
        used=set();matches={}
        if cost.size:
            rows,cols=linear_sum_assignment(cost)
            for row,col in zip(rows,cols):
                if cost[row,col]<1:
                    matches[col]=ids[row];used.add(col)
        for col,obs in enumerate(observations):
            if col not in used and obs['confidence']>=self.profile.get('pose_new_track_thresh',.3):
                tid=self.next_id;self.next_id+=1
                self.tracks[tid]=dict(last=time,anchor=obs['anchor'],scale=obs['scale'],history=deque(maxlen=8))
                matches[col]=tid
        result=[]
        for col,tid in matches.items():
            obs=observations[col];track=self.tracks[tid]
            # Long gaps must not pollute the next velocity estimate.
            if time-track['last']>.4:track['history'].clear()
            track['history'].append((time,obs['anchor']))
            track.update(last=time,anchor=obs['anchor'],scale=.8*track['scale']+.2*obs['scale'])
            result.append(dict(id=tid,xyxy=obs['xyxy'],confidence=obs['confidence'],anchor=obs['anchor'].tolist(),anchor_method=obs['anchor_method']))
        return result
