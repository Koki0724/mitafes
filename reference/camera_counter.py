"""Camera profiles with time-aware crossing confirmation and umbrella coverage auditing.

Umbrellas never create people. Their crossings are reconciled against observed-person
events and otherwise exported as review candidates.
"""
import argparse
import csv
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker
from pose_tracker import PoseAnchorTracker

ROOT = Path(__file__).resolve().parent


class TemporalCounter:
    def __init__(self, profile, width, fps):
        self.profile, self.width, self.fps = profile, width, fps
        self.states, self.events = {}, []

    def update(self, tid, x, y, frame):
        p = self.profile
        state = self.states.setdefault(tid, {"stable":None,"candidate":0,"n":0,"last_frame":None})
        if state['last_frame'] is not None:
            gap = (frame-state['last_frame'])/self.fps
            if gap > p['counter_max_gap_seconds']:
                state.update(candidate=0,n=0)
            if gap > p['counter_stale_seconds']:
                state['stable']=None
        state.update(last_x=x,last_y=y,last_frame=frame)
        distance=x/self.width-p['line_x']
        side=-1 if distance < -p['margin'] else (1 if distance > p['margin'] else 0)
        if side==0:
            state.update(candidate=0,n=0)
            return
        if state['candidate']!=side:
            state.update(candidate=side,n=1,candidate_start=frame)
        else:
            state['n']+=1
        if state['n']<2 or (frame-state['candidate_start'])/self.fps < .06:
            return
        if state['stable'] is not None and state['stable']!=side:
            self.events.append(dict(event_id=len(self.events)+1,track_id=tid,
                                    direction='ltr' if side==1 else 'rtl',frame=frame,
                                    seconds=round(frame/self.fps,3),x=round(x,1),y=round(y,1)))
        state['stable']=side

    def pending(self, final_frame, tolerance):
        result=[]
        for tid,s in self.states.items():
            x=s['last_x']/self.width-self.profile['line_x']
            if final_frame-s['last_frame']<=tolerance and ((s['stable']==1 and x<0) or (s['stable']==-1 and x>0)):
                result.append(dict(track_id=tid,direction='rtl' if s['stable']==1 else 'ltr',last_frame=s['last_frame']))
        return result


def foot(box):
    return ((box[0]+box[2])/2,box[3])


def person_quality(box, profile, width, height):
    """A short box at the physical bottom edge cannot provide a ground footpoint.

    Umbrella-hidden legs away from the image boundary remain usable. Substantial
    partially visible pedestrians are also retained, rather than masking the whole
    lower part of the walkway.
    """
    minimum=profile.get('bottom_fragment_min_height_fraction',0)
    if minimum and box[3]>=height*(1-profile.get('bottom_edge_tolerance_fraction',.01)):
        if box[3]-box[1]<minimum*height:
            return False,'bottom_fragment_no_ground_footpoint'
    return True,''


def inside(point, profile, w, h):
    x,y=point
    if not (0<=x<=w and 0<=y<=h):
        return False
    if profile['roi_mode']=='glass_ground':
        return y>.10*h and y>.82*x-.71*h
    polygon=np.asarray([(a*w,b*h) for a,b in profile['roi_polygon']],np.float32)
    return cv2.pointPolygonTest(polygon,(float(x),float(y)),False)>=0


def umbrella_point(box, profile):
    return ((box[0]+box[2])/2,box[3]+profile['umbrella_foot_offset']*(box[3]-box[1]))


def associate_umbrellas(umbrellas, people, profile):
    """One-to-one spatial association; reject assignments with close alternatives.

    This is a heuristic association, not proof of umbrella ownership or occupancy.
    """
    if not umbrellas or not people:
        return []
    costs=np.zeros((len(umbrellas),len(people)))
    for i,u in enumerate(umbrellas):
        ux,uy=umbrella_point(u['xyxy'],profile)
        uw=max(u['xyxy'][2]-u['xyxy'][0],1)
        uh=max(u['xyxy'][3]-u['xyxy'][1],1)
        for j,person in enumerate(people):
            px,py=foot(person['xyxy'])
            costs[i,j]=np.hypot((px-ux)/uw,(py-uy)/uh)
    rows,cols=linear_sum_assignment(costs)
    result=[]
    for i,j in zip(rows,cols):
        alternatives=[costs[i,k] for k in range(len(people)) if k!=j]
        alternatives.extend(costs[k,j] for k in range(len(umbrellas)) if k!=i)
        gap=min(alternatives)-costs[i,j] if alternatives else float('inf')
        if costs[i,j]<=profile['umbrella_match_max_cost'] and gap>=profile['umbrella_match_margin']:
            result.append((umbrellas[i]['id'],people[j]['id'],float(costs[i,j])))
    return result


def make_tracker(fps, profile, umbrella=False):
    args=SimpleNamespace(track_high_thresh=.35 if umbrella else profile.get('track_high_thresh',.25),track_low_thresh=.1,
                         new_track_thresh=.45 if umbrella else profile.get('new_track_thresh',.25),
                         track_buffer=40,match_thresh=.8 if umbrella else profile.get('track_match_thresh',.8),
                         fuse_score=True if umbrella else profile.get('track_fuse_score',True))
    tracker=BYTETracker(args,frame_rate=fps)
    # Set duration explicitly rather than relying on a 30fps-normalized buffer.
    tracker.max_time_lost=max(1,round(fps*profile['tracker_lost_seconds']))
    return tracker


def objects(tracker, detections, shape):
    a=np.asarray(detections,np.float32).reshape(-1,6)
    result=[]
    for row in tracker.update(Boxes(a,shape),img=None):
        result.append(dict(id=int(row[4]),xyxy=row[:4].tolist(),confidence=float(row[5])))
    return result


def csv_write(path, rows, fields):
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)


def run(args):
    cv2.setNumThreads(1)
    profile=json.loads(args.profile.read_text())
    cache=json.loads(args.cache.read_text())
    w,h,fps,stride=cache['width'],cache['height'],cache['fps'],cache['stride']
    effective_fps=fps/stride
    pose_mode=profile.get('tracker')=='pose_anchor'
    if pose_mode and any(len(item.get('keypoints',[]))!=len(item['detections']) for item in cache['frames']):
        raise ValueError('Pose tracking requires keypoints aligned with every detection')
    person_tracker=PoseAnchorTracker(fps,profile) if pose_mode else make_tracker(effective_fps,profile)
    umbrella_tracker=make_tracker(effective_fps,profile,umbrella=True)
    people_counter=TemporalCounter(profile,w,fps)
    umbrella_counter=TemporalCounter(profile,w,fps)
    links=defaultdict(Counter)
    link_rows,records,rejected=[],[],[]
    for item in cache['frames']:
        frame=item['index']
        person_dets=[];person_keypoints=[]
        for det_index,d in enumerate(item['detections']):
            if int(d[5])!=0 or not inside(foot(d[:4]),profile,w,h):continue
            accepted,reason=person_quality(d[:4],profile,w,h)
            if accepted:
                person_dets.append(d)
                if pose_mode:person_keypoints.append(item['keypoints'][det_index])
            else:
                rejected.append(dict(frame=frame,seconds=round(frame/fps,3),reason=reason,
                                     x1=d[0],y1=d[1],x2=d[2],y2=d[3],confidence=d[4]))
        umbrella_dets=[d for d in item['detections'] if int(d[5])==25 and inside(umbrella_point(d[:4],profile),profile,w,h)]
        people=person_tracker.update(person_dets,person_keypoints,frame) if pose_mode else objects(person_tracker,person_dets,(h,w))
        umbrellas=objects(umbrella_tracker,umbrella_dets,(h,w))
        # Predictions can also approach the physical border; apply the same gate
        # before allowing them to vote on a crossing.
        people=[person for person in people if person_quality(person['xyxy'],profile,w,h)[0]]
        for person in people:
            x,y=person.get('anchor',foot(person['xyxy']))
            people_counter.update(person['id'],x,y,frame)
        for umbrella in umbrellas:
            x,y=umbrella_point(umbrella['xyxy'],profile)
            umbrella_counter.update(umbrella['id'],x,y,frame)
        matches=associate_umbrellas(umbrellas,people,profile)
        for uid,pid,cost in matches:
            links[uid][pid]+=1
            link_rows.append(dict(frame=frame,umbrella_track_id=uid,person_track_id=pid,cost=round(cost,4)))
        records.append(dict(frame=frame,people=people,umbrellas=umbrellas,matches=matches))
    audits=[]
    for event in umbrella_counter.events:
        candidate_ids={pid for pid,n in links[event['track_id']].items() if n>=2}
        covered=[e['event_id'] for e in people_counter.events if e['track_id'] in candidate_ids and e['direction']==event['direction'] and abs(e['seconds']-event['seconds'])<=profile['umbrella_event_tolerance_seconds']]
        audits.append(dict(umbrella_event_id=event['event_id'],umbrella_track_id=event['track_id'],
                           direction=event['direction'],seconds=event['seconds'],
                           associated_person_event_ids=covered,
                           status='person_observed' if len(covered)==1 else 'needs_review',
                           reason='Matched observed-person crossing; umbrella itself adds zero people' if len(covered)==1 else 'No unique observed-person crossing; occupancy cannot be inferred from an umbrella'))
    pending=people_counter.pending(cache['source_frames']-1,stride+1)
    summary=dict(profile=profile,source=cache['source'],model=cache['model'],imgsz=cache['imgsz'],
                 person_tracker='pose_anchor' if pose_mode else 'bytetrack',
                 count_anchor='head_keypoints_with_fallback' if pose_mode else 'box_bottom_center',
                 umbrella_model=cache.get('umbrella_model',cache['model']),
                 classes=cache.get('classes',[0]),gamma=cache.get('gamma',1),fps=effective_fps,
                 synthetic_test=cache.get('synthetic_test'),
                 ltr=sum(e['direction']=='ltr' for e in people_counter.events),
                 rtl=sum(e['direction']=='rtl' for e in people_counter.events),pending_at_end=pending,
                 umbrella_crossings=len(audits),umbrella_review_candidates=sum(e['status']=='needs_review' for e in audits),
                 rejected_person_observations=len(rejected),
                 inference_seconds=cache['inference_seconds'],
                 limitation='Umbrella detection and carrier association are heuristic; a fully hidden person or shared umbrella may remain uncountable. Review candidates are not added to counts.')
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    (args.output/'tracks.json').write_text(json.dumps(records))
    (args.output/'umbrella_audit.json').write_text(json.dumps(audits,indent=2))
    csv_write(args.output/'events.csv',people_counter.events,['event_id','track_id','direction','frame','seconds','x','y'])
    csv_write(args.output/'umbrella_links.csv',link_rows,['frame','umbrella_track_id','person_track_id','cost'])
    csv_write(args.output/'rejected_observations.csv',rejected,['frame','seconds','reason','x1','y1','x2','y2','confidence'])
    if args.render:
        render(cache,profile,records,people_counter.events,pending,args.output)
    print(json.dumps(summary,indent=2),flush=True)


def render(cache,profile,records,events,pending,output):
    w,h,fps,stride=cache['width'],cache['height'],cache['fps'],cache['stride']
    cap=cv2.VideoCapture(cache['source'])
    writer=cv2.VideoWriter(str(output/'preview_raw.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps/stride,(1440,810))
    if not writer.isOpened():raise RuntimeError('Video writer failed')
    sx,sy=1440/w,810/h
    trails=defaultdict(lambda:deque(maxlen=20))
    source_frame=-1
    for record in records:
        index=record['frame']
        while source_frame<index:
            ok,im=cap.read();source_frame+=1
            if not ok:raise RuntimeError('Source decode failed')
        im=cv2.resize(im,(1440,810))
        if profile['roi_polygon']:
            polygon=np.array([(int(x*1440),int(y*810)) for x,y in profile['roi_polygon']],np.int32)
            cv2.polylines(im,[polygon],True,(140,140,140),1)
        for offset,color in [(0,(0,255,255)),(-profile['margin'],(120,120,120)),(profile['margin'],(120,120,120))]:
            xx=int((profile['line_x']+offset)*1440)
            cv2.line(im,(xx,40),(xx,810),color,3 if offset==0 else 1)
        recent={e['track_id']:e for e in events if 0<=index/fps-e['seconds']<.8}
        for person in record['people']:
            tid=person['id'];b=person['xyxy']
            x,y=person.get('anchor',foot(b));point=(int(x*sx),int(y*sy));trails[tid].append(point)
            color=(0,130,255) if tid in recent else (60,230,60)
            a=(int(b[0]*sx),int(b[1]*sy));z=(int(b[2]*sx),int(b[3]*sy))
            cv2.rectangle(im,a,z,color,2)
            cv2.putText(im,f'P{tid}',(a[0],max(20,a[1]-5)),cv2.FONT_HERSHEY_SIMPLEX,.6,color,2)
            cv2.circle(im,point,5,color,-1)
            if len(trails[tid])>1:cv2.polylines(im,[np.array(trails[tid],np.int32)],False,color,2)
            if tid in recent:
                event=recent[tid]
                # Keep the crossing label visible for bottom-clipped people.
                label_y=min(z[1]+22,im.shape[0]-10)
                cv2.putText(im,f"#{event['event_id']} {event['direction'].upper()}",(a[0],label_y),cv2.FONT_HERSHEY_SIMPLEX,.6,color,2)
        pmap={p['id']:p for p in record['people']}
        umap={u['id']:u for u in record['umbrellas']}
        for u in record['umbrellas']:
            b=u['xyxy'];a=(int(b[0]*sx),int(b[1]*sy));z=(int(b[2]*sx),int(b[3]*sy))
            cv2.rectangle(im,a,z,(255,0,255),2)
            cv2.putText(im,f"U{u['id']} (audit)",(a[0],max(20,a[1]-5)),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,0,255),2)
        for uid,pid,cost in record['matches']:
            ub=umap[uid]['xyxy'];px,py=foot(pmap[pid]['xyxy'])
            cv2.line(im,(int((ub[0]+ub[2])/2*sx),int(ub[3]*sy)),(int(px*sx),int(py*sy)),(255,0,255),2)
        seen=[e for e in events if e['frame']<=index]
        ltr=sum(e['direction']=='ltr' for e in seen);rtl=len(seen)-ltr
        cv2.rectangle(im,(10,55),(750,145),(0,0,0),-1)
        cv2.putText(im,f"{profile['name']} | L -> R: {ltr}   R -> L: {rtl}",(25,90),cv2.FONT_HERSHEY_SIMPLEX,.75,(255,255,255),2)
        cv2.putText(im,f"clip +{index/fps:.2f}s | umbrellas are audit cues, never extra people",(25,124),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
        if index==records[-1]['frame'] and pending:
            cv2.rectangle(im,(10,150),(750,190),(0,0,0),-1)
            cv2.putText(im,f"Clip end: {len(pending)} person crossing pending",(25,178),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,220,255),2)
        writer.write(im)
    cap.release();writer.release()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--render',action='store_true')
    run(parser.parse_args())
