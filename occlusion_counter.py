"""Observed line crossings for tracks born inside the confirmation margin."""
from collections import deque
from camera_counter import TemporalCounter


class OcclusionCounter(TemporalCounter):
    def update(self,tid,x,y,frame):
        previous=len(self.events)
        super().update(tid,x,y,frame)
        state=self.states[tid]
        if len(self.events)>previous:state['has_crossed']=True
        samples=state.setdefault('birth_samples',deque(maxlen=16))
        if state.get('birth_done'):return
        samples.append((frame,float(x),float(y)))
        if frame-samples[0][0]>round(.8*self.fps):state['birth_done']=True;return
        if state.get('has_crossed') or len(samples)<4:return
        line=self.profile['line_x']*self.width;margin=self.profile['margin']*self.width
        origin=samples[0][1]-line
        # The first observation must actually be on the other side, while
        # too close to the line for the ordinary two-observation margin rule.
        if not self.width*.003<=abs(origin)<margin:return
        direction=1 if origin<0 else -1
        if direction*(x-samples[0][1])<self.width*.03:return
        target=1 if direction>0 else -1
        if state['stable']!=target or state['candidate']!=target or state['n']<2:return
        if (frame-state['candidate_start'])/self.fps<.06:return
        dx=[direction*(b[1]-a[1]) for a,b in zip(samples,list(samples)[1:])]
        if sum(v>=-5 for v in dx)/len(dx)<.7:return
        if abs(y-samples[0][2])>max(90,.5*abs(x-samples[0][1])):return
        state['has_crossed']=True;state['birth_done']=True
        self.events.append(dict(event_id=len(self.events)+1,track_id=tid,
                                direction='ltr' if direction>0 else 'rtl',frame=frame,
                                seconds=round(frame/self.fps,3),x=round(x,1),y=round(y,1)))
