"""Render saved observations and automatic totals on the original video."""
import argparse,csv,json
from collections import Counter
from pathlib import Path
import cv2
from video_io import PreviewSink

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--start',type=float,default=0);p.add_argument('--end',type=float,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    records={}
    for line in (a.run/'tracks.jsonl').open():
        d=json.loads(line)
        if a.start<=d['frame']/25<a.end:records[d['frame']]=d
    events=list(csv.DictReader((a.run/'events.csv').open()));counts=Counter()
    cap=cv2.VideoCapture(str(a.source),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,4]);cap.set(cv2.CAP_PROP_POS_FRAMES,round(a.start*25))
    sink=PreviewSink(a.output,2880,1620,12.5,dict(line_x=.6),'nvenc','0');cursor=0
    try:
        for frame in range(round(a.start*25),round(a.end*25)):
            if frame not in records:
                if not cap.grab():break
                continue
            ok,im=cap.read()
            if not ok:raise RuntimeError('Video ended before requested review interval')
            while cursor<len(events) and int(events[cursor]['frame'])<=frame:
                counts[events[cursor]['direction']]+=1;cursor+=1
            # IDs are anonymous trajectory labels, useful when inspecting splits.
            for person in records[frame]['people']:
                x,y=person['anchor'];cv2.putText(im,str(person['id']),(round(x),max(20,round(y)-15)),cv2.FONT_HERSHEY_SIMPLEX,.9,(40,240,40),2)
            sink.put(im,records[frame],counts,25)
    finally:cap.release();info=sink.close()
    info.update(start=a.start,end=a.end,counts_are_automatic=True,generated_online=False,rendered_from_saved_tracks=True)
    a.output.with_suffix('.json').write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info))

if __name__=='__main__':main()
