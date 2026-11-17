"""Offline CPU visualization; its encoding cost is excluded from inference timing."""
import argparse,csv,json,subprocess
from pathlib import Path
import cv2


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--result',type=Path,required=True);a=p.parse_args()
    config=json.loads((a.result/'config.json').read_text());args=config['args'];profile=config['profile']
    records=[json.loads(x) for x in (a.result/'tracks.jsonl').read_text().splitlines()]
    events=sorted(csv.DictReader((a.result/'events.csv').open()),key=lambda e:int(e['frame']))
    cap=cv2.VideoCapture(args['source'][0],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,2]);cv2.setNumThreads(1)
    source_fps=cap.get(5);width,height=int(cap.get(3)),int(cap.get(4))
    start=round(args['start_seconds']*source_fps);cap.set(1,start);cursor=start
    fps=source_fps/args['stride'];scale=1440/width;size=(1440,round(height*scale))
    output=a.result/'counting_preview.mp4'
    process=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pixel_format','bgr24','-video_size',f'{size[0]}x{size[1]}',
                              '-framerate',str(fps),'-i','pipe:0','-c:v','libx264','-preset','veryfast','-threads','4','-crf','21',
                              '-pix_fmt','yuv420p','-movflags','+faststart','-an',str(output)],stdin=subprocess.PIPE)
    counts=dict(ltr=0,rtl=0);event_index=0
    try:
        for record in records:
            target=start+record['frame']
            while cursor<target:
                assert cap.grab();cursor+=1
            ok,image=cap.read();assert ok;cursor+=1
            image=cv2.resize(image,size)
            kind=record.get('person_observation_kind','network');flow=kind=='optical_flow'
            color=(240,180,40) if flow else (60,220,60)
            for person in record['people']:
                x1,y1,x2,y2=[round(x*scale) for x in person['xyxy']]
                cv2.rectangle(image,(x1,y1),(x2,y2),color,2)
                cv2.putText(image,str(person['id']),(x1,max(15,y1-4)),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1)
                anchor=person.get('anchor')
                if anchor:cv2.circle(image,tuple(round(x*scale) for x in anchor),5,color,2)
            for umbrella in record['umbrellas']:
                x1,y1,x2,y2=[round(x*scale) for x in umbrella['xyxy']]
                cv2.rectangle(image,(x1,y1),(x2,y2),(255,100,200),2)
            while event_index<len(events) and int(events[event_index]['frame'])<=record['frame']:
                counts[events[event_index]['direction']]+=1;event_index+=1
            line=round(profile['line_x']*size[0]);cv2.line(image,(line,0),(line,size[1]-1),(0,230,255),2)
            label=f'CPU FP32 | {record["frame"]/source_fps:.2f}s | {kind} | LTR {counts["ltr"]}  RTL {counts["rtl"]}'
            cv2.putText(image,label,(14,30),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,0),4)
            cv2.putText(image,label,(14,30),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1)
            process.stdin.write(image.tobytes())
    finally:
        cap.release();process.stdin.close()
    if process.wait()!=0:raise RuntimeError('Video encoding failed')
    report=dict(output=str(output),frames=len(records),fps=fps,generated_offline=True,
                inference_timing_excludes_visualization=True,ltr=counts['ltr'],rtl=counts['rtl'],
                blue_boxes='checked optical flow',green_boxes='network observations')
    (a.result/'visualization.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))


if __name__=='__main__':main()
