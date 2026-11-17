"""Predetermined raw-source validation windows, separate from known failures."""
import json
from pathlib import Path
import cv2,numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'audit/holdout_raw';OUT.mkdir(parents=True,exist_ok=True)
SOURCE=ROOT.parent/'data/1219/ch1_20261106_121900-124148_vscode.mp4'
windows=[dict(start=s,score_start=s+2,score_end=s+18,
              sheets=[f'window_{s:04d}_0.jpg',f'window_{s:04d}_1.jpg'])
         for s in range(105,1320,60)]
(OUT/'plan.json').write_text(json.dumps(dict(source=str(SOURCE),
    selection='20 seconds every minute from 105s; middle 16 seconds scored; fixed before reviewing new inference',
    windows=windows),indent=2)+'\n')
cap=cv2.VideoCapture(str(SOURCE),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,4])
for window in windows:
    cap.set(1,window['start']*25);panels=[]
    for k in range(500):
        if k%10:assert cap.grab();continue
        ok,im=cap.read();assert ok
        patch=cv2.resize(im[:,1280:2176],(320,579))
        cv2.line(patch,(160,0),(160,578),(0,255,255),1)
        patch=cv2.copyMakeBorder(patch,28,0,0,0,cv2.BORDER_CONSTANT)
        cv2.putText(patch,f'{window["start"]+k/25:.2f}s',(8,20),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
        panels.append(cv2.copyMakeBorder(patch,3,3,3,3,cv2.BORDER_CONSTANT,value=(190,100,0)))
    for index,name in enumerate(window['sheets']):
        chunk=panels[index*25:(index+1)*25]
        cv2.imwrite(str(OUT/name),np.vstack([np.hstack(chunk[i:i+5]) for i in range(0,25,5)]),[cv2.IMWRITE_JPEG_QUALITY,94])
cap.release()
print(f'Created {len(windows)} holdout windows, {len(windows)*20} source seconds')
