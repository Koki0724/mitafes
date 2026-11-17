"""Native CPU decode throughput and full-pixel checks across decoder threads."""
import hashlib,json,time
from pathlib import Path
import cv2

ROOT=Path(__file__).resolve().parent


def main():
    cv2.setNumThreads(1);results=[];reference=None
    source=ROOT.parent/'data/1219/ch1_20261106_121900-124148_vscode.mp4'
    for threads in [2,4,8,12]:
        cap=cv2.VideoCapture(str(source),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,threads]);hashes={};count=0;start=time.perf_counter()
        for frame in range(1500):
            if frame%2:assert cap.grab();continue
            ok,image=cap.read();assert ok;count+=1
            if frame in [0,696,1238,1498]:hashes[frame]=hashlib.sha256(image.tobytes()).hexdigest()
        cap.release();duration=time.perf_counter()-start
        if reference is None:reference=hashes
        assert hashes==reference
        result=dict(threads=threads,decode_seconds=duration,selected_frames=count,sampled_full_frame_hashes_identical=True,hashes=hashes)
        results.append(result);print(json.dumps(result),flush=True)
        (ROOT/'runs/decode_benchmark.json').write_text(json.dumps(results,indent=2))


if __name__=='__main__':main()
