"""Real GPU resume checks for people and the first umbrella audit."""
import argparse,csv,json,os,subprocess,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',default='1');a=p.parse_args()
    results=[]
    for camera,source,start,split,end,baseline in [
        ('glass','1219/ch1_20261106_121900-124148_vscode.mp4',0,.4,3.2,'glass_trt_fp32_preview_b1'),
        ('overhead','1300/ch1_20261106_130000-133423_vscode.mp4',670,20,23.2,'overhead_trt_fp32_preview_b1')]:
        out=ROOT/'runs'/f'resume_{camera}';out.mkdir(exist_ok=True)
        command=[sys.executable,str(ROOT/'run.py'),'--backend','trt_graph','--camera',camera,
                 '--source',str(ROOT.parent/'data'/source),'--start-seconds',str(start),'--device',a.device,
                 '--record-tracks','--record-detections','--output',str(out)]
        for name,tail in [('before',['--max-seconds',str(split)]),('after',['--resume','--max-seconds',str(end)])]:
            with (out/f'{name}.log').open('w') as log:
                subprocess.run(command+tail,stdout=log,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
        load=lambda path:[json.loads(line) for line in path.read_text().splitlines()]
        actual=load(out/'tracks.jsonl');expected=[r for r in load(ROOT/'runs'/baseline/'tracks.jsonl') if r['frame']<round(end*25)]
        assert [r['frame'] for r in actual]==list(range(0,round(end*25),2))
        assert len(actual)==len(expected)
        maximum_error=0
        for current,reference in zip(actual,expected):
            for key in ['people','umbrellas']:
                assert [person['id'] for person in current[key]]==[person['id'] for person in reference[key]]
                for new,old in zip(current[key],reference[key]):
                    maximum_error=max(maximum_error,max(abs(x-y) for x,y in zip(new['xyxy'],old['xyxy'])))
                    np.testing.assert_allclose(new['xyxy'],old['xyxy'],atol=.002,rtol=0)
                    if 'anchor' in new:np.testing.assert_allclose(new['anchor'],old['anchor'],atol=.002,rtol=0)
        row=dict(camera=camera,observations=len(actual),unique_complete_frames=True,track_ids_identical=True,
                 maximum_box_error_px=maximum_error,split_seconds=split,end_seconds=end,
                 umbrella_audit=json.loads((out/'umbrella_audit.json').read_text()))
        if camera=='overhead':assert row['umbrella_audit'][0]['status']=='needs_review'
        results.append(row);print(json.dumps(row),flush=True)
    (ROOT/'runs/resume_validation.json').write_text(json.dumps(results,indent=2))


if __name__=='__main__':main()
