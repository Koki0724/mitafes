"""Event and trajectory comparison; references are evaluation-only, never runtime inputs."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT=Path(__file__).resolve().parent


def rows(path):
    return list(csv.DictReader(path.open()))


def point(person):
    return person.get('anchor',[(person['xyxy'][0]+person['xyxy'][2])/2,person['xyxy'][3]])


def evaluate(camera,result,max_distance=30):
    config=json.loads((result/'config.json').read_text())
    args=config['args']
    if args['camera']!=camera or abs((args.get('max_seconds') or 0)-60)>1e-9:
        raise ValueError('Reference comparison requires the correct camera and a 60-second run')
    if not args['record_tracks'] or args.get('resume'):
        raise ValueError('Reference comparison requires --record-tracks and a fresh run')
    summary=json.loads((result/'summary.json').read_text())
    if (summary['width'],summary['height'],summary['source_fps'],summary['stride'])!=(2880,1620,25,2):
        raise ValueError('Reference comparison requires 2880x1620, 25 FPS, stride 2')
    if summary['observations']!=750 or not summary['all_observations_processed']:
        raise ValueError('Reference comparison requires all 750 observations')
    expected=json.loads((ROOT/f'reference/{camera}_expected.json').read_text())
    reference_path=ROOT/'reference/improvements'/('glass_including_bottom_result' if camera=='glass' else 'overhead_result')/'tracks.json'
    reference=json.loads(reference_path.read_text())
    with (result/'tracks.jsonl').open() as f:candidate=[json.loads(line) for line in f]
    reftracks={};newtracks={}
    for records,destination in [(reference,reftracks),(candidate,newtracks)]:
        for frame in records:
            for p in frame['people']:destination.setdefault(p['id'],{})[frame['frame']]=point(p)
    events=rows(result/'events.csv')
    costs=np.full((len(expected),len(events)),1e6);details={}
    line=2880*(.6 if camera=='glass' else .36);margin=2880*.0125
    for i,ref in enumerate(expected):
        observations=reftracks[int(ref['track_id'])]
        center=int(ref['frame'])
        for j,event in enumerate(events):
            if event['direction']!=ref['direction'] or abs(float(event['seconds'])-float(ref['seconds']))>2:continue
            new=newtracks[int(event['track_id'])]
            common=sorted(f for f in observations if center-75<=f<=center+75 and f in new)
            if len(common)<3:continue
            distances=np.array([np.linalg.norm(np.asarray(observations[f])-new[f]) for f in common])
            before=sum(d<=max_distance and observations[f][0]<line-margin and new[f][0]<line-margin for f,d in zip(common,distances))
            after=sum(d<=max_distance and observations[f][0]>line+margin and new[f][0]>line+margin for f,d in zip(common,distances))
            median=float(np.median(distances))
            details[i,j]=dict(median_anchor_error_px=round(median,3),common_observations=len(common),matching_left_observations=int(before),matching_right_observations=int(after))
            if median<=max_distance and before>=2 and after>=2:
                costs[i,j]=median/max_distance+.02*abs(float(event['seconds'])-float(ref['seconds']))
    matched={};used=set()
    if costs.size:
        ii,jj=linear_sum_assignment(costs)
        for i,j in zip(ii,jj):
            if costs[i,j]<1.1:matched[i]=j;used.add(j)
    verification=[]
    for i,ref in enumerate(expected):
        j=matched.get(i)
        verification.append(dict(reference_id=ref['reference_id'],description=ref['description'],direction=ref['direction'],
                                 reference_seconds=ref['seconds'],status='matched' if j is not None else 'unmatched_needs_review',
                                 candidate_event_id=events[j]['event_id'] if j is not None else '',
                                 candidate_track_id=events[j]['track_id'] if j is not None else '',
                                 **(details[i,j] if j is not None else {})))
    extras=[e for j,e in enumerate(events) if j not in used]
    critical=['23','43','47','62','61','53'] if camera=='glass' else []
    report=dict(camera=camera,expected=len(expected),automatic=len(events),matched_trajectories=len(matched),
                unmatched_references=[v for v in verification if v['status']!='matched'],unmatched_automatic_events=extras,
                critical_cases={v['reference_id']:v['status'] for v in verification if v['reference_id'] in critical},
                max_matching_anchor_error_px=max_distance,independent_test=False,
                method='One-to-one assignment using shared-frame anchor trajectories; <=30px median, >=2 matching observations on BOTH sides. Totals alone do not pass.')
    (result/'verification.json').write_text(json.dumps(verification,indent=2))
    (result/'accuracy.json').write_text(json.dumps(report,indent=2));return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--camera',choices=['glass','overhead'],required=True);p.add_argument('--result',type=Path,required=True)
    a=p.parse_args()
    if a.result:print(json.dumps(evaluate(a.camera,a.result),indent=2))
