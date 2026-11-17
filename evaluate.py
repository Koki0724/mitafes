"""Event and trajectory comparison; references are evaluation-only, never runtime inputs."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT=Path(__file__).resolve().parent
POC=ROOT.parent/'counting_poc'


def rows(path):
    return list(csv.DictReader(path.open()))


def prepare():
    names={x['reference_id']:x for x in rows(ROOT/'reference/improvements/glass_manual_revised.csv')}
    old_events={e['event_id']:e for e in rows(POC/'improvements/glass_result/events.csv')}
    mapping={}
    key=lambda e:(e['direction'],e['frame'],e['x'],e['y'])
    for v in rows(ROOT/'reference/improvements/glass_result/verification.csv'):
        if v['status']=='matched':mapping[key(old_events[v['event_id']])]=v['reference_id']
    expected=[]
    for e in rows(ROOT/'reference/improvements/glass_including_bottom_result/events.csv'):
        rid=mapping.get(key(e))
        if rid is None:
            if e['direction']=='ltr' and 27<float(e['seconds'])<29:rid='61'
            elif e['direction']=='rtl' and 36<float(e['seconds'])<38:rid='53'
            else:raise ValueError('Unexpected baseline event')
        expected.append(e|dict(reference_id=rid,description=names[rid]['description']))
    assert len(expected)==len({e['reference_id'] for e in expected})==62
    (ROOT/'reference/glass_expected.json').write_text(json.dumps(expected,indent=2))
    overhead={e['event_id']:e for e in rows(ROOT/'reference/improvements/overhead_result/events.csv')}
    expected=[]
    for v in rows(ROOT/'reference/improvements/overhead_result/verification.csv'):
        if v['status']=='matched':expected.append(overhead[v['event_id']]|dict(reference_id=v['reference_id'],description=v['description']))
    (ROOT/'reference/overhead_expected.json').write_text(json.dumps(expected,indent=2))


def point(person):
    return person.get('anchor',[(person['xyxy'][0]+person['xyxy'][2])/2,person['xyxy'][3]])


def evaluate(camera,result,max_distance=30):
    expected=json.loads((ROOT/f'reference/{camera}_expected.json').read_text())
    reference_path=ROOT/'reference/improvements'/('glass_including_bottom_result' if camera=='glass' else 'overhead_result')/'tracks.json'
    if not reference_path.exists():reference_path=POC/'improvements/overhead_result/tracks.json'
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
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',action='store_true')
    p.add_argument('--camera',choices=['glass','overhead']);p.add_argument('--result',type=Path)
    a=p.parse_args()
    if a.prepare:prepare()
    if a.result:print(json.dumps(evaluate(a.camera,a.result),indent=2))
