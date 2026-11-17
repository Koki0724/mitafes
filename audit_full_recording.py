"""Offline validation aids. These do not modify runtime counts or models.

Uniform windows are selected by time, before looking at counter results.
Agreement between inference configurations is not ground-truth accuracy.
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def read_events(folder):
    with (Path(folder) / 'events.csv').open() as f:
        return list(csv.DictReader(f))


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sample_windows(source, output):
    output.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(source), cv2.CAP_FFMPEG,
                           [cv2.CAP_PROP_N_THREADS, 4])
    assert cap.isOpened()
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    windows = []
    for start in range(90, int(duration) - 10, 60):
        panels = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, round(start * fps))
        for k in range(250):
            if k % 10:
                assert cap.grab()
                continue
            ok, im = cap.read()
            assert ok
            # Raw source, full image height; no detections, IDs or counts.
            panel = cv2.resize(im[:, 1280:2176], (320, 579))
            panel = cv2.copyMakeBorder(panel, 28, 0, 0, 0,
                                      cv2.BORDER_CONSTANT)
            cv2.line(panel, (160, 28), (160, 606), (0, 255, 255), 1)
            cv2.putText(panel, f'{start + k / fps:.2f}s', (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
            panels.append(panel)
        sheet = np.vstack([np.hstack(panels[i:i + 5])
                           for i in range(0, 25, 5)])
        name = f'window_{start:04d}.jpg'
        cv2.imwrite(str(output / name), sheet,
                    [cv2.IMWRITE_JPEG_QUALITY, 94])
        windows.append(dict(start_seconds=start, end_seconds=start + 10,
                            sheet=name, sample_interval_seconds=.4))
    cap.release()
    dump(output / 'windows.json', dict(
        source=str(source.resolve()), duration_seconds=duration,
        selection='10 seconds at minute midpoint, once per minute after the known first minute',
        input='Raw native source with a fixed gate crop, full height; no model overlay',
        windows=windows))
    print(f'Created {len(windows)} uniform windows ({len(windows) * 10}s)')


def tracks(folder):
    result = {}
    with (Path(folder) / 'tracks.jsonl').open() as f:
        for line in f:
            row = json.loads(line)
            for p in row['people']:
                result.setdefault(p['id'], {})[row['frame']] = p['anchor']
    return result


def compare(candidate, comparator, output):
    output.mkdir(parents=True, exist_ok=True)
    aa, bb = read_events(candidate), read_events(comparator)
    ta, tb = tracks(candidate), tracks(comparator)
    cost = np.full((len(aa), len(bb)), 1e6)
    details = {}
    for i, a in enumerate(aa):
        pa = ta[int(a['track_id'])]
        for j, b in enumerate(bb):
            dt = abs(float(a['seconds']) - float(b['seconds']))
            if a['direction'] != b['direction'] or dt > 2:
                continue
            pb = tb[int(b['track_id'])]
            center = int(a['frame'])
            frames = sorted(f for f in pa if abs(f - center) <= 50 and f in pb)
            if len(frames) < 3:
                continue
            dist = np.array([np.linalg.norm(np.asarray(pa[f]) - pb[f])
                             for f in frames])
            left = sum(d <= 30 and pa[f][0] < 1692 and pb[f][0] < 1692
                       for f, d in zip(frames, dist))
            right = sum(d <= 30 and pa[f][0] > 1764 and pb[f][0] > 1764
                        for f, d in zip(frames, dist))
            median = float(np.median(dist))
            if median <= 30 and left >= 2 and right >= 2:
                cost[i, j] = median / 30 + .02 * dt
                details[i, j] = dict(median_anchor_error_px=median,
                                     matching_left=int(left), matching_right=int(right))
    ii, jj = linear_sum_assignment(cost)
    pairs = [(int(i), int(j)) for i, j in zip(ii, jj) if cost[i, j] < 1.1]
    ma, mb = {i for i, _ in pairs}, {j for _, j in pairs}
    report = dict(
        candidate=str(candidate), comparator=str(comparator),
        candidate_events=len(aa), comparator_events=len(bb),
        matched_trajectories=len(pairs),
        candidate_only=[a for i, a in enumerate(aa) if i not in ma],
        comparator_only=[b for j, b in enumerate(bb) if j not in mb],
        pairs=[dict(candidate_event_id=aa[i]['event_id'],
                    comparator_event_id=bb[j]['event_id'], **details[i, j])
               for i, j in pairs],
        independent_ground_truth=False,
        caveat='Same trained pose model and counter, different image context. Agreement is not accuracy.')
    dump(output / 'configuration_agreement.json', report)
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ['pairs', 'candidate_only', 'comparator_only']}))
    print('Unmatched:', len(report['candidate_only']), len(report['comparator_only']))


def first_minute(folder, output):
    """Evaluate the frozen known window, without counting later events as extras."""
    from evaluate import evaluate
    output.mkdir(parents=True, exist_ok=True)
    events = [e for e in read_events(folder) if float(e['seconds']) < 60]
    with (output / 'events.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(events[0]))
        writer.writeheader()
        writer.writerows(events)
    with (folder / 'tracks.jsonl').open() as src, (output / 'tracks.jsonl').open('w') as dst:
        for line in src:
            if json.loads(line)['frame'] >= 1500:
                break
            dst.write(line)
    print(json.dumps(evaluate('glass', output)))


def discrepancy_sheets(source, agreement_path, output):
    output.mkdir(parents=True, exist_ok=True)
    report = json.loads(agreement_path.read_text())
    items = [dict(e, side=side) for side in ['candidate_only', 'comparator_only']
             for e in report[side]]
    items.sort(key=lambda e: float(e['seconds']))
    groups = []
    for e in items:
        found = next((g for g in groups
                      if abs(float(e['seconds']) - g['seconds']) < 1.5
                      and abs(float(e['y']) - g['y']) < 180), None)
        if found is None:
            groups.append(dict(group_id=len(groups) + 1,
                               seconds=float(e['seconds']), y=float(e['y']),
                               events=[e]))
        else:
            found['events'].append(e)
    cap = cv2.VideoCapture(str(source), cv2.CAP_FFMPEG,
                           [cv2.CAP_PROP_N_THREADS, 4])
    fps = cap.get(cv2.CAP_PROP_FPS)
    panels = []
    for group in groups:
        frames = []
        for offset in [-.64, -.16, .32]:
            target = max(0, round((group['seconds'] + offset) * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            ok, im = cap.read()
            assert ok
            x1 = 1278
            y1 = int(np.clip(group['y'] - 450, 0, 720))
            patch = im[y1:y1 + 900, x1:x1 + 900].copy()
            cv2.line(patch, (450, 0), (450, 899), (0, 255, 255), 2)
            patch = cv2.resize(patch, (320, 320))
            patch = cv2.copyMakeBorder(patch, 25, 0, 0, 0,
                                      cv2.BORDER_CONSTANT)
            cv2.putText(patch, f'{target / fps:.2f}s', (6, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
            frames.append(patch)
        row = np.hstack(frames)
        row = cv2.copyMakeBorder(row, 28, 0, 0, 0, cv2.BORDER_CONSTANT)
        description = ' '.join(f"{'A' if e['side']=='candidate_only' else 'B'}#{e['event_id']} {e['direction']}"
                               for e in group['events'])
        cv2.putText(row, f"G{group['group_id']} {description}", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, .52, (255, 255, 255), 1)
        panels.append(row)
    cap.release()
    for i in range(0, len(panels), 5):
        cv2.imwrite(str(output / f'discrepancy_{i // 5 + 1:02d}.jpg'),
                    np.vstack(panels[i:i + 5]), [cv2.IMWRITE_JPEG_QUALITY, 94])
    dump(output / 'discrepancy_groups.json', groups)
    print(f'{len(groups)} discrepancy groups in {(len(groups) + 4) // 5} sheets')


def score_manual(candidate, annotations_path, output):
    output.mkdir(parents=True, exist_ok=True)
    annotation = json.loads(annotations_path.read_text())
    events = read_events(candidate)
    points = tracks(candidate)
    for event in events:
        history = points[int(event['track_id'])]
        center = int(event['frame'])
        frames = sorted(f for f in history if center - 75 <= f <= center + 8)
        crosses = []
        for f0, f1 in zip(frames, frames[1:]):
            x0, x1 = history[f0][0], history[f1][0]
            if f1 - f0 > 6 or x0 == x1:
                continue
            crossed = ((x0 < 1728 <= x1) if event['direction'] == 'ltr'
                       else (x1 <= 1728 < x0))
            if crossed:
                crosses.append((f0 + (1728 - x0) / (x1 - x0) * (f1 - f0)) / 25)
        event['line_seconds'] = min(crosses, key=lambda s: abs(s - float(event['seconds']))) if crosses else float(event['seconds'])
        event['line_time_method'] = 'observed_anchor_interpolation' if crosses else 'reported_event_time_fallback'
    results = []
    for window in annotation['windows']:
        left, right = window['score_start'], window['score_end']
        gt = window['events']
        auto = [e for e in events if left - .8 <= e['line_seconds'] <= right + .8]
        cost = np.full((len(gt), len(auto)), 1e6)
        for i, g in enumerate(gt):
            for j, e in enumerate(auto):
                dt = abs(g[0] - e['line_seconds'])
                if g[1] == e['direction'] and dt <= .8:
                    cost[i, j] = dt
        ii, jj = linear_sum_assignment(cost)
        pairs = [(int(i), int(j)) for i, j in zip(ii, jj) if cost[i, j] < 1]
        mg, ma = {i for i, _ in pairs}, {j for _, j in pairs}
        contexts = window.get('boundary_context', [])
        unscored_boundary = [e for j, e in enumerate(auto) if j not in ma and
                             any(c[1] == e['direction'] and abs(c[0] - e['line_seconds']) <= .6
                                 for c in contexts)]
        excluded_ids = {e['event_id'] for e in unscored_boundary}
        extras = [e for j, e in enumerate(auto) if j not in ma
                  and left <= e['line_seconds'] < right
                  and e['event_id'] not in excluded_ids]
        results.append(dict(
            start=window['start'], score_start=left, score_end=right,
            expected=len(gt), matched=len(pairs),
            manual_ltr=window['ltr'], manual_rtl=window['rtl'],
            matches=[dict(manual=gt[i], automatic=auto[j], time_error_seconds=cost[i, j])
                     for i, j in pairs],
            missing=[g for i, g in enumerate(gt) if i not in mg],
            extras=extras, unscored_boundary=unscored_boundary))
    expected = sum(w['expected'] for w in results)
    matched = sum(w['matched'] for w in results)
    extras = sum(len(w['extras']) for w in results)
    report = dict(
        reviewed_raw_seconds=sum(w.get('raw_seconds',10) for w in annotation['windows']),
        scoring_window_seconds=sum(w['score_end'] - w['score_start'] for w in results),
        windows=len(results), expected=expected, matched=matched,
        false_negatives=expected - matched, false_positives=extras,
        recall=matched / expected if expected else None,
        precision=matched / (matched + extras) if matched + extras else None,
        method='Assistant visual annotations from raw .4-second snapshots, direction and one-to-one time matching <=.8s. Model anchors are interpolated to the line because counter emission follows a hysteresis margin. Boundary ambiguities explicitly excluded. This is a sampled audit, not full-recording ground truth.',
        independent_human_double_review=False, full_recording_accuracy=False,
        details=results)
    dump(output / 'manual_sample_accuracy.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'details'}))
    for w in results:
        if w['missing'] or w['extras']:
            print(json.dumps({k: v for k, v in w.items() if k != 'matches'}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['samples', 'compare', 'first-minute', 'discrepancies', 'score-manual'])
    p.add_argument('--source', type=Path)
    p.add_argument('--candidate', type=Path)
    p.add_argument('--comparator', type=Path)
    p.add_argument('--agreement', type=Path)
    p.add_argument('--annotations', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.action == 'samples':
        sample_windows(a.source, a.output)
    elif a.action == 'compare':
        compare(a.candidate, a.comparator, a.output)
    elif a.action == 'discrepancies':
        discrepancy_sheets(a.source, a.agreement, a.output)
    elif a.action == 'score-manual':
        score_manual(a.candidate, a.annotations, a.output)
    else:
        first_minute(a.candidate, a.output)
