"""Collect regression-scored runs; never used by the counting runtime."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def main():
    experiments=[]
    for summary in sorted((ROOT/'runs').glob('*/summary.json')):
        folder=summary.parent;s=json.loads(summary.read_text())
        if not (folder/'accuracy.json').exists():continue
        a=json.loads((folder/'accuracy.json').read_text())
        experiments.append(dict(name=folder.name,seconds=s['processing_seconds'],startup_seconds=s['startup_seconds'],
                                gpus=s['gpu_count_used'],batch=s['batch'],backend=s['backend'],
                                precision=s['person_graph_precision'],decoder=s.get('decoder','cv2'),
                                decode_threads=s.get('decode_threads',2),preview=bool(s.get('preview')),
                                process_mode=s.get('worker_mode','thread'),ltr=s['ltr'],rtl=s['rtl'],
                                events=a['automatic'],trajectory_matches=a['matched_trajectories'],expected=a['expected'],
                                regression_pass=a['automatic']==a['matched_trajectories']==a['expected']))
    report=dict(experiments=experiments,live=json.loads((ROOT/'runs/live_final/validation.json').read_text()),
                offline_two_camera_split=json.loads((ROOT/'runs/offline_split/validation.json').read_text()),
                resumes=json.loads((ROOT/'runs/resume_validation.json').read_text()),
                independent_crowd_accuracy_test=False,eight_hour_soak_test=False)
    (ROOT/'runs/results.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(experiments,indent=2))


if __name__=='__main__':main()
