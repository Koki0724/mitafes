"""Measure one or two camera recordings on MPS; optionally run both concurrently."""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

from evaluate import evaluate
from mac_runtime import require_mps

ROOT = Path(__file__).resolve().parent


def latency_windows(path, fps, paced):
    windows = []
    bucket = None
    latencies = []

    def append():
        if latencies:
            windows.append(dict(source_start_seconds=bucket * 10,
                                observations=len(latencies),
                                latency_p50_seconds=float(np.median(latencies)),
                                latency_p95_seconds=float(np.percentile(latencies, 95))))

    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            current = int(row['frame'] / fps // 10)
            if current != bucket:
                append()
                bucket = current
                latencies = []
            latencies.append(row['latency_seconds'])
    append()
    return dict(latency_basis='source schedule' if paced else 'decoded frame arrival',
                windows=windows,
                last_minus_first_window_p95_seconds=(
                    windows[-1]['latency_p95_seconds'] - windows[0]['latency_p95_seconds']
                    if len(windows) >= 2 and paced else None))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--glass-source', type=Path)
    parser.add_argument('--overhead-source', type=Path)
    parser.add_argument('--glass-start', type=float, default=0)
    parser.add_argument('--overhead-start', type=float, default=670)
    parser.add_argument('--seconds', type=float, default=60)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--pace', action='store_true')
    parser.add_argument('--concurrent', action='store_true')
    parser.add_argument('--compare-reference', action='store_true',
                        help='Only for the documented two reference recording intervals')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.glass_source and not args.overhead_source:
        parser.error('Specify --glass-source and/or --overhead-source')
    if not math.isfinite(args.seconds) or args.seconds <= 0 or args.threads < 1:
        parser.error('Seconds and threads must be positive')
    if args.compare_reference and args.seconds != 60:
        parser.error('--compare-reference requires --seconds 60')
    require_mps()
    args.output.mkdir(parents=True, exist_ok=False)
    jobs = []
    report = dict(device='mps', paced=args.pace, concurrent=args.concurrent,
                  requested_seconds=args.seconds, cameras={}, failures={},
                  note='File replay; rendering, real RTSP and long-term thermal stability are not tested.')

    def launch(camera, source, start):
        output = args.output / camera
        command = [sys.executable, str(ROOT / 'run.py'), '--camera', camera,
                   '--source', str(source.resolve()), '--start-seconds', str(start),
                   '--max-seconds', str(args.seconds), '--threads', str(args.threads),
                   '--output', str(output.resolve())]
        if args.pace:
            command.append('--pace')
        if args.compare_reference:
            command.append('--record-tracks')
        log = (args.output / f'{camera}.log').open('w')
        print(f'{camera}: starting ({"paced" if args.pace else "maximum speed"})', flush=True)
        try:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        except BaseException:
            log.close()
            raise
        job = (camera, process, log, output)
        jobs.append(job)
        return job

    def collect(job):
        camera, process, log, output = job
        code = process.wait()
        log.close()
        if code:
            report['failures'][camera] = dict(exit_code=code, log=str(log.name))
            print(f'{camera}: failed; see {log.name}', flush=True)
            return
        summary = json.loads((output / 'summary.json').read_text())
        complete = (summary['all_observations_processed']
                    and abs(summary['video_seconds'] - args.seconds) <= 1 / summary['source_fps'])
        result = dict(summary=summary, requested_duration_complete=complete,
                      fits_source_duration=(complete and summary['realtime_factor'] >= 1)
                      if not args.pace else None,
                      latency=latency_windows(output / 'timings.jsonl', summary['source_fps'], args.pace))
        if args.compare_reference:
            if not complete:
                result['accuracy_skipped'] = 'Requested reference interval was not fully processed'
            else:
                result['accuracy'] = evaluate(camera, output)
        report['cameras'][camera] = result
        print(f"{camera}: RTF={summary['realtime_factor']:.3f}, "
              f"latency p95={summary['latency_p95_seconds']:.3f}s, "
              f"events={summary['events']}, complete={complete}", flush=True)

    try:
        for camera, source, start in [
            ('glass', args.glass_source, args.glass_start),
            ('overhead', args.overhead_source, args.overhead_start),
        ]:
            if source is not None:
                job = launch(camera, source, start)
                if not args.concurrent:
                    collect(job)
        if args.concurrent:
            for job in jobs:
                collect(job)
    finally:
        for _, process, log, _ in jobs:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            log.close()
        (args.output / 'report.json').write_text(json.dumps(report, indent=2))
    print(args.output / 'report.json')
    return 1 if report['failures'] else 0


if __name__ == '__main__':
    sys.exit(main())
