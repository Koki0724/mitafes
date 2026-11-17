"""Portable control-flow tests. These do not measure or emulate MPS performance."""
import contextlib
import io
import json
import queue
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

import mac_runtime
import run
from benchmark import latency_windows


class MacRuntimeTests(unittest.TestCase):
    def test_unavailable_mps_stops_without_cpu_fallback(self):
        with patch('mac_runtime.platform.system', return_value='Darwin'), \
             patch('mac_runtime.platform.machine', return_value='arm64'), \
             patch('mac_runtime.torch.backends.mps.is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'CPU'):
                mac_runtime.require_mps()

    def test_rosetta_stops_before_allocating_on_mps(self):
        with patch('mac_runtime.platform.system', return_value='Darwin'), \
             patch('mac_runtime.platform.machine', return_value='x86_64'), \
             patch('mac_runtime.torch.ones') as allocate:
            with self.assertRaisesRegex(RuntimeError, 'Rosetta'):
                mac_runtime.require_mps()
            allocate.assert_not_called()

    def test_rss_units_on_mac_and_linux(self):
        with patch('mac_runtime.resource.getrusage', return_value=SimpleNamespace(ru_maxrss=1048576)):
            with patch('mac_runtime.sys.platform', 'darwin'):
                self.assertEqual(mac_runtime.peak_rss_mb(), 1)
            with patch('mac_runtime.sys.platform', 'linux'):
                self.assertEqual(mac_runtime.peak_rss_mb(), 1024)

    def test_full_video_buffer_can_be_cancelled(self):
        buffer = queue.Queue(1)
        buffer.put('occupied')
        stop = threading.Event()
        thread = threading.Thread(target=run.put_buffer, args=(buffer, None, stop))
        thread.start()
        stop.set()
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())

    def test_failure_from_reader_reaches_consumer(self):
        buffer = queue.Queue(2)
        stop = threading.Event()
        args = SimpleNamespace(source=['missing.mp4'], start_seconds=0)
        run.produce(args, buffer, stop, dict(fps=25), time.perf_counter(), 0)
        self.assertIsInstance(buffer.get_nowait(), ValueError)

    def test_latency_drift_uses_source_windows_only_in_paced_mode(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'timings.jsonl'
            path.write_text('\n'.join(json.dumps(dict(frame=f, latency_seconds=lag))
                                      for f, lag in [(0, .1), (2, .1), (250, 2), (252, 2)]))
            self.assertAlmostEqual(latency_windows(path, 25, True)['last_minus_first_window_p95_seconds'], 1.9)
            self.assertIsNone(latency_windows(path, 25, False)['last_minus_first_window_p95_seconds'])


class PersistenceIntegrationTest(unittest.TestCase):
    def test_synthetic_reader_and_stub_detector_resume_without_replaying_observations(self):
        # Actual video decoding, tracking and SQLite; inference is explicitly stubbed.
        class EmptyDetector:
            def __init__(self, *args):pass
            def warmup(self):pass
            def infer(self, batch):
                return [dict(index=item['index'], detections=[]) for item in batch]

        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / 'sample.avi'
            output = Path(td) / 'result'
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 25, (64, 64))
            self.assertTrue(writer.isOpened())
            for _ in range(13):writer.write(np.zeros((64, 64, 3), np.uint8))
            writer.release()
            base = ['run.py', '--camera', 'overhead', '--source', str(source),
                    '--output', str(output), '--record-tracks']
            with patch('run.require_mps'), patch('run.synchronize'), \
                 patch('run.Detector', EmptyDetector), \
                 patch('run.environment', return_value={'test': 'stubbed inference'}), \
                 patch('run.torch.mps.current_allocated_memory', return_value=0), \
                 patch('run.torch.mps.driver_allocated_memory', return_value=0), \
                 contextlib.redirect_stdout(io.StringIO()):
                with patch.object(sys, 'argv', base + ['--max-seconds', '.24']):run.main()
                # Simulate an uncommitted debug record from a crash.
                with (output / 'tracks.jsonl').open('a') as handle:
                    handle.write(json.dumps(dict(frame=6, people=[], umbrellas=[])) + '\n')
                with patch.object(sys, 'argv', base + ['--max-seconds', '.52', '--resume']):run.main()
            frames = [json.loads(line)['frame'] for line in (output / 'tracks.jsonl').read_text().splitlines()]
            self.assertEqual(frames, [0, 2, 4, 6, 8, 10, 12])
            summary = json.loads((output / 'summary.json').read_text())
            self.assertEqual(summary['observations'], 4)
            self.assertTrue(summary['all_observations_processed'])
            self.assertAlmostEqual(summary['video_seconds'], .28)


if __name__ == '__main__':
    unittest.main()
