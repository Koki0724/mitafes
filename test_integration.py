"""Operational integration tests; GPU inference is explicitly stubbed here."""
import contextlib,io,json,os,queue,sqlite3,sys,tempfile,threading,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch,MagicMock
import cv2,numpy as np
import run
import runtime_control as control


class RuntimeTests(unittest.TestCase):
    def test_full_queue_can_be_cancelled(self):
        q=queue.Queue(1);q.put('full');stop=threading.Event()
        t=threading.Thread(target=control.put_buffer,args=(q,None,stop));t.start();stop.set();t.join(2)
        self.assertFalse(t.is_alive())

    def test_reader_failure_is_not_reported_as_normal_eof(self):
        cap=MagicMock();cap.get.side_effect=lambda k:{3:64,4:64,5:25.,7:12}[k];cap.read.return_value=(False,None)
        q=queue.Queue(2);a=SimpleNamespace(source=['broken.mp4'],stride=2,max_seconds=None,batch=1,pace=False)
        with patch('run.capture',return_value=cap):run.produce(a,q,threading.Event(),dict(width=64,height=64,fps=25),time.perf_counter(),0)
        self.assertIsInstance(q.get_nowait(),RuntimeError);cap.release.assert_called()

    def test_reader_seek_failure_reaches_consumer(self):
        cap=MagicMock();cap.get.side_effect=lambda k:{3:64,4:64,5:25.,7:12}[k];cap.set.return_value=False
        q=queue.Queue(2);a=SimpleNamespace(source=['sample.mp4'],stride=2,max_seconds=None,batch=1,pace=False)
        with patch('run.capture',return_value=cap):run.produce(a,q,threading.Event(),dict(width=64,height=64,fps=25),time.perf_counter(),2)
        self.assertIn('seek',str(q.get_nowait()));cap.release.assert_called()

    def test_missing_recording_and_invalid_time_fail_before_inference(self):
        for source,start,limit in [(['missing.mp4'],0,None),(['missing.mp4'],float('nan'),None),(['missing.mp4'],0,float('inf'))]:
            with self.assertRaises(ValueError):control.validate_sources(source,start,limit)

    def test_mps_unavailable_does_not_fall_back_to_cpu(self):
        with patch('runtime_control.platform.system',return_value='Darwin'),patch('runtime_control.platform.machine',return_value='arm64'),patch('torch.backends.mps.is_available',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'CPU'):control.select_platform('mps')

    def test_rosetta_fails_before_allocating_tensors(self):
        with patch('runtime_control.platform.system',return_value='Darwin'),patch('runtime_control.platform.machine',return_value='x86_64'),patch('torch.ones') as allocate:
            with self.assertRaisesRegex(RuntimeError,'arm64'):control.select_platform('mps')
            allocate.assert_not_called()

    def test_cuda_is_preferred_when_available(self):
        with patch('torch.cuda.is_available',return_value=True):self.assertEqual(control.select_platform('auto'),'cuda')

    def test_mps_adapter_preserves_shape_and_fp32(self):
        from mps_backend import MpsModel
        yolo=MagicMock();yolo.task='pose';yolo.predictor.model.device.type='mps'
        result=MagicMock();yolo.predict.return_value=[result]
        with patch('mps_backend.YOLO') as ctor,patch('mps_backend.torch.mps.synchronize') as sync:
            ctor.return_value.to.return_value=yolo
            model=MpsModel('test.pt',[736,512]);output=model.predict(['image'],736,.1,.7,[0])
            self.assertEqual(output,[result.cpu.return_value]);sync.assert_called_once()
            self.assertEqual(yolo.predict.call_args.kwargs['imgsz'],(736,512))
            self.assertFalse(yolo.predict.call_args.kwargs['half'])
            self.assertFalse(yolo.predict.call_args.kwargs['rect'])

    def test_rss_units(self):
        with patch('runtime_control.resource.getrusage',return_value=SimpleNamespace(ru_maxrss=1048576)):
            with patch('runtime_control.sys.platform','darwin'):self.assertEqual(control.peak_rss_mb(),1)
            with patch('runtime_control.sys.platform','linux'):self.assertEqual(control.peak_rss_mb(),1024)

    def test_models_and_inputs_have_distinct_resume_identities(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'weights.pt';p.write_bytes(b'first');before=control.model_identities([p]);p.write_bytes(b'other')
            self.assertNotEqual(before,control.model_identities([p]))
            before=control.recording_identity([p]);p.write_bytes(b'longer recording')
            self.assertNotEqual(before,control.recording_identity([p]))

    def test_truncated_and_uncommitted_logs_are_removed(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'tracks.jsonl';p.write_text('{"frame":0}\n{"frame":2}\n{"frame":4}\n{"frame":')
            control.trim_uncommitted_logs(td,2)
            self.assertEqual([json.loads(l)['frame'] for l in p.read_text().splitlines()],[0,2])

    def test_latency_windows_exclude_previous_invocation(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'timings.jsonl';p.write_text('\n'.join(json.dumps(dict(frame=f,latency_seconds=l)) for f,l in [(0,.1),(250,2.),(500,4.)]))
            self.assertAlmostEqual(control.latency_windows(p,25,True)['last_minus_first_window_p95_seconds'],3.9)
            self.assertAlmostEqual(control.latency_windows(p,25,True,250)['last_minus_first_window_p95_seconds'],2.)
            self.assertIsNone(control.latency_windows(p,25,False)['last_minus_first_window_p95_seconds'])


class ResumeTests(unittest.TestCase):
    @staticmethod
    def make_video(path):
        w=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),25,(64,64))
        for _ in range(13):w.write(np.zeros((64,64,3),np.uint8))
        w.release()

    @contextlib.contextmanager
    def stubbed_gpu(self,detector):
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch('run.select_platform',return_value='cuda'))
            stack.enter_context(patch('run.environment',return_value=dict(inference='explicitly stubbed')))
            stack.enter_context(patch('run.Detector',detector))
            for name in ['max_memory_allocated','memory_allocated','memory_reserved','reset_peak_memory_stats','empty_cache']:
                stack.enter_context(patch('run.torch.cuda.'+name,return_value=0))
            stack.enter_context(patch('run.torch.cuda.device'))
            stack.enter_context(patch('run.torch.cuda.get_device_name',return_value='stub GPU'))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()));yield

    @staticmethod
    def detector(fail=False):
        class EmptyDetector:
            def __init__(self,*a):self.person=SimpleNamespace();self.flow=None;self.calls=0
            def warmup(self):pass
            def snapshot(self):return None
            def restore(self,state):pass
            def infer(self,batch):
                self.calls+=1
                if fail and self.calls==3:raise RuntimeError('injected inference failure')
                return [dict(index=r['index'],detections=[]) for r in batch]
        return EmptyDetector

    def test_resume_preserves_order_and_rejects_replaced_weights(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);video=root/'test.avi';self.make_video(video);model=root/'stub.pt';model.write_bytes(b'weights')
            out=root/'run';args=['run.py','--camera','overhead','--source',str(video),'--model',str(model),'--output',str(out),'--record-tracks']
            with self.stubbed_gpu(self.detector()):
                with patch.object(sys,'argv',args+['--max-seconds','.24']):run.main()
                with (out/'tracks.jsonl').open('a') as f:f.write('{"frame":6,"people":[]}\n{"frame":')
                with patch.object(sys,'argv',args+['--max-seconds','.52','--resume']):run.main()
                frames=[json.loads(l)['frame'] for l in (out/'tracks.jsonl').read_text().splitlines()]
                self.assertEqual(frames,[0,2,4,6,8,10,12])
                s=json.loads((out/'summary.json').read_text());self.assertEqual(s['observations'],4);self.assertTrue(s['all_observations_processed'])
                self.assertAlmostEqual(s['video_seconds'],.28)
                with patch('run.dependency_versions',return_value={'changed_dependency':'different version'}),patch.object(sys,'argv',args+['--max-seconds','.52','--resume']):
                    with self.assertRaisesRegex(ValueError,'match'):run.main()
                with patch('run.file_sha256',return_value='changed implementation'),patch.object(sys,'argv',args+['--max-seconds','.52','--resume']):
                    with self.assertRaisesRegex(ValueError,'match'):run.main()
                with patch.object(sys,'argv',args+['--max-seconds','.24','--resume']):
                    with self.assertRaisesRegex(ValueError,'earlier'):run.main()
                model.write_bytes(b'replaced weights')
                with patch.object(sys,'argv',args+['--max-seconds','.52','--resume']):
                    with self.assertRaisesRegex(ValueError,'match'):run.main()
                model.write_bytes(b'weights');stat=video.stat();os.utime(video,ns=(stat.st_atime_ns,stat.st_mtime_ns+1000000))
                with patch.object(sys,'argv',args+['--max-seconds','.52','--resume']):
                    with self.assertRaisesRegex(ValueError,'match'):run.main()

    def test_failed_inference_rolls_back_and_resume_replays_only_uncommitted_frames(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);video=root/'test.avi';self.make_video(video);model=root/'stub.pt';model.write_bytes(b'weights');out=root/'run'
            args=['run.py','--camera','overhead','--source',str(video),'--model',str(model),'--output',str(out),'--record-tracks']
            with self.stubbed_gpu(self.detector(fail=True)),patch.object(sys,'argv',args):
                with self.assertRaisesRegex(RuntimeError,'injected'):run.main()
            with self.stubbed_gpu(self.detector()),patch.object(sys,'argv',args+['--resume']):run.main()
            frames=[json.loads(l)['frame'] for l in (out/'tracks.jsonl').read_text().splitlines()];self.assertEqual(frames,list(range(0,13,2)))
            self.assertTrue(json.loads((out/'summary.json').read_text())['all_observations_processed'])


class PipelineTests(unittest.TestCase):
    @staticmethod
    def metadata():return dict(width=2880,height=1620,fps=25,end_frame=1500,available_frames=1500,requested_duration_available=True)

    def test_incomplete_cached_observations_fail_before_publishing_results(self):
        import run_accuracy
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);video=root/'source.mp4';video.write_bytes(b'video stub');cache=root/'cache.jsonl';cache.write_text('');out=root/'result'
            def incomplete(command,log,resume=False):
                folder=out/log.stem;folder.mkdir();(folder/'summary.json').write_text(json.dumps(dict(observations=1)))
            args=['run_accuracy.py','--source',str(video),'--output',str(out),'--cached-full',str(cache),'--cached-roi',str(cache),'--max-seconds','60']
            with patch.object(sys,'argv',args),patch('runtime_control.validate_sources',return_value=self.metadata()),patch('run_accuracy.call',side_effect=incomplete) as calls:
                with self.assertRaisesRegex(RuntimeError,'planned'):run_accuracy.main()
            self.assertEqual(calls.call_count,2);self.assertFalse((out/'summary.json').exists());self.assertFalse((out/'result').exists())
            self.assertEqual(json.loads((out/'pipeline_status.json').read_text())['state'],'failed')

    def test_existing_output_is_not_overwritten(self):
        import run_accuracy
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);video=root/'source.mp4';video.write_bytes(b'stub');out=root/'result';out.mkdir();existing=out/'keep.txt';existing.write_text('preserve')
            args=['run_accuracy.py','--source',str(video),'--output',str(out),'--cached-full',str(video),'--cached-roi',str(video)]
            with patch.object(sys,'argv',args),patch('runtime_control.validate_sources',return_value=self.metadata()):
                with self.assertRaises(FileExistsError):run_accuracy.main()
            self.assertEqual(existing.read_text(),'preserve')


if __name__=='__main__':unittest.main()
