import contextlib,csv,json,pickle,queue,tempfile,threading,time,unittest
from pathlib import Path
from types import SimpleNamespace
import cv2,numpy as np
from engine import StreamingCounter,BaseTrack
from run import produce

ROOT=Path(__file__).resolve().parent


def fixture(camera):
    profile='glass_1219_heads_including_bottom.json' if camera=='glass' else 'overhead_1300_v2.json'
    cache='glass_detections.json' if camera=='glass' else 'overhead_detections.json'
    p=json.loads((ROOT/'reference/profiles'/profile).read_text());d=json.loads((ROOT/'tests/fixtures'/cache).read_text())
    return StreamingCounter(p,d['width'],d['height'],d['fps'],d['stride']),d


def process(engine,frames):
    events=[];audits=[]
    for frame in frames:
        _,e,a=engine.update(frame);events.extend(e);audits.extend(a)
    return events,audits


def read_events(path):
    with path.open() as f:return list(csv.DictReader(f))


class StreamingTests(unittest.TestCase):
    def test_glass_matches_all_62_frozen_events(self):
        engine,data=fixture('glass');events,_=process(engine,data['frames'])
        reference=read_events(ROOT/'reference/improvements/glass_including_bottom_result/events.csv')
        self.assertEqual([(e['direction'],e['frame'],e['x'],e['y']) for e in events],
                         [(e['direction'],int(e['frame']),float(e['x']),float(e['y'])) for e in reference])

    def test_overhead_events_and_both_umbrella_audits_match(self):
        engine,data=fixture('overhead');events,audits=process(engine,data['frames']);audits.extend(engine.audit(60,final=True))
        reference=read_events(ROOT/'reference/improvements/overhead_result/events.csv')
        self.assertEqual([(e['direction'],e['frame'],e['x'],e['y']) for e in events],
                         [(e['direction'],int(e['frame']),float(e['x']),float(e['y'])) for e in reference])
        self.assertEqual([a['status'] for a in audits],['needs_review','person_observed'])

    def test_checkpoint_inside_bottom_crossing_preserves_all_events(self):
        engine,data=fixture('glass');split=345
        before,_=process(engine,data['frames'][:split]);blob=pickle.dumps(engine.snapshot())
        BaseTrack._count=0
        engine=StreamingCounter.restore(pickle.loads(blob));after,_=process(engine,data['frames'][split:])
        reference=read_events(ROOT/'reference/improvements/glass_including_bottom_result/events.csv')
        self.assertEqual([e['frame'] for e in before+after],[int(e['frame']) for e in reference])
        self.assertEqual([e['event_id'] for e in before+after],list(range(1,63)))

    def test_checkpoint_restores_external_bytetrack_allocator(self):
        engine,_=fixture('overhead');BaseTrack._count=57
        blob=pickle.dumps(engine.snapshot());BaseTrack._count=0
        StreamingCounter.restore(pickle.loads(blob))
        self.assertEqual(BaseTrack.next_id(),58)

    def test_ended_states_retire_over_repeated_time_intervals(self):
        engine,data=fixture('glass');sample=data['frames'][0]
        for i in range(100):engine.update(sample|dict(index=i*250))
        self.assertLessEqual(len(engine.people_counter.states),12)
        self.assertLessEqual(len(engine.people_tracker.tracks),12)

    def test_file_boundary_preserves_global_stride_and_frame_order(self):
        with tempfile.TemporaryDirectory() as td:
            paths=[]
            for segment in range(2):
                path=Path(td)/f'{segment}.avi';paths.append(str(path))
                writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),25,(64,64))
                for frame in range(3):writer.write(np.full((64,64,3),20+segment*50+frame*5,np.uint8))
                writer.release()
            args=SimpleNamespace(source=paths,max_seconds=None,pace=False,batch=2,stride=2)
            q=queue.Queue(2);stop=threading.Event()
            producer=threading.Thread(target=produce,args=(args,q,stop,dict(width=64,height=64,fps=25),time.perf_counter(),0));producer.start()
            observed=[]
            while True:
                batch=q.get(timeout=5)
                if batch is None:break
                if isinstance(batch,BaseException):raise batch
                observed.extend(x['index'] for x in batch)
            producer.join();self.assertEqual(observed,[0,2,4])


if __name__=='__main__':unittest.main()
