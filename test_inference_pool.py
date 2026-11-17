"""Parallel completion must never reorder tracking or hide a failed batch."""
import json,queue,time,unittest
from pathlib import Path
from inference_pool import inferred_batches
from engine import StreamingCounter

ROOT=Path(__file__).resolve().parent


class DelayedDetector:
    def __init__(self,delay=0,fail=False):self.delay=delay;self.fail=fail
    def infer(self,batch):
        time.sleep(self.delay)
        if self.fail:raise RuntimeError('simulated worker failure')
        return batch


class PoolTests(unittest.TestCase):
    def test_reordering_preserves_all_62_events(self):
        data=json.loads((ROOT/'tests/fixtures/glass_detections.json').read_text())
        profile=json.loads((ROOT/'reference/profiles/glass_1219_heads_including_bottom.json').read_text())
        frames=data['frames'][:80]
        q=queue.Queue()
        for start in range(0,len(frames),8):q.put(frames[start:start+8])
        q.put(None);received=[]
        for batch,items,_ in inferred_batches(q,[DelayedDetector(.01),DelayedDetector()]):received.extend(items)
        self.assertEqual(received,frames)
        # Run the full golden sequence across workers and compare all events.
        q=queue.Queue()
        for start in range(0,len(data['frames']),8):q.put(data['frames'][start:start+8])
        q.put(None);engine=StreamingCounter(profile,2880,1620,25,2);actual=[]
        for _,items,_ in inferred_batches(q,[DelayedDetector(),DelayedDetector()]):
            for item in items:_,events,_=engine.update(item);actual.extend(events)
        self.assertEqual(engine.summary()['events'],62)
        self.assertEqual(engine.summary()['ltr'],50);self.assertEqual(engine.summary()['rtl'],12)
        self.assertEqual([e['event_id'] for e in actual],list(range(1,63)))

    def test_worker_failure_is_propagated(self):
        q=queue.Queue();q.put([{'index':0}]);q.put([{'index':2}]);q.put(None)
        with self.assertRaisesRegex(RuntimeError,'simulated worker failure'):
            list(inferred_batches(q,[DelayedDetector(),DelayedDetector(fail=True)]))


if __name__=='__main__':unittest.main()
