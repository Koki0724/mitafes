"""Parallel independent neural batches, yielded in original video order."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import queue,time


def inferred_batches(buffer,detectors):
    def infer(detector,batch):
        started=time.perf_counter();items=detector.infer(batch)
        return items,time.perf_counter()-started
    if len(detectors)==1:
        while True:
            batch=buffer.get()
            if isinstance(batch,BaseException):raise batch
            if batch is None:return
            items,duration=infer(detectors[0],batch)
            yield batch,items,duration
    else:
        pending=deque();ended=False;sequence=0
        with ThreadPoolExecutor(len(detectors)) as executor:
            while not ended or pending:
                if pending and (pending[0][1].done() or len(pending)==len(detectors) or ended):
                    batch,future=pending.popleft();items,duration=future.result()
                    yield batch,items,duration
                    continue
                try:batch=buffer.get(timeout=.005)
                except queue.Empty:continue
                if isinstance(batch,BaseException):raise batch
                if batch is None:ended=True;continue
                future=executor.submit(infer,detectors[sequence%len(detectors)],batch)
                pending.append((batch,future));sequence+=1
