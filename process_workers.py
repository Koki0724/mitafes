"""One isolated neural process per GPU, with small preprocessed IPC packets."""
import copy,multiprocessing,resource,traceback
from multiprocessing import shared_memory
from types import SimpleNamespace
import cv2,numpy as np,torch
from ultralytics.data.augment import LetterBox


def worker_main(args,width,height,connection):
    mapped={}
    try:
        from run import Detector
        torch.set_num_threads(args.threads);cv2.setNumThreads(1);torch.backends.cudnn.benchmark=args.cudnn_benchmark
        detector=Detector(args,width,height);detector.warmup();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(int(args.device))
        connection.send(dict(ready=True,precision=getattr(detector.person,'graph_precision','FP32'),trt=getattr(detector.person,'tensorrt_version',None)))
        while True:
            packet=connection.recv()
            if packet is None:break
            if isinstance(packet,str) and packet=='stats':
                connection.send(dict(device=args.device,torch_allocated_mb=torch.cuda.memory_allocated(int(args.device))/1048576,
                                     torch_peak_mb=torch.cuda.max_memory_allocated(int(args.device))/1048576,
                                     peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024));continue
            def array(key):
                name,shape=packet[key]
                if name not in mapped:
                    block=shared_memory.SharedMemory(name=name);mapped[name]=(block,np.ndarray(shape,dtype=np.uint8,buffer=block.buf))
                return mapped[name][1]
            detector.person.enqueue_prepared(array('person_shm'),packet['person_shapes'])
            if detector.umbrella:detector.umbrella.enqueue_prepared(array('umbrella_shm'),packet['image_shapes'])
            people=detector.person.collect(.1,.7,[0] if detector.umbrella else [0,25])
            umbrellas=detector.umbrella.collect(.1,.7,[25]) if detector.umbrella else None
            x1,y1,_,_=packet['crop'];ih,iw=packet['image_shapes'][0][:2];items=[]
            for i,(index,p) in enumerate(zip(packet['indices'],people)):
                det=p.boxes.data.numpy().copy();points=p.keypoints.data.numpy().copy() if p.keypoints is not None else None
                if len(det):det[:,[0,2]]=(det[:,[0,2]]+x1)*width/iw;det[:,[1,3]]=(det[:,[1,3]]+y1)*height/ih
                if points is not None and len(points):points[:,:,0]=(points[:,:,0]+x1)*width/iw;points[:,:,1]=(points[:,:,1]+y1)*height/ih
                det=det.tolist();points=points.tolist() if points is not None else None
                if umbrellas is not None:
                    u=umbrellas[i].boxes.data.numpy().copy()
                    if len(u):u[:,[0,2]]*=width/iw;u[:,[1,3]]*=height/ih
                    u=u.tolist();det.extend(u);points.extend([[[0,0,0] for _ in range(17)] for _ in u])
                items.append(dict(index=index,detections=det,umbrella_observed=True,**({'keypoints':points} if points is not None else {})))
            connection.send(dict(items=items))
    except BaseException:
        try:connection.send(dict(error=traceback.format_exc()))
        except (BrokenPipeError,EOFError,OSError):pass
    finally:
        for block,_ in mapped.values():block.close()
        connection.close()


class RemoteDetector:
    def __init__(self,args,width,height):
        if args.backend=='onnx' or args.umbrella_every!=1 or args.flow_every!=1:raise ValueError('Process workers require fixed GPU backends and full cadence')
        self.args=args;self.width=width;self.height=height;self.flow=None;self.transport='shared_memory_v1'
        context=multiprocessing.get_context('spawn');self.connection,child=context.Pipe()
        self.process=context.Process(target=worker_main,args=(args,width,height,child),daemon=True);self.process.start();child.close()
        ready=self.receive(120)
        if not ready.get('ready'):raise RuntimeError('GPU worker did not initialize')
        self.person=SimpleNamespace(graph_precision=ready['precision'],tensorrt_version=ready['trt'])
        self.shared={}
        for name,shape in [('person_shm',args.person_shape)]+([('umbrella_shm',args.umbrella_shape)] if args.camera=='glass' else []):
            shape=(args.batch,3,*shape);block=shared_memory.SharedMemory(create=True,size=int(np.prod(shape)))
            self.shared[name]=(block,np.ndarray(shape,np.uint8,buffer=block.buf))

    def receive(self,timeout=120):
        if not self.connection.poll(timeout):raise TimeoutError('GPU worker timed out')
        result=self.connection.recv()
        if 'error' in result:raise RuntimeError(result['error'])
        return result

    @staticmethod
    def prepare(images,shape,batch,target=None):
        array=np.zeros((batch,3,*shape),np.uint8) if target is None else target
        if target is not None:array.fill(0)
        for i,image in enumerate(images):
            transformed=LetterBox(shape,auto=False,stride=32)(image=image)
            array[i]=transformed[...,::-1].transpose(2,0,1)
        return array

    def infer(self,batch):
        images=[r['image'] for r in batch];h,w=images[0].shape[:2]
        crop=[round(v*s) for v,s in zip(self.args.crop,[w,h,w,h])];x1,y1,x2,y2=crop
        person=[image[y1:y2,x1:x2] for image in images]
        self.prepare(person,tuple(self.args.person_shape),self.args.batch,self.shared['person_shm'][1])
        packet=dict(person_shapes=[im.shape for im in person],image_shapes=[im.shape for im in images],crop=crop,
                    indices=[r['index'] for r in batch])
        if self.args.camera=='glass':self.prepare(images,tuple(self.args.umbrella_shape),self.args.batch,self.shared['umbrella_shm'][1])
        for name,(block,array) in self.shared.items():packet[name]=(block.name,array.shape)
        self.connection.send(packet);return self.receive()['items']

    def stats(self):self.connection.send('stats');return self.receive()
    def snapshot(self):return None
    def restore(self,state):
        if state is not None:raise ValueError('Neural process workers are stateless')
    def close(self):
        try:
            if self.process.is_alive():self.connection.send(None)
        except (BrokenPipeError,EOFError,OSError):pass
        self.process.join(timeout=10)
        if self.process.is_alive():self.process.terminate();self.process.join(timeout=5)
        self.connection.close()
        for block,_ in self.shared.values():block.close();block.unlink()
