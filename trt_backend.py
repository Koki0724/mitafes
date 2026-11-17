"""Isolated TensorRT 8.6 FP32 execution with shared preprocessing/postprocessing."""
import json
from pathlib import Path
import torch
from gpu_backend import GpuModel
from trt_support import load_trt


class TrtModel(GpuModel):
    def __init__(self,path,device,shape,backend='trt_graph',batch=1,half=False):
        if half:raise ValueError('Engine precision is fixed at build time; leave --half off')
        self.device=torch.device(f'cuda:{int(device)}');torch.cuda.set_device(self.device)
        self.backend=backend;self.half=False;self.batch=batch;self.graph=None
        trt=load_trt();self.tensorrt_version=trt.__version__
        metadata=json.loads(Path(path).with_suffix('.json').read_text())
        self.graph_precision=metadata['precision']
        self.task=metadata['task'];self.names={int(k):v for k,v in metadata['names'].items()};self.nc=len(self.names)
        self.kpt_shape=metadata['kpt_shape'];self.logger=trt.Logger(trt.Logger.WARNING)
        self.runtime=trt.Runtime(self.logger);self.engine=self.runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if self.engine is None:raise RuntimeError('TensorRT engine failed to load')
        self.context=self.engine.create_execution_context()
        inputs=[self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors) if self.engine.get_tensor_mode(self.engine.get_tensor_name(i))==trt.TensorIOMode.INPUT]
        outputs=[self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors) if self.engine.get_tensor_mode(self.engine.get_tensor_name(i))==trt.TensorIOMode.OUTPUT]
        if len(inputs)!=1 or len(outputs)!=1:raise ValueError('Expected one dense detector output')
        if any(self.engine.get_tensor_dtype(name)!=trt.float32 for name in inputs+outputs):raise ValueError('This wrapper requires FP32 engine IO')
        self.shape=tuple(self.engine.get_tensor_shape(inputs[0]))
        if self.shape!=(batch,3,*shape):raise ValueError(f'Engine shape {self.shape} differs from configured {(batch,3,*shape)}')
        self.host=torch.empty(self.shape,dtype=torch.uint8,pin_memory=True)
        self.bytes=torch.zeros(self.shape,dtype=torch.uint8,device=self.device)
        self.input_fp=torch.empty(self.shape,dtype=torch.float32,device=self.device)
        self.prediction=torch.empty(tuple(self.engine.get_tensor_shape(outputs[0])),dtype=torch.float32,device=self.device)
        self.context.set_tensor_address(inputs[0],self.input_fp.data_ptr())
        self.context.set_tensor_address(outputs[0],self.prediction.data_ptr())
        self.stream=torch.cuda.Stream(device=self.device)
        with torch.inference_mode(),torch.cuda.stream(self.stream):
            for _ in range(3):self._forward()
        self.stream.synchronize()
        if backend=='trt_graph':
            self.graph=torch.cuda.CUDAGraph()
            with torch.inference_mode(),torch.cuda.graph(self.graph,stream=self.stream):self._forward()
        self.stream.synchronize()

    def _forward(self):
        self.input_fp.copy_(self.bytes).div_(255.)
        if not self.context.execute_async_v3(self.stream.cuda_stream):raise RuntimeError('TensorRT inference failed')
        return self.prediction
