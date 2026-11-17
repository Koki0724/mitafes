"""FP32 ONNX CUDA inference with the same letterbox, GPU NMS and scaling as YOLO.

GPU I/O binding avoids copying dense model outputs to CPU before NMS.
Only the local, fixed-size exported models are accepted.
"""
import ast
import cv2
import numpy as np
import torch
import onnxruntime as ort
from ultralytics.data.augment import LetterBox
from ultralytics.engine.results import Results
from ultralytics.utils import ops
from ultralytics.utils.nms import non_max_suppression


class OnnxModel:
    def __init__(self,path,device):
        self.device=int(device)
        self.torch_device=torch.device(f'cuda:{self.device}')
        stream=torch.cuda.current_stream(self.device)
        options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(path),sess_options=options,providers=[('CUDAExecutionProvider',{
            'device_id':self.device,'user_compute_stream':str(stream.cuda_stream),
            'cudnn_conv_algo_search':'HEURISTIC','cudnn_conv_use_max_workspace':'1'})])
        if 'CUDAExecutionProvider' not in self.session.get_providers():raise RuntimeError('ONNX CUDA provider did not initialize; refusing silent CPU fallback')
        self.session.disable_fallback()
        metadata=self.session.get_modelmeta().custom_metadata_map
        self.task=metadata['task'];self.names=ast.literal_eval(metadata['names']);self.nc=len(self.names)
        self.kpt_shape=ast.literal_eval(metadata['kpt_shape']) if self.task=='pose' else None
        inp=self.session.get_inputs()[0];out=self.session.get_outputs()[0]
        if any(not isinstance(x,int) for x in inp.shape+out.shape):raise ValueError('Use a fixed-size ONNX export')
        self.shape=inp.shape;self.output=torch.empty(out.shape,device=self.torch_device,dtype=torch.float32)
        self.input_name=inp.name;self.output_name=out.name

    def predict(self,images,imgsz,conf,iou,classes,device,half=False,verbose=False):
        if half:raise ValueError('This backend deliberately preserves FP32')
        output=[]
        for image in images:
            transformed=LetterBox(new_shape=tuple(self.shape[-2:]),auto=False,stride=32)(image=image)
            array=np.ascontiguousarray(transformed[...,::-1].transpose(2,0,1)[None])
            tensor=torch.from_numpy(array).to(self.torch_device).float()/255.
            if list(tensor.shape)!=self.shape:raise ValueError(f'Expected {self.shape}, got {list(tensor.shape)}; export this input size')
            binding=self.session.io_binding()
            binding.bind_input(self.input_name,'cuda',self.device,np.float32,self.shape,tensor.data_ptr())
            binding.bind_output(self.output_name,'cuda',self.device,np.float32,list(self.output.shape),self.output.data_ptr())
            self.session.run_with_iobinding(binding)
            pred=non_max_suppression(self.output,conf_thres=conf,iou_thres=iou,classes=classes,nc=self.nc,max_det=300)[0]
            pred[:,:4]=ops.scale_boxes(tensor.shape[2:],pred[:,:4],image.shape)
            keypoints=None
            if self.kpt_shape:
                keypoints=pred[:,6:].view(-1,*self.kpt_shape)
                keypoints=ops.scale_coords(tensor.shape[2:],keypoints,image.shape)
            output.append(Results(orig_img=image,path='',names=self.names,boxes=pred[:,:6],keypoints=keypoints))
        return output
