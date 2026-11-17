"""MPS adapter; preserves the CUDA preset's input shape and thresholds.

Actual MPS execution must be validated on Apple Silicon. Linux tests only cover
dispatch/control flow, not MPS numerical agreement or performance.
"""
from ultralytics import YOLO
import torch


class MpsModel:
    graph_precision='FP32'
    def __init__(self,path,shape):
        self.yolo=YOLO(str(path)).to('mps');self.task=self.yolo.task;self.names=self.yolo.names;self.shape=tuple(shape)

    def predict(self,images,imgsz,conf,iou,classes,device=None,half=False,verbose=False):
        result=self.yolo.predict(images,imgsz=self.shape,rect=False,conf=conf,iou=iou,classes=classes,device='mps',half=False,verbose=False)
        torch.mps.synchronize()
        if self.yolo.predictor.model.device.type!='mps':raise RuntimeError('YOLO inference did not run on MPS')
        return [r.cpu() for r in result]
