"""Build local FP32 TensorRT engines, with explicit bounded workspace."""
import argparse,ast,hashlib,json,time
from pathlib import Path
import torch,onnx,cv2,numpy as np
from ultralytics.data.augment import LetterBox
from trt_support import load_trt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--onnx',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workspace-mb',type=int,default=512)
    p.add_argument('--precision',choices=['fp32','fp16','int8'],default='fp32')
    p.add_argument('--crop',type=float,nargs=4,default=[.4,0,.8,1])
    a=p.parse_args();torch.cuda.set_device(0);trt=load_trt()
    logger=trt.Logger(trt.Logger.WARNING);builder=trt.Builder(logger)
    network=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser=trt.OnnxParser(network,logger)
    if not parser.parse(a.onnx.read_bytes()):raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config=builder.create_builder_config();config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,a.workspace_mb*1048576)
    config.clear_flag(trt.BuilderFlag.TF32);config.max_aux_streams=0
    calibration_rows=[]
    if a.precision=='fp16':config.set_flag(trt.BuilderFlag.FP16)
    if a.precision=='int8':
        config.set_flag(trt.BuilderFlag.INT8)
        root=Path(__file__).resolve().parent
        calibration_rows=[r for r in json.loads((root.parent/'cpu_counter/calibration/selected.json').read_text()) if r['camera']=='glass'][:24]
        shape=tuple(network.get_input(0).shape)
        class Calibrator(trt.IInt8EntropyCalibrator2):
            def __init__(self):
                super().__init__();self.position=0
                self.device_input=torch.empty(shape,device='cuda:0',dtype=torch.float32)
            def get_batch_size(self):return shape[0]
            def get_batch(self,names):
                if self.position+shape[0]>len(calibration_rows):return None
                images=[]
                for row in calibration_rows[self.position:self.position+shape[0]]:
                    image=cv2.imread(row['path']);h,w=image.shape[:2]
                    x1,y1,x2,y2=[round(v*s) for v,s in zip(a.crop,[w,h,w,h])]
                    image=LetterBox(shape[-2:],auto=False,stride=32)(image=image[y1:y2,x1:x2])
                    images.append(np.ascontiguousarray(image[...,::-1].transpose(2,0,1)).astype(np.float32)/255.)
                self.device_input.copy_(torch.from_numpy(np.stack(images)));torch.cuda.synchronize()
                self.position+=shape[0];return [self.device_input.data_ptr()]
            def read_calibration_cache(self):return None
            def write_calibration_cache(self,data):a.output.with_suffix('.calibration').write_bytes(data)
        calibrator=Calibrator();config.int8_calibrator=calibrator
        # Preserve the entire position/keypoint/class head; only the backbone
        # and neck are eligible for quantization. No scoring labels are used.
        for index in range(network.num_layers):
            layer=network.get_layer(index)
            if '/model.23/' in layer.name and layer.type in [trt.LayerType.CONVOLUTION,trt.LayerType.ACTIVATION,trt.LayerType.ELEMENTWISE]:layer.precision=trt.float32
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
    config.builder_optimization_level=3
    cache_path=a.output.parent/'tactics.cache';a.output.parent.mkdir(parents=True,exist_ok=True)
    cache=config.create_timing_cache(cache_path.read_bytes() if cache_path.exists() else b'');config.set_timing_cache(cache,False)
    started=time.perf_counter();serialized=builder.build_serialized_network(network,config)
    if serialized is None:raise RuntimeError('TensorRT engine construction failed')
    a.output.write_bytes(bytes(serialized));cache_path.write_bytes(bytes(config.get_timing_cache().serialize()))
    m=onnx.load(a.onnx);metadata={x.key:x.value for x in m.metadata_props}
    report=dict(source=str(a.onnx),source_sha256=hashlib.sha256(a.onnx.read_bytes()).hexdigest(),tensorrt=trt.__version__,
                names=ast.literal_eval(metadata['names']),task=metadata['task'],
                kpt_shape=ast.literal_eval(metadata['kpt_shape']) if metadata['task']=='pose' else None,
                precision={'fp32':'FP32','fp16':'Mixed FP16 with FP32 IO','int8':'INT8 backbone/neck with FP32 head and IO'}[a.precision],
                calibration=calibration_rows,workspace_mb=a.workspace_mb,build_seconds=time.perf_counter()-started,
                gpu=torch.cuda.get_device_name(0),capability=torch.cuda.get_device_capability(0),
                fast_fp16=builder.platform_has_fast_fp16,fast_int8=builder.platform_has_fast_int8)
    a.output.with_suffix('.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
