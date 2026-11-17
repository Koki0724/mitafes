"""Load isolated TensorRT 8.6 libraries without changing earlier environments."""
import ctypes,sys
from pathlib import Path


def load_trt():
    root=Path(__file__).resolve().parent/'vendor/trt'
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    # PyTorch uses cuDNN 9. TensorRT 8.6 needs cuDNN 8; distinct SONAMEs coexist.
    ctypes.CDLL(str(root/'nvidia/cudnn/lib/libcudnn.so.8'),mode=ctypes.RTLD_GLOBAL)
    import tensorrt_bindings as trt
    return trt
