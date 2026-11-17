"""Apple Silicon runtime checks shared by the runner and environment diagnostic."""
import os
import platform
import resource
import sys
from importlib.metadata import version

import torch


def require_mps():
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise RuntimeError('Apple Silicon Mac + arm64 Python が必要です。Rosetta/x86_64 Python は使用しません。')
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS が利用できません。macOS と PyTorch の対応を確認し、python doctor.py を実行してください。CPU には切り替えません。')
    # Check an actual allocation and operation, not just the backend flag.
    value = (torch.ones(4, device='mps') * 2).sum()
    torch.mps.synchronize()
    if value.item() != 8:
        raise RuntimeError('MPS 演算の確認に失敗しました。')


def synchronize():
    torch.mps.synchronize()


def peak_rss_mb():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports KiB. Both are converted to MiB.
    return value / (1024 ** 2 if sys.platform == 'darwin' else 1024)


def environment():
    return dict(
        system=platform.system(), macos=platform.mac_ver()[0],
        machine=platform.machine(), processor=platform.processor(),
        python=sys.version, executable=sys.executable, cpu_count=os.cpu_count(),
        packages={name: version(name) for name in
                  ['torch', 'torchvision', 'ultralytics', 'opencv-python', 'numpy', 'scipy', 'lap']},
        mps_built=torch.backends.mps.is_built(),
        mps_available=torch.backends.mps.is_available(),
        device='mps', precision='float32',
        mps_environment={name: os.environ.get(name) for name in
                         ['PYTORCH_ENABLE_MPS_FALLBACK', 'PYTORCH_MPS_FAST_MATH',
                          'PYTORCH_MPS_HIGH_WATERMARK_RATIO', 'PYTORCH_MPS_PREFER_METAL']},
    )
