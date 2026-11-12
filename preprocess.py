"""Native fixed-length normalization; portable NumPy fallback."""
import ctypes
from pathlib import Path
import numpy as np
from preprocess_numpy import preprocess as numpy_preprocess
try:
    _library=ctypes.CDLL(str(Path(__file__).resolve().parent/'preprocess_native.so'))
    _native=_library.gdg_preprocess
    _native.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t]
    _native.restype=None
except (OSError,AttributeError):
    _native=None
def preprocess(raw):
    if _native is None or np.shape(raw)[-1]!=256:
        return numpy_preprocess(raw)
    a=np.asarray(raw,dtype=np.float32,order='C')
    out=np.empty(a.shape,dtype=np.float32)
    _native(a.ctypes.data,out.ctypes.data,a.size//256)
    return out

def preprocess_into(raw,out):
    if _native is None or np.shape(raw)[-1]!=256:
        np.copyto(out,numpy_preprocess(raw))
        return
    a=np.asarray(raw,dtype=np.float32,order='C')
    if a.shape!=out.shape:raise ValueError('Unexpected signal shape')
    _native(a.ctypes.data,out.ctypes.data,a.size//256)
