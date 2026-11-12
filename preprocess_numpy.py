"""Fixed-length vectorized median, with the same interpolation and float32 math."""
import numpy as np
def preprocess(raw):
 a=np.array(raw,dtype=np.float32,copy=True)
 for sample in a:
  for row in sample:
   ok=np.isfinite(row)
   if not ok.all():row[~ok]=np.interp(np.flatnonzero(~ok),np.flatnonzero(ok),row[ok]) if ok.any() else 0
 n=a.shape[-1]
 if n!=256:
  a-=np.median(a,axis=-1,keepdims=True)
  a/=np.maximum(1.4826*np.median(np.abs(a),axis=-1,keepdims=True),1e-5)
 else:
  v=np.partition(a,(127,128),axis=-1)
  median=(v[...,127:128]+v[...,128:129])*np.float32(.5)
  a-=median
  v=np.partition(np.abs(a),(127,128),axis=-1)
  median=(v[...,127:128]+v[...,128:129])*np.float32(.5)
  a/=np.maximum(1.4826*median,1e-5)
 return np.clip(a,-8,8)
