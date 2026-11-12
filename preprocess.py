import numpy as np
def preprocess(raw):
    a=np.array(raw,dtype=np.float32,copy=True)
    for sample in a:
        for row in sample:
            ok=np.isfinite(row)
            if not ok.all():row[~ok]=np.interp(np.flatnonzero(~ok),np.flatnonzero(ok),row[ok]) if ok.any() else 0
    a-=np.median(a,axis=-1,keepdims=True)
    a/=np.maximum(1.4826*np.median(np.abs(a),axis=-1,keepdims=True),1e-5)
    return np.clip(a,-8,8)
