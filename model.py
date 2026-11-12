"""Portable fixed INT8/mixed-precision inference; no runtime calibration."""
import json,warnings
from pathlib import Path
import numpy as np
import torch
from preprocess import preprocess
HERE=Path(__file__).resolve().parent
class Model:
    def __init__(self):
        torch.set_num_threads(1)
        config=json.loads((HERE/'config.json').read_text())
        torch.backends.quantized.engine=config['backend']
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',FutureWarning)
            self.graph=torch.jit.load(str(HERE/config['graph']),map_location='cpu').eval()
        self.family=config['family'];self.bias=np.asarray(config['bias']);self.thresholds=np.asarray(config['thresholds'])
        self.weights=tuple(config.get('weights',[.5,.5]));self.first_index=config.get('first_index',0)
        with torch.inference_mode():self.graph(torch.zeros(1,2,256))
    def predict(self,sample):
        x=torch.from_numpy(preprocess(sample.series[None,:2]))
        with torch.inference_mode():result=self.graph(x)
        if self.family=='conditional':
            first,second,early=result
            if early!=-2:return int(early)
            probs=(first,second) if self.first_index==0 else (second,first)
            p=np.zeros(8)
            for w,prob in zip(self.weights,probs):p+=w*prob.numpy()
        else:p=result.numpy().astype(np.float64)
        p*=self.bias;p/=p.sum();label=int(p.argmax())
        return -1 if p[label]<self.thresholds[label] else label
