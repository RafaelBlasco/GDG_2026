from pathlib import Path
import torch
from preprocess import preprocess_into
HERE=Path(__file__).resolve().parent
class Model:
 def __init__(self):
  torch.set_num_threads(1);torch.backends.quantized.engine='fbgemm';self.graph=torch.jit.load(str(HERE/'controller.pt'),map_location='cpu').eval();self.buffer=torch.empty((1,2,256),dtype=torch.float32);self.array=self.buffer.numpy()
  with torch.inference_mode():self.graph(torch.zeros(1,2,256))
 def predict(self,sample):
  preprocess_into(sample.series[None,:2],self.array)
  with torch.inference_mode():return int(self.graph(self.buffer))
