from model.xDeepFM import xDeepFM
from config import Config
import torch
from models_engine import ModelEngine

model = xDeepFM(config=Config)
data = torch.ones((128, 9))
data = data.long()
a, b = model(data)
print(a.size())
print(b.size())