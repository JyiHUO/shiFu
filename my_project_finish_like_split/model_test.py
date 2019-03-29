# from model.xDeepFM_new import xDeepFM
from config import Config
import torch
from models_engine import ModelEngine
from model.deepfm import DeepFM
from model.DTFM import TF

model = DeepFM(config=Config)
data = torch.ones((128, 9))
data = data.long()
out = model(data)
print(out)