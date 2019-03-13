from model.xDeepFM import xDeepFM
import torch
from engine import Engine
from utils import use_cuda
from config import Config


class ModelEngine(Engine):
    """Engine for training & evaluating GMF model"""
    def __init__(self, config, model):
        self.model = model(config)
        super(ModelEngine, self).__init__()
        if Config["normal_config"]['pretrain']:
            self.model.load_pretrain_weights()
        if config["normal_config"]['use_cuda'] is True:
            use_cuda(True, Config["normal_config"]['device_id'])
            self.model.cuda()
        print(self.model)