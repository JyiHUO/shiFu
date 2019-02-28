import torch


class BaseModel:
    def __init__(self):
        self.config = None
        model_config = None
        data_config = None

    def init_weight(self):
        pass

    def load_pretrain_weights(self):
        self.load_state_dict(torch.load(self.config["normal_config"]["pretrain_model_dir"]))
        self.eval()






