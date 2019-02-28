# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 16:52:51 2019

@author: v_fdwang
"""

import torch as t
from torch.autograd import Variable as V
from torch import nn
from torch.nn import functional as F
from torch import optim
from .base_model import BaseModel


class xDeepFM(nn.Module, BaseModel):
    def __init__(self, config):
        super(xDeepFM, self).__init__()
        # store some config
        self.config = config
        self.model_config = config["model_config"]["xDeepFM_config"]
        self.data_config = config["data_config"]
        model_config = config["model_config"]["xDeepFM_config"]
        data_config = config["data_config"]

        # model structure
        self.emb_layers = [nn.Embedding(data_config[key], model_config["CIN"]["D"]) for key in data_config]
        self.cin = CIN(model_config["CIN"])
        self.dnn = DNN(model_config["DNN"])
        dim = model_config["CIN"]["m"] * model_config["CIN"]["D"] +\
              model_config["CIN"]["k"] * model_config["CIN"]["H"] +\
              model_config["DNN"]["out_dim_list"][-1]
        self.lin = nn.Linear(dim, 2)
        
    def forward(self, x):
        """
            batch first!
            input: a tensor
            x: batchsize * [uid_num, user_city_num, item_id_num, author_id_num, item_city_num, channel_num, music_id_num, device_num]
            output: two float in (0 - 1) for finish, like
            res: 0.5, 0.5
        """
        batch_size, num_features = x.size()
        if num_features != self.model_config["CIN"]["m"]:
            raise ValueError("Check the dimention of your features, " \
                             "Expect %d, but got %d!"%(self.model_config["CIN"]["m"], num_features))
        
         
        x_1 = []
        for i, emb in enumerate(self.emb_layers):
            x_1.append(emb(x[:,i]).unsqueeze(1))
        
        # Input for Linear
        x_21 = t.cat(x_1, 2) # batch * 1 * (m*D)
        
        # Input for CIN 
        x_22 = t.cat(x_1, 1) # batch * m * D 
        
        # Input for DNN
        x_23 = t.cat(x_1, 2) # batch * 1 * (m*D)
        
        # Output for CIN
        x_32 = self.cin(x_22) # batch * 1 * (H*k) 
        
        # Output for DNN
        x_33 = self.dnn(x_23) # batch * 1 * outdim=100
        
        x_cat = t.cat([x_21, x_32, x_33], 2) # batch * 1 * (m*D + H*k + outdim)
        y = t.sigmoid(self.lin(x_cat)).squeeze(1)

        return y[:, 0], y[:, 1]

    def init_weight(self):
        pass

    
class CIN(nn.Module):
    def __init__(self, conf):
        super(CIN, self).__init__()
        self.conf = conf
        self.k = self.conf["k"]
        self.m = self.conf["m"]
        self.D = self.conf["D"]
        self.H = self.conf["H"]
        self.lin = nn.Linear(self.m*self.m, self.H, bias=False)
        self.layers = []
        for i in range(1, self.k):
            self.layers.append(nn.Linear(self.m*self.H, self.H, bias=False))
    
    def forward(self, x):
        """
            Input: batch * m * D
            Output: batch * 1 *(k*H)
        """
        x_0 = x.permute(0, 2, 1).unsqueeze(2).expand([-1, self.D, self.m, self.m])
        x_0 = x_0 * x_0
        x_0 = x_0.contiguous().view((-1, self.D, self.m*self.m))
        x_1 = self.lin(x_0) # -1, self.D, self.H
        x_list = [x_1]
        for layer in self.layers:
            x_k = x_1.unsqueeze(2).expand([-1, self.D, self.m, self.H])
            x_0 = x.permute(0, 2, 1).unsqueeze(3).expand([-1, self.D, self.m, self.H])
            z_k = x_0 * x_k
            z_k = z_k.contiguous().view(-1, self.D, self.m*self.H)
            x_k = layer(z_k)
            x_list.append(x_k)
            x_1 = x_k
        all_x = t.cat(x_list, 2) # -1, self.D, self.H*self.k
        out = t.sum(all_x, 1).unsqueeze(1) # -1, 1, self.H*self.k
        return out

    
class DNN(nn.Module):
    def __init__(self, conf):
        super(DNN, self).__init__()
        self.conf = conf
        self.layers = []
        start = self.conf["in_dim"]
        for i in range(self.conf["num_layers"]):
            end = self.conf["out_dim_list"][i]
            self.layers.append(nn.Linear(start, end))
            start = end
        
    def forward(self, x):
        """
            input: Variable
                size : batch * 1 * (m*D) 
            output: Variable
                size : batch * 1 * out_dim=100
        """
        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)
        return x
        

