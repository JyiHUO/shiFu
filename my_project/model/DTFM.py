# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 10:10:45 2019

@author: v_fdwang
"""
import numpy as np
import torch as t
from torch.autograd import Variable as V
from torch import nn
from torch.nn import functional as F
from torch import optim


"""
   config: for the parameters, may be later load from a json file 
"""


class DTFM(nn.Module):
    def __init__(self, config):
        super(DTFM, self).__init__()
        config = config["model_config"]["DTFM"]
        self.config = config
        self.emb_layers = nn.ModuleList([nn.Embedding(config["id"][key],config["CIN"]["D"]) for key in self.config["id"]])
        self.cin = CIN(config["CIN"])
        self.dnn = DNN(config["DNN"])
        self.tf = TF(config["TF"])
        dim = config["CIN"]["m"] * config["CIN"]["D"] + config["CIN"]["k"] * config["CIN"]["H"] + \
                config["DNN"]["out_dim_list"][-1] + config["CIN"]["D"] * config["TF"]["head_num_list_length"]
        self.lin = nn.Linear(dim, 2)
        
    def forward(self, x):
        """
            batch first!
            input: a tensor
            x: [uid_num, user_city_num, item_id_num, author_id_num, item_city_num, channel_num, music_id_num, device_num]
            output: two float in (0 - 1) for finish, like
            res: 0.5, 0.5
        """
        batch_size, num_features = x.size()
        if num_features != self.config["CIN"]["m"]:
            raise ValueError("Check the dimention of your features, " \
                             "Expect %d, but got %d!"%(self.config["CIN"]["m"], num_features))
        
         
        x_1 = []
        for i, emb in enumerate(self.emb_layers):
            x_1.append(emb(x[:,i]).unsqueeze(1))
        
        # Input for Linear
        x_21 = t.cat(x_1, 2) # batch * 1 * (m*D)
        
        # Input for CIN 
        x_22 = t.cat(x_1, 1) # batch * m * D 
        
        # Input for DNN
        x_23 = t.cat(x_1, 2) # batch * 1 * (m*D)
        
        # Input for TF
        x_24 = t.cat(x_1, 1) # batch * m * D
        
        # Output for CIN
        x_32 = self.cin(x_22) # batch * 1 * (H*k) 
        
        # Output for DNN
        x_33 = self.dnn(x_23) # batch * 1 * outdim=100
        
        # Output for TF
        x_34 = self.tf(x_24) # batch * 1 * (head_list_length*D)
        
        x_cat = t.cat([x_21, x_32, x_33, x_34], 2) # batch * 1 * (m*D + H*k + outdim + head_list_length*D)
        y = t.sigmoid(self.lin(x_cat)).squeeze(1)
        
        return y[:, 0], y[:, 1]
    
    
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
        self.layers = nn.ModuleList(self.layers)
    
    def forward(self, x):
        """
            Input: batch * m * D
            Output: batch * 1 *(k*H)
        """
        x_0 = x.permute(0, 2, 1).unsqueeze(2).expand([-1, self.D, self.m, self.m])
        x_0 = x_0 * x_0
        x_0 = x_0.contiguous().view((-1, self.D, self.m*self.m))
        x_1 = self.lin(x_0) # -1, self.D, self.H
        x_1 = F.relu(x_1)
        x_list = [x_1]
        
        for layer in self.layers:
            x_k = x_1.unsqueeze(2).expand([-1, self.D, self.m, self.H])
            x_0 = x.permute(0, 2, 1).unsqueeze(3).expand([-1, self.D, self.m, self.H])
            z_k = x_0 * x_k
            z_k = z_k.contiguous().view(-1, self.D, self.m*self.H)
            x_k = layer(z_k)
            x_k = F.relu(x_k)
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
        self.layers = nn.ModuleList(self.layers)
        
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
    
class TF(nn.Module):
    def __init__(self, conf):
        super(TF, self).__init__()
        self.conf = conf
        self.build_model()
        
    def build_model(self):
        self.layers = []
        for i in range(self.conf["num_layers"]):
            layer = []
            for head_num in self.conf["head_num_list"]:
                d_i = self.conf["input_dim"]
                d_k = int(self.conf["input_dim"] / head_num)
                layer_list = [nn.Linear(d_i, d_k, bias=False) for _ in range(head_num)]
                layer_list.append(nn.Linear(d_i, self.conf["forward_dim"]))
                layer_list.append(nn.Linear(self.conf["forward_dim"], d_i))
                layer_list = nn.ModuleList(layer_list)
                layer.append(layer_list)
            layer = nn.ModuleList(layer)
            self.layers.append(layer)
        self.layers = nn.ModuleList(self.layers)
    
    def forward(self, x):
        """
            input: x batch * m * D
            output: batch * 1 * (D*head_num_list_length)
        """
        x_list = [x for _ in range(self.conf["head_num_list_length"])]
        for layer in self.layers:
            z_list = []
            for i, layer_list in enumerate(layer):
                x = x_list[i]
                att_list = layer_list[:-2]
                lin_list = layer_list[-2:]
                temp = []
                batch_size = -1
                for att in att_list:
                    x_att = att(x) # batch * m * d_k
                    batch_size, _, d_k = x_att.size()
                    temp.append(x_att)
                y = t.cat(temp, 2) # batch * m * D
                
                # attention op
                k = y.permute(0, 2, 1)
                v = y.matmul(k) / np.sqrt(d_k)
                z = F.softmax(v, dim=2).matmul(y) # batch * m *  d_i 
                #print(z.size(), x.size())
                z = z + x
                
                # forward op
                out = lin_list[0](z) # batch * m * forward_dim
                out = F.relu(out)     
                out = lin_list[1](out) # batch * m * d_i
                z = out + z
                z_list.append(z)
            x_list = z_list
        batch_size = x.size()[0]
        output = t.sum(t.cat(x_list, 2), 1).contiguous().view(batch_size, 1, -1) # batch *
