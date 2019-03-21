# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 10:14:54 2019

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


class NFFM(nn.Module):
    def __init__(self, config):
        super(NFFM, self).__init__()
        config = config["model_config"]["NFFM"]
        self.config = config
        self.emb_size = self.config['emb_size']
        # 2 order feature
        emb_list_2 = []
        for i,first in enumerate(self.config['field_max_num_list']):
            for j,second in enumerate(self.config['field_max_num_list'][i:]):
                conf = {'first': first, 'second': second, 'emb_size': self.emb_size}
                temp_model = Interac(conf)
                emb_list_2.append(temp_model)
        self.emb_layers_2 = nn.ModuleList(emb_list_2)
        
        # 1 order feature
        emb_list_1 = []
        for i,field in enumerate(self.config['field_max_num_list']):
            emb_list_1.append(nn.Embedding(field, self.emb_size))
        self.emb_layers_1 = nn.ModuleList(emb_list_1)
        
        dim = self.emb_size * self.config['field_size'] * (self.config['field_size'] - 1) / 2 \
              + self.emb_size * self.config['field_size'] * 2
        self.lin = nn.Linear(int(dim), 2)
        
    def forward(self, x):
        """
            batch first!
            input: a tensor
            x: [uid_num, user_city_num, item_id_num, author_id_num, item_city_num, channel_num, music_id_num, device_num]
            output: two float in (0 - 1) for finish, like
            res: 0.5, 0.5
        """
        batch_size, seq_len = x.size()
        if seq_len != self.config['field_size']:
            raise ValueError("Check your input size!")
        one_order_features = []
        for i in range(seq_len):
            temp = self.emb_layers_1[i](x[:, i])
            one_order_features.append(temp)
        
        second_order_features = []
        inc = 0
        for i in range(seq_len):
            for j in range(i, seq_len):
                temp = self.emb_layers_2[inc]([x[:, i], x[:, j]])
                second_order_features.append(temp)
                inc += 1
                
        
        total_features = []
        total_features.extend(one_order_features)
        total_features.extend(second_order_features)
        
        out = t.cat(total_features, dim=1)
        y = F.sigmoid(self.lin(F.relu(out)))
                 
        return y[:, 0], y[:, 1]
    

class Interac(nn.Module):
    def __init__(self, conf):
        super(Interac, self).__init__()
        self.conf = conf
        self.emb_size = self.conf['emb_size']
        self.first_size = self.conf['first']
        self.second_size = self.conf['second']
        self.emb1 = nn.Embedding(self.first_size, self.emb_size)
        self.emb2 = nn.Embedding(self.second_size, self.emb_size)
        
    def forward(self, x):
        """
        input:
            x batch_size * 2
        output:
            y batch_size * emb_size
        """
        first, second = x[0], x[1]
        first_emb = self.emb1(first)
        second_emb = self.emb2(second)
        y = first_emb*second_emb
        return y



