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
from .base_model import BaseModel


"""
   config: for the parameters, may be later load from a json file 
"""
# config = {
#         'total_size': 384,
#         'interactive_field_size': 9,
#         'interactive_field_max_num_list': [73974,397,4122689,850308,462,5,89779,75085,641],
#         'emb_size': 20,
#         'no_inter_field_max_num_list': [69, 102, 102, 93, 102, 102, 102, 45, 75],
#         'title_size': 134411,
#         }


class NFFM(nn.Module, BaseModel):
    def __init__(self, config):
        super(NFFM, self).__init__()
        self.config = config["model_config"]["NFFM"]
        self.path_config = config
        self.emb_size = self.config['emb_size']
        # interactive part 0 - 8
        # 2 order feature
        emb_list_2 = []
        for i,first in enumerate(self.config['interactive_field_max_num_list']):
            for j,second in enumerate(self.config['interactive_field_max_num_list'][i:]):
                conf = {'first': first, 'second': second, 'emb_size': self.emb_size}
                temp_model = Interac(conf)
                emb_list_2.append(temp_model)
        self.emb_layers_2 = nn.ModuleList(emb_list_2)
        
        # 1 order feature
        emb_list_1 = []
        for i,field in enumerate(self.config['interactive_field_max_num_list']):
            emb_list_1.append(nn.Embedding(field, self.emb_size))
        self.emb_layers_1 = nn.ModuleList(emb_list_1)
        
        # part2 not interactive 9 - 17
        emb_list_3 = []
        for i,field in enumerate(self.config['no_inter_field_max_num_list']):
            emb_list_3.append(nn.Embedding(field, self.emb_size))
        self.emb_layers_3 = nn.ModuleList(emb_list_3)
        
        # part3 title
        self.title_emb = nn.Embedding(self.config["title_size"], self.emb_size)
        
        
        dim = self.emb_size * self.config['interactive_field_size'] * (self.config['interactive_field_size'] - 1) / 2 \
              + self.emb_size * self.config['interactive_field_size'] * 2 + self.config['emb_size']*10 + 256

        start = dim
        linear_layer = []
        bn = []
        for end in self.config["layers"]:
            linear_layer.append(nn.Linear(start, end))
            if end != 1:
                bn.append(nn.BatchNorm1d(end))
        self.linear_layer = nn.ModuleList(linear_layer)
        self.bn = nn.ModuleList(bn)

        
    def forward(self, x):
        """
            batch first!
            input: a tensor
            x: [uid_num, user_city_num, item_id_num, author_id_num, item_city_num, channel_num, music_id_num, device_num]
            output: two float in (0 - 1) for finish, like
            res: 0.5, 0.5
        """
        batch_size, seq_len = x.size()
        if seq_len != self.config['total_size']:
            raise ValueError("Check your input size!")
        # interactive part
        interac_one_order_features = []
        for i in range(self.config['interactive_field_size']):
            temp = self.emb_layers_1[i](x[:, i].long())
            interac_one_order_features.append(temp)
        
        interac_second_order_features = []
        inc = 0
        for i in range(self.config['interactive_field_size']):
            for j in range(i, self.config['interactive_field_size']):
                temp = self.emb_layers_2[inc]([x[:, i].long(), x[:, j].long()])
                interac_second_order_features.append(temp)
                inc += 1
                
        # no interactive part
        no_inc_one_order_features = []
        for i in range(9):
            temp = self.emb_layers_3[i](x[:, i+9].long())
            no_inc_one_order_features.append(temp)
            
        # title part
        x_title = x[:, 18:28].long()
        title_feature = t.sum(self.title_emb(x_title), dim=1)
        
        # video feature
        x_video = x[:, 28:156].float()
        
        # autio feature
        x_autio = x[:, 156:].float()
        
        
        
        total_features = []
        total_features.extend(interac_one_order_features)
        total_features.extend(interac_second_order_features)
        total_features.extend(no_inc_one_order_features)
        total_features.append(title_feature)
        total_features.append(x_video)
        total_features.append(x_autio)
        
        out = t.cat(total_features, dim=1)

        # DNN part
        for i in range(len(self.linear_layer)):
            out = self.linear_layers[i](out)
            if i != len(self.linear_layers) - 1:
                out = self.bn[i](out)

        y = F.sigmoid(out)  
        return y
    

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

"""
model = NFFM(config)    
print(model)
x = t.FloatTensor(np.random.randint(1, 4, (3, 384)))
y = model(x)
print(y)
"""
