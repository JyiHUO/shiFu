import numpy as np
import torch as t
from torch.autograd import Variable as V
from torch import nn
from torch.nn import functional as F
from torch import optim
from .base_model import BaseModel

# config = \
# {
#     'total_size': 284,
#     'interactive_field_size': 9,
#     'interactive_field_max_num_list': [73974, 397, 4122689, 850308, 462, 5, 89779, 75085, 641],
#     'emb_size': 20,
#     'no_inter_field_max_num_list': [69, 102, 102, 93, 102, 102, 102, 45, 75],
#     'title_size': 134544+2,
#     "attention":
#     {
#         "input_dim": None,  # input_dim should be set automatically
#         "num_layers": 3,
#         "head_num": 10,
#         "att_emb_size": 20,
#         # "head_num_list": [10, 5, 2, 1],
#         "forward_dim": 200
#     }
# }


class AFFM(nn.Module, BaseModel):
    def __init__(self, config):
        super(AFFM, self).__init__()
        self.config = config["model_config"]["AFFM"]
        # self.config = config
        self.path_config = config
        self.emb_size = self.config['emb_size']
        # interactive part 0 - 8
        # 2 order feature
        emb_list_2 = []
        for i, first in enumerate(self.config['interactive_field_max_num_list']):
            for j, second in enumerate(self.config['interactive_field_max_num_list'][i:]):
                conf = {'first': first, 'second': second, 'emb_size': self.emb_size}
                temp_model = Interac(conf)
                emb_list_2.append(temp_model)
        self.emb_layers_2 = nn.ModuleList(emb_list_2)

        # 1 order feature
        emb_list_1 = []
        for i, field in enumerate(self.config['interactive_field_max_num_list']):
            emb_list_1.append(nn.Embedding(field, self.emb_size))
        self.emb_layers_1 = nn.ModuleList(emb_list_1)

        # part2 not interactive 9 - 17
        emb_list_3 = []
        for i, field in enumerate(self.config['no_inter_field_max_num_list']):
            emb_list_3.append(nn.Embedding(field, self.emb_size))
        self.emb_layers_3 = nn.ModuleList(emb_list_3)

        # part3 title
        self.title_emb = nn.Embedding(self.config["title_size"], self.emb_size)

        # part4 video + audio
        self.video_out = nn.Linear(128, self.emb_size)
        self.audio_out = nn.Linear(128, self.emb_size)

        # non_inc behaviour+face+title_length:9*2
        # inc: 9 * (9 - 1) / 2
        # title: 10
        # video+audio: 2
        # dim = self.emb_size * self.config['interactive_field_size'] * (self.config['interactive_field_size'] - 1) / 2 \
        #       + self.emb_size * self.config['interactive_field_size'] * 2 + self.emb_size * 10 + 2 * self.emb_size

        dim = self.emb_size * self.config['interactive_field_size'] * (self.config['interactive_field_size'] - 1) / 2 \
              + self.emb_size * self.config['interactive_field_size'] * 2 + self.emb_size + 2 * self.emb_size
        print(dim)  # something wrong in this place

        # this is deep part
        # start = int(dim)
        # linear_layer = []
        # bn = []
        # for end in self.config["layers"]:
        #     linear_layer.append(nn.Linear(start, end))
        #     start = end
        #     if end != 1:
        #         bn.append(nn.BatchNorm1d(end))
        # self.linear_layer = nn.ModuleList(linear_layer)
        # self.bn = nn.ModuleList(bn)

        # this is attention part
        self.config["attention"]["input_dim"] = int(dim)
        self.att1 = Attention(self.config)
        self.att2 = Attention(self.config)

        self.linear = nn.Linear(66*self.config['emb_size'],1)

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
            interac_one_order_features.append(temp.unsqueeze(1))

        interac_second_order_features = []
        inc = 0
        for i in range(self.config['interactive_field_size']):
            for j in range(i, self.config['interactive_field_size']):
                temp = self.emb_layers_2[inc]([x[:, i].long(), x[:, j].long()])
                interac_second_order_features.append(temp.unsqueeze(1))
                inc += 1

        # no interactive part
        no_inc_one_order_features = []
        for i in range(9):
            temp = self.emb_layers_3[i](x[:, i + 9].long())
            no_inc_one_order_features.append(temp.unsqueeze(1))

        # title part
        x_title = x[:, 18:28].long()
        title_feature = t.mean(self.title_emb(x_title), dim=1)

        # video feature
        x_video = self.video_out(x[:, 28:156].float())
        print("video input")
        print(x[:, 28:156].size())

        # autio feature
        x_autio = self.audio_out(x[:, 156:].float())
        print("audio input")
        print(x[:, 156:].size())

        total_features = []
        total_features.extend(interac_one_order_features)
        print(len(total_features))
        total_features.extend(interac_second_order_features)
        print(len(total_features))
        total_features.extend(no_inc_one_order_features)
        print(len(total_features))
        total_features.append(title_feature.unsqueeze(1))
        total_features.append(x_video.unsqueeze(1))
        total_features.append(x_autio.unsqueeze(1))

        out = t.cat(total_features, dim=1)  # batch * m * emb_size

        # DNN part
        # for i in range(len(self.linear_layer)):
        #     out = self.linear_layer[i](out)
        #     if i != len(self.linear_layer) - 1:
        #         out = self.bn[i](out)
        #
        # y = F.sigmoid(out)

        # attention part
        out = self.att1(out)
        out = self.att2(out)
        out = self.att2(out)

        batch, _, _ = out.size()
        out = out.view(batch, -1)
        y = self.linear(out)

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
        y = first_emb * second_emb
        return y


class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.config = config
        emb_size = self.config["emb_size"]
        self.head_num = self.config["attention"]["head_num"]
        self.att_emb_size = self.config["attention"]["att_emb_size"]
        self.w_query = nn.Linear(emb_size, self.att_emb_size*self.head_num, bias=False)
        self.w_keys = nn.Linear(emb_size, self.att_emb_size*self.head_num, bias=False)
        self.w_values = nn.Linear(emb_size, self.att_emb_size*self.head_num, bias=False)
        self.w_resnet = nn.Linear(emb_size, self.att_emb_size, bias=False)

    def forward(self, x):
        '''
        :param x:  batch * m, emb_size
        :return: y batch
        '''
        batch, m, _ = x.size()
        query = self.w_query(x).view(self.head_num, batch, m, self.att_emb_size)
        keys = self.w_keys(x).view(self.head_num, batch, self.att_emb_size, m)
        values = self.w_values(x).view(self.head_num, batch, m, self.att_emb_size)

        inner_pro = query.matmul(keys)  # head_num * batch * m * m
        att_score = F.softmax(inner_pro, dim=3)

        result = att_score.matmul(values)  # head_num * batch * m * att_emb_size

        # head compress, may need some modification
        result = t.mean(result, 0)  # batch * m * att_emb_size

        # theme from resnet
        result = F.relu(result + self.w_resnet(x))  # batch * m * att_emb_size

        return result


model = NFFM(config)
print(model)
x = t.FloatTensor(np.random.randint(1, 4, (3, 284)))
y = model(x)
print(y)




