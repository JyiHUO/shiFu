import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DeepFM(nn.Module):
    def __init__(self, config):
        super(DeepFM, self).__init__()
        # normal setting
        self.config = config
        self.model_setting = config["model_config"]["DeepFM"]
        self.data_config = config["data_config"]

        # init fm part
        self.bias = torch.nn.Parameter(torch.randn(1))
        self.fm_first_order_embedding = nn.ModuleList(
            [nn.Embedding(feature_unique_value, 2) for feature_unique_value in self.data_config.values()]
        )
        self.fm_second_order_embedding = nn.ModuleList(
            [nn.Embedding(feature_unique_value, self.model_setting["emb_size"]) for feature_unique_value in self.data_config.values()]
        )

        # init deep part
        all_dims = [self.model_setting["num_feature"]*self.model_setting["emb_size"]]+\
            self.model_setting["layers"]
        self.linear = nn.ModuleList(
            [nn.Linear(all_dims[i-1], all_dims[i]) for i in range(1, len(all_dims))]
        )
        self.bn = nn.ModuleList(
            [nn.BatchNorm1d(all_dims[i]) for i in range(1, len(all_dims)-1)]
        )

    def forward(self, x):
        field_size = x.size()[1]
        fm_first_order_emb_arr= [self.fm_first_order_embedding[i](x[:, i]) for i in range(field_size)]  # [N, 2]
        fm_first_order = sum(fm_first_order_emb_arr)  # .unsqueeze(1)  # N*2

        # for one sample <v_i, v_j> * x_i * x_j = sum( (v_i*x_i)^2 - v_i^2*x_i^2 for k in K )
        # value of x_i equal to zero
        fm_second_order_emb_arr = [self.fm_second_order_embedding[i](x[:, i]) for i in range(field_size)]  # [N*k, N*k ...]
        left_sub = sum(fm_second_order_emb_arr)  # N*k
        left = left_sub * left_sub  # N*k
        right = sum([emb*emb for emb in fm_second_order_emb_arr])  # [N*k, N*k ...] -> N*k
        fm_second_order = torch.sum((left - right) * 0.5, 1).unsqueeze(1)  # N

        # deep part
        deep_a = torch.cat(fm_second_order_emb_arr, 1)
        for i in range(len(self.linear)):
            deep_a = self.linear[i](deep_a)
            if i != len(self.linear) - 1:
                deep_a = self.bn[i](deep_a)

        total_sum = F.sigmoid(deep_a + fm_second_order + fm_first_order + self.bias)
        return total_sum[:, 0], total_sum[:, 1]


