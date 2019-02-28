import torch
from engine import Engine
from utils import use_cuda, resume_checkpoint
from ..config import Config


class MLP(torch.nn.Module):
    def __init__(self):
        # super(MLP, self).__init__()
        super().__init__()
        self.num_users = Config['num_users']
        self.num_items = Config['num_items']
        self.latent_dim = Config['latent_dim']

        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(Config['layers'][:-1], Config['layers'][1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        self.affine_output = torch.nn.Linear(in_features=Config['layers'][-1], out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        vector = torch.cat([user_embedding, item_embedding], dim=-1)  # the concat latent vector
        for idx, _ in enumerate(range(len(self.fc_layers))):
            vector = self.fc_layers[idx](vector)
            vector = torch.nn.ReLU()(vector)
            # vector = torch.nn.BatchNorm1d()(vector)
            # vector = torch.nn.Dropout(p=0.5)(vector)
        logits = self.affine_output(vector)
        pred = self.logistic(logits)
        return pred

    def init_weight(self):
        pass

    def load_pretrain_weights(self):
        """Loading weights from trained model"""
        resume_checkpoint(self, model_dir=Config['pretrain_model_dir'])


class MLPEngine(Engine):
    """Engine for training & evaluating GMF model"""
    def __init__(self, config):
        self.model = MLP(config)
        if Config['pretrain']:
            self.model.load_pretrain_weights()
        if config['use_cuda'] is True:
            use_cuda(True, Config['device_id'])
            self.model.cuda()
        super(MLPEngine, self).__init__(config)
        print(self.model)

