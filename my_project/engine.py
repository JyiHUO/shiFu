import torch
from torch import nn
from tensorboardX import SummaryWriter

from utils import save_checkpoint, use_optimizer
import numpy as np
from utils import cal_auc
from config import Config
import pandas as pd


class Engine(object):
    def __init__(self):
        self._writer = SummaryWriter(log_dir=Config["normal_config"]["model_log_dir"])  # tensorboard writer
        # self._writer.add_text('config', str(config), 0)
        self.opt = use_optimizer(self.model)
        self.crit = nn.CrossEntropyLoss()

    def batch_forward(self, batch):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        # user, item, target = Variable(batch[0]), Variable(batch[1]), Variable(batch[2])
        # [0, 1, 2, 3, 4, 5, 8, 9, 6, 7]
        # "user_id"0, "user_city"1, "item_id"2, "author_id"3, "item_city"4,
        # "channel"5, "finish"6, "like"7, "music_id"8, "device"9
        X = batch[:, :9]
        label = [batch[:, 6], batch[:, 7]]   # finish like
        target = batch[:, -1].squeeze()
        if Config["normal_config"]['use_cuda'] is True:
            X = X.cuda()
            target = target.cuda()

        pred = self.model(X)  # "fl_00 12", "fl_01 13", "fl_11 14", "fl_10 15"

        # cal loss
        loss = self.crit(pred, target)

        # cal auc
        pred_list = []
        pred_list.append(pred[:, 2] + pred[:, 3])
        pred_list.append(pred[:, 1] + pred[:, 2])
        auc = []
        for i in range(len(pred_list)):
            auc.append(cal_auc(label[i].cpu().detach().numpy(), pred_list[i].cpu().detach().numpy()))

        return label, pred, loss, auc

    def train_single_batch(self, batch):
        '''
        This function should be modify when I add new feature
        :param batch:
        first version:
        batch[0] = uid, batch[1] = user_city ...
        :return:
        '''
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        # clear gradient history
        self.opt.zero_grad()
        target, pred, loss, auc = self.batch_forward(batch)

        # gradient update
        loss.backward()
        self.opt.step()

        return loss.cpu().detach().numpy(), auc

    def train_an_epoch(self, train_loader, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.train()
        total_loss = []
        total_auc = []
        for batch_id, batch in enumerate(train_loader):
            # batch = torch.LongTensor(batch)
            assert isinstance(batch[0], torch.LongTensor)
            loss, auc = self.train_single_batch(batch)
            print("Training epoch_id: ", epoch_id, " batch_id: ", batch_id, " loss: ", loss, " auc: ", auc)
            total_loss.append(loss)
            if batch_id == 0:
                total_auc = [[] for i in auc]
            for i in range(len(auc)):
                total_auc[i].append(auc[i])

        self._writer.add_scalar('train_performance/total_loss', np.mean(total_loss), epoch_id)
        self._writer.add_scalar('train_performance/finish_auc', np.mean(total_auc[0]), epoch_id)
        self._writer.add_scalar('train_performance/like_auc', np.mean(total_auc[1]), epoch_id)

    def evaluate(self, evaluate_data, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.eval()
        total_loss = []
        total_auc = []
        total_pred = []
        total_target = []

        for batch_id, batch in enumerate(evaluate_data):
            assert isinstance(batch[0], torch.LongTensor)
            target, pred, loss, auc = self.batch_forward(batch)

            print("Evaluating epoch_id: ", epoch_id, " batch_id: ", batch_id, " loss: ", loss, " auc: ", auc)
            total_loss.append(loss.cpu().detach().numpy())
            if batch_id == 0:
                total_auc = [[] for i in auc]
                total_pred = [[] for i in auc]
                total_target = [[] for i in auc]

            for i in range(len(auc)):
                total_pred[i].append(pred[i].cpu().detach().numpy())
                total_target[i].append(target[i].cpu().detach().numpy())

        for i in range(len(total_auc)):
            total_pred[i] = np.concatenate(total_pred[i])
            total_target[i] = np.concatenate(total_target[i])

        # auc and loss
        total_loss = np.mean(total_loss)
        for i in range(len(total_auc)):
            total_auc[i].append(cal_auc(total_target[i], total_pred[i]))
        print("[**Evluating whole Epoch** ]: loss = ", total_loss, " auc = ", total_auc)

        self._writer.add_scalar('test_performance/total_loss', total_loss, epoch_id)
        self._writer.add_scalar('test_performance/finish_auc', np.mean(total_auc[0]), epoch_id)
        self._writer.add_scalar('test_performance/like_auc', np.mean(total_auc[1]), epoch_id)

        return "_".join(total_auc)

    def save(self, epoch_id, auc):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        model_dir = Config["normal_config"]['model_dir'].format(Config["normal_config"]["model_name"], auc, epoch_id)
        save_checkpoint(self.model, model_dir)

    def predict(self, test_data):
        '''
        may need some modify
        :param test_data:
        :return:
        '''
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.eval()
        uid = []
        item_id = []
        finish_pred_list = []
        like_pred_list = []

        for batch_id, batch in enumerate(test_data):
            assert isinstance(batch[0], torch.LongTensor)
            finish, finish_pred, like, like_pred, \
            loss, finish_loss, like_loss, \
            finish_auc, like_auc = self.batch_forward(batch)

            uid.append(batch[:, 0].cpu().detach().numpy())
            item_id.append(batch[:, 2].cpu().detach().numpy())
            finish_pred_list.append(finish_pred.cpu().detach().numpy())
            like_pred_list.append(like_pred.cpu().detach().numpy())

        uid = np.concatenate(uid).astype(int)
        item_id = np.concatenate(item_id).astype(int)
        finish_probability = np.concatenate(finish_pred_list)
        like_probability = np.concatenate(like_pred_list)

        data = np.concatenate([uid[:, None], item_id[:, None],
                               finish_probability[:, None], like_probability[:, None]], 1)
        df = pd.DataFrame(data, columns=["uid", "item_id",
                                         "finish_probability", "like_probability"])
        df.to_csv(Config["normal_config"]["predict_file"], index=None, float_format="%.6f")
