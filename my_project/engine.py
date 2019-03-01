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
        self.crit = nn.BCELoss()

    def batch_forward(self, batch):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        # user, item, target = Variable(batch[0]), Variable(batch[1]), Variable(batch[2])
        # [0, 1, 2, 3, 4, 5, 8, 9, 6, 7]
        # "user_id"0, "user_city"1, "item_id"2, "author_id"3, "item_city"4,
        # "channel"5, "finish"6, "like"7, "music_id"8, "device"9

        # uid = batch[0]
        # user_city = batch[1]
        # item_id = batch[2]
        # author_id = batch[3]
        # item_city = batch[4]
        # channel = batch[5]
        # music_id = batch[8]
        # device = batch[9]
        X = batch[:, :8]
        finish = batch[:, -2]
        like = batch[:, -1]
        if Config["normal_config"]['use_cuda'] is True:
            X = X.cuda()
            finish = finish.cuda()
            like = like.cuda()
        finish = finish.float()
        like = like.float()

        finish_pred, like_pred = self.model(X)

        # print(finish_pred.size())
        # print(like_pred.size())
        # print(finish.size())
        # print(like.size())
        # cal loss
        finish_loss = self.crit(finish_pred, finish)
        like_loss = self.crit(like_pred, like)
        loss = 0.7 * finish_loss + 0.3 * like_loss

        # cal auc
        # finish_auc
        finish_auc = cal_auc(finish.cpu().detach().numpy(), finish_pred.cpu().detach().numpy())

        # like_auc
        like_auc = cal_auc(like.cpu().detach().numpy(), like_pred.cpu().detach().numpy())

        return finish, finish_pred, like, like_pred, loss, finish_loss, like_loss, finish_auc, like_auc

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
        finish, finish_pred, like, like_pred,\
        loss, finish_loss, like_loss, \
        finish_auc, like_auc = self.batch_forward(batch)

        # gradient update
        loss.backward()
        self.opt.step()

        return loss.cpu().detach().numpy(), finish_loss.cpu().detach().numpy(), like_loss.cpu().detach().numpy(), finish_auc, like_auc

    def train_an_epoch(self, train_loader, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.train()
        total_loss = []
        total_finish_loss = []
        total_like_loss = []
        avg_finish_auc = []
        avg_like_auc = []
        for batch_id, batch in enumerate(train_loader):
            assert isinstance(batch[0], torch.LongTensor)
            loss, finish_loss, like_loss, finish_auc, like_auc = self.train_single_batch(batch)
            print('[Training Epoch {}] Batch {}, Loss {}, finish_loss {}, like_loss {}, finish_auc {}, like_auc {}'.format(
                epoch_id, batch_id, loss, finish_loss, like_loss, finish_auc, like_auc))
            total_loss.append(loss)
            total_finish_loss.append(finish_loss)
            total_like_loss.append(like_loss)
            avg_finish_auc.append(finish_auc)
            avg_like_auc.append(like_auc)
        self._writer.add_scalar('train_performance/total_loss', np.mean(total_loss), epoch_id)
        self._writer.add_scalar('train_performance/total_finish_loss', np.mean(total_finish_loss), epoch_id)
        self._writer.add_scalar('train_performance/total_like_loss', np.mean(total_like_loss), epoch_id)
        self._writer.add_scalar('train_performance/finish_auc', np.mean(avg_finish_auc), epoch_id)
        self._writer.add_scalar('train_performance/like_auc', np.mean(avg_like_auc), epoch_id)

    def evaluate(self, evaluate_data, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.eval()
        total_loss = []
        total_finish_loss = []
        total_like_loss = []
        finish_pred_list = []
        finish_list = []
        like_pred_list = []
        like_list = []

        for batch_id, batch in enumerate(evaluate_data):
            assert isinstance(batch[0], torch.LongTensor)
            finish, finish_pred, like, like_pred, \
            loss, finish_loss, like_loss, \
            finish_auc, like_auc = self.batch_forward(batch)

            print('[Training Epoch {}] Batch {}, Loss {}, finish_loss {}, like_loss {}, finish_auc {}, like_auc {}'.format(
                    epoch_id, batch_id, loss, finish_loss, like_loss, finish_auc, like_auc))

            total_loss.append(loss.cpu().detach().numpy())
            total_finish_loss.append(finish_loss.cpu().detach().numpy())
            total_like_loss.append(like_loss.cpu().detach().numpy())
            finish_pred_list.append(finish_pred.cpu().detach().numpy())
            finish_list.append(finish.cpu().detach().numpy())
            like_pred_list.append(like_pred.cpu().detach().numpy())
            like_list.append(like.cpu().detach().numpy())

        finish_list = np.concatenate(finish_list)
        finish_pred_list = np.concatenate(finish_pred_list)
        like_list = np.concatenate(like_list)
        like_pred_list = np.concatenate(like_pred_list)

        # auc and loss
        total_loss = np.mean(total_loss)
        total_finish_loss = np.mean(total_finish_loss)
        total_like_loss = np.mean(total_like_loss)
        total_finish_auc = cal_auc(finish_list, finish_pred_list)
        total_like_auc = cal_auc(like_list, like_pred_list)
        print('[**Evluating whole Epoch** {}] loss = {:.4f}, finish_loss = {:.4f}, like_loss = {:.4f}, finish_auc = {:.4f}, like_auc = {:.4f}'.format(
            epoch_id, total_loss, total_finish_loss, total_like_loss, total_finish_auc, total_like_auc))

        self._writer.add_scalar('test_performance/total_loss', total_loss, epoch_id)
        self._writer.add_scalar('test_performance/total_finish_loss', total_finish_loss, epoch_id)
        self._writer.add_scalar('test_performance/total_like_loss', total_like_loss, epoch_id)
        self._writer.add_scalar('test_performance/finish_auc', total_finish_auc, epoch_id)
        self._writer.add_scalar('test_performance/like_auc', total_like_auc, epoch_id)
        return total_finish_auc, total_like_auc

    def save(self, epoch_id, auc):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        model_dir = Config["normal_config"]['model_dir'].format(Config["normal_config"]["model_name"], auc[0], auc[1], epoch_id)
        save_checkpoint(self.model, model_dir)

    def predict(self, test_data):
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

        uid = np.concatenate(uid)
        item_id = np.concatenate(item_id)
        finish_probability = np.concatenate(finish_pred_list)
        like_probability = np.concatenate(like_pred_list)

        data = np.concatenate([uid[:, None], item_id[:, None],
                               finish_probability[:, None], like_probability[:, None]], 1)
        df = pd.DataFrame(data, columns=["uid", "item_id",
                                         "finish_probability", "like_probability"])
        df.to_csv(Config["normal_config"]["predict_file"], index=None, float_format="%.6f")
