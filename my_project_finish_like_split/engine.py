import torch
from torch import nn
from tensorboardX import SummaryWriter

from utils import save_checkpoint, use_optimizer, use_scheduler
import numpy as np
from utils import cal_auc
from config import Config
import pandas as pd


class Engine(object):
    def __init__(self):
        self._writer = SummaryWriter(log_dir=Config["normal_config"]["model_log_dir"])  # tensorboard writer
        # self._writer.add_text('config', str(config), 0)
        self.opt = use_optimizer(self.model)
        self.scheduler = use_scheduler(self.opt)
        self.crit = nn.BCELoss()

    def batch_forward(self, batch):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        # user, item, target = Variable(batch[0]), Variable(batch[1]), Variable(batch[2])
        # [0, 1, 2, 3, 4, 5, 8, 9, 6, 7]
        # "user_id"0, "user_city"1, "item_id"2, "author_id"3, "item_city"4,
        # "channel"5, "finish"6, "like"7, "music_id"8, "device"9
        # print(batch.size())
        X = batch[:, :-1]
        target = batch[:, -1]
        if Config["normal_config"]['use_cuda'] is True:
            X = X.cuda()
            target = target.cuda()
        target = target.float()

        pred = self.model(X)

        # cal loss
        loss = self.crit(pred, target)

        # cal auc
        # finish_auc
        auc = cal_auc(target.cpu().detach().numpy(), pred.cpu().detach().numpy())

        return target, pred, loss, auc

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
        self.scheduler.step()

        return loss.cpu().detach().numpy(), auc

    def train_an_epoch(self, train_loader, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.train()
        total_loss = []
        avg_auc = []

        for batch_id, batch in enumerate(train_loader):
            # assert isinstance(batch[0], torch.LongTensor)
            loss, auc = self.train_single_batch(batch)
            print('[Training Epoch {}] Batch {}, Loss {}, lr {}, task {}, auc {}'.format(
                epoch_id, batch_id, loss, self.opt.param_groups[0]["lr"], Config["normal_config"]["task"], auc))
            total_loss.append(loss)
            avg_auc.append(auc)

            self._writer.add_scalar('train_performance/total_loss_batch', loss, batch_id)
            self._writer.add_scalar('train_performance/avg_auc_batch', auc, batch_id)

        self._writer.add_scalar('train_performance/total_loss', np.mean(total_loss), epoch_id)
        self._writer.add_scalar('train_performance/avg_auc', np.mean(avg_auc), epoch_id)

    def evaluate(self, evaluate_data, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.eval()
        total_loss = []
        pred_list = []
        target_list = []

        for batch_id, batch in enumerate(evaluate_data):
            # assert isinstance(batch[0], torch.LongTensor)
            target, pred, loss, auc = self.batch_forward(batch)

            print('[Evaluating Epoch {}] Batch {}, Loss {}, task {}, auc {}'.format(
                epoch_id, batch_id, loss, Config["normal_config"]["task"], auc))

            total_loss.append(loss.cpu().detach().numpy())
            pred_list.append(pred.cpu().detach().numpy())
            target_list.append(target.cpu().detach().numpy())

        target_list = np.concatenate(target_list)
        pred_list = np.concatenate(pred_list)

        # auc and loss
        total_loss = np.mean(total_loss)
        total_auc = cal_auc(target_list, pred_list)
        print('[Evaluating Epoch {}] Loss {}, task {}, auc {}'.format(
            epoch_id, total_loss, Config["normal_config"]["task"], total_auc))

        self._writer.add_scalar('test_performance/total_loss', total_loss, epoch_id)
        self._writer.add_scalar('test_performance/total_auc', total_auc, epoch_id)
        return total_auc

    def save(self, epoch_id, auc):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        model_dir = Config["normal_config"]['model_dir'].format(Config["normal_config"]["model_name"]+"_"+Config["normal_config"]["task"],
                                                                auc, epoch_id)
        save_checkpoint(self.model, model_dir)

    def predict(self, test_data):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.eval()
        uid = []
        item_id = []
        pred_list = []

        for batch_id, batch in enumerate(test_data):
            # assert isinstance(batch[0], torch.LongTensor)
            print("batch_id: ", batch_id)
            target, pred, loss, auc = self.batch_forward(batch)

            uid.append(batch[:, 0].cpu().detach().numpy())
            item_id.append(batch[:, 2].cpu().detach().numpy())
            pred_list.append(pred.cpu().detach().numpy())
            print(uid[0].shape)
            print(item_id[0].shape)
            print(pred_list[0].shape)

        uid = np.concatenate(uid).astype(int)
        item_id = np.concatenate(item_id).astype(int)
        pred_probability = np.concatenate(pred_list)
        print(uid.shape)
        print(item_id.shape)
        print(pred_probability.shape)

        data = np.concatenate([uid[:, None], item_id[:, None], pred_probability], 1)
        df = pd.DataFrame(data, columns=["uid", "item_id","pred_probability"])
        # df.to_csv(Config["normal_config"]["predict_file"], index=None, float_format="%.6f")
        return df