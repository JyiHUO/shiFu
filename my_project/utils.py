"""
    Some handy functions for pytroch model training ...
"""
import torch
from sklearn import metrics
from config import Config


# Checkpoints
def save_checkpoint(model, model_dir):
    torch.save(model.state_dict(), model_dir)


def resume_checkpoint(model, model_dir):
    model.load_state_dict(torch.load(model_dir))
    model.eval()


# Hyper params
def use_cuda(enabled, device_id=0):
    if enabled:
        assert torch.cuda.is_available(), 'CUDA is not available'
        torch.cuda.set_device(device_id)


def use_optimizer(network):
    if Config["training_config"]['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), lr=Config["training_config"]['adam_lr'],
                                     weight_decay=Config["training_config"]['l2_regularization'])
    elif Config["training_config"]['optimizer'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(network.parameters(),
                                        lr=Config["training_config"]['rmsprop_lr'],
                                        alpha=Config["training_config"]['rmsprop_alpha'],
                                        momentum=Config["training_config"]['rmsprop_momentum'])
    return optimizer


def cal_auc(y_true, y_pred):
    fpr, tpr, t = metrics.roc_curve(y_true, y_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc