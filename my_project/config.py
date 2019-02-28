
# for track2
from collections import OrderedDict

track = "track2"
Config = {}

if track == "track2":
    normal_setting = {
        "large_file": False,
        'use_cuda': True,
        'device_id': 0,
        'pretrain': False,
        'pretrain_model_dir': '',
        'model_dir': '../../checkpoints/{}_auc_{}_Epoch{}.model',
        "train_path": "../../cache/track2/tmp/train.csv",  # or ../../data/track2/final_track2_train.txt # train + val
        "val_path": "../../cache/track2/tmp/val.csv",
        "test_path": "../../cache/track2/tmp/test.csv",
        'hd5_train_path': "../../cache/track2/tmp/hd5_train.hd5",
        "hd5_val_path": "../../cache/track2/tmp/hd5_val.hd5",
        "raw_test_path": '../../data/track2/final_track2_test_no_anwser.txt',
        "raw_data_path": "../../data/track2/final_track2_train.txt"
    }

    model_setting = {
        'model_name': 'xDeepFM',
        'num_epoch': 100,
        'batch_size': 1024,
        'optimizer': 'adam',
        'adam_lr': 1e-2,
        'latent_dim': 8,
        'num_negative': 4,
        'layers': [128,64,64,48,32,16,16,8],  # layers[0] is the concat of latent user vector & latent item vector
        'l2_regularization': 0.0000001,
    }

    data_setting = OrderedDict({
        'uid': 73974,
        'user_city': 396,
        'item_id': 4122689,
        'author_id': 850308,
        'item_city': 461,
        'channel': 5,
        'music_id': 89778,
        'device': 75085,
    })

    Config.update(normal_setting)
    Config.update(model_setting)
    Config["data_setting"] = data_setting
else:
    # this is for track1
    normal_setting = {
        "large_file": False,
        'use_cuda': True,
        'device_id': 0,
        'pretrain': False,
        'pretrain_model_dir': '',
        'model_dir': '../../checkpoints/{}_auc_{}_Epoch{}.model',
        "train_path": "../../cache/track1/tmp/train.csv",  # or ../../data/track2/final_track2_train.txt # train + val
        "val_path": "../../cache/track1/tmp/val.csv",
        "test_path": "../../cache/track1/tmp/test.csv",
        'hd5_train_path': "../../cache/track1/tmp/hd5_train.hd5",
        "hd5_val_path": "../../cache/track1/tmp/hd5_val.hd5",
        "raw_test_path": '../../data/track1/final_track1_test_no_anwser.txt',
        "raw_data_path": "../../data/track1/final_track1_train.txt"
    }

    model_setting = {
        'model_name': 'xDeepFM',
        'num_epoch': 100,
        'batch_size': 1024,
        'optimizer': 'adam',
        'adam_lr': 1e-2,
        'latent_dim': 8,
        'num_negative': 4,
        'layers': [128, 64, 64, 48, 32, 16, 16, 8],
        'l2_regularization': 0.0000001,
    }


    data_setting = {
        'uid': 73974,
        'user_city' :  396,
        'item_id' :  4122604,
        'author_id' :  850299,
        'item_city' :  461,
        'channel' :  5,
        'music_id': 89778,
        'device': 75085,
    }

    Config.update(normal_setting)
    Config.update(data_setting)
    Config.update(model_setting)