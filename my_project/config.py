
# for track2
from collections import OrderedDict

track = "track2"
Config = {}

if track == "track2":
    normal_config = {
        "model_name": "DTFM",
        "large_file": False,
        'use_cuda': False,
        'device_id': 0,
        "num_workers": 8,
        'pretrain': False,
        'pretrain_model_dir': '../../../checkpoints/',
        'model_dir': '../../cache/track2/checkpoints/{}_finish_auc_{}_like_auc_{}_Epoch{}.model',
        "model_log_dir": "../../cache/track2/runs/",
        "train_path": "../../cache/track2/tmp/train.csv",  # or ../../data/track2/final_track2_train.txt # train + val
        "val_path": "../../cache/track2/tmp/val.csv",
        "test_path": "../../cache/track2/tmp/test.csv",
        "all_data_path": "../../cache/track2/tmp/all_data.csv",
        'hd5_train_path': "../../cache/track2/tmp/hd5_train.hd5",
        "hd5_val_path": "../../cache/track2/tmp/hd5_val.hd5",
        "raw_test_path": '../../data/track2/final_track2_test_no_anwser.txt',
        "raw_data_path": "../../data/track2/final_track2_train.txt",
        "predict_file": "../../cache/track2/result/result.csv"
    }

    model_config = {
        'xDeepFM_config': {
            "CIN": {
                "k": 5,
                "m": 8,
                "D": 100,
                "H": 20
            },

            "DNN": {
                "num_layers": 3,
                "in_dim": 8 * 100,
                "out_dim_list": [200, 100, 100]
            }
        },

        'MLP_config': {
            "k": 50,
            "layers": [200, 64, 32, 16, 2]
        },

        "DTFM": {
            "id":{
                "uid_num": 73974,
                "user_city_num": 397,
                "item_id_num": 4122689,
                "author_id_num": 850308,
                "item_city_num": 462,
                "channel_num": 5,
                "music_id_num": 89779,
                "device_num": 75085
                },
            "CIN":{
                    "k": 5,
                    "m": 8,
                    "D": 100,
                    "H": 20
                    },

            "DNN":{
                    "num_layers": 3,
                    "in_dim": 8 * 100,
                    "out_dim_list": [200, 100, 100]
                    },

            "TF":{
                    "input_dim": 100,
                    "num_layers": 1,
                    "head_num_list": [10, 5, 2, 1],
                    "head_num_list_length": 4,
                    "forward_dim": 200
                    }
        }
    }

    training_config = {
        'num_epoch': 100,
        'batch_size': 64,
        'optimizer': 'adam',
        'adam_lr': 0.001,
        'l2_regularization': 0.0000001
    }

    data_config = OrderedDict({
        'uid': 73974,
        'user_city': 397,
        'item_id': 4122689,
        'author_id': 850308,
        'item_city': 462,
        'channel': 5,
        'music_id': 89779,
        'device': 75085,
    })

    Config["normal_config"] = normal_config
    Config["model_config"] = model_config
    Config["training_config"] = training_config
    Config["data_config"] = data_config
else:
    # this is for track1
    normal_config = {
        "large_file": False,
        'use_cuda': True,
        'device_id': 0,
        'pretrain': False,
        'pretrain_model_dir': '',
        'model_dir': '../../cache/checkpoints/{}_auc_{}_Epoch{}.model',
        "train_path": "../../cache/track1/tmp/train.csv",  # or ../../data/track2/final_track2_train.txt # train + val
        "val_path": "../../cache/track1/tmp/val.csv",
        "test_path": "../../cache/track1/tmp/test.csv",
        'hd5_train_path': "../../cache/track1/tmp/hd5_train.hd5",
        "hd5_val_path": "../../cache/track1/tmp/hd5_val.hd5",
        "raw_test_path": '../../data/track1/final_track1_test_no_anwser.txt',
        "raw_data_path": "../../data/track1/final_track1_train.txt"
    }

    model_config = {
        'model_name': 'MLP',
        'num_epoch': 100,
        'batch_size': 1024,
        'optimizer': 'adam',
        'adam_lr': 1e-2,
        'latent_dim': 8,
        'num_negative': 4,
        'layers': [128, 64, 64, 48, 32, 16, 16, 8],
        'l2_regularization': 0.0000001,
    }

    data_config = {
        'uid': 73974,
        'user_city' :  396,
        'item_id' :  4122604,
        'author_id' :  850299,
        'item_city' :  461,
        'channel' :  5,
        'music_id': 89778,
        'device': 75085,
    }

    training_config = {
        'num_epoch': 100,
        'batch_size': 1024,
        'optimizer': 'adam',
        'adam_lr': 1e-2,
        'l2_regularization': 0.0000001
    }

    Config["normal_config"] = normal_config
    Config["model_config"] = model_config
    Config["training_config"] = training_config
    Config["data_config"] = data_config