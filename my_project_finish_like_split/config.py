
# for track2
from collections import OrderedDict

track = "track2"
compute = "cpu"
Config = {}

if compute == "cpu":
    if track == "track2":
        normal_config = {
            "task": "finish",  # or like
            "model_name": "xDeepFm",
            "large_file": False,
            'use_cuda': False,
            'device_id': 0,
            "num_workers": 0,
            'pretrain': False,
            'pretrain_model_dir': '../../../checkpoints/',
            'model_dir': '../../cache/track2/checkpoints/{}_auc:{}_Epoch{}.model',
            "model_log_dir": "../../cache/track2/runs/",
            "train_path": "../../cache/track2/tmp/train.csv",  # or ../../data/track2/final_track2_train.txt # train + val
            "val_path": "../../cache/track2/tmp/val.csv",
            "test_path": "../../cache/track2/tmp/test.csv",
            "all_data_path": "../../cache/track2/tmp/all_data.csv",
            'hd5_train_path': "../../cache/track2/tmp/hd5_train.hd5",
            "hd5_val_path": "../../cache/track2/tmp/hd5_val.hd5",
            "raw_test_path": '../../data/track2/final_track2_test_no_anwser.txt',
            "raw_data_path": "../../data/track2/final_track2_train.txt",
            "predict_file": "../../cache/track2/result/result.csv",
            # for lgb
            "save_test_path": "../../cache/track2/tmp/test_lgb.csv",
            "save_all_data_path": "../../cache/track2/tmp/all_data_lgb.csv",
            "save_train_path": "../../cache/track2/tmp/train_lgb.csv",
            "save_val_path": "../../cache/track2/tmp/val_lgb.csv",
            # title data
            "raw_title_data_path": "../../data/track2/track2_title.txt",
            "title_data_path": "../../cache/track2/tmp/title_data.json",
            # face data
            "raw_face_data_path": "../../cache/track2/tmp/track2_face_attrs_fill.csv",
            "face_data_path": "../../cache/track2/tmp/face_data.json",
            # video data
            "raw_video_data_path": "../../data/track2/track2_video_features.txt",
            "video_data_path": "../../cache/track2/tmp/video_data.json",
            # audio data
            "raw_audio_data_path": "../../data/track2/track2_audio_features.txt",
            "audio_data_path": "../../cache/track2/tmp/audio_data.json"
        }

        model_config = {
            'xDeepFM_config': {
                "CIN": {
                    "k": 5,
                    "m": 9,
                    "D": 100,
                    "H": 20
                },

                "DNN": {
                    "num_layers": 3,
                    "in_dim": 9 * 100,
                    "out_dim_list": [200, 100, 100]
                }
            },

            'MLP_config': {
                "k":  [50, 10, 50, 50, 10, 5, 25, 25, 10],  # 50
                "layers": [256, 256, 128, 128, 64, 32, 1]
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
            },
            "DeepFM": {
                "emb_size": 15,
                "num_feature": 9,
                "layers": [256, 256, 128, 128, 64, 32, 1]
            },
            "NFFM": {
                'total_size': 284,
                'interactive_field_size': 9,
                'interactive_field_max_num_list': [73974, 397, 4122689, 850308, 462, 5, 89779, 75085, 641],
                'emb_size': 20,
                'no_inter_field_max_num_list': [69, 102, 102, 93, 102, 102, 102, 45, 75],
                'title_size': 134544+2
            }
        }

        training_config = {
            'num_epoch': 10,
            'batch_size': 64,
            'optimizer': 'adam',
            'adam_lr': 0.0001,
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
            "duration_time": 641
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
else:
    if track == "track2":
        normal_config = {
            "task": "finish",  # or like
            "model_name": "nffm",
            "large_file": False,
            'use_cuda': True,
            'device_id': 0,
            "num_workers": 0,
            'pretrain': False,
            'pretrain_model_dir': '../../../checkpoints/',
            'model_dir': '../../cache/track2/checkpoints/{}_auc:{}_Epoch{}.model',
            "model_log_dir": "../../cache/track2/runs/",
            "train_path": "../../cache/track2/tmp/train.csv",
            # or ../../data/track2/final_track2_train.txt # train + val
            "val_path": "../../cache/track2/tmp/val.csv",
            "test_path": "../../cache/track2/tmp/test.csv",
            "all_data_path": "../../cache/track2/tmp/all_data.csv",
            'hd5_train_path': "../../cache/track2/tmp/hd5_train.hd5",
            "hd5_val_path": "../../cache/track2/tmp/hd5_val.hd5",
            "raw_test_path": '../../data/track2/final_track2_test_no_anwser.txt',
            "raw_data_path": "../../data/track2/final_track2_train.txt",
            "predict_file": "../../cache/track2/result/result.csv",
            # for lgb
            "save_test_path": "../../cache/track2/tmp/test_lgb.csv",
            "save_all_data_path": "../../cache/track2/tmp/all_data_lgb.csv",
            "save_train_path": "../../cache/track2/tmp/train_lgb.csv",
            "save_val_path": "../../cache/track2/tmp/val_lgb.csv",
            # title data
            "raw_title_data_path": "../../data/track2/track2_title.txt",
            "title_data_path": "../../cache/track2/tmp/title_data.json",
            # face data
            "raw_face_data_path": "../../cache/track2/tmp/track2_face_attrs_fill.csv",
            "face_data_path": "../../cache/track2/tmp/face_data.json",
            # video data
            "raw_video_data_path": "../../data/track2/track2_video_features.txt",
            "video_data_path": "../../cache/track2/tmp/video_data.json",
            # audio data
            "raw_audio_data_path": "../../data/track2/track2_audio_features.txt",
            "audio_data_path": "../../cache/track2/tmp/audio_data.json"
        }

        model_config = {
            'xDeepFM_config': {
                "CIN": {
                    "k": 5,
                    "m": 9,
                    "D": 100,
                    "H": 20
                },

                "DNN": {
                    "num_layers": 3,
                    "in_dim": 9 * 100,
                    "out_dim_list": [200, 100, 100]
                }
            },

            'MLP_config': {
                "k":  [50, 10, 50, 50, 10, 5, 25, 25, 10],  # 50
                "layers": [256, 256, 128, 128, 64, 32, 1]
            },

            "DTFM": {
                "id": {
                    "uid_num": 73974,
                    "user_city_num": 397,
                    "item_id_num": 4122689,
                    "author_id_num": 850308,
                    "item_city_num": 462,
                    "channel_num": 5,
                    "music_id_num": 89779,
                    "device_num": 75085
                },
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
                },

                "TF": {
                    "input_dim": 100,
                    "num_layers": 1,
                    "head_num_list": [10, 5, 2, 1],
                    "head_num_list_length": 4,
                    "forward_dim": 200
                }
            },
            "DeepFM": {
                "emb_size": 15,
                "num_feature": 9,
                "layers": [256, 256, 128, 128, 64, 32, 1]
            },
            "NFFM": {
                'total_size': 284,
                'interactive_field_size': 9,
                'interactive_field_max_num_list': [73974, 397, 4122689, 850308, 462, 5, 89779, 75085, 641],
                'emb_size': 20,
                'no_inter_field_max_num_list': [69, 102, 102, 93, 102, 102, 102, 45, 75],
                'title_size': 134544+2
            }
        }

        training_config = {
            'num_epoch': 10,
            'batch_size': 1024 * 2,
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
            "duration_time": 641
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
            "train_path": "../../cache/track1/tmp/train.csv",
            # or ../../data/track2/final_track2_train.txt # train + val
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
            'user_city': 396,
            'item_id': 4122604,
            'author_id': 850299,
            'item_city': 461,
            'channel': 5,
            'music_id': 89778,
            'device': 75085,
        }

        training_config = {
            'num_epoch': 10,
            'batch_size': 1024,
            'optimizer': 'adam',
            'adam_lr': 1e-2,
            'l2_regularization': 0.0000001
        }

        Config["normal_config"] = normal_config
        Config["model_config"] = model_config
        Config["training_config"] = training_config
        Config["data_config"] = data_config