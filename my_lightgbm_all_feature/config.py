Config = {
        "model_name": "xDeepFm",
        "large_file": False,
        'use_cuda': False,
        'device_id': 0,
        "num_workers": 8,
        'pretrain': False,
        'pretrain_model_dir': '../../../checkpoints/',
        'model_dir': '../../cache/track2/checkpoints/{}_auc:{}_{}_Epoch{}.model',
        "model_log_dir": "../../cache/track2/runs/",
        "train_path": "../../cache/track2/tmp/train.csv",  # or ../../data/track2/final_track2_train.txt # train + val
        "val_path": "../../cache/track2/tmp/val.csv",
        "test_path": "../../cache/track2/tmp/test.csv",
        "all_data_path": "../../cache/track2/tmp/all_data.csv",
        "raw_test_path": '../../data/track2/final_track2_test_no_anwser.txt',
        "raw_data_path": "../../data/track2/final_track2_train.txt",
        "predict_file": "../../cache/track2/result/result.csv",
        "save_test_path": "../../cache/track2/tmp/test_lgb.csv",
        "save_all_data_path": "../../cache/track2/tmp/all_data_lgb.csv"
    }