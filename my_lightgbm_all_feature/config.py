Config = {
        "model_name": "xDeepFm",
        "large_file": False,
        'use_cuda': False,
        'device_id': 0,
        "num_workers": 8,
        'pretrain': False,
        'model_dir': '../../cache/track2/checkpoints/lgb.model',
        "train_path": "../../cache/track2/tmp/train.csv",  # or ../../data/track2/final_track2_train.txt # train + val
        "val_path": "../../cache/track2/tmp/val.csv",
        "test_path": "../../cache/track2/tmp/test.csv",
        "all_data_path": "../../cache/track2/tmp/all_data.csv",
        "raw_data_path": "../../data/track2/final_track2_train.txt",
        "predict_file": "../../cache/track2/result/result.csv",
        "save_test_path": "../../cache/track2/tmp/test_lgb.csv",
        "save_all_data_path": "../../cache/track2/tmp/all_data_lgb.csv",
        "save_train_path": "../../cache/track2/tmp/train_lgb.csv",
        "save_val_path": "../../cache/track2/tmp/val_lgb.csv"
    }