import lightgbm as lgb
from config import Config
import pandas as pd
import pickle

train = pd.read_csv(Config["save_train_path"])
val = pd.read_csv(Config["save_val_path"])
# all_data = pd.read_csv(Config["save_all_data_path"])

# train
train_like_label = train["like"]
train_finish_label = train["finish"]
train = train.drop(["finish", "like"], axis=1)

# val
val_finish_label = val["finish"]
val_like_label = val["like"]
val = val.drop(["finish", "like"], axis=1)


# all_data
# all_data_like_label = all_data["like"]
# all_data_finish_label = all_data["finish"]
# all_data = all_data.drop(["finish", "like"], axis=1)


clf_finish = lgb.LGBMClassifier(
    boosting_type="gbdt", num_leaves=31, reg_alpha=0.0, reg_lambda=1,
    max_depth=-1, n_estimators=1500, objective="binary",
    subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
    learning_rate=0.05, min_child_weight=50, random_state=2019, n_jobs=-1
)

clf_like = lgb.LGBMClassifier(
    boosting_type="gbdt", num_leaves=31, reg_alpha=0.0, reg_lambda=1,
    max_depth=-1, n_estimators=1500, objective="binary",
    subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
    learning_rate=0.05, min_child_weight=50, random_state=2019, n_jobs=-1
)

# for offline validation
if True:
    clf_like.fit(train, train_like_label, eval_set=[(val, val_like_label)],
            eval_metric="auc", early_stopping_rounds=100, verbose=True)
    with open(Config["model_dir"]+"_offline_"+"like", "wb") as f:
        pickle.dump(clf_like, f)
    clf_finish.fit(train, train_finish_label, eval_set=[(val, val_finish_label)],
            eval_metric="auc", early_stopping_rounds=100, verbose=True)
    with open(Config["model_dir"]+"_offline_"+"finish", "wb") as f:
        pickle.dump(clf_finish, f)

# for online submission
# if True:
#     # for like
#     clf_like.fit(all_data, all_data_like_label,
#                    eval_metric="auc", verbose=True, early_stopping_rounds=100)
#     with open(Config["model_dir"]+"_online_"+"like", "wb") as f:
#         pickle.dump(clf_finish, f)
#     # for finish
#     clf_finish.fit(all_data, all_data_finish_label,
#                    eval_metric="auc", verbose=True, early_stopping_rounds=100)
#     with open(Config["model_dir"]+"_online_"+"finish", "wb") as f:
#         pickle.dump(clf_finish, f)
