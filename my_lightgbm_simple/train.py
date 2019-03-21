import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn import metrics
from config import Config


def cal_auc(y_true, y_pred):
    fpr, tpr, t = metrics.roc_curve(y_true, y_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc


train = pd.read_csv(Config["train_path"])
val = pd.read_csv("../../cache/track2/tmp/val.csv")

train_data = train[["uid", "user_city", "item_id", "author_id", "item_city",
               "channel", "music_id", "device",  "duration_time"]]
train_like_label = train["like"]
train_finish_label = train["finish"]

val_data = val[["uid", "user_city", "item_id", "author_id", "item_city",
               "channel", "music_id", "device",  "duration_time"]]
val_finish_label = val["finish"]
val_like_label = val["like"]


print("start training")
params = {}
params["learning_rate"] = 0.003
params["objective"] = "binary"
params["metric"] = "auc"

clf = lgb.LGBMClassifier(
    boosting_type="gbdt", num_leaves=31, reg_alpha=0.0, reg_lambda=1,
    max_depth=-1, n_estimators=1500, objective="binary",
    subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
    learning_rate=0.05, min_child_weight=50, random_state=2019, n_jobs=-1
)


clf.fit(train_data, train_like_label, eval_set=[(val_data, val_like_label)],
        eval_metric="auc", early_stopping_rounds=100, verbose=True
        )


