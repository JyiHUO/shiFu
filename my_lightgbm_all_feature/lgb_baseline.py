import pandas as pd
import lightgbm as lgb
from config import Config
from sklearn.preprocessing import OneHotEncoder

print("start reading data")
all_data = pd.read_csv(Config["all_data_path"])
all_data = all_data[["uid", "user_city", "item_id", "author_id", "item_city",
               "channel", "music_id", "device",  "duration_time", "finish", "like"]]
test = pd.read_csv(Config["test_path"])
test = test[["uid", "user_city", "item_id", "author_id", "item_city",
               "channel", "music_id", "device",  "duration_time", "finish", "like"]]


val = pd.read_csv(Config["val.csv"])
print("finish reading data")
