import pandas
import h5py
from config import Config
import pandas as pd

'''
1. split data into train or validation
2. convert larget file to hd5file
'''

columns = ["uid", "user_city", "item_id", "author_id", "item_city",
            "channel", "finish", "like", "music_id", "device",
            "create_time", "duration_time"]

data = pd.read_csv(Config["normal_config"]["raw_data_path"], sep='\t', names=columns)
test_data = pd.read_csv(Config["normal_config"]['raw_test_path'], sep='\t', names=columns)
print("finish reading")


# outline data
data[data["duration_time"] == 70000] = data.duration_time.median()
data[data["finish"] == 10] = 0
data[data["like"] == 10] = 0
data = data.astype(int)
test_data = test_data.astype(int)

# output field size for each feature
for c in data.columns:
    max_id = max(data[c].max(), test_data[c].max())
    min_id = min(data[c].max(), test_data[c].min())
    print(c, ':', " min: ", min_id, " max: ", max_id+1)

# data clean
# user_city, item_city, music_id should plus one
data["user_city"] = data["user_city"] + 1
data["item_city"] = data["item_city"] + 1
data["music_id"] = data["music_id"] + 1
test_data["user_city"] = test_data["user_city"] + 1
test_data["item_city"] = test_data["item_city"] + 1
test_data["music_id"] = test_data["music_id"] + 1

# split target to finish_lisk: 01, 11, 10, 00
data["fl_00"] = 0
data["fl_01"] = 0
data["fl_11"] = 0
data["fl_10"] = 0

test_data["fl_00"] = 0
test_data["fl_01"] = 0
test_data["fl_11"] = 0
test_data["fl_10"] = 0

data["fl_00"][(data.finish == 0)&(data.like == 0)] = 0
data["fl_01"][(data.finish == 0)&(data.like == 1)] = 1
data["fl_11"][(data.finish == 1)&(data.like == 1)] = 2
data["fl_10"][(data.finish == 1)&(data.like == 0)] = 3

data["target"] = data["fl_00"] + data["fl_01"] + data["fl_11"] + data["fl_10"]

# add title data


# split
train_size = int(data.shape[0] * (1 - 0.2))
train = data.iloc[:train_size]
val = data.iloc[train_size:]
train.to_csv(Config["normal_config"]["train_path"], index=False)
print("finish train")
val.to_csv(Config["normal_config"]["val_path"], index=False)
print("finish val")
test_data.to_csv(Config["normal_config"]["test_path"], index=False)
print("finish test")
data.to_csv(Config["normal_config"]["all_data_path"], index=False)



