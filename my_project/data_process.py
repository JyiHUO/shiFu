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

data = pd.read_csv(Config["raw_data_path"], sep='\t', names=columns)
test_data = pd.read_csv(Config['raw_test_path'], sep='\t', names=columns)
print("finish reading")

# output field size for each feature
for c in data.columns:
    max_id = max(data[c].max(), test_data[c].max())
    min_id = min(data[c].max(), test_data[c].min())
    print(c, ':', " min: ", min_id, " max: ", max_id+1)

# split
# train_size = int(data.shape[0] * (1 - 0.2))
# train = data.iloc[:train_size]
# val = data.iloc[train_size:]
# train.to_csv(Config["train_path"], index=False)
# print("finish train")
# val.to_csv(Config["val_path"], index=False)
# print("finish val")
# test_data.to_csv(Config["test_path"], index=False)
# print("finish test")


