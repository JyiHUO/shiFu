import pandas as pd
import lightgbm as lgb
from config import Config
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold

'''
user_id_num: 70711
item_id_num:  3687156
author_id: 778113
item_city: 456
music_id: 82841
device_id: 71681
Duration: 128
'''

print("start reading data")
columns = ["uid", "user_city", "item_id", "author_id", "item_city",
               "channel", "music_id", "device",  "duration_time", "finish", "like"]
all_data = pd.read_csv(Config["all_data_path"])
all_data = all_data[columns]
test = pd.read_csv(Config["test_path"])
test = test[columns]
# train = pd.read_csv(Config["train_path"])
# train = train[columns]
#
# val = pd.read_csv(Config["val_path"])
# val = val[columns]

print("finish reading data")

print("start feature extraction")


'''
onehot：
item_city
duration_time

对于vector特征：
"uid", "user_city", "item_id", "author_id",
"channel", "music_id", "device"
1.提取每个id的数量
2.每个id的正样本数
3.每个id的转化率
4.user_id 跟 item_id组合出现次数
5.组合正样本次数
'''


def g_train_first_order(data):
    columns = data.columns
    data = data.copy().values
    vector_f = ["uid", "user_city", "item_id", "author_id", "item_city",
                "channel", "music_id", "device", "duration_time"]
    kf = KFold(n_splits=7, shuffle=False)
    new_data = []
    for train_index, test_index in kf.split(data):
        train = pd.DataFrame(data=data[train_index], columns=columns)
        test = pd.DataFrame(data=data[test_index], columns=columns)
        for i in range(len(vector_f)):
            print("first_order_feature: " + vector_f[i])
            for label in ["finish", "like"]:
                f = vector_f[i]
                label_rate = train.groupby(f)[label].mean().rename(f + "_" + label + "_rate").reset_index()  # 转化率
                label_sum = train.groupby(f)[label].sum().rename(f + "_" + label + "_num").reset_index()  # 正样本个数
                test = test.merge(right=label_rate, how="left", left_on=f, right_on=f)
                test = test.merge(right=label_sum, how="left", left_on=f, right_on=f)

            feature_num = train.groupby(f)[label].count().rename(f + "_num").reset_index()
            test = test.merge(right=feature_num, how="left", left_on=f, right_on=f)

            print("second_order_feature")
            for j in range(i+1, len(vector_f)):
                f1 = vector_f[i]
                f2 = vector_f[j]
                print("start: "+f1+"_"+f2)
                for label in ["finish", "like"]:
                    label_rate = train.groupby([f1, f2])[label].\
                        mean().rename(f1+"_"+f2+"_"+label+"_rate").reset_index().drop(f2, axis=1)
                    label_sum = train.groupby([f1, f2])[label].\
                        sum().rename(f1+"_"+f2+"_"+label+"_sum").reset_index().drop(f2, axis=1)
                    test = test.merge(right=label_rate, how="left", left_on=f1, right_on=f1)
                    test = test.merge(right=label_sum, how="left", left_on=f1, right_on=f1)
                feature_num = train.groupby([f1, f2])[label]\
                    .count().rename(f1+"_"+f2+"_count").reset_index().drop(f2, axis=1)

                test = test.merge(right=feature_num, how="left", left_on=f1, right_on=f1)

        new_data.append(test)
    new_data = pd.concat(new_data, 0)
    new_data.fillna(0, inplace=True)
    return new_data


def g_test_first_order_two(train, test):
    vector_f = ["uid", "user_city", "item_id", "author_id", "item_city",
                "channel", "music_id", "device", "duration_time"]
    for i in range(len(vector_f)):
        print("first_order_feature: " + vector_f[i])
        for label in ["finish", "like"]:
            f = vector_f[i]
            label_rate = train.groupby(f)[label].mean().rename(f + "_" + label + "_rate").reset_index()  # 转化率
            label_sum = train.groupby(f)[label].sum().rename(f + "_" + label + "_num").reset_index()  # 正样本个数
            test = test.merge(right=label_rate, how="left", left_on=f, right_on=f)
            test = test.merge(right=label_sum, how="left", left_on=f, right_on=f)

        feature_num = train.groupby(f)[label].count().rename(f + "_num").reset_index()
        test = test.merge(right=feature_num, how="left", left_on=f, right_on=f)

        print("second_order_feature")
        for j in range(i + 1, len(vector_f)):
            f1 = vector_f[i]
            f2 = vector_f[j]
            print("start: " + f1 + "_" + f2)
            for label in ["finish", "like"]:
                label_rate = train.groupby([f1, f2])[label]. \
                    mean().rename(f1 + "_" + f2 + "_" + label + "_rate").reset_index().drop(f2, axis=1)
                label_sum = train.groupby([f1, f2])[label]. \
                    sum().rename(f1 + "_" + f2 + "_" + label + "_sum").reset_index().drop(f2, axis=1)
                test = test.merge(right=label_rate, how="left", left_on=f1, right_on=f1)
                test = test.merge(right=label_sum, how="left", left_on=f1, right_on=f1)
            feature_num = train.groupby([f1, f2])[label] \
                .count().rename(f1 + "_" + f2 + "_count").reset_index().drop(f2, axis=1)

            test = test.merge(right=feature_num, how="left", left_on=f1, right_on=f1)

    test.fillna(0, inplace=True)
    return test


def get_feature(data, test):
    '''
    :param train: can be train.csv or all_data.csv
    :param test: can be val.csv or test.csv
    (train.csv, val_csv), (all_data.csv, test.csv)
    :return:
    '''
    data = data.copy()
    test = test.copy()

    test = g_test_first_order_two(data, test)
    data = g_train_first_order(data)

    test.to_csv(Config["save_test_path"], index=False)
    data.to_csv(Config["save_all_data_path"], index=False)




get_feature(all_data, test)