import pandas as pd
import lightgbm as lgb
from config import Config
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
import numpy as np
'''
user_id_num: 70711
item_id_num:  3687156
author_id: 778113
item_city: 456
music_id: 82841
device_id: 71681
Duration: 128
'''


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
def g_deep():
    columns = ["uid", "user_city", "item_id", "author_id", "item_city",
                "channel", "finish", "like", "music_id", "device",
                "create_time", "duration_time"]

    data = pd.read_csv(Config["raw_data_path"], sep='\t', names=columns)
    test_data = pd.read_csv(Config['raw_test_path'], sep='\t', names=columns)
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
    train.to_csv(Config["train_path"], index=False)
    print("finish train")
    val.to_csv(Config["val_path"], index=False)
    print("finish val")
    test_data.to_csv(Config["test_path"], index=False)
    print("finish test")
    data.to_csv(Config["all_data_path"], index=False)

def g_train(data):
    columns = data.columns
    data = data.copy().values
    vector_f = ["uid", "user_city", "item_id", "author_id", "item_city",
                "channel", "music_id", "device", "duration_time"]
    kf = KFold(n_splits=7, shuffle=False)
    count = 0
    for train_index, test_index in kf.split(data):
        count += 1
        print("*************"+str(count)+"*****************")
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
                        mean().rename(f1+"_"+f2+"_"+label+"_rate").reset_index()
                    label_sum = train.groupby([f1, f2])[label].\
                        sum().rename(f1+"_"+f2+"_"+label+"_sum").reset_index()
                    print("start merge")
                    test = test.merge(right=label_rate, how="left", on=[f1, f2])
                    print(test.shape)
                    test = test.merge(right=label_sum, how="left", on=[f1, f2])
                    print(test.shape)
                feature_num = train.groupby([f1, f2])[label]\
                    .count().rename(f1+"_"+f2+"_count").reset_index()

                test = test.merge(right=feature_num, how="left", on=[f1, f2])
                print(test.head(1))
                print("check null number")
                print(np.sum(test.isnull())/test.shape[0])


        test.fillna(0, inplace=True)
        test.to_csv(Config["save_all_data_path"]+str(count), index=False)


def g_test_two(train, test):
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
                    mean().rename(f1 + "_" + f2 + "_" + label + "_rate").reset_index()
                label_sum = train.groupby([f1, f2])[label]. \
                    sum().rename(f1 + "_" + f2 + "_" + label + "_sum").reset_index()
                print("start merge")
                test = test.merge(right=label_rate, how="left", on=[f1, f2])
                print(test.shape)
                test = test.merge(right=label_sum, how="left", on=[f1, f2])
                print(test.shape)
            feature_num = train.groupby([f1, f2])[label] \
                .count().rename(f1 + "_" + f2 + "_count").reset_index()

            test = test.merge(right=feature_num, how="left", on=[f1, f2])
            print(test.head(1))
            print("check null number")
            print(np.sum(test.isnull()) / test.shape[0])

    test.fillna(0, inplace=True)
    test.to_csv(Config["save_test_path"], index=False)


def get_feature():
    '''
    :param train: can be train.csv or all_data.csv
    :param test: can be val.csv or test.csv
    (train.csv, val_csv), (all_data.csv, test.csv)
    :return:
    '''
    print("start reading data")
    columns = ["uid", "user_city", "item_id", "author_id", "item_city",
               "channel", "music_id", "device", "duration_time", "finish", "like"]
    data = pd.read_csv(Config["all_data_path"])
    data = data[columns]
    test = pd.read_csv(Config["test_path"])
    test = test[columns]

    print("finish reading data")

    print("start feature extraction")

    g_train(data)
    print("finish train data")
    g_test_two(data, test)
    print("finish test data")


def collect_train_val_split():
    '''
    查看样本有多大，才决定是否合并all_data的所有分片
    :return:
    '''
    data = pd.read_csv(Config["save_all_data_path"])
    L = data.shape[0]
    train = data[:int(0.8*L)]
    val = data[int(0.8*L):]
    train.to_csv(Config["save_train_path"], index=False)
    val.to_csv(Config["save_test_path"], index=False)


# 生成train,test,val,all_data作为deepFM等等的输入，只有9个特征
# g_deep()

# 生成一阶和二阶的特征来作为lgb的训练
get_feature()

# 将生成的all_data_lgb进行train和val的切分
collect_train_val_split()

# face_data的合并
