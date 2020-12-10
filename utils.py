import os
import gc
import sys
import math
import torch  # PyTorch
import random
import joblib
import warnings
import numpy as np
import pandas as pd

# categorical columns
CAT_FEATURES = ["pm_type", "station", "summary", "icon"]  # ,"month","source","baseline_month"]

# continues columns
CONT_FEATURES = ["hour", "dayofyear",  # "day",
                 # "elevation",
                 # "weekend",
                 "weekday", "year",
                 # "latitude","longitude",
                 # "precipIntensity","precipProbability",
                 "temperature", "apparentTemperature",
                 "dewPoint", "humidity", "windSpeed", "windBearing",
                 "cloudCover", "uvIndex", "visibility",
                 # "baseline_mean","baseline_std",
                 # "random_noise",
                 ]


# target
TARGET = "target"


def read_data(root_folder):
    """
    docstring
    """

    df_train = pd.read_csv(os.path.join(root_folder, "pm_train.csv"))
    df_test = pd.read_csv(os.path.join(root_folder, "pm_test.csv"))
    df_weather = pd.read_csv(os.path.join(root_folder, "weather.csv"))
    df_sub = pd.read_csv(os.path.join(root_folder, "sample_submission.csv"))

    df_train = df_train.rename(columns={"type": "pm_type"})
    df_test = df_test.rename(columns={"type": "pm_type"})

    df_train['date'] = pd.to_datetime(df_train['date'])
    df_test['date'] = pd.to_datetime(df_test['date'])
    df_weather['date'] = pd.to_datetime(df_weather['date'])

    print("NaN in `summary`", df_weather[df_weather.summary.isna()].shape[0])
    print("NaN in `icon`", df_weather[df_weather.icon.isna()].shape[0])

    print("DATA SHAPES:", df_train.shape, df_test.shape, df_weather.shape, df_sub.shape)

    df_train['dayofyear'] = df_train['date'].dt.dayofyear
    df_test['dayofyear'] = df_test['date'].dt.dayofyear
    df_weather['dayofyear'] = df_weather['date'].dt.dayofyear

    df_train['hour'] = df_train['date'].dt.hour
    df_test['hour'] = df_test['date'].dt.hour
    df_weather['hour'] = df_weather['date'].dt.hour

    df_train['year'] = df_train['date'].dt.year
    df_test['year'] = df_test['date'].dt.year
    df_weather['year'] = df_weather['date'].dt.year

    df_train['month'] = df_train['date'].dt.month
    df_test['month'] = df_test['date'].dt.month
    df_weather['month'] = df_weather['date'].dt.month

    df_train['day'] = df_train['date'].dt.day
    df_test['day'] = df_test['date'].dt.day
    df_weather['day'] = df_weather['date'].dt.day

    df_train['weekday'] = df_train['date'].dt.weekday
    df_test['weekday'] = df_test['date'].dt.weekday

    df_train['weekend'] = df_train['weekday'].apply(lambda x: 1 if x > 5 else 0)
    df_test['weekend'] = df_test['weekday'].apply(lambda x: 1 if x > 5 else 0)

    df_train = pd.merge(df_train, df_weather.drop(["month", "day"], axis=1).iloc[:, 2:], on=[
                        "year", "dayofyear", "hour"], how='left')
    df_test = pd.merge(df_test, df_weather.drop(["month", "day"], axis=1).iloc[:, 2:],  on=[
                       "year", "dayofyear", "hour"], how='left')

    months_in_test = df_test.month.unique()

    print("Months in test:", months_in_test)

    # indices_2015_end   = df_train[df_train.apply(lambda x: x.year==2015 and x.month in [10,11,12], axis=1)].index
    # indices_2016_early = df_train[df_train.apply(lambda x: x.year==2016 and x.month in [1,2,3] , axis=1)].index
    # indices_2016_end   = df_train[df_train.apply(lambda x: x.year==2016 and x.month in [10,11,12], axis=1)].index
    # indices_2017_early = df_train[df_train.apply(lambda x: x.year==2017 and x.month in [1,2,3] , axis=1)].index
    # indices_2017_end   = df_train[df_train.apply(lambda x: x.year==2017 and x.month in [10,11,12], axis=1)].index
    # indices_2018_early = df_train[df_train.apply(lambda x: x.year==2018 and x.month in [1,2,3,10] , axis=1)].index

    indices_2019_early = df_test[df_test.apply(lambda x: x.year == 2019 and x.month in [1, 2, 3], axis=1)].index
    indices_2019_end = df_test[df_test.apply(lambda x: x.year == 2019 and x.month in [10, 11, 12], axis=1)].index
    indices_2020_early = df_test[df_test.apply(lambda x: x.year == 2020 and x.month in [1, 2, 3], axis=1)].index
    indices_2020_end = df_test[df_test.apply(lambda x: x.year == 2020 and x.month in [10, 11, 12], axis=1)].index

    # month
    df_test.loc[indices_2019_early, "season"] = "indices_2019_early"
    df_test.loc[indices_2019_end, "season"] = "indices_2019_end"
    df_test.loc[indices_2020_early, "season"] = "indices_2020_early"
    df_test.loc[indices_2020_end, "season"] = "indices_2020_end"

    assert df_test.season.isna().sum() == 0

    return df_train, df_test, df_sub, df_weather


def get_folds(train):

    train["fold"] = -1

    tmp = train[["year", "month"]].copy()
    tmp = tmp[tmp.year.isin([2015, 2016])]
    fold1 = tmp[tmp.apply(lambda x: (x.year == 2015 and x.month in [10, 11, 12])
                          or (x.year == 2016 and x.month in [1, 2, 3]), axis=1)].index

    tmp = train[["year", "month"]].copy()
    tmp = tmp[tmp.year.isin([2016, 2017])]
    fold2 = tmp[tmp.apply(lambda x: (x.year == 2016 and x.month in [10, 11, 12])
                          or (x.year == 2017 and x.month in [1, 2, 3]), axis=1)].index

    tmp = train[["year", "month"]].copy()
    tmp = tmp[tmp.year.isin([2017, 2018])]
    fold3 = tmp[tmp.apply(lambda x: (x.year == 2017 and x.month in [10, 11, 12])
                          or (x.year == 2018 and x.month in [1, 2, 3, 10]), axis=1)].index

    train.loc[fold1, "fold"] = 0
    train.loc[fold2, "fold"] = 1
    train.loc[fold3, "fold"] = 2

    print("DONE FOLDS...", fold1.shape, fold2.shape, fold3.shape)

    return train


# NAN HANDLER
def add_isna_cols(all_data, cat_feat, cont_feat):
    for col in cat_feat + cont_feat:
        if all_data[col].isna().sum() > 0:

            if col in cat_feat:
                all_data[col] = all_data[col].astype("str")
            else:  # which means it is a continous feature
                all_data[col] = all_data[col].fillna(all_data[col].median())

            new_col = f"{col}_isna"
            all_data[new_col] = 0
            all_data.loc[all_data[all_data[col].isna()].index, new_col] = 1

            cont_feat.append(new_col)

    return all_data, cat_feat, cont_feat


def simple_label_encode(all_data, cat_feat):
    """
    docstring
    """
    cat_input_dims = {}

    # SIMPLE LABEL ENCODE
    for col in cat_feat:
        ll = [x for x in all_data[col].unique().tolist() if not pd.isnull(x)]
        dd = {l: i for i, l in enumerate(ll)}
        all_data[col] = all_data[col].map(dd)
        all_data[col] = all_data[col].astype(int)
        cat_input_dims[col] = len(ll)

    return all_data, cat_input_dims


def one_hot_encode(df, cols):

    df_cols = df[cols]

    df = pd.get_dummies(df, columns=cols)
    df[cols] = df_cols

    new_cols = []
    for col in cols:
        new_cols += [x for x in df.columns.tolist() if x.startswith(f"{col}_")]

    return df, new_cols

# TRANSFORM TARGET


class TargetTransform():

    def __init__(self, transform_power=1, epsilon=1e-15):
        self.transform_power = transform_power
        self.epsilon = epsilon

    def transform_target(self, pd_series):
        return np.power(pd_series.clip(self.epsilon), 1/self.transform_power)

    def inverse_transform_target(self, pd_series):
        return np.power(pd_series, self.transform_power)


def read_and_preprocess(targetTransform, root_folder="./data"):
    """
    docstring
    """
    train, test, sub, weather = read_data(root_folder)
    train = get_folds(train)

    train["train"] = 1
    test["train"] = 0
    all_data = pd.concat([train, test]).reset_index(drop=True)

    all_data, cat_feat, cont_feat = add_isna_cols(all_data, CAT_FEATURES, CONT_FEATURES)

    # all_data, cat_input_dims = simple_label_encode(all_data, CAT_FEATURES)
    # print(cat_input_dims)

    all_data, new_cols = one_hot_encode(all_data, CAT_FEATURES)
    cont_feat += new_cols

    train = all_data[all_data.train == 1]
    test = all_data[all_data.train == 0]
    del train["train"]
    del test["train"]
    del all_data

    train["target"] = targetTransform.transform_target(train.aqi)

    # print(train.aqi)
    # print(train.target)
    # print(targetTransform.inverse_transform_target(train.target))
    # print("DONE TARGET TRANSFORMING...\n" + "-"*20)

    return train, test, cont_feat


def create_model_dir(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    targetTransform = TargetTransform(transform_power=2)
    read_and_preprocess(targetTransform)
