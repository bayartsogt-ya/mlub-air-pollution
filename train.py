import torch

import numpy as np
import pandas as pd
from time import time
from tqdm.auto import tqdm

from sklearn.metrics import mean_squared_error
from utils import TargetTransform, read_and_preprocess, create_model_dir, seed_everything
from train_utils import train_fold

import argparse




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments.')
    parser.add_argument('--data_dir', default="./data", type=str ,help='path to data directory')
    args = parser.parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", DEVICE)

    st = time()
    targetTransform = TargetTransform(transform_power=1)
    train, test, cat_input_dims = read_and_preprocess(targetTransform, root_folder=args.data_dir)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    print("DONE READING DATA %4.1f sec" % (time() - st))

    PARAMS = {
        "BATCH_SIZE": 512,
        "EPOCHS": 10,
        "LEARNING_RATE": 1e-1,
        "WEIGHT_DECAY": 1e-3,
        "VERBOSE": 1,
        "MODEL_DIR": "./output/models",
    }

    NFOLD = 3
    # SEEDS = [42,101, 111, 1111, 8888]
    SEEDS = [42]

    # containers
    oof = np.zeros(train.shape[0],)
    pred = np.zeros(test.shape[0],)

    create_model_dir(PARAMS["MODEL_DIR"])

    for seed in SEEDS:

        print("-"*15)
        print("-- SEED:", seed)
        print("-"*10)

        seed_everything(seed)  # set a seed to every libraries

        # some containers
        scores = []
        oof_ = np.zeros(train.shape[0],)
        pred_ = np.zeros(test.shape[0],)

        for fold in range(NFOLD):

            # GET FOLD DATA
            train_idx, valid_idx = train[train["fold"] !=
                                         fold].index, train[train["fold"] == fold].index
            train_, valid_ = train.loc[train_idx, ], train.loc[valid_idx, ]
            train_, valid_ = train_.reset_index(drop=True), valid_.reset_index(drop=True)

            print(train_.shape, valid_.shape)

            # TRAIN AND TEST
            y_valid_pred, y_test_pred = train_fold(PARAMS, fold, train_, valid_,
                                                   test, seed, targetTransform, cat_input_dims, DEVICE)

            # EVALUATE
            oof_[valid_idx] += targetTransform.inverse_transform_target(y_valid_pred)
            pred_ += targetTransform.inverse_transform_target(y_test_pred)/NFOLD

            score = np.sqrt(mean_squared_error(valid_.aqi.values,
                                               targetTransform.inverse_transform_target(y_valid_pred)))
            scores.append(score)

            print("RMSE: %2.4f" % score)

        score = np.sqrt(mean_squared_error(train.loc[oof_idx, "aqi"].values, oof_[oof_idx]))

        oof += oof_ / len(SEEDS)
        pred += pred_ / len(SEEDS)

        print("OOF %2.4f | CV %2.4f | STD %2.4f" % (score, np.mean(scores), np.std(scores)))

    # calculate competition metric
    oof_idx = np.array(train[train["fold"] != -1].index.tolist())
    train.loc[oof_idx, "aqi_pred"] = oof[oof_idx]
    score = np.sqrt(mean_squared_error(train.loc[oof_idx, "aqi"].values, oof[oof_idx]))

    print("FINAL SCORE")
    print("- "*10)
    print("OOF %2.4f" % (score))
    print("-"*25)

    print("Preparing Submission...")
    sub = test[["ID"]].copy()
    sub["aqi"] = pred
    sub.loc[test[~test.aqi.isna()].index, "aqi"] = test.loc[test[~test.aqi.isna()].index, "aqi"].values
    sub.to_csv("submission.csv", index=False)

    print(sub.sample(5))