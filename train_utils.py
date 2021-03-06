import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from utils import CAT_FEATURES, CONT_FEATURES
from model import MLP
from dataset import TrainDataset, TestDataset


def train_epoch(model, train_loader, optimizer, loss_fn, device):
    train_loss = 0
    model.train()
    for batch in tqdm(train_loader):
        cont_feat = batch["cont_feat"].to(device)
        y_true = batch["y"].to(device)

        optimizer.zero_grad()

        # FORWARD
        y_pred = model(cont_feat)
        loss = loss_fn(y_true, y_pred)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    return train_loss


def validate_on(model, valid_loader, loss_fn, device):
    # VALIDATION
    valid_loss = 0
    model.eval()
    for batch in tqdm(valid_loader):
        # FORWARD
        cont_feat = batch["cont_feat"].to(device)
        y_true = batch["y"].to(device)

        with torch.no_grad():
            y_pred = model(cont_feat)
        loss = loss_fn(y_true, y_pred)

        valid_loss += loss.item()

    valid_loss /= len(valid_loader)

    return valid_loss


def predict_on(model, test_loader, device):
    # VALIDATION
    y_pred_list = []
    model.eval()
    for batch in tqdm(test_loader):
        cont_feat = batch["cont_feat"].to(device)
        
        # FORWARD
        with torch.no_grad():
            y_pred = model(cont_feat)
        y_pred_list.append(y_pred.cpu().detach().numpy())

    return np.concatenate(y_pred_list, 0)


def train_fold(PARAMS, fold, train_, valid_, test,
               seed, targetTransform, device, cat_feat=CAT_FEATURES, cont_feat=CONT_FEATURES):

    train_loader = torch.utils.data.DataLoader(
        TrainDataset(train_, cont_feat),
        batch_size=PARAMS["BATCH_SIZE"],
        shuffle=True)
    valid_loader = torch.utils.data.DataLoader(
        TrainDataset(valid_, cont_feat),
        batch_size=PARAMS["BATCH_SIZE"],
        shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        TestDataset(test, cont_feat),
        batch_size=PARAMS["BATCH_SIZE"],
        shuffle=False)

    loss_fn = nn.MSELoss()
    model_path = os.path.join(PARAMS["MODEL_DIR"], "model_%d.pth" % (fold))
    model = MLP(cont_feat=cont_feat)
    model.to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=PARAMS["LEARNING_RATE"], weight_decay=PARAMS["WEIGHT_DECAY"])

    best_loss = np.inf
    print("epoch | train loss | valid loss")
    print("-----   ----------   ----------")
    for epoch in range(1, PARAMS["EPOCHS"] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        # valid_loss = validate_on(model, valid_loader, loss_fn, device)
        y_valid_pred = predict_on(model, valid_loader, device)
        valid_loss = np.sqrt(mean_squared_error(
            valid_.aqi.values, targetTransform.inverse_transform_target(y_valid_pred)))

        # LOG THE TRAIN PROGRESS
        if PARAMS["VERBOSE"] != None and epoch % PARAMS["VERBOSE"] == 0:
            print("%5d | %10.1f | %10.1f" % (epoch, train_loss, valid_loss))

        # SAVE BEST MODEL
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), model_path)

    print("Testing...")
    # model = MLP(cont_feat=cont_feat)
    # model.to(device=device)
    model.load_state_dict(torch.load(model_path))

    y_valid_pred = predict_on(model, valid_loader, device)
    y_test_pred = predict_on(model, test_loader, device)
    print("RMSE: %5.1f\n------------------------" %
          (np.sqrt(mean_squared_error(valid_.aqi.values, targetTransform.inverse_transform_target(y_valid_pred)))))

    return y_valid_pred, y_test_pred
