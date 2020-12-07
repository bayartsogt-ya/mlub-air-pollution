import torch
from torch import tensor, float32

class TrainDataset:
    def __init__(self, train, cat_feat, cont_feat):
        self.train = train
        self.cat_feat = cat_feat
        self.cont_feat = cont_feat
        
    def __len__(self):
        return (self.train.shape[0])
    
    def __getitem__(self, idx):
        dct = {}
        for col in self.cat_feat:
            dct[col] = tensor(self.train.loc[idx, col], dtype=float32)
        dct['cont_feat'] = tensor(self.train.loc[idx, self.cont_feat], dtype=float32)
        dct['y'] = tensor(self.train.loc[idx, "target"], dtype=float32)

        return dct
    
class TestDataset:
    def __init__(self, test, cat_feat, cont_feat):
        self.test = test
        self.cat_feat = cat_feat
        self.cont_feat = cont_feat
        
    def __len__(self):
        return (self.test.shape[0])
    
    def __getitem__(self, idx):
        dct = {}
        for col in self.cat_feat:
            dct[col] = tensor(self.test.loc[idx, col], dtype=float32)
        dct['cont_feat'] = tensor(self.test.loc[idx, self.cont_feat], dtype=float32)

        return dct