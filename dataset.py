import torch

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
            dct[col] = torch.tensor(self.train.loc[idx, col], dtype=torch.long)
        dct['cont_feat'] = torch.tensor(self.train.loc[idx, self.cont_feat], dtype=torch.float32)
        dct['y'] = torch.tensor(self.train.loc[idx, "target"], dtype=torch.float32)

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
            dct[col] = torch.tensor(self.test.loc[idx, col], dtype=torch.long)
        dct['cont_feat'] = torch.tensor(self.test.loc[idx, self.cont_feat], dtype=torch.float32)

        return dct