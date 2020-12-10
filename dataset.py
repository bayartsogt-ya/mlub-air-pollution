import torch


class TrainDataset:
    def __init__(self, train, cont_feat):
        self.train = train
        self.cont_feat = cont_feat

    def __len__(self):
        return (self.train.shape[0])

    def __getitem__(self, idx):
        dct = {}
        dct['cont_feat'] = torch.tensor(self.train.loc[idx, self.cont_feat], dtype=torch.float32)
        dct['y'] = torch.tensor(self.train.loc[idx, ["target"]], dtype=torch.float32)

        return dct


class TestDataset:
    def __init__(self, test, cont_feat):
        self.test = test
        self.cont_feat = cont_feat

    def __len__(self):
        return (self.test.shape[0])

    def __getitem__(self, idx):
        dct = {}
        dct['cont_feat'] = torch.tensor(self.test.loc[idx, self.cont_feat], dtype=torch.float32)

        return dct
