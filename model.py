import torch
import torch.nn.functional as F
from torch.nn import ModuleDict, Embedding, Linear, Dropout, BatchNorm1d


class MLP(torch.nn.Module):
    """
    docstring
    """

    def __init__(self, cat_feat_dims, cont_feat, hidden_size=64, dropout_rate=0.1, device="cpu"):
        super(MLP, self).__init__()

        self.device = device

        # EMBEDDING OF CATEGORICAL FEATURES
        embedding_dict = {}
        for cat_col, cat_dim in cat_feat_dims.items():
            embedding_dict[cat_col] = Embedding(cat_dim, hidden_size)
        self.embedding_dict = ModuleDict(embedding_dict)

        # LINEAR LAYERS
        self.linear_cont = Linear(len(cont_feat), hidden_size)
        self.linear_out1 = Linear(hidden_size*(len(cat_feat_dims) + 1), hidden_size)
        self.linear_out2 = Linear(hidden_size, 1)

        # DROPOUTS
        self.dropout_emb = Dropout(dropout_rate)
        self.dropout_cont = Dropout(dropout_rate)
        self.dropout_out1 = Dropout(dropout_rate)

    def forward(self, inp_dct):
        xs = []
        # categorical features
        for cat_col, cat_emb in self.embedding_dict.items():
            x = cat_emb(inp_dct[cat_col].to(self.device))
            x = F.relu(self.dropout_emb(x))
            xs.append(torch.squeeze(x, 1))

        # continous features
        x = self.linear_cont(inp_dct['cont_feat'].to(self.device))
        x = self.dropout_cont(x)
        xs.append(F.relu(x))

        # CONCAT ALL EMBEDDINGS + CONTINOUS DATA
        x = torch.cat(xs, dim=1)
        x = self.dropout_out1(x)
        x = self.linear_out1(x)
        x = self.dropout_out1(x)
        x = self.linear_out2(x)

        return x


if __name__ == "__main__":
    model = MLP(
        cat_feat_dims={"f1": 5, "f2": 2},
        cont_feat=["f1", "f2", "f3"])

    dict_ = {
        "f1": torch.randint(5, size=(32,)),
        "f2": torch.randint(2, size=(32,)),
        "cont_feat": torch.rand(size=(32, 3), dtype=torch.float),
    }

    print(model.parameters)

    a = model(dict_)

    print(a.size())
