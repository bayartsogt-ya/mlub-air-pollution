import torch
import torch.nn.functional as F
from torch.nn import ModuleDict, Embedding, Linear, Dropout, BatchNorm1d


class MLP(torch.nn.Module):
    """
    docstring
    """

    def __init__(self, cont_feat, hidden_size=64, dropout_rate=0.1):
        super(MLP, self).__init__()

        # LINEAR LAYERS
        self.linear_cont = Linear(len(cont_feat), hidden_size)
        self.linear_out1 = Linear(hidden_size, hidden_size)
        self.linear_out2 = Linear(hidden_size, 1)

        # DROPOUTS
        self.dropout = Dropout(dropout_rate)

    def forward(self, cont_features):
        # CONTINOUS INPUT
        x = F.relu(self.dropout(self.linear_cont(cont_features)))
        x = F.relu(self.dropout(self.linear_out1(x)))
        x = self.linear_out2(x)

        return x

if __name__ == "__main__":
    model = MLP(
        cont_feat=["f1", "f2", "f3"]
    )
    
    cont_feat = torch.rand(size=(32, 3), dtype=torch.float)

    print(model.parameters)
    a = model(cont_feat)
    print(a.size())
