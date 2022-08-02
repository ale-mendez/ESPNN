import torch.nn.functional as F
from torch import nn


class Model(nn.Module):

    def __init__(self,
                 num_features,
                 num_targets,
                 hidden_size1=10,
                 hidden_size2=24,
                 drop_rate1=0.2,
                 drop_rate2=0.5,
                 drop_rate3=0.5
                 ):
        super(Model, self).__init__()
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size1, bias=False))

        self.dropout2 = nn.Dropout(drop_rate2)
        self.dense2 = nn.utils.weight_norm(nn.Linear(hidden_size1, hidden_size2))

        self.dropout3 = nn.Dropout(drop_rate3)
        self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_size2, 32))

        self.dropout4 = nn.Dropout(drop_rate3)
        self.dense4 = nn.utils.weight_norm(nn.Linear(32, hidden_size2))

        self.dropout5 = nn.Dropout(drop_rate3)
        self.dense5 = nn.utils.weight_norm(nn.Linear(hidden_size2, hidden_size1))

        self.batch_norm6 = nn.BatchNorm1d(hidden_size1)

        self.dense6 = nn.Linear(hidden_size1, num_targets)

    def forward(self, x):
        x = F.leaky_relu(self.dense1(x))

        x = self.dropout2(x)
        x = F.leaky_relu(self.dense2(x))

        x = self.dropout3(x)
        x = F.leaky_relu(self.dense3(x))

        x = self.dropout4(x)
        x = F.leaky_relu(self.dense4(x))

        x = self.dropout5(x)
        x = F.leaky_relu(self.dense5(x))

        x = self.dense6(x)

        return x
