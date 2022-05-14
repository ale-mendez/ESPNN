import torch.nn.functional as F
from torch import nn


class Model(nn.Module):
    def __init__(
        self,
        num_features,
        num_targets,
        hidden_size1=40,
        hidden_size2=80,
        drop_rate1=0.2,
        drop_rate2=0.5,
        drop_rate3=0.5,
    ):
        super(Model, self).__init__()

        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dropout1 = nn.Dropout(drop_rate1)
        self.dense1 = nn.utils.weight_norm(
            nn.Linear(
                num_features,
                hidden_size1
            )
        )

        self.batch_norm2 = nn.BatchNorm1d(hidden_size1)
        self.dropout2 = nn.Dropout(drop_rate2)
        self.dense2 = nn.utils.weight_norm(
            nn.Linear(
                hidden_size1,
                hidden_size2
            )
        )

        self.batch_norm3 = nn.BatchNorm1d(hidden_size2)
        self.dropout3 = nn.Dropout(drop_rate3)
        self.dense3 = nn.utils.weight_norm(
            nn.Linear(
                hidden_size2,
                128
            )
        )

        self.batch_norm4 = nn.BatchNorm1d(64)
        self.dropout4 = nn.Dropout(drop_rate3)
        self.dense4 = nn.utils.weight_norm(
            nn.Linear(
                128,
                hidden_size2
            )
        )

        self.batch_norm5 = nn.BatchNorm1d(hidden_size2)
        self.dropout5 = nn.Dropout(drop_rate3)
        self.dense5 = nn.utils.weight_norm(
            nn.Linear(
                hidden_size2,
                hidden_size1
            )
        )

        self.batch_norm6 = nn.BatchNorm1d(hidden_size1)
        self.dense6 = nn.Linear(hidden_size1, num_targets)

    def forward(self, x):

        x = F.leaky_relu(self.dense1(x))

        x = F.leaky_relu(self.dense2(x))

        x = F.leaky_relu(self.dense3(x))

        x = F.leaky_relu(self.dense4(x))

        x = F.leaky_relu(self.dense5(x))

        x = self.dense6(x)

        return x
