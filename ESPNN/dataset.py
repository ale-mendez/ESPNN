import torch
from torch.utils.data.dataset import Dataset


class IaeaDataset(Dataset):
    def __init__(self, features, targets):

        self.features = features
        self.targets = targets

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        dct = {
            "x": torch.tensor(self.features[idx, :], dtype=torch.float),
            "y": torch.tensor(self.targets[idx, :], dtype=torch.float),
        }
        return dct


class TestDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        dct = {"x": torch.tensor(self.features[idx, :], dtype=torch.float)}
        return dct


class InferenceDataset(Dataset):
    """Receives a dataframe containing the following features in this order
    - target atomic mass(amu)
    - projectile atomic mass (amu)
    - projectile Z
    - target Z
    - normalized stopping power (in MeV)
    """

    def __init__(self, features):

        self.features = features

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        dct = {
            "x": torch.tensor(self.features[idx, :], dtype=torch.float),
        }
        return dct
