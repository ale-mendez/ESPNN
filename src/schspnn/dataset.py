import numpy as np
import torch
from torch.utils.data.dataset import Dataset

from ..utils.utils import get_ionisation_projectile, get_mass, match_symbol_to_Z


class SolidSchnet(Dataset):
    def __init__(
        self, df, schnet_feat, features_to_use, target_name="normalized_stopping_power"
    ):

        self.df = df[features_to_use].values
        self.idx_ls = df["idx"].values
        self.schnet_feat = schnet_feat
        self.targets = df[target_name].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        x = self.df[idx]
        y = self.targets[idx]
        infer_df = self.schnet_feat[self.idx_ls[idx]]
        feat_vec = infer_df["feat_vec"]

        # print(feat_vec.shape)
        # print(x.shape)

        atom_list = [match_symbol_to_Z(Z) for Z in infer_df["atom_num"]]
        # print(atom_list)
        x = self.add_feat_vector(x, feat_vec, atom_list)

        x = self.add_element_Z(x, infer_df["atom_num"])

        x = self.add_element_mass(x, atom_list)

        properties = {
            "x": torch.Tensor(x),
            "y": torch.Tensor([y]),
            "_atomic_numbers": torch.Tensor(infer_df["atom_num"]),
        }

        return properties

    def add_feat_vector(self, x, feat_vec, atom_list):

        x = np.concatenate(
            [np.tile(x.reshape(1, x.shape[0]), (len(atom_list), 1)), feat_vec], axis=1
        )

        return x

        # merge(infer_df[idx]['feat_vec'],how='left',on=['target_formula'])

    def add_element_Z(self, x, atom_list):

        x = np.concatenate(
            [x, np.array(atom_list).reshape((len(atom_list), 1))], axis=1
        )

        return x

    # def add_target_mass(self,):

    # X[33,:]

    def add_element_mass(self, x, atom_list):

        elem_mass = [get_mass(sym) for sym in atom_list]

        x = np.concatenate(
            [x, np.array(elem_mass).reshape((len(elem_mass), 1))], axis=1
        )

        return x

    def add_element_ionization(self, x, atom_list):

        elem_ion = [get_ionisation_projectile(sym) for sym in atom_list]

        # x[:,33]

        x = np.append([x["out_vec"][0], np.array(elem_ion, shape=(-1, 1))])

    def add_energy_and_stopping(
        self,
    ):
        # this one is per molecule per instance
        return None
