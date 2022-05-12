import numpy as np
import torch
from torch.utils import DataLoader
import schnetpack as spk

from .model import Model
from .dataset import SolidSchnet
from .dataloader import _collate_aseatoms
from ..utils.utils import seed_everything


def inference_fn(model, dataloader, device):
    model.eval()
    preds = []

    for data in dataloader:
        inputs = {nam: prop.to(device) for nam, prop in data.items()}

        with torch.no_grad():
            outputs = model(inputs)["y"]

        preds.append(outputs.detach().cpu().numpy())

    preds = np.concatenate(preds)

    return preds


def run_inference(
    valid,
    feature_cols,
    target_cols,
    schnet_feat,
    fold,
    seed,
    device,
    model_dir="/model",
    exp_name="default_exp",
    verbose=False,
    **kwargs,
):

    BATCH_SIZE = 32

    seed_everything(seed)

    test_dataset = SolidSchnet(valid, schnet_feat, feature_cols)

    validloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=_collate_aseatoms,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    model_sarasa = Model(num_features=39, num_targets=1)

    model = spk.atomistic.Atomwise(
        n_in=None, aggregation_mode="avg", outnet=model_sarasa
    )

    model.to(device)

    model.load_state_dict(torch.load(f"../results/FOLD{fold}_{exp_name}.pth"))

    oof = inference_fn(model, validloader, device)

    return oof
