import numpy as np
import torch

from .dataset import TestDataset
from .SPNN import Model
from .utils import seed_everything


def inference_fn(model, dataloader, device):
    """
    Given a model a data loader and a target device executes inference with
    the model on the dataloader
    """
    model.eval()
    preds = []

    for data in dataloader:
        inputs = data["x"].to(device)

        with torch.no_grad():
            outputs = model(inputs)

        preds.append(outputs.detach().cpu().numpy())

    preds = np.concatenate(preds)

    return preds


def run_inference(
    X_valid,
    fold,
    seed,
    device,
    verbose=False,
    model_dir="/model",
    exp_name="default_exp",
):
    """Executes inference on X_valid dataset"""

    seed_everything(seed)

    valid_dataset = TestDataset(X_valid)

    validloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=256, shuffle=False
    )

    model = Model(num_features=X_valid.shape[1], num_targets=1)

    model.to(device)

    model.load_state_dict(
        torch.load(
            f"{model_dir}/FOLD{fold}_{exp_name}.pth",
            map_location=torch.device(device),
        )
    )

    oof = inference_fn(model, validloader, device)

    return oof
