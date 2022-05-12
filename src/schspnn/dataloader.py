import numpy as np
import torch


def _collate_aseatoms(examples):
    """
    Build batch from systems and properties & apply padding
    Args:
        examples (list):
    Returns:
        dict[str->torch.Tensor]: mini-batch of atomistic systems
    """
    properties = examples[0]

    # initialize maximum sizes
    max_size = {
        prop: np.array(val.size(), dtype=np.int) for prop, val in properties.items()
    }

    for properties in examples[1:]:
        for prop, val in properties.items():
            max_size[prop] = np.maximum(
                max_size[prop], np.array(val.size(), dtype=np.int)
            )

    # initialize batch

    batch = {
        p: torch.zeros(len(examples), *[int(ss) for ss in size]).type(
            examples[0][p].type()
        )
        for p, size in max_size.items()
    }
    #     batch = {
    #         p: torch.zeros(len(examples), *[int(ss) for ss in size]).type(
    #             examples[0][p].type()
    #         )

    # for p, size in max_size.items()
    #     }
    has_atom_mask = "_atom_mask" in batch.keys()

    if not has_atom_mask:
        batch["_atom_mask"] = torch.zeros_like(batch["_atomic_numbers"]).float()

    # build batch and pad
    for k, properties in enumerate(examples):
        for prop, val in properties.items():
            shape = val.size()
            s = (k,) + tuple([slice(0, d) for d in shape])
            batch[prop][s] = val

        if not has_atom_mask:
            z = properties["_atomic_numbers"]
            shape = z.size()
            s = (k,) + tuple([slice(0, d) for d in shape])
            batch["_atom_mask"][s] = z > 0

    return batch
