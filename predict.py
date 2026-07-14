"""
Standalone inference entry point for PerX2CT.

This script builds and loads the pretrained PerX2CT model outside of the
training/evaluation machinery in main.py (no Trainer, no DataModule, no
LIDC dataset classes). It reuses the repository's existing config-driven
instantiation and checkpoint-loading utilities rather than duplicating
them.

Phase 6.2A scope: build a ready-to-use, eval-mode model from a config +
checkpoint via a clean CLI. Preprocessing, batch construction, inference,
CT reconstruction, and output saving are intentionally left as TODOs for
the next implementation phase.
"""
import argparse

import torch
from omegaconf import OmegaConf

from main import instantiate_from_config


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="PerX2CT standalone inference entry point.",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="+",
        required=True,
        metavar="config.yaml",
        help="Path(s) to model config file(s), e.g. configs/PerX2CT.yaml. "
        "Multiple paths are merged left-to-right, same as main.py.",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the pretrained .ckpt file to load.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help='Device to run inference on ("cpu" or "cuda"). '
        "Defaults to cuda if available, otherwise cpu.",
    )
    return parser.parse_args()


def load_configuration(config_paths):
    """Load and merge one or more OmegaConf YAML configs, same pattern as main.py."""
    configs = [OmegaConf.to_container(OmegaConf.load(cfg), resolve=True) for cfg in config_paths]
    config = OmegaConf.merge(*configs)
    # main.py pops "lightning" off the top-level config before touching
    # config.model; not needed here since we never build a Trainer, but the
    # pop is harmless/idempotent if a lightning: block happens to be present.
    config.pop("lightning", None)
    return config


def build_model(config):
    """Instantiate the model from config.model, without loading a checkpoint yet."""
    model = instantiate_from_config(config.model)
    return model


def load_checkpoint(model, checkpoint_path):
    """Load pretrained weights via the model's own init_from_ckpt utility."""
    model.init_from_ckpt(checkpoint_path)
    return model


def main():
    opt = parse_arguments()

    device = opt.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    config = load_configuration(opt.base)
    model = build_model(config)
    model = load_checkpoint(model, opt.checkpoint)

    model = model.to(device)
    model.eval()

    print(f"Model built: {type(model).__name__}")
    print(f"Checkpoint loaded: {opt.checkpoint}")
    print(f"Device: {device}")
    print("Model is in eval mode and ready.")

    # TODO (next implementation phase): preprocessing
    #   Build the PA/Lateral input tensors from user-supplied X-ray images,
    #   replicating the exact dataset preprocessing chain (orientation
    #   flip/transpose, channel replication, /255 normalization, ToTensor).

    # TODO (next implementation phase): batch creation
    #   Assemble the standalone batch dictionary (PA, Lateral, PA_cam,
    #   Lateral_cam, ctslice placeholder, file_path_, image_key) as specified
    #   in the Phase 6.1 batch schema, and move all tensors to `device`.

    # TODO (next implementation phase): inference / CT reconstruction
    #   Call model.log_images(batch) (or an equivalent forward path) once per
    #   requested axis/slice to produce reconstructed CT slices.

    # TODO (next implementation phase): output saving
    #   Persist the reconstructed slice(s) to disk (e.g. PNG/NumPy/NIfTI)
    #   at a user-specified output location.


if __name__ == "__main__":
    main()
