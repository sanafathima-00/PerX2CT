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
import math

import imageio
import numpy as np
import torch
from omegaconf import OmegaConf

from main import instantiate_from_config
from x2ct_nerf.preprocessing.X2CT_transform_3d import List_Compose, Normalization, ToTensor

# Fixed X-ray value range used by every shipped config's dataset opt
# (data.params.train.params.opt.XRAY_MIN_MAX in configs/*.yaml). Reused here
# instead of duplicating the Normalization/ToTensor math by hand.
_XRAY_MIN_MAX = (0, 255)
_XRAY_PREPROCESSING = List_Compose([(Normalization(*_XRAY_MIN_MAX),), (ToTensor(),)])

# Axis names recognized by INREncoderZoomAxisInAlign.get_rays_for_no_rendering's
# file_path_ parser (x2ct_nerf/modules/INREncoderZoomAxisInAlign.py). Any string
# other than "sagittal"/"coronal" falls into that function's axial branch, but
# we validate against the full set here to catch typos early.
_VALID_AXES = {"sagittal", "coronal", "axial"}


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
    """Load pretrained weights directly, without depending on model.init_from_ckpt.

    model.init_from_ckpt still raises the PyTorch 2.6+ weights_only
    UnpicklingError despite calling torch.load the same way; this performs
    the load inline instead so predict.py has no dependency on it.
    Same backward-compatible weights_only fallback used in the other
    checkpoint-loading sites across the repository (PyTorch <1.13 lacks the
    weights_only kwarg entirely).
    """
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=False)
    print(f"Restored from {checkpoint_path}")
    return model


# ---------------------------------------------------------------------------
# Standalone batch-construction utilities.
#
# These replicate, field-by-field, what LIDCMultiInputMultiResTypes
# (x2ct_nerf/data/base.py) used to produce per dataset item, so that a batch
# dict can be built directly from user-supplied images without the LIDC
# dataset classes, DataLoader, or dataset_list files.
# ---------------------------------------------------------------------------

def load_xray_image(path):
    """Load an image from disk as a numpy array.

    Mirrors LIDCMultiInputMultiResTypes.get_image's png branch
    (imageio.imread + np.asarray), generalized to any format imageio/PIL
    supports rather than restricted to '.png'.
    """
    image = imageio.imread(path)
    return np.asarray(image)


def preprocess_pa_image(image):
    """Replicate the PA preprocessing chain exactly.

    Traced from LIDCMultiInputMultiResTypes.apply_preprocessing_xray_according2cam
    (src_camtype == "PA": np.fliplr) and __getitem__'s xray branch (channel
    replication to 3, then the dataset's own Normalization(0,255)+ToTensor
    pipeline, reused here via _XRAY_PREPROCESSING).

    Args:
        image: 2D grayscale numpy array (H, W), as returned by load_xray_image.

    Returns:
        float32 torch tensor, shape (H, W, 3), values in [0, 1] -- channel-last,
        matching a single dataset item's output before batch collation.
    """
    assert image.ndim == 2, f"expected a 2D grayscale X-ray image, got shape {image.shape}"
    image = np.fliplr(image)
    image = np.expand_dims(image, -1)
    image = np.concatenate((image, image, image), axis=-1)
    return _XRAY_PREPROCESSING(image)


def preprocess_lateral_image(image):
    """Replicate the Lateral preprocessing chain exactly.

    Traced from LIDCMultiInputMultiResTypes.apply_preprocessing_xray_according2cam
    (src_camtype == "Lateral": np.transpose((1, 0)) then np.flipud, in that
    order) and __getitem__'s xray branch (channel replication, then the same
    Normalization(0,255)+ToTensor pipeline as PA).

    Args:
        image: 2D grayscale numpy array (H, W), as returned by load_xray_image.

    Returns:
        float32 torch tensor, shape (W, H, 3) [transpose swaps the axes],
        values in [0, 1] -- channel-last, matching a single dataset item's
        output before batch collation.
    """
    assert image.ndim == 2, f"expected a 2D grayscale X-ray image, got shape {image.shape}"
    image = np.transpose(image, (1, 0))
    image = np.flipud(image)
    image = np.expand_dims(image, -1)
    image = np.concatenate((image, image, image), axis=-1)
    return _XRAY_PREPROCESSING(image)


def create_camera_tensors(batch_size, device):
    """Build the fixed PA/Lateral camera-pose tensors.

    Values traced from LIDCMultiInputMultiResTypes.__init__'s
    mapping_camera_type2pose: PA -> (0, 0), Lateral -> (pi/2, pi/2). These are
    fixed constants, identical for every sample -- not derived from any image.

    Dtype note: the dataset literal for PA_cam is int64 (torch.tensor([0, 0]))
    and for Lateral_cam is float32. Downstream, PerspectiveINRNet.encode
    torch.stacks both camera tensors together, which silently promotes the
    int64 PA_cam to float32 (required, since sample_camera_positions applies
    torch.sin/torch.cos to it, which int64 tensors do not support). Both
    tensors are constructed here directly as float32 to match what actually
    reaches that math, rather than relying on implicit promotion.

    Returns:
        (PA_cam, Lateral_cam): float32 tensors, each shape (batch_size, 2).
    """
    pa_cam = torch.tensor([0.0, 0.0], dtype=torch.float32, device=device).repeat(batch_size, 1)
    lateral_cam = torch.tensor([math.pi / 2, math.pi / 2], dtype=torch.float32, device=device).repeat(batch_size, 1)
    return pa_cam, lateral_cam


def create_placeholder_ct(batch_size, height, width, device):
    """Create the placeholder ctslice tensor structurally required by the encoder.

    Reasoning, traced through x2ct_nerf/modules/INREncoderZoomAxisInAlign.py:
      - INRAEZoomModel.log_images does `x = batch[self.image_key]`
        (image_key == "ctslice"), so the key must be present or this raises
        a KeyError -- its presence is structurally required.
      - get_rays_for_no_rendering reads
        `gt_ctslice = inputs[inputs['image_key']][b:b+1]` and passes it to
        rendering_from_ctslice, which only uses it for a `cropped_ctslice`
        entry in the returned dict; that entry is never read again by
        run_nerf/decode. The actual reconstruction path (transformed_points +
        latent_zs from PA/Lateral) never touches this tensor's values.
      - rendering_from_ctslice does read `ct_slice.shape[-2:]` to rescale the
        p0/p1 crop window into that tensor's own resolution -- so the SHAPE
        must be a valid 2D grid for F.grid_sample (any height/width >= 2
        works; grid_sample interpolates regardless of resolution). The
        function signature takes height/width explicitly rather than
        hardcoding a number, since no crash depends on a specific value.
      - The one concretely-defined resolution constant inside the encoder
        itself is `ct_res` (configs/*.yaml: `encoder_params.params.ct_res:
        ${input_ct_res}` = 128), as opposed to the dataset-side `ct_size:
        320` which the encoder never reads. 128 is therefore the recommended
        height/width to pass at the call site, traceable to config rather
        than guessed -- but it is not enforced here since the value is
        provided by the caller.

    Returns:
        float32 zeros tensor, shape (batch_size, height, width, 3) --
        channel-last, matching PA/Lateral's convention consumed by
        INRAEZoomModel.get_input.
    """
    return torch.zeros((batch_size, height, width, 3), dtype=torch.float32, device=device)


def create_file_path(axis, slice_index):
    """Generate the file_path_ string the encoder's parser expects.

    Traced from get_rays_for_no_rendering:
        file_path = inputs['file_path_'][b].split("/")[-1]
        recon_axis, slice_idx = os.path.splitext(file_path)[0].split("_")
        slice_idx = int(slice_idx)
    This requires a filename stem with exactly one underscore, splitting into
    an axis name and an integer slice index -- e.g. "coronal_104.h5", the
    exact form seen in the dataset's own path comment
    (x2ct_nerf/data/base.py, __getitem__).
    """
    if axis not in _VALID_AXES:
        raise ValueError(f"axis must be one of {sorted(_VALID_AXES)}, got {axis!r}")
    if not isinstance(slice_index, int) or slice_index < 0:
        raise ValueError(f"slice_index must be a non-negative int, got {slice_index!r}")
    return f"{axis}_{slice_index:03d}.h5"


def build_batch(pa_tensor, lateral_tensor, axis, slice_index, device, ct_height=128, ct_width=128):
    """Assemble the complete batch dictionary expected by model.log_images().

    Note: "image_key" is intentionally NOT included here. Tracing
    INRAEZoomModel.log_images (x2ct_nerf/models/zoom_aegan.py) shows it does
    `batch = self.get_input(batch)` followed immediately by
    `batch['image_key'] = self.image_key` -- the model injects this key
    itself before it is ever read, so a caller-supplied batch does not need
    to (and any caller-supplied value would be overwritten regardless).

    Args:
        pa_tensor: output of preprocess_pa_image, shape (H, W, 3), no batch dim.
        lateral_tensor: output of preprocess_lateral_image, shape (H, W, 3), no batch dim.
        axis: one of "sagittal", "coronal", "axial".
        slice_index: non-negative int slice index along `axis`.
        device: torch.device (or device string) all tensors are placed on.
        ct_height, ct_width: spatial size of the placeholder ctslice tensor
            (see create_placeholder_ct's docstring for why 128 is recommended).

    Returns:
        dict with keys "PA", "Lateral", "PA_cam", "Lateral_cam", "ctslice",
        "file_path_" -- ready to pass to model.log_images(batch). This
        function does not call model.log_images() itself.
    """
    batch_size = 1
    pa = pa_tensor.unsqueeze(0).to(device)
    lateral = lateral_tensor.unsqueeze(0).to(device)
    pa_cam, lateral_cam = create_camera_tensors(batch_size, device)
    ctslice = create_placeholder_ct(batch_size, ct_height, ct_width, device)
    file_path_ = [create_file_path(axis, slice_index)]

    return {
        "PA": pa,
        "Lateral": lateral,
        "PA_cam": pa_cam,
        "Lateral_cam": lateral_cam,
        "ctslice": ctslice,
        "file_path_": file_path_,
    }


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

    # Preprocessing and batch-construction utilities now exist above
    # (load_xray_image, preprocess_pa_image, preprocess_lateral_image,
    # create_camera_tensors, create_placeholder_ct, create_file_path,
    # build_batch), but are not yet wired into this CLI.

    # TODO (next implementation phase): CLI wiring
    #   Add arguments for the PA/Lateral image paths and the requested
    #   axis/slice_index, then call load_xray_image + preprocess_*_image +
    #   build_batch to construct the batch dict on `device`.

    # TODO (next implementation phase): inference / CT reconstruction
    #   Call model.log_images(batch) (or an equivalent forward path) on the
    #   constructed batch to produce reconstructed CT slices.

    # TODO (next implementation phase): output saving
    #   Persist the reconstructed slice(s) to disk (e.g. PNG/NumPy/NIfTI)
    #   at a user-specified output location.


if __name__ == "__main__":
    main()
