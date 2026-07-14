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
import os

import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from main import instantiate_from_config
from x2ct_nerf.preprocessing.X2CT_transform_3d import List_Compose, Normalization, ToTensor

# Fixed X-ray value range used by every shipped config's dataset opt
# (data.params.train.params.opt.XRAY_MIN_MAX in configs/*.yaml). Reused here
# instead of duplicating the Normalization/ToTensor math by hand.
_XRAY_MIN_MAX = (0, 255)
_XRAY_PREPROCESSING = List_Compose([(Normalization(*_XRAY_MIN_MAX),), (ToTensor(),)])

# data_preprocessing/3_make_h5_dataset.py resizes with PIL's Image.ANTIALIAS,
# which modern Pillow renamed to Image.Resampling.LANCZOS (plain Image.LANCZOS
# on older Pillow that lacks the Resampling namespace). Same filter, restored
# here for the standalone pipeline (Phase 6.2E/6.2F).
try:
    _XRAY_RESIZE_FILTER = Image.Resampling.LANCZOS
except AttributeError:
    _XRAY_RESIZE_FILTER = Image.LANCZOS

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
        default=["configs/PerX2CT.yaml"],
        metavar="config.yaml",
        help="Path(s) to model config file(s), e.g. configs/PerX2CT.yaml. "
        "Multiple paths are merged left-to-right, same as main.py.",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        default="checkpoints/PerX2CT.ckpt",
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
    parser.add_argument(
        "--input_pa",
        type=str,
        default="test_data/pa.png",
        help="Path to the PA X-ray image file.",
    )
    parser.add_argument(
        "--input_lateral",
        type=str,
        default="test_data/lateral.png",
        help="Path to the Lateral X-ray image file.",
    )
    parser.add_argument(
        "--axis",
        type=str,
        default="axial",
        help="Reconstruction axis: one of sagittal, coronal, axial.",
    )
    parser.add_argument(
        "--slice_index",
        type=int,
        default=64,
        help="Non-negative slice index along --axis.",
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

def load_xray_image(path, target_size=None):
    """Load an image from disk as a numpy array.

    Mirrors LIDCMultiInputMultiResTypes.get_image's png branch
    (imageio.imread + np.asarray), generalized to any format imageio/PIL
    supports rather than restricted to '.png'.

    Restores the offline resize step from data_preprocessing/3_make_h5_dataset.py
    (lines 102-112), which the standalone pipeline otherwise skips: that script
    grayscale-extracts (cv2.imread(...)[..., 0]) THEN resizes to a square
    (xray_size, xray_size) via Image.resize(..., Image.ANTIALIAS), before the
    dataset loader (x2ct_nerf/data/base.py) ever sees the PNG -- base.py itself
    never resizes. Reproduced here in the same order: grayscale, then resize,
    both applied identically to PA and Lateral before any orientation-specific
    (fliplr/transpose) processing, which is what previously let a non-square
    source image reach preprocess_pa_image/preprocess_lateral_image with
    mismatched PA/Lateral shapes and fail at torch.stack(src_imgs) inside
    INREncoderZoomAxisInAlign.forward.

    Args:
        target_size: if given, the square side length (int) to resize to,
            using the same Image.ANTIALIAS/LANCZOS filter the offline
            preprocessing script used. If None, no resize is performed
            (preserves the previous behavior for already-correctly-sized
            inputs).
    """
    image = imageio.imread(path)
    image = np.asarray(image)
    if image.ndim == 3:
        image = image[..., 0]
    elif image.ndim != 2:
        raise ValueError(f"expected a 2D grayscale or 3D RGB X-ray image, got shape {image.shape}")
    if target_size is not None:
        image = Image.fromarray(image).resize((target_size, target_size), _XRAY_RESIZE_FILTER)
        image = np.asarray(image)
    return image


def get_xray_target_size(config):
    """Read the square X-ray target resolution from the loaded config.

    Authoritative source (traced in Phase 6.2E): configs/*.yaml's top-level
    `input_img_size`, which is also what cond_encoder_params.cfg.input_img_size
    resolves to (the value ResNetEncoder/get_data_transform's own Resize uses)
    and equals data.params.train.params.opt.xray_size (the value
    data_preprocessing/3_make_h5_dataset.py bakes PA/Lateral PNGs to offline)
    in every shipped config. Read from the already-loaded config rather than
    hardcoded, so a different config's resolution is picked up automatically.
    """
    return int(config.input_img_size)


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
    if image.ndim == 3:
        image = image[..., 0]
    elif image.ndim != 2:
        raise ValueError(
            f"expected a 2D grayscale or 3D RGB X-ray image, got shape {image.shape}"
        )
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
    if image.ndim == 3:
        image = image[..., 0]
    elif image.ndim != 2:
        raise ValueError(
            f"expected a 2D grayscale or 3D RGB X-ray image, got shape {image.shape}"
        )
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
    if not isinstance(slice_index, int) or not 0 <= slice_index < 128:
        raise ValueError(f"slice_index must be between 0 and 127, got {slice_index}")
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

    if not os.path.isfile(opt.checkpoint):
        raise RuntimeError(f"Checkpoint not found: {opt.checkpoint}")
    if opt.axis not in _VALID_AXES:
        raise ValueError(f"--axis must be one of {sorted(_VALID_AXES)}, got {opt.axis!r}")
    if opt.slice_index < 0:
        raise ValueError(f"--slice_index must be non-negative, got {opt.slice_index}")

    device = opt.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if device.type != "cuda":
        raise RuntimeError(
            "PerX2CT currently requires a CUDA-enabled GPU because the repository still contains "
            "hardcoded .cuda() calls. CPU inference will be supported after the Phase 7 "
            "device-agnostic refactor."
        )

    print("Loading configuration...")
    config = load_configuration(opt.base)
    print("Building model...")
    model = build_model(config)
    print("Loading checkpoint...")
    model = load_checkpoint(model, opt.checkpoint)

    model = model.to(device)
    model.eval()

    print(f"Model built: {type(model).__name__}")
    print(f"Checkpoint loaded: {opt.checkpoint}")
    print(f"Device: {device}")
    print("Model is in eval mode and ready.")

    if not os.path.isfile(opt.input_pa):
        raise RuntimeError(f"PA X-ray image not found: {opt.input_pa}")
    if not os.path.isfile(opt.input_lateral):
        raise RuntimeError(f"Lateral X-ray image not found: {opt.input_lateral}")

    xray_target_size = get_xray_target_size(config)

    print("Loading PA image...")
    pa_image = load_xray_image(opt.input_pa, target_size=xray_target_size)

    print("Loading Lateral image...")
    lateral_image = load_xray_image(opt.input_lateral, target_size=xray_target_size)

    print(f"Loaded PA shape: {pa_image.shape}")
    print(f"Loaded Lateral shape: {lateral_image.shape}")

    print("Preprocessing...")
    pa_tensor = preprocess_pa_image(pa_image)
    lateral_tensor = preprocess_lateral_image(lateral_image)

    print(f"PA tensor shape: {tuple(pa_tensor.shape)}")
    print(f"Lateral tensor shape: {tuple(lateral_tensor.shape)}")

    print("Building inference batch...")
    batch = build_batch(pa_tensor, lateral_tensor, opt.axis, opt.slice_index, device)

    print("Running inference...")
    log = model.log_images(batch)

    if not isinstance(log, dict):
        raise RuntimeError(f"model.log_images(batch) returned {type(log).__name__}, expected dict.")
    if "reconstructions" not in log:
        raise RuntimeError(
            f"model.log_images(batch) result missing 'reconstructions' key. Keys present: {list(log.keys())}"
        )

    recon = log["reconstructions"]
    if not isinstance(recon, torch.Tensor):
        raise RuntimeError(
            f"model.log_images(batch) returned 'reconstructions' as {type(recon).__name__}, "
            "expected torch.Tensor."
        )
    if recon.ndim != 4:
        raise RuntimeError(
            f"model.log_images(batch) returned 'reconstructions' with {recon.ndim} dimensions, "
            "expected (B, C, H, W)."
        )
    if recon.shape[1] != 3:
        raise RuntimeError(
            f"model.log_images(batch) returned 'reconstructions' with {recon.shape[1]} channels, "
            "expected 3."
        )
    print(f"Available log keys: {list(log.keys())}")
    print(f"Reconstruction shape: {tuple(recon.shape)}")
    print(f"Reconstruction dtype: {recon.dtype}")
    print(f"Reconstruction device: {recon.device}")
    print(f"Reconstruction min: {recon.min().item()}")
    print(f"Reconstruction max: {recon.max().item()}")

    reconstruction_image = recon[0, 1].detach().cpu().numpy()
    reconstruction_min = reconstruction_image.min()
    reconstruction_max = reconstruction_image.max()
    if reconstruction_max > reconstruction_min:
        reconstruction_image = (reconstruction_image - reconstruction_min) / (
            reconstruction_max - reconstruction_min
        )
    else:
        reconstruction_image = np.zeros_like(reconstruction_image)
    reconstruction_image = (reconstruction_image * 255).astype(np.uint8)

    output_path = f"outputs/{opt.axis}_{opt.slice_index:03d}.png"
    print("Saving reconstruction...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imageio.imwrite(output_path, reconstruction_image)
    print(f"Reconstruction saved to {output_path}")
    print("Done.")


if __name__ == "__main__":
    main()
