# PerX2CT

PerX2CT reconstructs a selected 2D CT slice from paired PA and lateral X-ray images. It implements the perspective-projection method described in [Perspective Projection-Based 3D CT Reconstruction from Biplanar X-rays](https://arxiv.org/abs/2303.05297) (ICASSP 2023 Best Student Paper Award).

> This repository is research code. The standalone predictor creates one reconstructed slice for visualization; it does not generate a full CT volume or a clinical DICOM/NIfTI result.

![PerX2CT architecture](model.png)

## What is in this repository?

| Path | Purpose |
| --- | --- |
| `predict.py` | Standalone single-slice inference from `test_data/pa.png` and `test_data/lateral.png`. |
| `main.py` | Training entry point and shared configuration/model-instantiation utilities. |
| `main_test.py` | Dataset-based full-frame evaluation. |
| `main_test_zoom.py` | Dataset-based zoom-in evaluation. |
| `configs/` | YAML model and experiment configurations. |
| `checkpoints/` | Place downloaded pretrained weights here. |
| `test_data/` | Input location for standalone PA and lateral X-rays. |
| `outputs/` | Created by `predict.py` for reconstructed PNGs. |
| `x2ct_nerf/` | Model, projection, data, and loss implementation. |
| `taming/` | Decoder and supporting model components. |
| `data_preprocessing/` | Dataset preparation documentation and utilities. |
| `dataset_list/` | Dataset split/list files used by training and evaluation. |

## Requirements

- Python 3.8 (the repository's pinned environment target)
- NVIDIA GPU with CUDA support
- A compatible CUDA-enabled PyTorch installation

GPU execution is required for the current standalone predictor. Parts of the repository still contain hardcoded `.cuda()` calls, so CPU inference is intentionally rejected with a clear error.

## Setup

Clone the repository and create the expected Python environment:

```bash
git clone https://github.com/dek924/PerX2CT.git
cd PerX2CT

conda create -n perx2ct python=3.8
conda activate perx2ct
pip install --upgrade pip
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 \
  -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirement.txt
```

The pinned dependencies are from the original project. If a newer CUDA, PyTorch, or Python version is needed, validate training and inference before relying on it.

## Pretrained checkpoint

Download the pretrained weights from [Hugging Face](https://huggingface.co/KAISTEdlab/PerX2CT), then place the standard checkpoint here:

```text
checkpoints/PerX2CT.ckpt
```

`configs/PerX2CT.yaml` is the matching configuration for that checkpoint. Use `configs/PerX2CT_global_w_zoomin.yaml` only with the corresponding global/zoom-in checkpoint.

## Quick start: reconstruct one slice from two X-rays

1. Put a PA X-ray at `test_data/pa.png`.
2. Put a lateral X-ray at `test_data/lateral.png`.
3. Run:

```bash
python predict.py
```

The default request reconstructs axial slice 64 and saves:

```text
outputs/axial_064.png
```

The predictor accepts grayscale PNGs and RGB PNGs that visually contain grayscale X-rays. For RGB inputs, it uses the first channel.

### Standalone predictor options

```bash
python predict.py \
  --checkpoint checkpoints/PerX2CT.ckpt \
  --input_pa test_data/pa.png \
  --input_lateral test_data/lateral.png \
  --axis axial \
  --slice_index 64 \
  --device cuda
```

Supported axes are `axial`, `coronal`, and `sagittal`. Output names use the selected axis and zero-padded slice index, for example `outputs/coronal_031.png`.

The predictor reports the reconstruction tensor's shape, dtype, device, and value range. PNG normalization is only for visualization; it does not alter the tensor returned by the model.

## Training

Training requires a prepared LIDC-IDRI dataset and valid dataset-list paths. See [dataset preparation](data_preprocessing/prepare_datasets.md) before starting.

```bash
python main.py --train True --gpus <gpu_ids> --name <experiment_name> --base configs/PerX2CT.yaml
```

## Dataset-based evaluation

Use these scripts when evaluating against configured dataset splits rather than manually supplied X-rays:

```bash
python main_test.py \
  --ckpt_path checkpoints/PerX2CT.ckpt \
  --config_path configs/PerX2CT.yaml \
  --save_dir <results_directory> \
  --val_test test
```

For zoom-in evaluation, use the global model/checkpoint pair:

```bash
python main_test_zoom.py \
  --ckpt_path <global_checkpoint> \
  --config_path configs/PerX2CT_global_w_zoomin.yaml \
  --save_dir <results_directory> \
  --zoom_size <patch_size>
```

## Common issues

| Symptom | Check |
| --- | --- |
| `Checkpoint not found` | Confirm `checkpoints/PerX2CT.ckpt` exists or pass `--checkpoint`. |
| PA/lateral image not found | Confirm the filenames in `test_data/` or pass `--input_pa` and `--input_lateral`. |
| CPU/CUDA error | Use a CUDA-enabled PyTorch environment and run with `--device cuda`; CPU inference is not supported yet. |
| Unexpected image dimensions | Supply a 2D grayscale image or a 3D RGB image. |
| Out-of-memory error | This model is GPU-intensive; use a GPU with more memory or investigate model/runtime settings before changing architecture code. |

## Data

The original experiments use the [LIDC-IDRI dataset](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3041807/), available through [The Cancer Imaging Archive](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=1966254). Follow [prepare_datasets.md](data_preprocessing/prepare_datasets.md) to prepare it for training or evaluation.

## Acknowledgements

This implementation uses code from the [official X2CT-GAN implementation](https://github.com/kylekma/X2CT) and [Taming Transformers](https://github.com/CompVis/taming-transformers).

## Citation

```bibtex
@INPROCEEDINGS{kyung2023perx2ct,
  author={Kyung, Daeun and Jo, Kyungmin and Choo, Jaegul and Lee, Joonseok and Choi, Edward},
  booktitle={ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech, and Signal Processing},
  title={Perspective Projection-Based 3D CT Reconstruction from Biplanar X-Rays},
  year={2023},
  pages={1-5},
  doi={10.1109/ICASSP49357.2023.10096296}
}
```

## Contact

For questions about the original work, contact [kyungdaeun@kaist.ac.kr](mailto:kyungdaeun@kaist.ac.kr) or [bttkm@kaist.ac.kr](mailto:bttkm@kaist.ac.kr).
