# MCVMDL

Code and processed-data release accompanying the manuscript:

> **Millisecond-scale neural prediction of photon transport across anatomical domains for personalized optical medicine**

**Manuscript status:** in preparation; not yet submitted.

## Overview

MCVMDL is a tissue- and source-conditioned three-dimensional surrogate for photon transport. The model combines one-hot tissue channels with an explicit directional source field, processes them with a 3D U-Net, and applies bottleneck FiLM conditioning before producing a sigmoid-normalized absorption field. The released example uses processed ScatterBrains volumes and the selected Subject01 peak-weighted-MSE checkpoint.

This release is intended for reproducible training, evaluation, preprocessing, single-sample inference, light-delivery planning, and optional MCVM comparison. The Web application remains configurable for other compatible 3D tissue grids; ScatterBrains is the default example included with this package.

## ScatterBrains data source

The original ScatterBrains dataset and toolbox are maintained in the official [wumelissa/scatterBrains repository](https://github.com/wumelissa/scatterBrains). If you need the original data, download it from that repository. This release does not mirror the original 16-subject database; the included Subject01 and Subject02 files are processed derivatives.

## Release contents

- `train.py`: training entry point for the released MCVMDL configuration.
- `eval.py`: accelerated evaluation with MSE, MAE, RE, PSNR, slice-averaged SSIM, and inference-time reporting.
- `preprocess.py`: block-average and log-normalization utility for raw absorption volumes.
- `models/mcvmdl.py`: the checkpoint-compatible 3D U-Net and bottleneck FiLM model.
- `utils/data_loader.py`: processed ScatterBrains dataset loading and device-side batch preparation.
- `utils/losses.py`: MSE, MAE, and peak-weighted loss implementations.
- `web_app/`: generic realtime inference, light-delivery planning, and same-grid MCVM verification.
- `data/`: processed ScatterBrains Subject01 and Subject02 data.
- `weight/best_model.pth`: selected Subject01 checkpoint.
- `THIRD_PARTY_LICENSES/`: notices for ScatterBrains-derived data and the bundled MCVM executable.

Generated logs, predictions, plans, and MCVM outputs are not part of the release. They are written to the ignored `outputs/` directory when a command is run.

## Repository layout

```text
MCVMDL/
├── data/
│   ├── ScatterBrains-Subject01-Full/
│   └── ScatterBrains-Subject02-Full/
├── models/
├── utils/
├── web_app/
├── weight/
│   └── best_model.pth
├── eval.py
├── preprocess.py
├── train.py
├── requirements.txt
└── README.md
```

## Installation

Python 3.10 or newer is recommended.

```bash
python -m pip install -r requirements.txt
```

Install a PyTorch build compatible with the local CUDA runtime when GPU execution is required. The bundled `web_app/bin/MCVM.exe` is a Windows executable used only by the optional MCVM verification page.

## Data format

Each subject contains 160 training samples and 40 test samples. A sample directory contains:

- `Mask_Compressed.npz`: the source mask stored under the `arr_0` key.
- `Absorption_AvgCompressed.npz`: the normalized reference absorption field stored under the `arr_0` key.

Both sample arrays are finite 128 × 128 × 128 volumes. Each split contains one shared `Tissue_Compressed.npz` tissue-label volume. The released ScatterBrains labels are `{0, 1, 2, 6, 7, 8}`.

The preprocessing normalization is

```text
(log10(max(A, 1e-10)) + 10) / 10
```

where `A` is the absorption field after optional smoothing, zero-padding, and block averaging.

## Quick start

### Evaluate the released checkpoint

Run from the repository root:

```bash
python eval.py
```

The default input is the Subject01 `Test` split and the default checkpoint is `weight/best_model.pth`. Evaluation logs are written to `outputs/evaluation/`.

To evaluate Subject02 with the same Subject01 checkpoint:

```powershell
$env:MCS_TEST_DATA_DIR = "data/ScatterBrains-Subject02-Full/Test"
$env:MCS_EVAL_OUTPUT_DIR = "outputs/subject02"
$env:MCS_EVAL_LOG_DIR = "outputs/subject02"
python eval.py
```

### Launch the Web application

```bash
python web_app/server.py --host 127.0.0.1 --port 7860 --device auto
```

Open `http://127.0.0.1:7860/` in a browser. The three pages are:

1. **Realtime Simulation** — inspect the tissue, set a source position and direction, and run MCVMDL inference.
2. **Light-delivery Planning** — define a target region, screen source candidates, and select a plan.
3. **MCVM Verification** — run the bundled MCVM executable on the selected grid and compare the MCVM field with the model prediction.

The default example uses the Subject01 checkpoint, the Subject01 test tissue, 17 tissue channels, a 128³ grid, and 2 mm voxels. The prepared planning example uses ROI center `[75, 65, 20]`, radii `[3, 3, 3]`, 8 surface positions, 2 directions per position, and target/off-target/hotspot weights of `0.7/0.2/0.1`. These values remain editable. Subject02 can be selected without changing the checkpoint.

The Web system is not limited to ScatterBrains paths, labels, dimensions, or voxel sizes. Tissue files, model files, channel counts, voxel grids, beam settings, and optical properties can be supplied through the interface. MCVM verification uses the selected model grid directly in same-grid mode; it does not perform additional spatial compression or resampling.

## Training

The following command reproduces the released Subject01 configuration:

```bash
python train.py \
  --run-name scatterbrains_s01 \
  --data-dir data/ScatterBrains-Subject01-Full \
  --output-dir outputs/train_s01 \
  --batch-size 1 \
  --epochs 250 \
  --lr 5e-5 \
  --patience 15 \
  --num-channels 17 \
  --seed 42 \
  --same-tissue \
  --use-scheduler \
  --sched-factor 0.5 \
  --sched-patience 5 \
  --sched-min-lr 1e-6 \
  --loss-name peak_weighted_mse \
  --peak-lambda 4 \
  --peak-gamma 2
```

As in the original experiment, the Subject01 `Test` split is used for validation and early stopping. Training outputs are written only to the directory supplied with `--output-dir`.

## Average-compression preprocessing

The preprocessor scans immediate sample directories under the supplied root. Each input sample must contain `Absorption.npz` with its volume stored under the `arr_0` key.

```bash
python preprocess.py --data-dir path/to/samples --workers 8 --smooth-sigma 0
```

The processing sequence is optional Gaussian smoothing, zero-padding to 512³, a 4 × 4 × 4 block mean to 128³, truncation below `5e-9`, and the log normalization shown above. The output is written beside each input as `Absorption_AvgCompressed.npz`.

## Reference results

For the included Subject01 checkpoint and the released evaluation path, the full test-set reference is approximately:

| Metric | Subject01 Test |
| --- | ---: |
| MSE | `6.7144415e-05` |
| MAE | reported by `eval.py` |
| RE | `0.8194%` |
| SSIM | `0.9820` |

The evaluation defines full-scale relative error as `RE (%) = 100 × sqrt(MSE)`. Reported inference time depends on the PyTorch, CUDA, and hardware configuration.

## Reproducibility notes

- The checkpoint contains 22,733,873 trainable parameters.
- Evaluation metrics are computed over the complete normalized 128³ volume.
- CPU and GPU results can differ slightly because of floating-point and kernel implementation details.
- The release starts from processed data; it does not include the original 16-subject database or the MCVM data-generation pipeline.

## Citation

If you use this code or the processed release data, please cite the associated manuscript:

```bibtex
@unpublished{cao_mcvmdl,
  author = {Cao, Xinzhe and Yang, Songqi and Li, Ting},
  title = {Millisecond-scale neural prediction of photon transport across anatomical domains for personalized optical medicine},
  note = {Manuscript in preparation}
}
```

The paper link, final bibliographic fields, and DOI will be added after submission.

## License and data notice

The MCVMDL source code and documentation are released under the MIT License. ScatterBrains-derived data is governed by its original terms in `THIRD_PARTY_LICENSES/scatterBrains_LICENSE.txt`; see `DATA_LICENSE.md`. The bundled `web_app/bin/MCVM.exe` is third-party software and is not covered by the MIT License; see `THIRD_PARTY_LICENSES/MCVM_NOTICE.txt`.
