# Drone-YOLO reproduction

Independent engineering reproduction of **Drone-YOLO**, the model described in
“改进YOLOv11s的无人机图像小目标检测模型”. This directory does not modify the
existing `dcaf_reproduction` project, the global Ultralytics installation, or the
VisDrone dataset.

## Reproduction boundary

The paper reports the main components and training hyperparameters, but does not
publish source code, a model YAML, an Ultralytics version, augmentation settings,
or a random seed. The implementation therefore separates:

- reported facts, encoded in `configs/train/paper.yaml`;
- architecture hypotheses, documented beside each model YAML;
- locally fixed defaults needed to make runs auditable.

The paper text says that MF-FPN removes B5, adds a 160x160 P2 head, removes the
20x20 P5 head, and predicts on P2/P3/P4. Figure 1 still draws a 20x20 deep feature,
so parameter count and FLOPs are used as additional constraints when selecting the
reconstruction.

## Target ablation checkpoints

| Stage | Paper mAP50 | Paper mAP50-95 | Params | GFLOPs |
| --- | ---: | ---: | ---: | ---: |
| YOLO11s baseline | 37.9 | 22.7 | 9.416M | 21.3 |
| MF-FPN + P2 | 42.3 | 25.6 | 3.181M | 24.5 |
| MF-FPN + P2 + LSCD | 43.4 | 26.5 | 3.060M | 19.6 |
| Full model + Inner-WIoU | 44.3 | 26.9 | 3.060M | 19.6 |

## Safety and isolation

- Dataset access is read-only by convention and points to
  `E:/yolo_dcaf/data/VisDrone_YOLO`.
- Ultralytics may create derived `labels.cache` index files beside the labels;
  it does not alter images or annotation text files.
- All outputs are written below this project's `runs/` directory.
- Create `.venv` inside this directory before installing packages.
- No file in the existing DCAF project is imported or modified.

## Environment

Run from PowerShell:

```powershell
cd "C:\AllenX\OneDrive\文档\New project\drone_yolo_reproduction"
.\setup_env.ps1
```

This creates a project-local `.venv` inside this directory. To avoid replacing
the working CUDA stack with PyPI's CPU-only Windows wheel, it reads the existing
CUDA-enabled PyTorch/Ultralytics packages and writes any new packages only under
`.venv`. It never installs into or upgrades the global environment. The validated
engineering environment uses Python 3.12, PyTorch 2.8.0+cu129 and Ultralytics 8.4.110.
The paper instead reports Python 3.10.15, PyTorch 2.2.2 and CUDA 12.1, but omits
the Ultralytics version. The validated environment is therefore the default;
a historical-version comparison should be treated as a separate experiment.

## Structural audit

```powershell
.\.venv\Scripts\python.exe audit.py --stage all
```

The audit checks dataset counts, output strides, parameter counts and GFLOPs.
Current reconstruction values are expected to be close rather than artificially
identical because the paper does not publish its model YAML.

## Smoke test

Use one percent of the training set and a reduced image size before a long run:

```powershell
.\.venv\Scripts\python.exe train.py --stage full --weights none --epochs 1 --fraction 0.01 --imgsz 320 --batch 2 --workers 0 --name smoke_full
```

## Paper-setting runs

Run each stage separately so the ablation evidence remains auditable:

```powershell
.\.venv\Scripts\python.exe train.py --stage baseline
.\.venv\Scripts\python.exe train.py --stage mffpn
.\.venv\Scripts\python.exe train.py --stage lscd
.\.venv\Scripts\python.exe train.py --stage full
```

By default, compatible parameters are transferred from `yolo11s.pt`. Pass
`--weights none` to train from scratch. The paper does not say which choice it
used, so the choice must be recorded with every reported result.

Evaluate a checkpoint through the stage-aware loader:

```powershell
.\.venv\Scripts\python.exe evaluate.py --stage full --weights .\runs\full\weights\best.pt
```

Resume an interrupted run with its matching stage:

```powershell
.\.venv\Scripts\python.exe train.py --stage full --resume .\runs\full\weights\last.pt
```

## Status

The baseline, MF-FPN/P2, LSCD and Inner-WIoU stages are implemented. The MF-FPN
YAML contains one parameter-constrained P4 refinement block because Figure 1 and
the textual description conflict. LSCD follows Figure 3 and the public reference
topology: dedicated 1x1 Conv-GN alignment, two shared full 3x3 Conv-GN blocks,
shared 1x1 box/class predictors and per-level regression scales. The hidden width
of 32 is an explicit hypothesis because the paper does not report it; it preserves
the stated 32 GN groups and is the closest valid width to the reported GFLOPs.
These hypotheses are intentionally visible in code.
