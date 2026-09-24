# SwinCVS: Automated Assessment of Critical View of Safety in Laparoscopic Cholecystectomy

MSc dissertation project (UCL, AI & Medical Imaging, 2025–2026), extending the SwinCVS
architecture to investigate whether the field's standard dataset and metric for CVS
assessment actually measure what they're assumed to.

**Supervisors:** Matthew Clarkson, Franciszek Nowak

## Overview

The Critical View of Safety (CVS) is the standard safety checkpoint in laparoscopic
cholecystectomy, requiring three criteria before the cystic duct and artery are divided:
- **C1** — hepatocystic triangle cleared of fat and fibrous tissue
- **C2** — cystic duct and artery clearly identified
- **C3** — lower third of the gallbladder separated from the liver bed

This repository reproduces [SwinCVS](https://doi.org/10.1007/s11548-025-03354-9)
(Nowak et al., 2025) on the Endoscapes-CVS201 dataset, then uses it as a testbed to
interrogate the dataset/metric pairing the field has standardised on, rather than take
per-frame mAP at face value as a proxy for surgical safety assessment.

## Key findings

- Reproduced SwinCVS and **exceeded the published baseline**
- **Curation beats scale for C2** — a smaller, carefully curated label set outperformed
  simply adding more (noisier) labels, challenging the assumption that more data
  straightforwardly helps
- **Optimal annotator-disagreement handling is architecture-dependent** — the best
  strategy for resolving inter-annotator disagreement reverses between ViT and SwinCVS
  backbones; there's no single "correct" way to resolve label noise
- **The per-frame metric may be measuring the wrong thing** — 43.5% of frames labelled
  C1-negative in the standard dataset have both hepatocystic-triangle structures visible
  on inspection, suggesting frame-level ground truth doesn't cleanly capture the
  criterion it's meant to represent

## Repository structure

config/ # model & experiment configs
experiments/
frozen_backbone_mask_ablation/ # backbone-frozen ablation experiments
scoring/ # evaluation / metric scoring
scripts/ # training & data prep
SwinCVS.py # main training/inference entrypoint
consolidate.py # results consolidation
create_optimal_labels.py # label-strategy experiments (curation vs. scale)


## Setup

```bash
conda create --name swincvs python=3.9.19
conda activate swincvs
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```
Requires CUDA 12.1. The dataset (Endoscapes-CVS201) downloads automatically, or point
the script at an existing local copy.

## Usage

```bash
python3 SwinCVS.py --config_path config/SwinCVS_config.yaml
```

| Config flag | Purpose |
|---|---|
| `MODEL.LSTM` | `False` = SwinV2 backbone only · `True` = SwinCVS (SwinV2 + LSTM) |
| `MODEL.E2E` | `False` = frozen backbone · `True` = end-to-end training |
| `MODEL.MULTICLASSIFIER` | adds a classifier head before the LSTM |
| `MODEL.INFERENCE` | `True` skips training, test-only on provided weights |
| `BACKBONE.PRETRAINED` | `imagenet` or `endoscapes` |

## Acknowledgements

Built on the original SwinCVS architecture and codebase:
> Nowak, F., Mazomenos, E., Davidson, B., Clarkson, M. *SwinCVS: A Unified Approach to
> Classifying Critical View of Safety Structures in Laparoscopic Cholecystectomy.*
> Int J CARS (2025). https://doi.org/10.1007/s11548-025-03354-9

---
Abu Sufian Basith · MSc AI & Medical Imaging, UCL
