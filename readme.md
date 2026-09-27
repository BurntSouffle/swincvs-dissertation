# Evaluation of Labels and Metrics in Automated Surgical Safety Assessment

**UCL MSc dissertation | Artificial Intelligence and Medical Imaging | 2025–2026**  
**Author:** Abu Sufian Basith  
**Supervisors:** Prof. Matt Clarkson and Franciszek Nowak

This repository contains research code extending the published SwinCVS implementation to study annotation reliability, probability calibration and the difference between frame-level classification scores and operation-level decisions for the Critical View of Safety (CVS) in laparoscopic cholecystectomy.

**This is experimental research, not clinically validated or deployed surgical decision-support software.**

## Research question

The Endoscapes-CVS201 benchmark evaluates whether three surgical criteria are visible in individual frames:

- **C1:** Two and only two structures (cystic duct and cystic artery) enter the gallbladder.
- **C2:** The hepatocystic triangle is cleared of fatty and fibrous tissue.
- **C3:** The lower third of the gallbladder is separated from the liver bed, exposing the cystic plate.

Per-frame mean average precision (mAP) measures ranking quality, but does not directly assess whether a model would correctly declare that an entire operation had achieved CVS. My submitted thesis investigates both the labels and this evaluation gap.

## Findings from the submitted dissertation

Submitted 31 July 2026: *Evaluation of Labels and Metrics in the Automated Assessment of the Critical View of Safety in Laparoscopic Cholecystectomy.*

| Study | Result | Interpretation |
|---|---|---|
| SwinCVS reproduction | **65.81% test mAP across five seeds**, compared with the published **67.45%** two-stage configuration | Reproduction within 1.64 percentage points; **not** an improvement over the published result. |
| Operation-level analysis | **14/40** test operations achieved full CVS under aggregated annotated-frame labels. With validation-tuned thresholds and two consecutive annotated keyframes, the model declared full CVS in **29–33/40**, including **16–19/26** non-achieving operations. | Frame-level ranking performance does not establish reliable operation-level declarations. |
| Matched null experiments | Preserving the prediction firing pattern while mismatching it with operations reproduced the *count* of operations declared; under the two-frame rule, discrimination from the matched null improved (permutation **p = 0.023**). | Declaration counts alone are uninformative without correct operation matching. |
| Annotator audit | On C1, two annotators agreed more closely with one another than with a third annotator applying a different criterion relationship. | Majority vote hides a systematic difference in interpretation; it does not decide which annotator is clinically correct. |
| **Separate** soft-label experiment | Expert-vote-fraction targets changed test mAP from **62.19% to 64.48%** and expected calibration error from **16.10% to 8.93%** (three seeds). | Separate end-to-end experiment, **not** the five-seed two-stage SwinCVS reproduction above. |

The dissertation's main contribution is scrutiny of labels, metrics and decision rules, rather than a claim of a new state-of-the-art architecture.

## Repository contents

- `SwinCVS.py`: training/inference entry point derived from the upstream implementation.
- `config/`: experiment configurations.
- `scripts/` and `scoring/`: data, analysis and evaluation utilities.
- `experiments/`: historical ablation experiments.
- `create_optimal_labels.py`: exploratory label-strategy work; **not** a synonym for the thesis's soft vote-fraction target experiment.

Historical exploratory findings in this repository should not be conflated with the submitted dissertation results.

## Historical setup

Requires the separately obtained Endoscapes data, relevant pretrained weights, and a compatible CUDA/PyTorch environment. A historical setup recipe is:

```bash
conda create -n swincvs python=3.9.19
conda activate swincvs
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
python SwinCVS.py --config_path config/SwinCVS_config.yaml
```

The command is a **historical research entry point**, not a newly verified one-command reproduction of all thesis analyses. Dataset paths, pretrained weights and checkpoints need configuring; restricted data is not included here.

## Limits and attribution

This is a single-dataset research evaluation with a limited operation-level test set and without independent clinical validation of these model declarations. Ranking, calibration and operation-level discrimination are distinct properties. The thesis documents limitations including annotation uncertainty and some affected training/validation temporal-window ordering (test windows unaffected).

Adapted from the published SwinCVS architecture and code by F. Nowak, E. B. Mazomenos, B. Davidson and M. J. Clarkson: [SwinCVS (2025)](https://doi.org/10.1007/s11548-025-03354-9). Original contributions remain credited to their authors; this repository documents Abu Sufian Basith's experiments and analysis. See [License.txt](License.txt) (CC BY-NC-SA 4.0), as well as the separate [Endoscapes](https://github.com/CAMMA-public/Endoscapes) data terms.
