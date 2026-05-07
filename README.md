# HipXGen

Code release for **HipXGen: A Multimodal Vision-Language Benchmark for Hip X-ray Diagnosis with Generative Conditional Synthesis**.

## Repository Structure

```
hipxgen/
├── classifier/          # Multimodal classifier (HipXGen-Exam & HipXGen-Symp)
├── diffusion/           # Class-conditional latent diffusion model
├── baselines/           # TF-IDF + Logistic Regression baselines
└── README.md
```

---

## Classifier

Two model variants, both built on a frozen BiomedCLIP vision encoder and frozen SmolLM2-1.7B text encoder:

| Model | Description | Script |
|-------|-------------|--------|
| **HipXGen-Exam** | Full clinical notes · Text-primary fusion · RecA image backbone | `classifier/train_hipxgen_exam.py` |
| **HipXGen-Exam†** | Full clinical notes · No RecA (ablation) | `classifier/train_hipxgen_exam_norca.py` |
| **HipXGen-Symp** | Symptom-only text · Sequential cross-attention | `classifier/train_hipxgen_symp.py` |

### Requirements

```bash
pip install torch torchvision open_clip_torch transformers scikit-learn tqdm
```

### Training

```bash
# HipXGen-Exam
V2_DIR=~/data_v2 \
TEXT_JSON=~/data/patient_text_no_dx.json \
CHECKPOINT_DIR=~/checkpoints \
python3 classifier/train_hipxgen_exam.py

# HipXGen-Symp
V2_DIR=~/data_v2 \
TEXT_JSON=~/data/patient_text_symptom_only.json \
CHECKPOINT_DIR=~/checkpoints \
python3 classifier/train_hipxgen_symp.py
```

### Multi-seed evaluation (mean ± std)

```bash
cd classifier && python3 multirun.py
```

Runs each variant 3 times (seeds 42, 123, 456) and reports mean ± std on the test set.

### Vision backbone pretraining

```bash
# RecA (Recurrent Cross-Attention) backbone
python3 classifier/pretrain_reca.py

# MAE backbone (domain adaptation)
python3 classifier/pretrain_mae.py
```

---

## Diffusion Model

Class-conditional latent diffusion model fine-tuned from Stable Diffusion v1-5.
Text conditioning via CLIP: `"{class} {op_status} hip xray"`.

### Requirements

```bash
pip install diffusers accelerate transformers open_clip_torch
```

### Training

```bash
python3 diffusion/train_diffusion.py \
  --data_dir ~/data_v2 \
  --manifest ~/data_v2/manifest.json \
  --output_dir ~/diffusion_output \
  --num_train_steps 30000 \
  --checkpointing_steps 5000
```

### Generating samples

```bash
# 4 prompts × 4 seeds (paper figure)
python3 diffusion/generate_samples.py

# Larger batch for FID evaluation
python3 diffusion/generate_fid_samples.py
```

---

## Baselines

```bash
# TF-IDF + Logistic Regression (symptom-only text)
V2_DIR=~/data_v2 \
TEXT_JSON=~/data/patient_text_symptom_only.json \
python3 baselines/tfidf_baseline.py
```

---

## Data

The HipXGen dataset is released separately. Each sample is a `.pt` file containing:

```python
{
    "image":       np.ndarray,   # float32 [1, H, W], range [-1, 1]
    "train_label": int,          # 0=OA, 1=ONFH, 2=Normal, 3=Other
    "op_status":   str,          # "pre_op" | "post_op" | "not_applicable"
    "patient_id":  int,
}
```

---

## Citation

```bibtex
@inproceedings{hipxgen2026,
  title     = {HipXGen: A Multimodal Vision-Language Benchmark for Hip X-ray Diagnosis with Generative Conditional Synthesis},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
}
```
