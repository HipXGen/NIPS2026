"""
dataset.py — HipXGen dataset loader.

Loads CLAHE-normalized hip X-ray .pt files from the HipXGen dataset.
Applies augmentation (flip, ±10° rotation, brightness/contrast jitter) during training.
CLIP normalization is applied in the model, not here.
"""
from __future__ import annotations
import json, os, pickle
from collections import Counter
from typing import Optional, Set

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
import open_clip

OP_STATUS_MAP = {"pre_op": 0, "post_op": 1, "not_applicable": 2}

V2_DIR   = os.environ.get("V2_DIR",   os.path.expanduser("~/Desktop/hipxgen_unified/processed_v2_image_only"))
TEXT_JSON = os.environ.get("TEXT_JSON", os.path.expanduser("~/Desktop/hipxgen_unified/processed/patient_text_no_dx.json"))

_tokenizer_biomedclip = None

def get_tokenizer():
    global _tokenizer_biomedclip
    if _tokenizer_biomedclip is None:
        _tokenizer_biomedclip = open_clip.get_tokenizer(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
    return _tokenizer_biomedclip


class HipXGenDataset(Dataset):
    """
    Args:
        split: "train" | "val" | "test"
        op_status_filter: if set, only keep samples whose op_status is in this set
        augment: apply training augmentation (flip, rotation, jitter)
    """

    def __init__(
        self,
        split: str,
        op_status_filter: Optional[Set[str]] = None,
        augment: bool = False,
        tokenizer_tag: str = "biomedclip",
    ):
        assert split in ("train", "val", "test")
        self.split = split
        self.augment = augment

        manifest_path = os.path.join(V2_DIR, "manifest.json")
        with open(manifest_path) as f:
            manifest = json.load(f)

        self.entries = [
            e for e in manifest["entries"]
            if e["split"] == split
            and (op_status_filter is None or e["op_status"] in op_status_filter)
        ]

        with open(TEXT_JSON) as f:
            self.text_lookup = json.load(f)

        self.tokenizer_tag = tokenizer_tag
        if tokenizer_tag == "llama":
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                "HuggingFaceTB/SmolLM2-1.7B", use_fast=True
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = get_tokenizer()

    def __len__(self) -> int:
        return len(self.entries)

    def get_sample_weights(self):
        """Returns per-sample weights for WeightedRandomSampler (inverse class freq)."""
        labels = [e["train_label"] for e in self.entries]
        counts = Counter(labels)
        total = len(labels)
        return [total / (len(counts) * counts[l]) for l in labels]

    def _augment(self, img: torch.Tensor) -> torch.Tensor:
        # img: [1, H, W] in [0, 1]
        if torch.rand(()) < 0.5:
            img = TF.hflip(img)
        angle = (torch.rand(()).item() * 2 - 1) * 10.0
        img = TF.rotate(img, angle, fill=0.0)
        bf = 1.0 + (torch.rand(()).item() * 2 - 1) * 0.1
        cf = 1.0 + (torch.rand(()).item() * 2 - 1) * 0.1
        img = TF.adjust_brightness(img.clamp(0, 1), bf).clamp(0, 1)
        img = TF.adjust_contrast(img, cf).clamp(0, 1)
        return img

    def __getitem__(self, idx):
        e = self.entries[idx]
        pt_path = os.path.join(V2_DIR, e["pt_path"])
        with open(pt_path, "rb") as f:
            d = pickle.load(f)

        # Image: [1, 512, 512] float32 in [0, 1]
        img = torch.from_numpy(d["image"].astype(np.float32))  # [1, 512, 512]
        if self.augment:
            img = self._augment(img)
        img = img.repeat(3, 1, 1)  # [3, 512, 512] — CLIP normalization done in model

        # Text tokenization
        text = self.text_lookup.get(e["patient_id"], "")
        if self.tokenizer_tag == "llama":
            enc = self.tokenizer(
                text, padding="max_length", max_length=256,
                truncation=True, return_tensors="pt",
            )
            input_ids      = enc["input_ids"].squeeze(0).long()
            attention_mask = enc["attention_mask"].squeeze(0).long()
        else:
            tokens = self.tokenizer([text], context_length=256)
            if tokens.dim() == 2:
                tokens = tokens.squeeze(0)
            input_ids      = tokens.long()
            attention_mask = (input_ids != 0).long()

        op_status  = OP_STATUS_MAP[e["op_status"]]
        train_label = int(e["train_label"])

        return img, input_ids, attention_mask, op_status, train_label
