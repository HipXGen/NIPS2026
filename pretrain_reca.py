"""
pretrain_reca.py — Reconstruction Alignment (RecA) pretraining.

Unlike MAE (pixel reconstruction), RecA reconstructs the teacher's patch
features for masked positions. This preserves the semantic structure of
BiomedCLIP while adapting the backbone to hip X-ray domain.

Architecture:
  Teacher: frozen original BiomedCLIP ViT-B/16  → patch features [B, 196, 768]
  Student: trainable BiomedCLIP ViT-B/16        → visible patch features
  Decoder: 4-block transformer → predicts teacher features for masked patches
  Loss: MSE(predicted, teacher_features[masked]) — feature space, not pixels

Saves student vision trunk state_dict to checkpoints/reca_backbone.pt

Usage:
    python pretrain_reca.py
    python pretrain_reca.py --epochs 200
"""

import os
import math
import argparse
os.environ.setdefault("V2_DIR", os.path.expanduser("~/data_v2"))

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle
import json
import numpy as np

BIOMEDCLIP_TAG = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
V2_DIR         = os.environ.get("V2_DIR", os.path.expanduser("~/data_v2"))
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", os.path.join(os.path.dirname(__file__), "checkpoints"))
RECA_CKPT      = os.path.join(CHECKPOINT_DIR, "reca_backbone.pt")

_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275,  0.40821073]).view(1,3,1,1)
_CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1,3,1,1)

MASK_RATIO   = 0.75
DECODER_DIM  = 512
DECODER_DEPTH = 4
FEAT_DIM     = 768   # ViT-B/16 patch feature dim = teacher output dim

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


class HipXGenAllImages(Dataset):
    """All 2163 hip images across all splits — no labels needed."""
    def __init__(self):
        manifest_path = os.path.join(V2_DIR, "manifest.json")
        with open(manifest_path) as f:
            self.entries = json.load(f)["entries"]
        print(f"RecA dataset: {len(self.entries)} images")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        pt_path = os.path.join(V2_DIR, e["pt_path"])
        with open(pt_path, "rb") as f:
            d = pickle.load(f)
        img = torch.from_numpy(d["image"].astype(np.float32))  # [1, H, W]
        return img.repeat(3, 1, 1)  # [3, H, W]


class RecADecoderBlock(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff    = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        x2, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + x2
        x = x + self.ff(self.norm2(x))
        return x


class RecADecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj   = nn.Linear(FEAT_DIM, DECODER_DIM)
        self.blocks = nn.Sequential(*[RecADecoderBlock(DECODER_DIM) for _ in range(DECODER_DEPTH)])
        self.norm   = nn.LayerNorm(DECODER_DIM)
        self.head   = nn.Linear(DECODER_DIM, FEAT_DIM)  # predict teacher's 768-dim features
        self.mask_token = nn.Parameter(torch.zeros(1, 1, DECODER_DIM))
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, x_vis, ids_restore):
        # x_vis: [B, N_vis, FEAT_DIM]
        B, N_vis, _ = x_vis.shape
        N = ids_restore.shape[1]  # total patches
        N_mask = N - N_vis

        x = self.proj(x_vis)

        # append mask tokens
        mask_tokens = self.mask_token.expand(B, N_mask, -1)
        x_full = torch.cat([x, mask_tokens], dim=1)  # [B, N, DECODER_DIM]

        # unshuffle
        x_full = torch.gather(
            x_full, dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, DECODER_DIM)
        )

        x_full = self.blocks(x_full)
        x_full = self.norm(x_full)
        return self.head(x_full)  # [B, N, FEAT_DIM]


class RecABackbone(nn.Module):
    def __init__(self):
        super().__init__()

        clip_model, _ = open_clip.create_model_from_pretrained(BIOMEDCLIP_TAG)

        # Teacher — frozen original BiomedCLIP ViT
        self.teacher = clip_model.visual.trunk
        for p in self.teacher.parameters():
            p.requires_grad = False

        # Student — trainable copy, initialized from same BiomedCLIP weights
        clip_student, _ = open_clip.create_model_from_pretrained(BIOMEDCLIP_TAG)
        self.student = clip_student.visual.trunk
        for p in self.student.parameters():
            p.requires_grad = True

        self.decoder   = RecADecoder()
        self.mask_ratio = MASK_RATIO

        self.register_buffer("clip_mean", _CLIP_MEAN)
        self.register_buffer("clip_std",  _CLIP_STD)

    def _preprocess(self, imgs):
        x = F.interpolate(imgs, size=(224, 224), mode="bilinear", align_corners=False)
        return (x - self.clip_mean) / self.clip_std

    @torch.no_grad()
    def _get_teacher_features(self, x):
        feats = self.teacher.forward_features(x)  # [B, 197, 768]
        return feats[:, 1:, :]  # drop CLS → [B, 196, 768]

    def random_masking(self, x):
        B, N, D = x.shape
        len_keep = int(N * (1 - self.mask_ratio))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_vis = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        mask = torch.ones(B, N, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_vis, mask, ids_restore

    def forward(self, imgs):
        x = self._preprocess(imgs)

        # Teacher: full image patch features (no masking, no grad)
        teacher_feats = self._get_teacher_features(x)  # [B, 196, 768]

        # Normalize teacher targets per-patch (stabilizes training)
        target = (teacher_feats - teacher_feats.mean(dim=-1, keepdim=True)) / \
                 (teacher_feats.std(dim=-1, keepdim=True) + 1e-6)

        # Student: encode only visible patches
        student_feats_full = self.student.forward_features(x)  # [B, 197, 768]
        student_feats = student_feats_full[:, 1:, :]            # [B, 196, 768]
        x_vis, mask, ids_restore = self.random_masking(student_feats)

        # Decode: predict teacher features for ALL positions
        pred = self.decoder(x_vis, ids_restore)  # [B, 196, 768]

        # Loss only on masked patches
        loss = ((pred - target) ** 2)           # [B, 196, 768]
        loss = loss.mean(dim=-1)                 # [B, 196]
        loss = (loss * mask).sum() / (mask.sum() + 1e-6)
        return loss


def main(epochs=200, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Epochs: {epochs}  |  Batch: {batch_size}")
    print(f"Saving backbone to: {RECA_CKPT}")

    dataset = HipXGenAllImages()
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         num_workers=4, pin_memory=(device.type == "cuda"),
                         drop_last=True)

    model = RecABackbone().to(device)
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params — total: {total:,}  |  trainable: {trainable:,}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1.5e-4, weight_decay=0.05, betas=(0.9, 0.95)
    )

    warmup_epochs = 20
    def lr_lambda(ep):
        if ep < warmup_epochs:
            return ep / warmup_epochs
        progress = (ep - warmup_epochs) / (epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for imgs in tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            imgs = imgs.to(device)
            loss = model(imgs)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        scheduler.step()
        print(f"Epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.student.state_dict(), RECA_CKPT)
            print(f"  → Saved best backbone (loss={best_loss:.4f})")

    print(f"\nDone. Best loss: {best_loss:.4f}")
    print(f"RecA backbone saved to {RECA_CKPT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    main(epochs=args.epochs, batch_size=args.batch_size)
