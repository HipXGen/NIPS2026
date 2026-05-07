"""
pretrain_mae.py — MAE domain-adaptation on all 2163 HipXGen images.

Uses BiomedCLIP ViT-B/16 as encoder (starts from pretrained weights),
lightweight 4-block decoder reconstructs masked patches.

Saves: checkpoints/mae_backbone.pt  (ViT trunk state_dict only)

Usage:
    python pretrain_mae.py
    python pretrain_mae.py --smoke-test
"""

import os
os.environ.setdefault("V2_DIR",
    os.path.expanduser("~/data_v2"))

import argparse
import json
import math
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import open_clip
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

BIOMEDCLIP_TAG = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
V2_DIR         = os.environ.get("V2_DIR", os.path.expanduser("~/data_v2"))
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", os.path.join(os.path.dirname(__file__), "checkpoints"))
MAE_CKPT       = os.path.join(CHECKPOINT_DIR, "mae_backbone.pt")

MASK_RATIO    = 0.75
IMG_SIZE      = 224
PATCH_SIZE    = 16
NUM_PATCHES   = (IMG_SIZE // PATCH_SIZE) ** 2   # 196
ENCODER_DIM   = 768
DECODER_DIM   = 512
DECODER_DEPTH = 4
DECODER_HEADS = 16

BATCH_SIZE    = 32
LR            = 1.5e-4
WEIGHT_DECAY  = 0.05
EPOCHS        = 200
WARMUP_EPOCHS = 20

_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275,  0.40821073])
_CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711])

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ── Dataset ───────────────────────────────────────────────────────────────────

class HipXGenAllImages(Dataset):
    """All 2163 images from all splits — no labels needed."""

    def __init__(self, v2_dir, augment=True):
        manifest_path = os.path.join(v2_dir, "manifest.json")
        with open(manifest_path) as f:
            manifest = json.load(f)
        self.entries  = manifest["entries"]
        self.v2_dir   = v2_dir
        self.augment  = augment

    def __len__(self):
        return len(self.entries)

    def _augment(self, img):
        if torch.rand(()) < 0.5:
            img = TF.hflip(img)
        angle = (torch.rand(()).item() * 2 - 1) * 10.0
        img = TF.rotate(img, angle, fill=0.0)
        return img

    def __getitem__(self, idx):
        e = self.entries[idx]
        pt_path = os.path.join(self.v2_dir, e["pt_path"])
        with open(pt_path, "rb") as f:
            d = pickle.load(f)
        img = torch.from_numpy(d["image"].astype(np.float32))  # [1, 512, 512]
        if self.augment:
            img = self._augment(img)
        img = img.repeat(3, 1, 1)  # [3, 512, 512]
        return img


# ── MAE components ────────────────────────────────────────────────────────────

class MAEDecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim    = int(dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, dim)
        )

    def forward(self, x):
        n = self.norm1(x)
        x = x + self.attn(n, n, n, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class MAEDecoder(nn.Module):
    def __init__(self, encoder_dim=ENCODER_DIM, decoder_dim=DECODER_DIM,
                 depth=DECODER_DEPTH, num_heads=DECODER_HEADS,
                 num_patches=NUM_PATCHES, patch_size=PATCH_SIZE):
        super().__init__()
        self.decoder_embed    = nn.Linear(encoder_dim, decoder_dim, bias=True)
        self.mask_token       = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_dim)
        )
        self.blocks           = nn.ModuleList([
            MAEDecoderBlock(decoder_dim, num_heads) for _ in range(depth)
        ])
        self.norm             = nn.LayerNorm(decoder_dim)
        self.pred             = nn.Linear(decoder_dim, patch_size * patch_size * 3)

        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.decoder_pos_embed, std=0.02)

    def forward(self, x, ids_restore):
        # x: [B, 1+len_keep, encoder_dim]
        x = self.decoder_embed(x)
        D = x.shape[-1]

        mask_tokens = self.mask_token.expand(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], -1
        )
        x_ = torch.cat([x[:, 1:], mask_tokens], dim=1)  # drop cls
        x_ = torch.gather(x_, dim=1,
                          index=ids_restore.unsqueeze(-1).expand(-1, -1, D))
        x  = torch.cat([x[:, :1], x_], dim=1)  # re-attach cls

        x = x + self.decoder_pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = self.pred(x[:, 1:])  # remove cls → [B, N, patch_size²×3]
        return x


class MAEBackbone(nn.Module):
    def __init__(self, mask_ratio=MASK_RATIO):
        super().__init__()
        self.mask_ratio = mask_ratio

        clip_model, _ = open_clip.create_model_from_pretrained(BIOMEDCLIP_TAG)
        self.encoder = clip_model.visual.trunk
        for p in self.encoder.parameters():
            p.requires_grad = True

        self.decoder = MAEDecoder()

        mean = _CLIP_MEAN.view(1, 3, 1, 1)
        std  = _CLIP_STD.view(1, 3, 1, 1)
        self.register_buffer("clip_mean", mean)
        self.register_buffer("clip_std",  std)

    def _preprocess(self, imgs):
        x = F.interpolate(imgs, size=(IMG_SIZE, IMG_SIZE),
                          mode="bilinear", align_corners=False)
        return (x - self.clip_mean) / self.clip_std

    def patchify(self, imgs):
        """[B,3,H,W] → [B, N, patch_size²×3]"""
        p = PATCH_SIZE
        B, C, H, W = imgs.shape
        h, w = H // p, W // p
        x = imgs.reshape(B, C, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1)   # [B,h,w,p,p,C]
        x = x.reshape(B, h * w, p * p * C)
        return x

    def random_masking(self, x):
        B, N, D = x.shape
        len_keep = int(N * (1 - self.mask_ratio))

        noise       = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep    = ids_shuffle[:, :len_keep]

        x_vis = torch.gather(x, dim=1,
                             index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        mask = torch.ones(B, N, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_vis, mask, ids_restore

    def forward_encoder(self, x):
        x = self.encoder.patch_embed(x)                           # [B, N, D]
        x = x + self.encoder.pos_embed[:, 1:]                     # patch pos embed

        x_vis, mask, ids_restore = self.random_masking(x)

        cls = (self.encoder.cls_token + self.encoder.pos_embed[:, :1]).expand(
            x_vis.shape[0], -1, -1
        )
        x_vis = torch.cat([cls, x_vis], dim=1)

        for blk in self.encoder.blocks:
            x_vis = blk(x_vis)
        x_vis = self.encoder.norm(x_vis)
        return x_vis, mask, ids_restore

    def forward(self, imgs):
        x      = self._preprocess(imgs)
        latent, mask, ids_restore = self.forward_encoder(x)
        pred   = self.decoder(latent, ids_restore)

        target = self.patchify(x)
        # per-patch normalization (as in MAE paper)
        mean   = target.mean(dim=-1, keepdim=True)
        var    = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1e-6).sqrt()

        loss = ((pred - target) ** 2 * mask.unsqueeze(-1)).sum() / mask.sum()
        return loss


# ── LR schedule ───────────────────────────────────────────────────────────────

def lr_lambda(epoch):
    if epoch < WARMUP_EPOCHS:
        return (epoch + 1) / WARMUP_EPOCHS
    t = (epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS)
    return 0.5 * (1 + math.cos(math.pi * t))


# ── Main ──────────────────────────────────────────────────────────────────────

def main(smoke_test=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"MAE pretraining — {EPOCHS} epochs, mask_ratio={MASK_RATIO}")

    ds     = HipXGenAllImages(V2_DIR, augment=not smoke_test)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=4, pin_memory=(device.type == "cuda"),
                        drop_last=True)
    print(f"Dataset: {len(ds)} images  |  {len(loader)} batches/epoch")

    model     = MAEBackbone().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_loss = float("inf")
    for epoch in range(1, (3 if smoke_test else EPOCHS) + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()
        for batch_idx, imgs in enumerate(tqdm(loader, desc=f"epoch {epoch}", leave=False)):
            imgs = imgs.to(device)
            loss = model(imgs)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            if smoke_test and batch_idx >= 1:
                break

        scheduler.step()
        n = min(len(loader), 2) if smoke_test else len(loader)
        avg_loss = total_loss / max(n, 1)
        lr_now   = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d}/{EPOCHS}  loss={avg_loss:.4f}  "
              f"lr={lr_now:.2e}  ({time.time()-t0:.1f}s)")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.encoder.state_dict(), MAE_CKPT)
            print(f"  → Backbone saved (loss={best_loss:.4f})")

        if smoke_test:
            break

    print(f"\nDone. Best loss: {best_loss:.4f}")
    print(f"Backbone saved to: {MAE_CKPT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    main(smoke_test=args.smoke_test)
