"""
model_hipxgen_exam.py — HipXGen-Exam: text-primary fusion with RecA image correction.

Architecture:
  1. Frozen RecA ViT-B/16 → 49 visual tokens [B, 49, 768]
  2. Frozen SmolLM2-1.7B → mean-pooled hidden state [B, 2048]
       → learnable text_proj [B, 2048→768]  → text_cls
  3. ONE cross-attention: text_cls queries RecA patches → attended feature [B, 768]
       → image_proj [B, 768→128]  → image_feat (additive correction)
  4. cat([text_cls, image_feat, op_emb]) → head → 4 classes

Trainable: text_proj + cross_attn + image_proj + op_emb + head (~5M params)
Both encoders completely frozen.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from transformers import AutoModel

BIOMEDCLIP_TAG = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
LLAMA_TAG      = "HuggingFaceTB/SmolLM2-1.7B"
RECA_CKPT      = os.path.join(
    os.environ.get("CHECKPOINT_DIR", os.path.join(os.path.dirname(__file__), "checkpoints")),
    "reca_backbone.pt"
)

_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275,  0.40821073])
_CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711])

VIS_DIM      = 768
LLAMA_DIM    = 2048
FUSION_DIM   = 768
IMAGE_CORR   = 128   # small image correction dim — can't overwhelm text
N_VIS        = 49
OP_EMB_DIM   = 64
NUM_CLASSES  = 4
HEAD_IN_DIM  = FUSION_DIM + IMAGE_CORR + OP_EMB_DIM  # 768 + 128 + 64 = 960


class GuidedCrossAttention(nn.Module):
    """Text-guided spatial attention over visual patches."""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out    = nn.Linear(dim, dim)
        self.drop   = nn.Dropout(dropout)
        self.norm   = nn.LayerNorm(dim)

    def forward(self, query, context):
        B, N, D = context.shape
        H, Dh   = self.num_heads, self.head_dim

        q = self.q_proj(query).view(B, 1, H, Dh).transpose(1, 2)
        k = self.k_proj(context).view(B, N, H, Dh).transpose(1, 2)
        v = self.v_proj(context).view(B, N, H, Dh).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, 1, D)
        out = self.out(out).squeeze(1)
        out = self.norm(out + query)

        attn_map = attn.mean(dim=1).squeeze(1)  # [B, N]
        return out, attn_map


class HipXGenExam(nn.Module):
    def __init__(self, reca_ckpt=None):
        super().__init__()

        # Vision — RecA-pretrained ViT, completely frozen
        clip_model, _ = open_clip.create_model_from_pretrained(BIOMEDCLIP_TAG)
        self.vision_trunk = clip_model.visual.trunk
        ckpt_path = reca_ckpt or RECA_CKPT
        if os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location="cpu")
            self.vision_trunk.load_state_dict(state)
            print(f"Loaded RecA backbone from {ckpt_path}")
        else:
            print(f"WARNING: RecA checkpoint not found at {ckpt_path}, using vanilla BiomedCLIP")
        for p in self.vision_trunk.parameters():
            p.requires_grad = False

        # Text — SmolLM2-1.7B, completely frozen
        self.llama = AutoModel.from_pretrained(LLAMA_TAG)
        for p in self.llama.parameters():
            p.requires_grad = False

        # Learnable: text bridge 2048→768
        self.text_proj = nn.Linear(LLAMA_DIM, FUSION_DIM, bias=False)

        # ONE cross-attention: text queries image → small image correction
        self.cross_attn = GuidedCrossAttention(FUSION_DIM, num_heads=8)

        # Compress attended image feature to small correction vector
        self.image_proj = nn.Linear(FUSION_DIM, IMAGE_CORR, bias=False)

        self.op_status_emb = nn.Embedding(3, OP_EMB_DIM)

        # Head: text_cls (768) + image_feat (128) + op_emb (64) = 960
        self.head = nn.Sequential(
            nn.LayerNorm(HEAD_IN_DIM),
            nn.Linear(HEAD_IN_DIM, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, NUM_CLASSES),
        )

        self.register_buffer("clip_mean", _CLIP_MEAN.view(1, 3, 1, 1))
        self.register_buffer("clip_std",  _CLIP_STD.view(1, 3, 1, 1))

    @torch.no_grad()
    def _get_patch_features(self, images):
        x = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - self.clip_mean) / self.clip_std
        feats = self.vision_trunk.forward_features(x)  # [B, 197, 768]
        return feats[:, 1:, :]  # drop CLS → [B, 196, 768]

    def _pool_visual_tokens(self, patch_feats):
        B, N, D = patch_feats.shape
        x = patch_feats.reshape(B, 14, 14, D).permute(0, 3, 1, 2)
        x = F.adaptive_avg_pool2d(x, (7, 7))
        return x.permute(0, 2, 3, 1).reshape(B, N_VIS, D)

    @torch.no_grad()
    def _get_llama_features(self, input_ids, attention_mask):
        out  = self.llama(input_ids=input_ids, attention_mask=attention_mask)
        h    = out.last_hidden_state  # [B, seq, 2048]
        mask = attention_mask.unsqueeze(-1).float()
        return (h * mask).sum(1) / mask.sum(1).clamp(min=1e-6)  # [B, 2048]

    def forward(self, images, input_ids, attention_mask, op_status):
        patch_feats = self._get_patch_features(images)       # [B, 196, 768]
        vis_tokens  = self._pool_visual_tokens(patch_feats)  # [B, 49, 768]

        llama_feat = self._get_llama_features(input_ids, attention_mask)  # [B, 2048]
        text_cls   = self.text_proj(llama_feat)  # [B, 768] — direct text signal

        # Image correction: ONE attention pass → compressed to 128-dim
        attended, attn_map = self.cross_attn(text_cls, vis_tokens)  # [B, 768]
        image_feat = self.image_proj(attended)  # [B, 128]

        op_emb = self.op_status_emb(op_status)  # [B, 64]

        # text_cls always the primary signal; image_feat is additive correction
        combined = torch.cat([text_cls, image_feat, op_emb], dim=-1)  # [B, 960]
        out = self.head(combined)

        return out, attn_map

    def attention_entropy_loss(self, attn_map):
        entropy = -(attn_map * (attn_map + 1e-8).log()).sum(dim=-1)
        return -entropy.mean()
