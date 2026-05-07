"""
model_hipxgen_symp.py — HipXGen-Symp: sequential cross-attention fusion.

Architecture:
  1. Frozen RecA ViT-B/16 → 49 patch features [B, 49, 768]
  2. Frozen SmolLM2-1.7B → mean-pooled hidden state [B, 2048]
       → learnable text_proj [B, 2048→768]
  3. Cross-attention pass 1: text_cls queries visual patches → fused_1 [B, 768]
  4. Cross-attention pass 2: fused_1 queries visual patches → fused_2 [B, 768]
  5. fused_2 + op_emb [B, 64] → head → 4 classes

Trainable: text_proj + cross-attn layers + op_emb + head (~5M params)
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
MAE_CKPT       = os.path.join(
    os.environ.get("CHECKPOINT_DIR", os.path.join(os.path.dirname(__file__), "checkpoints")),
    "mae_backbone.pt"
)

_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275,  0.40821073])
_CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711])

VIS_DIM     = 768    # ViT-B/16 patch dim
LLAMA_DIM   = 2048   # Llama 3.2-1B hidden dim
FUSION_DIM  = 768    # shared fusion space
N_VIS       = 49     # 7×7 visual tokens after adaptive avg pool
OP_EMB_DIM  = 64
NUM_CLASSES = 4


class GuidedCrossAttention(nn.Module):
    """
    Text-guided spatial attention over visual patches.
    Query: text representation. Keys/Values: visual patch tokens.
    Returns attended visual feature + attention weights for entropy loss.
    """
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
        # query:   [B, dim]    — text (or fused) representation
        # context: [B, N, dim] — visual patch tokens
        B, N, D = context.shape
        H, Dh   = self.num_heads, self.head_dim

        q = self.q_proj(query).view(B, 1, H, Dh).transpose(1, 2)    # [B, H, 1, Dh]
        k = self.k_proj(context).view(B, N, H, Dh).transpose(1, 2)  # [B, H, N, Dh]
        v = self.v_proj(context).view(B, N, H, Dh).transpose(1, 2)  # [B, H, N, Dh]

        attn = (q @ k.transpose(-2, -1)) * self.scale   # [B, H, 1, N]
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)

        out  = (attn @ v).transpose(1, 2).reshape(B, 1, D)  # [B, 1, D]
        out  = self.out(out).squeeze(1)                       # [B, D]
        out  = self.norm(out + query)                         # residual

        attn_map = attn.mean(dim=1).squeeze(1)  # [B, N] — averaged over heads
        return out, attn_map


class HipXGenSymp(nn.Module):
    def __init__(self, mae_ckpt=None):
        super().__init__()

        # Vision — MAE-pretrained ViT, completely frozen
        clip_model, _ = open_clip.create_model_from_pretrained(BIOMEDCLIP_TAG)
        self.vision_trunk = clip_model.visual.trunk
        ckpt_path = mae_ckpt or MAE_CKPT
        if os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location="cpu")
            self.vision_trunk.load_state_dict(state)
            print(f"Loaded MAE backbone from {ckpt_path}")
        for p in self.vision_trunk.parameters():
            p.requires_grad = False

        # Text — Llama 3.2-1B, completely frozen
        self.llama = AutoModel.from_pretrained(LLAMA_TAG)
        for p in self.llama.parameters():
            p.requires_grad = False

        # Learnable bridge: Llama hidden → fusion dim
        self.text_proj = nn.Linear(LLAMA_DIM, FUSION_DIM, bias=False)

        # Pass 1: text CLS attends to visual patches
        self.cross_attn_1 = GuidedCrossAttention(FUSION_DIM, num_heads=8)

        # Pass 2: fused representation re-attends to refine focus
        self.cross_attn_2 = GuidedCrossAttention(FUSION_DIM, num_heads=8)

        self.op_status_emb = nn.Embedding(3, OP_EMB_DIM)

        self.head = nn.Sequential(
            nn.LayerNorm(FUSION_DIM + OP_EMB_DIM),
            nn.Linear(FUSION_DIM + OP_EMB_DIM, 256),
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
        """14×14 → 7×7 = 49 tokens via adaptive avg pool."""
        B, N, D = patch_feats.shape
        x = patch_feats.reshape(B, 14, 14, D).permute(0, 3, 1, 2)
        x = F.adaptive_avg_pool2d(x, (7, 7))
        return x.permute(0, 2, 3, 1).reshape(B, N_VIS, D)

    @torch.no_grad()
    def _get_llama_features(self, input_ids, attention_mask):
        out = self.llama(input_ids=input_ids, attention_mask=attention_mask)
        h   = out.last_hidden_state  # [B, seq, 2048]
        # Mean pool over non-padding tokens
        mask = attention_mask.unsqueeze(-1).float()
        return (h * mask).sum(1) / mask.sum(1).clamp(min=1e-6)  # [B, 2048]

    def forward(self, images, input_ids, attention_mask, op_status):
        # Frozen features
        patch_feats = self._get_patch_features(images)       # [B, 196, 768]
        vis_tokens  = self._pool_visual_tokens(patch_feats)  # [B, 49, 768]

        llama_feat  = self._get_llama_features(input_ids, attention_mask)  # [B, 2048]
        text_cls    = self.text_proj(llama_feat)  # [B, 768] — learnable projection

        # Pass 1: text queries where to look in image
        fused_1, attn_map_1 = self.cross_attn_1(text_cls, vis_tokens)

        # Pass 2: iterative refinement
        fused_2, attn_map_2 = self.cross_attn_2(fused_1, vis_tokens)

        op_emb = self.op_status_emb(op_status)
        out    = self.head(torch.cat([fused_2, op_emb], dim=-1))

        return out, attn_map_1, attn_map_2

    def attention_entropy_loss(self, attn_map):
        """Maximize entropy of attention → force model to spread focus."""
        entropy = -(attn_map * (attn_map + 1e-8).log()).sum(dim=-1)
        return -entropy.mean()
