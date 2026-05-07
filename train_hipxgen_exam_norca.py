"""
train_hipxgen_exam_norca.py — Ablation: HipXGen-Exam without RecA (vanilla BiomedCLIP).

Identical to train_hipxgen_exam.py but skips RecA weights,
using the original BiomedCLIP ViT-B/16 trunk to isolate RecA's contribution.
"""

import os
os.environ.setdefault("TEXT_JSON", os.path.expanduser("~/data/patient_text_no_dx.json"))
os.environ.setdefault("V2_DIR",    os.path.expanduser("~/data_v2"))

import argparse, csv, time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, accuracy_score, classification_report
from tqdm import tqdm

from dataset import HipXGenDataset
from model_hipxgen_exam import HipXGenExam

CHECKPOINT_DIR  = os.environ.get("CHECKPOINT_DIR", os.path.join(os.path.dirname(__file__), "checkpoints"))
LOG_CSV         = os.environ.get("LOG_CSV",        os.path.join(os.path.dirname(__file__), "training_log_exam_norca.csv"))
BEST_CKPT       = os.path.join(CHECKPOINT_DIR, "best_model_exam_norca.pt")

OP_FILTER_EVAL  = {"pre_op", "not_applicable"}
LABEL_NAMES     = {0: "OA", 1: "ONFH", 2: "Normal", 3: "Other"}

BATCH_SIZE      = 16
GRAD_ACCUM      = 2
MAX_EPOCHS      = 100
PATIENCE        = 20
LR_HEAD         = 5e-4
WEIGHT_DECAY    = 0.01
NUM_CLASSES     = 4
LABEL_SMOOTHING = 0.1
GRAD_CLIP       = 1.0
ENT_LOSS_WEIGHT = 0.05

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def compute_class_weights(dataset):
    from collections import Counter
    labels = [e["train_label"] for e in dataset.entries]
    counts = Counter(labels)
    total  = len(labels)
    weights = torch.zeros(NUM_CLASSES)
    for c in range(NUM_CLASSES):
        weights[c] = total / (NUM_CLASSES * max(counts.get(c, 1), 1))
    return weights


def run_epoch(model, loader, criterion, optimizer, device, smoke_test=False):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    for batch_idx, (imgs, ids, masks, ops, labels) in enumerate(
        tqdm(loader, desc="train", leave=False)
    ):
        imgs, ids, masks, ops, labels = (
            imgs.to(device), ids.to(device), masks.to(device),
            ops.to(device), labels.to(device),
        )
        logits, attn_map = model(imgs, ids, masks, ops)
        cls_loss = criterion(logits, labels)
        ent_loss = model.attention_entropy_loss(attn_map)
        loss = (cls_loss + ENT_LOSS_WEIGHT * ent_loss) / GRAD_ACCUM
        loss.backward()
        if (batch_idx + 1) % GRAD_ACCUM == 0 or (batch_idx + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad()
        total_loss += loss.item() * GRAD_ACCUM
        if smoke_test and batch_idx >= 1:
            break
    n = min(len(loader), 2) if smoke_test else len(loader)
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, smoke_test=False):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for batch_idx, (imgs, ids, masks, ops, labels) in enumerate(
        tqdm(loader, desc="val  ", leave=False)
    ):
        imgs, ids, masks, ops, labels = (
            imgs.to(device), ids.to(device), masks.to(device),
            ops.to(device), labels.to(device),
        )
        logits, _ = model(imgs, ids, masks, ops)
        total_loss += criterion(logits, labels).item()
        all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        if smoke_test and batch_idx >= 1:
            break
    n = min(len(loader), 2) if smoke_test else len(loader)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    return total_loss / max(n, 1), accuracy_score(all_labels, all_preds), weighted_f1


@torch.no_grad()
def run_test(model, device):
    print("\n" + "="*60)
    print("FINAL TEST EVALUATION (pre_op + not_applicable, n=118)")
    print("="*60)
    test_ds = HipXGenDataset("test", op_status_filter=OP_FILTER_EVAL,
                               augment=False, tokenizer_tag="llama")
    loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=2, pin_memory=True)
    model.eval()
    all_preds, all_labels = [], []
    for imgs, ids, masks, ops, labels in loader:
        logits, _ = model(imgs.to(device), ids.to(device), masks.to(device), ops.to(device))
        all_preds.extend(logits.argmax(-1).cpu().tolist())
        all_labels.extend(labels.tolist())

    acc         = accuracy_score(all_labels, all_preds)
    macro_f1    = f1_score(all_labels, all_preds, average="macro",    zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    print(f"Test Accuracy    : {acc:.4f}")
    print(f"Test Macro F1    : {macro_f1:.4f}")
    print(f"Test Weighted F1 : {weighted_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds,
                                target_names=[LABEL_NAMES[c] for c in sorted(LABEL_NAMES)],
                                zero_division=0))


def main(smoke_test=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"ABLATION: vanilla BiomedCLIP backbone (no RecA)")
    print(f"Text   : SmolLM2-1.7B (frozen) + learnable text_proj 2048→768")
    print(f"Fusion : text-primary — text_cls direct to head + image_proj 128-dim correction")

    train_ds = HipXGenDataset("train", op_status_filter={"pre_op", "not_applicable"},
                                augment=True, tokenizer_tag="llama")
    val_ds   = HipXGenDataset("val", op_status_filter=OP_FILTER_EVAL,
                                augment=False, tokenizer_tag="llama")
    print(f"Train: {len(train_ds)} samples  |  Val: {len(val_ds)} samples")

    sample_weights = train_ds.get_sample_weights()
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=2, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                              num_workers=2, pin_memory=(device.type == "cuda"))

    # Pass nonexistent path → falls back to vanilla BiomedCLIP with a warning
    model = HipXGenExam(reca_ckpt="/tmp/no_reca").to(device)
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params — total: {total:,}  |  trainable: {trainable:,}")

    weights   = compute_class_weights(train_ds).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=LABEL_SMOOTHING)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-6)

    with open(LOG_CSV, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_acc", "val_weighted_f1"])

    best_f1, patience_left = -1.0, PATIENCE

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, smoke_test)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device, smoke_test)
        scheduler.step()

        print(f"Epoch {epoch:3d}/{MAX_EPOCHS}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  val_weighted_f1={val_f1:.4f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}  ({time.time()-t0:.1f}s)")

        with open(LOG_CSV, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss, val_acc, val_f1])

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_left = PATIENCE
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_f1": val_f1, "val_acc": val_acc}, BEST_CKPT)
            print(f"  → Best checkpoint saved (val_weighted_f1={best_f1:.4f})")
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"Early stopping at epoch {epoch}.")
                break

        if smoke_test:
            print("Smoke test complete.")
            break

    print(f"\nDone. Best val weighted F1: {best_f1:.4f}")
    ckpt = torch.load(BEST_CKPT, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Best checkpoint: epoch {ckpt['epoch']}, val_weighted_f1={ckpt['val_f1']:.4f}")
    run_test(model, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    main(smoke_test=args.smoke_test)
