"""
clause_classifier.py — Multi-label legal clause classifier.

Supports three backbone models:
  - Legal-BERT:       nlpaueb/legal-bert-base-uncased
  - DeBERTa-v3:       microsoft/deberta-v3-base
  - Legal-RoBERTa-L:  lexlms/legal-roberta-large

All backbones share identical training logic (AsymmetricLoss, AdamW,
linear warmup, early stopping). Model selection and per-class threshold
optimisation are handled by ModelSelector, which evaluates all single
backbones and every 2- and 3-model ensemble combination. Production
inference uses the saved production_config.json for deterministic,
tuning-free deployment.

Usage:
    # Train individual backbones
    python src/clause_classifier.py --mode train --backbone legalbert
    python src/clause_classifier.py --mode train --backbone deberta
    python src/clause_classifier.py --mode train --backbone legalroberta

    # Select best model / ensemble and optimise thresholds
    python src/clause_classifier.py --mode select

    # Final evaluation on test set (no tuning)
    python src/clause_classifier.py --mode evaluate

    # Single-clause prediction
    python src/clause_classifier.py --mode predict --text "The Licensor shall..."
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger.remove()
logger.add(config.LOGS_DIR / "classifier.log", rotation="10 MB", level="DEBUG")
logger.add(sys.stderr, level="INFO")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BACKBONE_MODEL_MAP: Dict[str, str] = {
    "legalbert":    config.LEGAL_BERT_MODEL,
    "deberta":      config.DEBERTA_MODEL,
    "legalroberta": config.LEGAL_ROBERTA_MODEL,
}

BACKBONE_DIR_MAP: Dict[str, Path] = {
    "legalbert":    config.CLASSIFIER_LEGALBERT_DIR,
    "deberta":      config.CLASSIFIER_DEBERTA_DIR,
    "legalroberta": config.CLASSIFIER_LEGALROBERTA_DIR,
}

# Per-backbone training hyperparameters
# DeBERTa-v3 NaN root causes: disentangled attention sensitivity to high LR,
# head gradient explosion, and AdamW default eps=1e-8 (must be 1e-6 per paper).
# Legal-RoBERTa-large uses the same settings as Legal-BERT — RoBERTa's pooler
# is stable at 2e-5 and the large model handles fp16 without NaN issues.
BACKBONE_LR_MAP: Dict[str, float] = {
    "legalbert":    2e-5,
    "deberta":      3e-6,   # 2e-6 was stable but too slow; 5e-6 caused NaN — 3e-6 is the midpoint
    "legalroberta": 1e-5,   # slightly lower than BERT-base due to larger model size
}
BACKBONE_EPOCHS_MAP: Dict[str, int] = {
    "legalbert":    90,
    "deberta":      90,
    "legalroberta": 90,
}
BACKBONE_HEAD_MULT_MAP: Dict[str, int] = {
    "legalbert":    10,
    "deberta":      1,    # any head multiplier causes NaN for DeBERTa
    "legalroberta": 10,
}
BACKBONE_ADAM_EPS_MAP: Dict[str, float] = {
    "legalbert":    1e-8,
    "deberta":      1e-6,  # required by DeBERTa paper — prevents NaN
    "legalroberta": 1e-8,
}
BACKBONE_GRAD_CLIP_MAP: Dict[str, float] = {
    "legalbert":    1.0,
    "deberta":      0.5,
    "legalroberta": 1.0,
}

# DeBERTa-v3: use mean-pooling over last hidden states instead of the default
# pooler. The AutoModelForSequenceClassification pooler adds instability at
# LR=3e-6 and can produce all-zero predictions for many epochs.
# Legal-RoBERTa-large: standard CLS pooler via AutoModelForSequenceClassification
# is stable — no mean-pooling needed.
BACKBONE_USE_MEAN_POOL: Dict[str, bool] = {
    "legalbert":    False,
    "deberta":      True,
    "legalroberta": False,
}
# DeBERTa at LR=2e-6 produces near-zero F1 for the first ~15 epochs while the
# encoder representations stabilise. Early stopping must not fire during warmup.
BACKBONE_MIN_EPOCHS_MAP: Dict[str, int] = {
    "legalbert":    10,
    "deberta":      20,
    "legalroberta": 10,
}
# Per-backbone early stopping patience (epochs without val F1 improvement).
BACKBONE_PATIENCE_MAP: Dict[str, int] = {
    "legalbert":    6,
    "deberta":      8,
    "legalroberta": 6,
}
# Legal-RoBERTa-large (355M params) needs smaller batch to fit in 8GB VRAM.
# Gradient accumulation steps compensate so effective batch = CLASSIFIER_BATCH_SIZE.
BACKBONE_BATCH_SIZE_MAP: Dict[str, int] = {
    "legalbert":    config.CLASSIFIER_BATCH_SIZE,      # 16
    "deberta":      config.CLASSIFIER_BATCH_SIZE,      # 16
    "legalroberta": config.CLASSIFIER_BATCH_SIZE // 2, # 8 — large model, 8GB VRAM
}
BACKBONE_GRAD_ACCUM_MAP: Dict[str, int] = {
    "legalbert":    2,
    "deberta":      2,
    "legalroberta": 4,  # 8 × 4 = effective batch 32 (slightly larger than 16×2 — helps large model)
}


# ---------------------------------------------------------------------------
# 1. DATASET
# ---------------------------------------------------------------------------

class ClauseDataset(Dataset):
    """PyTorch Dataset with pre-tokenization and sliding-window chunking.

    Short clauses (≤ max_length tokens) are batch-tokenized upfront — zero
    CPU work per batch during training.

    Long clauses (> max_length tokens) are split into overlapping chunks of
    max_length tokens (stride = max_length // 2). Each chunk inherits the
    parent clause's labels. This ensures the model sees the full text of long
    clauses including exceptions, limitations, and conditions that often appear
    beyond the first max_length tokens.

    Construction is a one-time cost. After that __getitem__ is pure tensor
    indexing.

    Args:
        texts: List of clause text strings.
        labels: List of multi-hot label vectors (length = NUM_CLAUSE_TYPES).
        tokenizer: HuggingFace tokenizer instance.
        max_length: Maximum token sequence length per chunk.
        stride: Overlap between consecutive chunks (tokens).
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[List[int]],
        tokenizer: AutoTokenizer,
        max_length: int = config.CLASSIFIER_MAX_LENGTH,
        stride: int = 128,
    ):
        logger.info(f"Pre-tokenizing {len(texts):,} texts (runs once)...")

        cls_id    = tokenizer.cls_token_id
        sep_id    = tokenizer.sep_token_id
        pad_id    = tokenizer.pad_token_id
        max_chunk = max_length - 2  # reserve 2 positions for [CLS] and [SEP]

        # ── Step 1: tokenize all texts without padding to get lengths ─────
        raw = tokenizer(
            texts,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )
        raw_ids = raw["input_ids"]  # list of variable-length lists

        # ── Step 2: partition into short and long ─────────────────────────
        short_texts:  List[str]       = []
        short_labels: List[List[int]] = []
        long_ids_list:   List[List[int]]       = []
        long_labels_list: List[List[int]]      = []

        for i, ids in enumerate(raw_ids):
            if len(ids) <= max_length:
                short_texts.append(texts[i])
                short_labels.append(labels[i])
            else:
                long_ids_list.append(ids)
                long_labels_list.append(labels[i])

        # ── Step 3: batch-tokenize short texts with padding (fast) ────────
        parts_ids:   List[torch.Tensor] = []
        parts_masks: List[torch.Tensor] = []
        parts_lbls:  List[torch.Tensor] = []

        if short_texts:
            enc = tokenizer(
                short_texts,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            parts_ids.append(enc["input_ids"])
            parts_masks.append(enc["attention_mask"])
            parts_lbls.append(torch.tensor(short_labels, dtype=torch.float32))

        # ── Step 4: sliding-window chunk long texts ────────────────────────
        chunk_ids:   List[List[int]] = []
        chunk_masks: List[List[int]] = []
        chunk_lbls:  List[List[int]] = []
        n_long_chunks = 0

        for ids, label in zip(long_ids_list, long_labels_list):
            body  = ids[1:-1]  # strip [CLS] and [SEP] added by tokenizer
            start = 0
            while start < len(body):
                chunk  = body[start : start + max_chunk]
                seq    = [cls_id] + chunk + [sep_id]
                pad_n  = max_length - len(seq)
                chunk_ids.append(seq + [pad_id] * pad_n)
                chunk_masks.append([1] * len(seq) + [0] * pad_n)
                chunk_lbls.append(label)
                n_long_chunks += 1
                next_start = start + max_chunk - stride
                if next_start >= len(body):
                    break
                start = next_start

        if chunk_ids:
            parts_ids.append(torch.tensor(chunk_ids,   dtype=torch.long))
            parts_masks.append(torch.tensor(chunk_masks, dtype=torch.long))
            parts_lbls.append(torch.tensor(chunk_lbls,  dtype=torch.float32))

        self.input_ids      = torch.cat(parts_ids,   dim=0)
        self.attention_mask = torch.cat(parts_masks, dim=0)
        self.labels         = torch.cat(parts_lbls,  dim=0)

        logger.info(
            f"Pre-tokenization complete. "
            f"{len(texts):,} clauses → {len(self.labels):,} training examples "
            f"({len(long_ids_list):,} long clauses expanded into "
            f"{n_long_chunks:,} chunks via sliding window, stride={stride})."
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels":         self.labels[idx],
        }


# ---------------------------------------------------------------------------
# 2. MODEL
# ---------------------------------------------------------------------------

class ClauseClassifierModel(nn.Module):
    """Standard backbone classifier using AutoModelForSequenceClassification.

    Used for Legal-BERT and any backbone where the default pooler is stable.

    Args:
        model_name: HuggingFace model identifier or local checkpoint path.
        num_labels: Number of output labels (default: NUM_CLAUSE_TYPES).
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int = config.NUM_CLAUSE_TYPES,
    ):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            problem_type="multi_label_classification",
            ignore_mismatched_sizes=True,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return raw logits (batch, num_labels)."""
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


class MeanPoolClassifierModel(nn.Module):
    """Mean-pooling classifier — used for DeBERTa-v3.

    DeBERTa-v3's AutoModelForSequenceClassification pooler is unstable at
    the low learning rates required to avoid NaN (LR ≤ 3e-6). Mean-pooling
    the last hidden states produces a more stable representation for the
    classification head.

    Architecture:
        DeBERTa encoder → mean-pool last hidden states over non-pad tokens
        → LayerNorm → Dropout(0.1) → Linear(hidden, num_labels)

    Args:
        model_name: HuggingFace model identifier or local checkpoint path.
        num_labels: Number of output labels.
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int = config.NUM_CLAUSE_TYPES,
    ):
        super().__init__()
        from transformers import AutoConfig, AutoModel
        cfg = AutoConfig.from_pretrained(model_name, ignore_mismatched_sizes=True)
        self.encoder   = AutoModel.from_pretrained(model_name, config=cfg)
        hidden         = cfg.hidden_size
        self.norm      = nn.LayerNorm(hidden)
        self.dropout   = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden, num_labels)
        # Initialise head to small values — prevents large logits on epoch 1
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return raw logits (batch, num_labels) via mean-pooled encoder output."""
        outputs     = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden      = outputs.last_hidden_state          # (B, T, H)
        # Mask padding tokens before mean-pooling
        mask_exp    = attention_mask.unsqueeze(-1).float()
        summed      = (hidden * mask_exp).sum(dim=1)
        lengths     = mask_exp.sum(dim=1).clamp(min=1e-9)
        pooled      = summed / lengths                   # (B, H)
        pooled      = self.dropout(self.norm(pooled))
        return self.classifier(pooled)                   # (B, num_labels)

    def save_pretrained(self, save_dir: str) -> None:
        """Save encoder + head weights in a HuggingFace-compatible layout."""
        import os, json as _json
        os.makedirs(save_dir, exist_ok=True)
        self.encoder.save_pretrained(save_dir)
        torch.save(
            {"norm": self.norm.state_dict(),
             "dropout": self.dropout.state_dict(),
             "classifier": self.classifier.state_dict()},
            os.path.join(save_dir, "head_weights.pt"),
        )

    @classmethod
    def from_pretrained(cls, save_dir: str, num_labels: int = config.NUM_CLAUSE_TYPES):
        """Load a saved MeanPoolClassifierModel checkpoint."""
        import os
        model = cls(save_dir, num_labels=num_labels)
        head_path = os.path.join(save_dir, "head_weights.pt")
        if os.path.exists(head_path):
            state = torch.load(head_path, map_location="cpu")
            model.norm.load_state_dict(state["norm"])
            model.classifier.load_state_dict(state["classifier"])
        return model


def build_model(backbone: str, model_name: str) -> nn.Module:
    """Instantiate the correct model class for a given backbone.

    Args:
        backbone: 'legalbert' or 'deberta'.
        model_name: HuggingFace identifier or local checkpoint path.

    Returns:
        Initialised model (ClauseClassifierModel or MeanPoolClassifierModel).
    """
    if BACKBONE_USE_MEAN_POOL.get(backbone, False):
        return MeanPoolClassifierModel(model_name)
    return ClauseClassifierModel(model_name)


# ---------------------------------------------------------------------------
# Asymmetric Loss for multi-label classification
# ---------------------------------------------------------------------------

class AsymmetricLoss(nn.Module):
    """Asymmetric Loss (ASL) for multi-label classification.

    Reference: Ben-Baruch et al. (2021) "Asymmetric Loss For Multi-Label
    Classification" (ICCV 2021).

    ASL addresses the extreme positive/negative imbalance in multi-label
    problems by applying:
      - A higher focusing parameter (gamma_neg) to easy negatives to
        down-weight their contribution
      - A lower focusing parameter (gamma_pos) to positives to preserve
        gradient from rare labels
      - An optional probability margin (clip) that shifts negative
        probabilities by 'm', effectively ignoring very easy negatives

    Default parameters follow the paper's recommended values for
    fine-grained multi-label classification.

    Args:
        gamma_neg: Focusing exponent for negative samples (default 4).
        gamma_pos: Focusing exponent for positive samples (default 1).
        clip: Probability shift for easy-negative clipping (default 0.05).
        reduction: 'mean' or 'sum'.
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip      = clip
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute ASL.

        Args:
            logits: Raw model logits (batch, num_labels).
            targets: Soft or hard targets in [0, 1] (batch, num_labels).

        Returns:
            Scalar loss.
        """
        probs    = torch.sigmoid(logits)
        probs_m  = probs                                 # negative branch

        if self.clip is not None and self.clip > 0:
            probs_m = (probs_m + self.clip).clamp(max=1)

        # Binary cross-entropy for both branches
        loss_pos = targets       * torch.log(probs.clamp(min=1e-8))
        loss_neg = (1 - targets) * torch.log((1 - probs_m).clamp(min=1e-8))

        # Asymmetric focusing
        with torch.no_grad():
            pt0 = probs_m.detach()
            pt1 = probs.detach()
            pt  = pt1 * targets + pt0 * (1 - targets)
            w_p = torch.pow(1 - pt1, self.gamma_pos)
            w_n = torch.pow(pt0,     self.gamma_neg)
            w   = w_p * targets + w_n * (1 - targets)

        loss = -w * (loss_pos + loss_neg)

        return loss.mean() if self.reduction == "mean" else loss.sum()


# Backward-compatible alias so any existing imports keep working.
LegalBERTClassifier = ClauseClassifierModel


# ---------------------------------------------------------------------------
# 3. DATA LOADING UTILITIES
# ---------------------------------------------------------------------------

def _load_split(split_name: str) -> Tuple[List[str], List[List[int]]]:
    """Load a train/val/test CSV and convert clause_type to multi-hot vectors.

    Args:
        split_name: One of 'train', 'val', 'test'.

    Returns:
        Tuple of (texts, labels) where labels are multi-hot integer lists.

    Raises:
        FileNotFoundError: If the split CSV does not exist.
    """
    csv_path = config.PROCESSED_DIR / f"{split_name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{split_name}.csv not found. Run data_pipeline.py --mode split first."
        )
    df     = pd.read_csv(csv_path)
    texts  = df["clause_text"].fillna("").tolist()
    labels = [_encode_labels(row) for row in df["clause_type"].fillna("").tolist()]
    return texts, labels


def _encode_labels(clause_type_str: str) -> List[int]:
    """Convert a pipe-separated clause type string to a multi-hot vector.

    Args:
        clause_type_str: e.g. 'Indemnification|Governing Law'.

    Returns:
        Multi-hot integer list of length NUM_CLAUSE_TYPES.
    """
    vec = [0] * config.NUM_CLAUSE_TYPES
    if not clause_type_str.strip():
        return vec
    type_to_idx = {ct: i for i, ct in enumerate(config.CUAD_CLAUSE_TYPES)}
    for ct in clause_type_str.split("|"):
        ct = ct.strip()
        if ct in type_to_idx:
            vec[type_to_idx[ct]] = 1
    return vec




def _build_optimizer_param_groups(
    hf_model: nn.Module,
    learning_rate: float,
    no_decay: List[str],
    head_mult: int = 10,
) -> List[Dict]:
    """Build backbone-agnostic optimizer parameter groups.

    Works for any HuggingFace model regardless of internal attribute names
    (bert, deberta, roberta, etc.) by identifying the classification head
    via standard attribute names and treating everything else as the encoder.

    Applies:
      - head_mult × LR to the classification head (10 for Legal-BERT, 1 for DeBERTa)
      - Weight decay only to non-bias / non-LayerNorm encoder params

    Args:
        hf_model: The AutoModelForSequenceClassification instance.
        learning_rate: Base learning rate for encoder layers.
        no_decay: Parameter name substrings that should have zero weight decay.
        head_mult: LR multiplier for the classification head. Use 1 for DeBERTa
            (any higher multiplier causes NaN during DeBERTa training).

    Returns:
        List of param group dicts for AdamW.
    """
    all_named = list(hf_model.named_parameters())

    head_ids: set = set()
    for attr in ("classifier", "pooler"):
        sub = getattr(hf_model, attr, None)
        if sub is not None:
            for p in sub.parameters():
                head_ids.add(id(p))

    enc_decay    = [p for n, p in all_named if id(p) not in head_ids and not any(nd in n for nd in no_decay)]
    enc_no_decay = [p for n, p in all_named if id(p) not in head_ids and     any(nd in n for nd in no_decay)]
    head_params  = [p for n, p in all_named if id(p) in head_ids]

    return [
        {"params": enc_decay,    "lr": learning_rate,             "weight_decay": 0.01},
        {"params": enc_no_decay, "lr": learning_rate,             "weight_decay": 0.0},
        {"params": head_params,  "lr": learning_rate * head_mult, "weight_decay": 0.01},
    ]


# ---------------------------------------------------------------------------
# 4. TRAINER
# ---------------------------------------------------------------------------

class ClauseClassifierTrainer:
    """Manages the full training loop for a single classifier backbone.

    Both Legal-BERT and DeBERTa use identical training logic:
      - AsymmetricLoss (gamma_neg=4, gamma_pos=1, clip=0.05)
      - AdamW with backbone-agnostic layer-wise LR
      - Linear warmup + linear decay scheduler
      - Gradient accumulation (effective batch = BATCH_SIZE × 2)
      - Best checkpoint saved by validation F1 macro
      - Per-backbone early stopping patience (legalbert=6, deberta=8)
      - Min epochs before early stopping (legalbert=10, deberta=20)
      - Auto-resume: saves training_state.json + training_state.pt after every
        epoch; if these files exist at startup, training resumes automatically
        from the next epoch with optimizer and scheduler state fully restored.

    Args:
        backbone: Either 'legalbert' or 'deberta'.
    """

    def __init__(self, backbone: str = "legalbert"):
        if backbone not in BACKBONE_MODEL_MAP:
            raise ValueError(
                f"backbone must be one of {list(BACKBONE_MODEL_MAP)}, got '{backbone}'"
            )
        self.backbone   = backbone
        self.model_name = BACKBONE_MODEL_MAP[backbone]
        self.ckpt_dir   = BACKBONE_DIR_MAP[backbone]
        self.device     = "cuda" if torch.cuda.is_available() else "cpu"

        # Per-backbone hyperparameters
        self.lr            = BACKBONE_LR_MAP[backbone]
        self.epochs        = BACKBONE_EPOCHS_MAP[backbone]
        self.head_mult     = BACKBONE_HEAD_MULT_MAP[backbone]
        self.adam_eps      = BACKBONE_ADAM_EPS_MAP[backbone]
        self.clip_norm     = BACKBONE_GRAD_CLIP_MAP[backbone]
        self.patience      = BACKBONE_PATIENCE_MAP[backbone]
        self.min_epochs    = BACKBONE_MIN_EPOCHS_MAP[backbone]
        self.batch_size    = BACKBONE_BATCH_SIZE_MAP[backbone]
        self.grad_accum    = BACKBONE_GRAD_ACCUM_MAP[backbone]
        # label_smooth removed — ASL handles noisy labels via gamma focusing;
        # soft targets (label smoothing) cause fp16 log overflow → Train Loss: inf

        logger.info(
            f"Backbone: {backbone} ({self.model_name}) | "
            f"lr={self.lr}  epochs={self.epochs}  eps={self.adam_eps}  clip={self.clip_norm}"
        )
        logger.info(f"Training device: {self.device}")

        # Legal-BERT requires use_fast=False in certain environments
        use_fast = (backbone != "legalbert")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=use_fast)
        self.model: Optional[ClauseClassifierModel] = None

    def train(self) -> Dict:
        """Run the full training pipeline and return final evaluation metrics.

        Returns:
            Dict with 'backbone', 'best_val_f1', 'best_epoch', 'epochs_trained',
            and per-epoch 'history'.
        """
        state_json = self.ckpt_dir / "training_state.json"
        state_pt   = self.ckpt_dir / "training_state.pt"
        best_dir   = self.ckpt_dir / "best"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # ── Auto-resume detection ──────────────────────────────────────────
        resuming = state_json.exists() and best_dir.exists()
        if resuming:
            saved       = json.loads(state_json.read_text())
            start_epoch = saved["epoch"] + 1
            best_val_f1 = saved["best_val_f1"]
            best_epoch  = saved["best_epoch"]
            no_improve  = saved["no_improve"]
            history     = saved["history"]
            logger.info(
                f"[{self.backbone}] RESUMING from epoch {start_epoch} "
                f"(best F1={best_val_f1:.4f} at epoch {best_epoch})"
            )
            use_fast = (self.backbone != "legalbert")
            self.tokenizer = AutoTokenizer.from_pretrained(str(best_dir), use_fast=use_fast)
            # MeanPoolClassifierModel has a custom from_pretrained that reloads
            # the head weights stored in head_weights.pt alongside the encoder.
            if BACKBONE_USE_MEAN_POOL.get(self.backbone, False):
                self.model = MeanPoolClassifierModel.from_pretrained(str(best_dir)).to(self.device)
            else:
                self.model = ClauseClassifierModel(str(best_dir)).to(self.device)
        else:
            start_epoch = 1
            best_val_f1 = 0.0
            best_epoch  = 0
            no_improve  = 0
            history: List[Dict] = []
            logger.info(f"[{self.backbone}] Loading training and validation data...")
            self.model = build_model(self.backbone, self.model_name).to(self.device)
            # Gradient checkpointing: recompute activations during backward instead
            # of storing them — cuts activation memory ~60% for large models.
            # Only enabled for legalroberta-large which is VRAM-bound on 8GB.
            if self.backbone == "legalroberta":
                if hasattr(self.model, "model"):
                    self.model.model.gradient_checkpointing_enable()
                elif hasattr(self.model, "encoder"):
                    self.model.encoder.gradient_checkpointing_enable()

        train_texts, train_labels = _load_split("train")
        val_texts,   val_labels   = _load_split("val")

        # Training: ClauseDataset chunks long clauses into overlapping windows.
        # Each chunk is an independent training example — correct for training.
        train_dataset = ClauseDataset(train_texts, train_labels, self.tokenizer)
        train_loader  = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(self.device == "cuda"),
        )
        # Validation: raw texts + labels kept separately so _evaluate_aggregated
        # can run sliding window per clause and max-pool predictions back into
        # one prediction per original clause — matching the inference path exactly.

        # AsymmetricLoss (ASL) handles positive/negative imbalance via asymmetric
        # focusing (gamma_neg > gamma_pos) — replaces BCEWithLogitsLoss+pos_weight.
        criterion = AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0, clip=0.05)

        no_decay = ["bias", "LayerNorm.weight"]
        # MeanPoolClassifierModel exposes the head directly on self.model;
        # ClauseClassifierModel wraps AutoModelForSequenceClassification at self.model.model.
        hf_model_for_params = (
            self.model
            if BACKBONE_USE_MEAN_POOL.get(self.backbone, False)
            else self.model.model
        )
        optimizer_params = _build_optimizer_param_groups(
            hf_model_for_params, self.lr, no_decay, head_mult=self.head_mult
        )
        optimizer = torch.optim.AdamW(optimizer_params, eps=self.adam_eps)

        total_steps = (
            len(train_loader) // self.grad_accum
        ) * self.epochs

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.CLASSIFIER_WARMUP_STEPS,
            num_training_steps=total_steps,
        )

        # Restore optimiser + scheduler state if resuming
        if resuming and state_pt.exists():
            pt = torch.load(str(state_pt), map_location=self.device)
            optimizer.load_state_dict(pt["optimizer"])
            scheduler.load_state_dict(pt["scheduler"])
            logger.info(f"[{self.backbone}] Optimizer and scheduler state restored.")

        for epoch in range(start_epoch, self.epochs + 1):
            train_loss = self._train_epoch(train_loader, criterion, optimizer, scheduler, epoch)

            # NaN abort: _train_epoch returns float('nan') when >50 NaN batches hit.
            # Skip val evaluation and no_improve increment — model state is corrupt for
            # this epoch; the best checkpoint from a prior epoch is still valid.
            if math.isnan(train_loss):
                logger.warning(
                    f"[{self.backbone}] Epoch {epoch} aborted (NaN) — "
                    "skipping val eval, no_improve unchanged."
                )
                history.append({"epoch": epoch, "train_loss": train_loss,
                                 "f1_macro": float("nan"), "precision_macro": float("nan"),
                                 "recall_macro": float("nan")})
                continue

            val_metrics = self._evaluate_aggregated(val_texts, val_labels)
            val_f1      = val_metrics["f1_macro"]

            logger.info(
                f"Epoch {epoch}/{self.epochs} — "
                f"Train Loss: {train_loss:.4f} | "
                f"Val F1: {val_f1:.4f} | "
                f"Val Precision: {val_metrics['precision_macro']:.4f} | "
                f"Val Recall: {val_metrics['recall_macro']:.4f}"
            )
            history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch  = epoch
                no_improve  = 0
                self._save_checkpoint(is_best=True)
                logger.info(f"  → New best checkpoint saved (F1={val_f1:.4f})")
            else:
                no_improve += 1

            # Save full training state after every epoch (enables resume after crash)
            torch.save(
                {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
                str(state_pt),
            )
            state_json.write_text(json.dumps({
                "epoch":       epoch,
                "best_val_f1": best_val_f1,
                "best_epoch":  best_epoch,
                "no_improve":  no_improve,
                "history":     history,
            }))

            if epoch >= self.min_epochs and no_improve >= self.patience:
                logger.info(
                    f"Early stopping at epoch {epoch} "
                    f"(patience={self.patience}, min_epochs={self.min_epochs})"
                )
                break

        logger.info(
            f"Training complete [{self.backbone}]. "
            f"Best val F1: {best_val_f1:.4f} at epoch {best_epoch}"
        )
        # Clean up resume files — training finished normally
        for fp in [state_json, state_pt]:
            if fp.exists():
                fp.unlink()

        return {
            "backbone":       self.backbone,
            "best_val_f1":    best_val_f1,
            "best_epoch":     best_epoch,
            "epochs_trained": len(history),
            "history":        history,
        }

    def _train_epoch(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        epoch: int,
    ) -> float:
        """Run one training epoch and return average loss.

        Args:
            loader: Training DataLoader.
            criterion: AsymmetricLoss instance.
            optimizer: AdamW optimiser.
            scheduler: Linear warmup scheduler.
            epoch: Current epoch number (for progress bar label).

        Returns:
            Average training loss over the epoch.
        """
        self.model.train()
        total_loss  = 0.0
        nan_batches = 0
        optimizer.zero_grad()

        use_amp = (self.device == "cuda")
        scaler  = torch.amp.GradScaler(device="cuda", enabled=use_amp, init_scale=2**10)

        pbar = tqdm(loader, desc=f"Epoch {epoch} [{self.backbone}]", leave=False)
        for step, batch in enumerate(pbar):
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels         = batch["labels"].to(self.device)

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits = self.model(input_ids, attention_mask)

            # NaN guard — critical for DeBERTa-v3 early in training
            if torch.isnan(logits).any():
                nan_batches += 1
                optimizer.zero_grad()
                if nan_batches > 50:
                    logger.error(
                        f"[{self.backbone}] >50 NaN batches in epoch {epoch} — aborting."
                    )
                    return float("nan")  # sentinel: caller skips val + no_improve
                continue

            # ASL loss in fp32 — log ops in ASL overflow in fp16; cast logits/labels
            # to float32 before loss computation regardless of autocast context
            loss = criterion(logits.float(), labels.float()) / self.grad_accum

            # Guard against residual NaN/inf loss
            if not torch.isfinite(loss):
                nan_batches += 1
                optimizer.zero_grad()
                if nan_batches > 50:
                    logger.error(
                        f"[{self.backbone}] >50 NaN/inf loss batches in epoch {epoch} — aborting."
                    )
                    return float("nan")
                continue

            scaler.scale(loss).backward()
            total_loss += loss.item() * self.grad_accum

            if (step + 1) % self.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            pbar.set_postfix({"loss": f"{loss.item() * self.grad_accum:.4f}"})

        if nan_batches > 0:
            logger.warning(f"[{self.backbone}] Epoch {epoch}: skipped {nan_batches} NaN batches")

        return total_loss / max(len(loader) - nan_batches, 1)

    @torch.no_grad()
    def _evaluate_aggregated(
        self,
        texts: List[str],
        labels: List[List[int]],
        batch_size: int = 64,
    ) -> Dict:
        """Evaluate on raw clause texts with sliding-window aggregation.

        Each clause — regardless of length — produces exactly ONE prediction.
        Long clauses are split into overlapping 256-token chunks, all chunks
        are run through the model, and per-class probabilities are max-pooled
        back into a single prediction per clause.

        This matches the inference path exactly, ensuring F1 is measured per
        original clause and not inflated by repeated chunk predictions.

        Args:
            texts:      Raw clause text strings.
            labels:     Ground-truth multi-hot label lists.
            batch_size: Number of chunks to process per GPU batch.

        Returns:
            Dict with f1_macro, precision_macro, recall_macro.
        """
        self.model.eval()
        use_amp   = (self.device == "cuda")
        max_len   = config.CLASSIFIER_MAX_LENGTH
        stride    = 128
        max_chunk = max_len - 2
        cls_id    = self.tokenizer.cls_token_id
        sep_id    = self.tokenizer.sep_token_id
        pad_id    = self.tokenizer.pad_token_id

        all_preds:  List[np.ndarray] = []
        all_labels: List[np.ndarray] = []

        for text, label in tqdm(
            zip(texts, labels), total=len(texts), desc="Evaluating", leave=False
        ):
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)

            if len(token_ids) <= max_len - 2:
                # Short clause — single forward pass
                enc = self.tokenizer(
                    text,
                    max_length=max_len,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).to(self.device)
                with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                    logits = self.model(enc["input_ids"], enc["attention_mask"])
                probs = torch.sigmoid(logits).float().cpu().numpy()[0]
            else:
                # Long clause — sliding window, max-pool across chunks
                chunks_ids:  List[List[int]] = []
                chunks_mask: List[List[int]] = []
                start = 0
                while start < len(token_ids):
                    chunk = token_ids[start : start + max_chunk]
                    seq   = [cls_id] + chunk + [sep_id]
                    pad_n = max_len - len(seq)
                    chunks_ids.append(seq + [pad_id] * pad_n)
                    chunks_mask.append([1] * len(seq) + [0] * pad_n)
                    next_start = start + max_chunk - stride
                    if next_start >= len(token_ids):
                        break
                    start = next_start

                chunk_probs: List[np.ndarray] = []
                for i in range(0, len(chunks_ids), batch_size):
                    ids_t  = torch.tensor(chunks_ids[i:i+batch_size],  dtype=torch.long).to(self.device)
                    mask_t = torch.tensor(chunks_mask[i:i+batch_size], dtype=torch.long).to(self.device)
                    with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                        logits = self.model(ids_t, mask_t)
                    chunk_probs.append(torch.sigmoid(logits).float().cpu().numpy())

                probs = np.max(np.vstack(chunk_probs), axis=0)  # max-pool across chunks

            all_preds.append((probs >= 0.5).astype(int))
            all_labels.append(label)

        y_pred = np.vstack(all_preds)
        y_true = np.vstack(all_labels)

        return {
            "f1_macro":        f1_score(y_true, y_pred, average="macro", zero_division=0),
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_macro":    recall_score(y_true, y_pred, average="macro", zero_division=0),
        }

    def _save_checkpoint(self, is_best: bool = False) -> None:
        """Save model weights and tokeniser to the backbone-specific checkpoint dir.

        Args:
            is_best: If True, saves to backbone/best/; otherwise backbone/last/.
        """
        save_dir = self.ckpt_dir / ("best" if is_best else "last")
        save_dir.mkdir(parents=True, exist_ok=True)
        # MeanPoolClassifierModel has a custom save_pretrained that also writes
        # head_weights.pt alongside the encoder files.
        if BACKBONE_USE_MEAN_POOL.get(self.backbone, False):
            self.model.save_pretrained(str(save_dir))
        else:
            self.model.model.save_pretrained(str(save_dir))
        self.tokenizer.save_pretrained(str(save_dir))
        logger.debug(f"Checkpoint saved to {save_dir}")


# ---------------------------------------------------------------------------
# 5. MODEL SELECTOR
# ---------------------------------------------------------------------------

class ModelSelector:
    """Selects the best production model and optimises per-class thresholds.

    Evaluation is performed exclusively on the validation set.
    The test set is never touched during selection.

    Process:
      1. Load available checkpoints (Legal-BERT, DeBERTa, or both).
      2. Compute sigmoid probabilities on the full validation set.
      3. Search ensemble weights 0.0–1.0 (step 0.1) when both are available.
      4. For each candidate, optimise per-class F1 thresholds on val.
      5. Pick the winner by highest macro F1 on val.
      6. Save production_config.json for deterministic inference.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def select(self) -> Dict:
        """Run full model selection and save production config.

        Returns:
            Saved production config dict.
        """
        val_texts, val_labels = _load_split("val")
        y_true = np.array(val_labels, dtype=np.float32)

        # --- Collect probabilities from available checkpoints ---------------
        available: Dict[str, np.ndarray] = {}
        for backbone in ("legalbert", "deberta", "legalroberta"):
            ckpt = BACKBONE_DIR_MAP[backbone] / "best"
            if not ckpt.exists():
                logger.info(f"No checkpoint for {backbone} — skipping")
                continue
            logger.info(f"Computing val probabilities: {backbone}")
            available[backbone] = self._get_probs(backbone, val_texts)

        if not available:
            raise RuntimeError(
                "No trained checkpoints found. Run --mode train --backbone <name> first."
            )

        # --- Evaluate each single backbone ----------------------------------
        results: Dict[str, Dict] = {}

        for backbone, probs in available.items():
            temperature = self._find_temperature(probs, y_true)
            cal_probs   = self._calibrate(probs, temperature)
            thresholds  = self._optimise_thresholds(cal_probs, y_true)
            f1 = f1_score(y_true, cal_probs >= thresholds, average="macro", zero_division=0)
            results[backbone] = {
                "thresholds":  thresholds,
                "probs":       probs,
                "temperature": temperature,
                "f1":          f1,
                "w_lb":        1.0 if backbone == "legalbert"    else 0.0,
                "w_db":        1.0 if backbone == "deberta"      else 0.0,
                "w_lr":        1.0 if backbone == "legalroberta" else 0.0,
            }
            logger.info(
                f"  {backbone} val F1 (T={temperature:.1f}, per-class thr): {f1:.4f}"
            )

        # --- Evaluate all ensemble combinations when ≥2 checkpoints exist --
        # Weight search step = 0.1; all weights ≥ 0 and sum to 1.0.
        # For 2-model ensembles: w_a ∈ {0.0, 0.1, …, 1.0}, w_b = 1 - w_a.
        # For 3-model ensemble:  w_lb, w_db ∈ grid with w_lb+w_db ≤ 1,
        #                        w_lr = 1 - w_lb - w_db.
        weight_steps = [round(x * 0.1, 1) for x in range(0, 11)]

        def _eval_combo(probs_combined: np.ndarray) -> Tuple[float, np.ndarray, float]:
            """Return (f1, thresholds, temperature) for a blended prob array."""
            temp = self._find_temperature(probs_combined, y_true)
            cal  = self._calibrate(probs_combined, temp)
            thr  = self._optimise_thresholds(cal, y_true)
            f1   = f1_score(y_true, cal >= thr, average="macro", zero_division=0)
            return f1, thr, temp

        # 2-model pairs
        pairs = [
            ("legalbert",    "deberta",      "w_lb", "w_db"),
            ("legalbert",    "legalroberta", "w_lb", "w_lr"),
            ("deberta",      "legalroberta", "w_db", "w_lr"),
        ]
        for bk_a, bk_b, field_a, field_b in pairs:
            if bk_a not in available or bk_b not in available:
                continue
            best_f1, best_wa = -1.0, 0.5
            for wa in weight_steps:
                wb = round(1.0 - wa, 1)
                combined = wa * available[bk_a] + wb * available[bk_b]
                f1, _, _ = _eval_combo(combined)
                if f1 > best_f1:
                    best_f1, best_wa = f1, wa
            best_wb = round(1.0 - best_wa, 1)
            combined = best_wa * available[bk_a] + best_wb * available[bk_b]
            f1, thr, temp = _eval_combo(combined)
            key = f"ensemble_{bk_a}_{bk_b}"
            results[key] = {
                "thresholds":  thr,
                "probs":       combined,
                "temperature": temp,
                "f1":          f1,
                "w_lb":        best_wa if bk_a == "legalbert"    else (best_wb if bk_b == "legalbert"    else 0.0),
                "w_db":        best_wa if bk_a == "deberta"      else (best_wb if bk_b == "deberta"      else 0.0),
                "w_lr":        best_wa if bk_a == "legalroberta" else (best_wb if bk_b == "legalroberta" else 0.0),
            }
            logger.info(
                f"  {key} ({field_a}={best_wa:.1f}, {field_b}={best_wb:.1f}) val F1: {f1:.4f}"
            )

        # 3-model ensemble
        if all(b in available for b in ("legalbert", "deberta", "legalroberta")):
            best_f1 = -1.0
            best_weights = (0.34, 0.33, 0.33)
            for w_lb in weight_steps:
                for w_db in weight_steps:
                    w_lr = round(1.0 - w_lb - w_db, 1)
                    if w_lr < 0:
                        continue
                    combined = (
                        w_lb * available["legalbert"] +
                        w_db * available["deberta"] +
                        w_lr * available["legalroberta"]
                    )
                    f1, _, _ = _eval_combo(combined)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_weights = (w_lb, w_db, w_lr)
            w_lb, w_db, w_lr = best_weights
            combined = (
                w_lb * available["legalbert"] +
                w_db * available["deberta"] +
                w_lr * available["legalroberta"]
            )
            f1, thr, temp = _eval_combo(combined)
            results["ensemble"] = {
                "thresholds":  thr,
                "probs":       combined,
                "temperature": temp,
                "f1":          f1,
                "w_lb":        w_lb,
                "w_db":        w_db,
                "w_lr":        w_lr,
            }
            logger.info(
                f"  ensemble (w_lb={w_lb:.1f}, w_db={w_db:.1f}, w_lr={w_lr:.1f}) val F1: {f1:.4f}"
            )

        # --- Pick winner ----------------------------------------------------
        winner_key = max(results, key=lambda k: results[k]["f1"])
        winner     = results[winner_key]
        temperature = winner["temperature"]
        logger.info(
            f"Winner: {winner_key}  (val F1 = {winner['f1']:.4f}, T={temperature:.1f})"
        )

        prod_config = {
            "model_type":           winner_key,
            "weight_legalbert":     winner["w_lb"],
            "weight_deberta":       winner["w_db"],
            "weight_legalroberta":  winner.get("w_lr", 0.0),
            "thresholds":           {
                ct: float(winner["thresholds"][i])
                for i, ct in enumerate(config.CUAD_CLAUSE_TYPES)
            },
            "temperature":          temperature,
            "unknown_threshold":    config.UNKNOWN_PROB_THRESHOLD,
            "val_f1_macro":         round(winner["f1"], 6),
        }

        config.CLASSIFIER_DIR.mkdir(parents=True, exist_ok=True)
        with open(config.CLASSIFIER_PRODUCTION_CONFIG, "w") as f:
            json.dump(prod_config, f, indent=2)
        logger.info(f"Production config saved → {config.CLASSIFIER_PRODUCTION_CONFIG}")
        return prod_config

    def _get_probs(self, backbone: str, texts: List[str]) -> np.ndarray:
        """Load a backbone checkpoint and return sigmoid probs for all texts.

        Args:
            backbone: 'legalbert' or 'deberta'.
            texts: List of clause strings.

        Returns:
            Array of shape (N, NUM_CLAUSE_TYPES).
        """
        ckpt_dir  = BACKBONE_DIR_MAP[backbone] / "best"
        tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir))
        use_mean_pool = BACKBONE_USE_MEAN_POOL.get(backbone, False)
        if use_mean_pool:
            model = MeanPoolClassifierModel.from_pretrained(str(ckpt_dir)).to(self.device)
        else:
            model = ClauseClassifierModel(str(ckpt_dir)).to(self.device)
        model.eval()

        all_probs: List[np.ndarray] = []
        batch_size = config.CLASSIFIER_BATCH_SIZE * 2

        with torch.no_grad():
            for start in tqdm(
                range(0, len(texts), batch_size),
                desc=f"  {backbone} probs",
                leave=False,
            ):
                batch   = texts[start : start + batch_size]
                encoded = tokenizer(
                    batch,
                    max_length=config.CLASSIFIER_MAX_LENGTH,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.device)
                # Both model classes take (input_ids, attention_mask) and return
                # raw logits directly — no .logits attribute access needed.
                logits = model(encoded["input_ids"], encoded["attention_mask"])
                all_probs.append(torch.sigmoid(logits).cpu().numpy())

        return np.vstack(all_probs)

    @staticmethod
    def _optimise_thresholds(probs: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Find the per-class threshold that maximises per-class F1.

        Searches thresholds in [0.1, 0.2, ..., 0.9] independently for each
        of the 100 classes. Falls back to 0.5 if no positive examples exist.

        Args:
            probs: Probability array (N, NUM_CLAUSE_TYPES).
            y_true: Ground-truth multi-hot array (N, NUM_CLAUSE_TYPES).

        Returns:
            Threshold array of shape (NUM_CLAUSE_TYPES,).
        """
        search     = [round(t * 0.1, 1) for t in range(1, 10)]
        thresholds = np.full(config.NUM_CLAUSE_TYPES, 0.5)

        for i in range(config.NUM_CLAUSE_TYPES):
            if y_true[:, i].sum() == 0:
                continue  # no positives — leave at 0.5
            best_f1 = -1.0
            for t in search:
                preds = (probs[:, i] >= t).astype(int)
                f1 = f1_score(y_true[:, i], preds, average="binary", zero_division=0)
                if f1 > best_f1:
                    best_f1        = f1
                    thresholds[i]  = t

        return thresholds

    @staticmethod
    def _find_temperature(probs: np.ndarray, y_true: np.ndarray) -> float:
        """Find temperature T that minimises binary cross-entropy on the val set.

        Temperature scaling: calibrated_prob = sigmoid(logit(p) / T)
          T > 1  →  softens probabilities (model was overconfident)
          T < 1  →  sharpens probabilities (model was underconfident)
          T = 1  →  no change (already well-calibrated)

        Searches T ∈ {0.5, 0.6, …, 2.0} and picks the value that minimises
        mean binary cross-entropy across all classes and all val examples.

        Args:
            probs: Sigmoid probability array (N, NUM_CLAUSE_TYPES).
            y_true: Ground-truth multi-hot array (N, NUM_CLAUSE_TYPES).

        Returns:
            Optimal temperature scalar (float).
        """
        import torch.nn.functional as F

        probs_clipped = np.clip(probs, 1e-7, 1 - 1e-7)
        logits_t = torch.tensor(
            np.log(probs_clipped / (1 - probs_clipped)), dtype=torch.float32
        )
        labels_t = torch.tensor(y_true, dtype=torch.float32)

        best_T, best_nll = 1.0, float("inf")
        for T in [round(t * 0.1, 1) for t in range(5, 21)]:  # 0.5 … 2.0
            nll = F.binary_cross_entropy_with_logits(
                logits_t / T, labels_t
            ).item()
            if nll < best_nll:
                best_nll, best_T = nll, T

        return best_T

    @staticmethod
    def _calibrate(probs: np.ndarray, temperature: float) -> np.ndarray:
        """Apply temperature scaling to a probability array (numpy).

        Args:
            probs: Raw probability array, any shape.
            temperature: Temperature scalar T.

        Returns:
            Calibrated probability array, same shape.
        """
        if temperature == 1.0:
            return probs
        clipped = np.clip(probs, 1e-7, 1 - 1e-7)
        logits  = np.log(clipped / (1 - clipped))
        return 1.0 / (1.0 + np.exp(-logits / temperature))


# ---------------------------------------------------------------------------
# 6. INFERENCE
# ---------------------------------------------------------------------------

class ClauseClassifierInference:
    """Production inference using saved production_config.json.

    Applies in order for each clause:
      1. Low-confidence check: if max(probs) < unknown_threshold → "Other"
      2. Per-class thresholds from production config
      3. No-label fallback: if nothing passes thresholds → "Other"
      4. Return multi-label list (or ["Other"])

    Long clauses (>CLASSIFIER_MAX_LENGTH tokens) are handled via
    sliding window with max-pooling aggregation.

    Args:
        checkpoint_dir: Legacy override — loads this directory directly as
            a Legal-BERT model, bypassing production config. Preserves
            backward compatibility with any existing callers.
    """

    def __init__(self, checkpoint_dir: Optional[Path] = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # ------------------------------------------------------------------
        # Resolve production config vs legacy direct-checkpoint path
        # ------------------------------------------------------------------
        if checkpoint_dir is not None:
            # Legacy path: treat as a single Legal-BERT checkpoint
            self.model_type  = "legalbert"
            self.w_lb        = 1.0
            self.w_db        = 0.0
            self.w_lr        = 0.0
            self.thresholds  = np.full(config.NUM_CLAUSE_TYPES, config.CLASSIFIER_THRESHOLD)
            self.temperature = 1.0
            self.unknown_thr = config.UNKNOWN_PROB_THRESHOLD
            self._load_backbone("legalbert", checkpoint_dir)
            return

        if config.CLASSIFIER_PRODUCTION_CONFIG.exists():
            with open(config.CLASSIFIER_PRODUCTION_CONFIG) as f:
                prod = json.load(f)
            self.model_type  = prod["model_type"]
            self.w_lb        = prod.get("weight_legalbert",    0.0)
            self.w_db        = prod.get("weight_deberta",      0.0)
            self.w_lr        = prod.get("weight_legalroberta", 0.0)
            thr = prod["thresholds"]
            if isinstance(thr, dict):
                self.thresholds = np.array([thr[ct] for ct in config.CUAD_CLAUSE_TYPES])
            else:
                self.thresholds = np.array(thr)
            self.temperature = float(prod.get("temperature", 1.0))
            self.unknown_thr = prod.get("unknown_threshold", config.UNKNOWN_PROB_THRESHOLD)
            logger.info(
                f"Loaded production config: {self.model_type} "
                f"(val F1 = {prod.get('val_f1_macro', 'n/a')})"
            )
        else:
            # Fallback: Legal-BERT best checkpoint with default thresholds
            logger.warning(
                "No production_config.json found. "
                "Run --mode select after training to optimise thresholds. "
                "Falling back to Legal-BERT best checkpoint with threshold=0.5."
            )
            self.model_type  = "legalbert"
            self.w_lb        = 1.0
            self.w_db        = 0.0
            self.w_lr        = 0.0
            self.thresholds  = np.full(config.NUM_CLAUSE_TYPES, config.CLASSIFIER_THRESHOLD)
            self.temperature = 1.0
            self.unknown_thr = config.UNKNOWN_PROB_THRESHOLD

        # ------------------------------------------------------------------
        # Load required backbone(s)
        # ------------------------------------------------------------------
        self.model_lb: Optional[nn.Module] = None
        self.tok_lb:   Optional[AutoTokenizer] = None
        self.model_db: Optional[nn.Module] = None
        self.tok_db:   Optional[AutoTokenizer] = None
        self.model_lr: Optional[nn.Module] = None
        self.tok_lr:   Optional[AutoTokenizer] = None

        if self.w_lb > 0:
            ckpt = config.CLASSIFIER_LEGALBERT_DIR / "best"
            if not ckpt.exists():
                raise FileNotFoundError(
                    f"Legal-BERT checkpoint not found at {ckpt}. "
                    "Run: python src/clause_classifier.py --mode train --backbone legalbert"
                )
            self._load_backbone("legalbert", ckpt)

        if self.w_db > 0:
            ckpt = config.CLASSIFIER_DEBERTA_DIR / "best"
            if not ckpt.exists():
                raise FileNotFoundError(
                    f"DeBERTa checkpoint not found at {ckpt}. "
                    "Run: python src/clause_classifier.py --mode train --backbone deberta"
                )
            self._load_backbone("deberta", ckpt)

        if self.w_lr > 0:
            ckpt = config.CLASSIFIER_LEGALROBERTA_DIR / "best"
            if not ckpt.exists():
                raise FileNotFoundError(
                    f"Legal-RoBERTa checkpoint not found at {ckpt}. "
                    "Run: python src/clause_classifier.py --mode train --backbone legalroberta"
                )
            self._load_backbone("legalroberta", ckpt)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        texts: List[str],
        threshold: Optional[float] = None,
    ) -> List[Dict]:
        """Predict clause types for a list of clause texts.

        Args:
            texts: List of clause text strings.
            threshold: Single threshold override (uses per-class if None).

        Returns:
            List of dicts, each with:
                'clause_types': list of predicted labels (may be ["Other"])
                'probabilities': dict mapping each clause type to its probability
        """
        probs_lb = self._backbone_probs("legalbert",    texts) if self.model_lb is not None else None
        probs_db = self._backbone_probs("deberta",      texts) if self.model_db is not None else None
        probs_lr = self._backbone_probs("legalroberta", texts) if self.model_lr is not None else None

        active = [
            (p, w) for p, w in (
                (probs_lb, self.w_lb),
                (probs_db, self.w_db),
                (probs_lr, self.w_lr),
            ) if p is not None and w > 0
        ]
        if not active:
            raise RuntimeError("No backbone loaded — check production_config.json weights.")
        probs = sum(w * p for p, w in active)

        probs = self._apply_temperature(probs)

        results = []
        for prob_vec in probs:
            clause_types = self._decode(prob_vec, threshold)
            prob_dict    = {
                config.CUAD_CLAUSE_TYPES[i]: float(prob_vec[i])
                for i in range(config.NUM_CLAUSE_TYPES)
            }
            results.append({"clause_types": clause_types, "probabilities": prob_dict})

        return results

    def predict_single(
        self, text: str, threshold: Optional[float] = None
    ) -> Dict:
        """Convenience wrapper for single-clause prediction.

        Args:
            text: Single clause text string.
            threshold: Optional single-value threshold override.

        Returns:
            Prediction dict with 'clause_types' and 'probabilities'.
        """
        return self.predict([text], threshold=threshold)[0]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_temperature(self, probs: np.ndarray) -> np.ndarray:
        """Apply temperature scaling using the production temperature.

        Delegates to ModelSelector._calibrate for consistent behaviour.
        """
        return ModelSelector._calibrate(probs, self.temperature)

    def _decode(
        self, prob_vec: np.ndarray, threshold_override: Optional[float]
    ) -> List[str]:
        """Apply open-set detection and thresholds to one probability vector.

        Logic (evaluated in order):
          1. max(prob) < unknown_threshold  → ["Other"]  (too uncertain overall)
          2. Apply per-class thresholds
          3. No label passed               → ["Other"]  (no confident prediction)
          4. Return the list of passing labels

        Args:
            prob_vec: Probability array of length NUM_CLAUSE_TYPES.
            threshold_override: If set, replaces per-class thresholds uniformly.

        Returns:
            List of predicted clause type strings.
        """
        # Step 1: global confidence gate
        if float(prob_vec.max()) < self.unknown_thr:
            return ["Other"]

        # Step 2: per-class (or override) thresholds
        if threshold_override is not None:
            mask = prob_vec >= threshold_override
        else:
            mask = prob_vec >= self.thresholds

        # Step 3: no-label fallback
        if not mask.any():
            return ["Other"]

        return [config.CUAD_CLAUSE_TYPES[i] for i, hit in enumerate(mask) if hit]

    def _load_backbone(self, backbone: str, ckpt_dir: Path) -> None:
        """Load tokeniser and model for one backbone into instance attributes.

        Args:
            backbone: 'legalbert' or 'deberta'.
            ckpt_dir: Path to the HuggingFace checkpoint directory.
        """
        logger.info(f"Loading {backbone} checkpoint from {ckpt_dir}")
        tok   = AutoTokenizer.from_pretrained(str(ckpt_dir))
        if BACKBONE_USE_MEAN_POOL.get(backbone, False):
            model = MeanPoolClassifierModel.from_pretrained(str(ckpt_dir)).to(self.device)
        else:
            model = ClauseClassifierModel(str(ckpt_dir)).to(self.device)
        model.eval()

        if backbone == "legalbert":
            self.tok_lb   = tok
            self.model_lb = model
        elif backbone == "deberta":
            self.tok_db   = tok
            self.model_db = model
        else:  # legalroberta
            self.tok_lr   = tok
            self.model_lr = model

    def _backbone_probs(self, backbone: str, texts: List[str]) -> np.ndarray:
        """Run a backbone over all texts, using sliding window when needed.

        Args:
            backbone: 'legalbert' or 'deberta'.
            texts: List of clause text strings.

        Returns:
            Probability array of shape (N, NUM_CLAUSE_TYPES).
        """
        if backbone == "legalbert":
            model, tok = self.model_lb, self.tok_lb
        elif backbone == "deberta":
            model, tok = self.model_db, self.tok_db
        else:
            model, tok = self.model_lr, self.tok_lr

        all_probs: List[np.ndarray] = []
        for text in texts:
            n_tokens = len(tok.encode(text, add_special_tokens=True))
            if n_tokens > config.CLASSIFIER_MAX_LENGTH:
                p = self._sliding_window(text, model, tok)
            else:
                p = self._forward_single(text, model, tok)
            all_probs.append(p)

        return np.array(all_probs)

    def _forward_single(
        self,
        text: str,
        model: nn.Module,
        tok: AutoTokenizer,
    ) -> np.ndarray:
        """Single forward pass for one clause text.

        Args:
            text: Clause string.
            model: Loaded HuggingFace model.
            tok: Corresponding tokeniser.

        Returns:
            Probability vector of length NUM_CLAUSE_TYPES.
        """
        encoded = tok(
            text,
            max_length=config.CLASSIFIER_MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).to(self.device)
        logits = model(encoded["input_ids"], encoded["attention_mask"])
        return torch.sigmoid(logits).cpu().numpy()[0]

    def _sliding_window(
        self,
        text: str,
        model: nn.Module,
        tok: AutoTokenizer,
        stride: int = 128,
    ) -> np.ndarray:
        """Sliding window inference for clauses that exceed max token length.

        Splits the token sequence into overlapping chunks (stride=128),
        runs each chunk through the model, and aggregates with max pooling
        (highest probability per class across all chunks).

        Args:
            text: Full clause text.
            model: Loaded HuggingFace model.
            tok: Corresponding tokeniser.
            stride: Overlap between consecutive windows (tokens).

        Returns:
            Aggregated probability vector of length NUM_CLAUSE_TYPES.
        """
        cls_id    = tok.cls_token_id
        sep_id    = tok.sep_token_id
        token_ids = tok.encode(text, add_special_tokens=False)
        max_chunk = config.CLASSIFIER_MAX_LENGTH - 2  # reserve [CLS] + [SEP]

        chunk_probs: List[np.ndarray] = []
        start = 0
        while start < len(token_ids):
            chunk     = token_ids[start : start + max_chunk]
            input_ids = torch.tensor([[cls_id] + chunk + [sep_id]], device=self.device)
            attn_mask = torch.ones_like(input_ids)
            logits    = model(input_ids=input_ids, attention_mask=attn_mask)
            chunk_probs.append(torch.sigmoid(logits).cpu().numpy()[0])

            next_start = start + max_chunk - stride
            if next_start >= len(token_ids):
                break
            start = next_start

        if not chunk_probs:
            return np.zeros(config.NUM_CLAUSE_TYPES)

        return np.max(chunk_probs, axis=0)


# ---------------------------------------------------------------------------
# 7. EVALUATOR
# ---------------------------------------------------------------------------

class ClauseClassifierEvaluator:
    """Evaluates production inference on the held-out test set.

    Uses the model and thresholds from production_config.json.
    The test set must not be used for any tuning — evaluation only.
    """

    def __init__(self):
        self.inference = ClauseClassifierInference()

    def evaluate(self) -> Dict:
        """Run evaluation on the test set and save metrics to evaluation_results.json.

        Returns:
            Dict with macro metrics, per-class metrics, and model_type.
        """
        test_texts, test_labels_hot = _load_split("test")
        logger.info(f"Evaluating on {len(test_texts)} test clauses...")

        predictions = self.inference.predict(test_texts)
        type_to_idx = {ct: i for i, ct in enumerate(config.CUAD_CLAUSE_TYPES)}

        pred_hot = []
        for pred in predictions:
            vec = [0] * config.NUM_CLAUSE_TYPES
            for ct in pred["clause_types"]:
                if ct in type_to_idx:
                    vec[type_to_idx[ct]] = 1
            pred_hot.append(vec)

        y_true = np.array(test_labels_hot)
        y_pred = np.array(pred_hot)

        report = classification_report(
            y_true, y_pred,
            target_names=config.CUAD_CLAUSE_TYPES,
            output_dict=True,
            zero_division=0,
        )

        results = {
            "classifier": {
                "model_type":      self.inference.model_type,
                "f1_macro":        f1_score(y_true, y_pred, average="macro", zero_division=0),
                "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
                "recall_macro":    recall_score(y_true, y_pred, average="macro", zero_division=0),
                "per_class":       report,
                "test_size":       len(test_texts),
            }
        }

        existing: Dict = {}
        if config.EVAL_RESULTS_PATH.exists():
            with open(config.EVAL_RESULTS_PATH) as f:
                existing = json.load(f)
        existing.update(results)
        with open(config.EVAL_RESULTS_PATH, "w") as f:
            json.dump(existing, f, indent=2)

        logger.info(f"Evaluation results saved → {config.EVAL_RESULTS_PATH}")
        logger.info(
            f"F1 Macro: {results['classifier']['f1_macro']:.4f}  |  "
            f"Precision: {results['classifier']['precision_macro']:.4f}  |  "
            f"Recall: {results['classifier']['recall_macro']:.4f}"
        )
        return results


# ---------------------------------------------------------------------------
# 8. CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Command-line interface for training, model selection, evaluation, and prediction."""
    parser = argparse.ArgumentParser(
        description="Legal Clause Classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/clause_classifier.py --mode train --backbone legalbert
  python src/clause_classifier.py --mode train --backbone deberta
  python src/clause_classifier.py --mode train --backbone legalroberta
  python src/clause_classifier.py --mode select
  python src/clause_classifier.py --mode evaluate
  python src/clause_classifier.py --mode predict --text "The Licensor shall..."
""",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "select", "evaluate", "predict"],
        required=True,
        help="Pipeline stage to run",
    )
    parser.add_argument(
        "--backbone",
        choices=["legalbert", "deberta", "legalroberta"],
        default="legalbert",
        help="Backbone for --mode train (default: legalbert)",
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Clause text for --mode predict",
    )
    args = parser.parse_args()

    if args.mode == "train":
        trainer = ClauseClassifierTrainer(backbone=args.backbone)
        metrics = trainer.train()
        print(f"\nTraining complete [{args.backbone}]. Best val F1: {metrics['best_val_f1']:.4f}")

    elif args.mode == "select":
        selector    = ModelSelector()
        prod_config = selector.select()
        print(f"\nModel selection complete.")
        print(f"  Winner:         {prod_config['model_type']}")
        print(f"  Val F1 (macro): {prod_config['val_f1_macro']:.4f}")
        print(f"  Config saved:   {config.CLASSIFIER_PRODUCTION_CONFIG}")

    elif args.mode == "evaluate":
        evaluator = ClauseClassifierEvaluator()
        results   = evaluator.evaluate()
        c = results["classifier"]
        print(f"\nF1 Macro:        {c['f1_macro']:.4f}")
        print(f"Precision Macro: {c['precision_macro']:.4f}")
        print(f"Recall Macro:    {c['recall_macro']:.4f}")
        print(f"Model type:      {c['model_type']}")

    elif args.mode == "predict":
        if not args.text:
            parser.error("--text is required with --mode predict")
        inference = ClauseClassifierInference()
        result    = inference.predict_single(args.text)
        print(f"\nPredicted clause types: {result['clause_types']}")
        top5 = sorted(result["probabilities"].items(), key=lambda x: x[1], reverse=True)[:5]
        print("Top 5 probabilities:")
        for ct, prob in top5:
            print(f"  {ct}: {prob:.4f}")


if __name__ == "__main__":
    main()
