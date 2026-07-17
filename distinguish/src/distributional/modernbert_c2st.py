"""Optional C2ST variant: a fine-tuned ModernBERT sequence classifier.

Same author-level GroupKFold as the linear C2ST, but each fold fine-tunes a
fresh AutoModelForSequenceClassification on the raw prompt texts, so the
classifier can exploit signals the frozen sentence embedding misses. There is
deliberately NO permutation test here: one null draw costs cv_folds full
fine-tunes, so a 200-permutation null would mean ~1000 fine-tunes. Statistical
significance therefore comes from the linear variant; this one only reports a
held-out accuracy (its verdict carries p_value=None).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.model_selection import GroupKFold
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding

from src.common.base_schema import BaseSchema
from src.common.logging_utils import log
from src.common.run_config import DistributionalConfig
from src.inference.residual_stream_extractor import get_torch_device

_BATCH_SIZE = 16
_MAX_LENGTH = 128


@dataclass
class ModernbertC2stOutcome(BaseSchema):
    """Pooled held-out accuracy of the fine-tuned classifier."""

    accuracy: float
    fold_accuracies: list[float]
    model_name: str
    n_epochs: int
    # Pooled held-out P(class 1) with true labels, prompt-aligned, so this
    # variant joins the ROC overlay alongside the linear ones.
    scores: list[float] = field(default_factory=list)
    true_labels: list[int] = field(default_factory=list)


def _encode_batch(
    tokenizer: AutoTokenizer, batch_texts: list[str], device: str
) -> BatchEncoding:
    """Tokenize one batch of texts onto the training device."""
    return tokenizer(
        batch_texts,
        truncation=True,
        max_length=_MAX_LENGTH,
        padding=True,
        return_tensors="pt",
    ).to(device)


def _fine_tune_fold(
    tokenizer: AutoTokenizer,
    train_texts: list[str],
    train_labels: NDArray[np.integer],
    config: DistributionalConfig,
    device: str,
    fold_index: int,
) -> AutoModelForSequenceClassification:
    """Fresh classifier fine-tuned on one fold's training authors."""
    # Seed torch so the randomly-initialised classification head and dropout are
    # reproducible per fold — otherwise the held-out accuracy drifts run to run.
    torch.manual_seed(fold_index)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.modernbert_model_name, num_labels=2
    ).to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.modernbert_learning_rate
    )
    label_tensor = torch.as_tensor(train_labels, dtype=torch.long)
    order_rng = np.random.default_rng(fold_index)  # deterministic batch order
    for _ in range(config.modernbert_epochs):
        order = order_rng.permutation(len(train_texts))
        for start in range(0, len(order), _BATCH_SIZE):
            batch = order[start : start + _BATCH_SIZE]
            encoded = _encode_batch(tokenizer, [train_texts[i] for i in batch], device)
            loss = model(**encoded, labels=label_tensor[batch].to(device)).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    return model


@torch.inference_mode()
def _score_fold(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    eval_texts: list[str],
    device: str,
) -> NDArray[np.floating]:
    """Held-out P(class 1) for one fold's texts (thresholding 0.5 = argmax)."""
    model.eval()
    scores = []
    for start in range(0, len(eval_texts), _BATCH_SIZE):
        encoded = _encode_batch(
            tokenizer, eval_texts[start : start + _BATCH_SIZE], device
        )
        probabilities = model(**encoded).logits.softmax(dim=-1)
        scores.append(probabilities[:, 1].float().cpu().numpy())
    return np.concatenate(scores)


def run_modernbert_c2st(
    texts: list[str],
    labels: NDArray[np.integer],
    author_ids: list[str],
    config: DistributionalConfig,
) -> ModernbertC2stOutcome:
    """Author-grouped CV accuracy of a per-fold fine-tuned ModernBERT."""
    labels = np.asarray(labels)
    device = get_torch_device()
    log(f"ModernBERT C2ST on {device}: {config.modernbert_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.modernbert_model_name)
    n_splits = min(config.cv_folds, len(set(author_ids)))
    folds = list(GroupKFold(n_splits=n_splits).split(texts, labels, author_ids))

    scores = np.zeros(len(labels), dtype=float)
    correct = np.zeros(len(labels), dtype=bool)
    fold_accuracies: list[float] = []
    for fold_index, (train_index, test_index) in enumerate(
        tqdm(folds, desc="modernbert folds", leave=False)
    ):
        model = _fine_tune_fold(
            tokenizer,
            [texts[i] for i in train_index],
            labels[train_index],
            config,
            device,
            fold_index,
        )
        fold_scores = _score_fold(
            model, tokenizer, [texts[i] for i in test_index], device
        )
        scores[test_index] = fold_scores
        hits = (fold_scores > 0.5).astype(labels.dtype) == labels[test_index]
        correct[test_index] = hits
        fold_accuracies.append(float(hits.mean()))
        del model  # fresh model per fold; free device memory before the next
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()

    return ModernbertC2stOutcome(
        accuracy=float(correct.mean()),
        fold_accuracies=fold_accuracies,
        model_name=config.modernbert_model_name,
        n_epochs=config.modernbert_epochs,
        scores=[float(s) for s in scores],
        true_labels=[int(label) for label in labels],
    )
