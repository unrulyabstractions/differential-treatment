"""NeuroBiber wrapper: 96 binary Biber-style syntactic features per text.

NeuroBiber (Blablablab/neurobiber) is a RoBERTa multi-label classifier over
Biber's inventory of syntactic/register features; sigmoid(logit) > 0.5 yields
one binary indicator per feature per text.
"""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.common.logging_utils import log
from src.inference.residual_stream_extractor import get_torch_device

# Prompts are single chatbot questions, far below 512 tokens; truncation only
# guards against pathological inputs and never clips realistic prompts.
_MAX_TOKENS = 512
_DECISION_THRESHOLD = 0.5


class NeurobiberExtractor:
    """Batched binary style-feature extraction with the NeuroBiber tagger."""

    def __init__(self, model_name: str, batch_size: int = 16):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = get_torch_device()
        log(f"Loading {model_name} on {self.device} for syntactic features")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        id2label = self.model.config.id2label
        self.feature_names: list[str] = [id2label[i] for i in range(len(id2label))]

    def extract(self, texts: list[str]) -> NDArray[np.int64]:
        """0/1 feature indicators of shape (len(texts), n_features).

        Interior whitespace is collapsed to single spaces before tokenizing —
        NeuroBiber's own preprocessing splits on whitespace and re-joins, and the
        RoBERTa byte-level tokenizer otherwise assigns different tokens to
        newlines/tabs/runs of spaces, drifting the features from the intended
        path. A text with no word tokens yields the all-zeros row the tagger
        produces for empty input.
        """
        normalized = [" ".join(text.split()) for text in texts]
        result = np.zeros((len(texts), len(self.feature_names)), dtype=np.int64)
        nonempty_index = [i for i, text in enumerate(normalized) if text]
        nonempty_texts = [normalized[i] for i in nonempty_index]
        rows = []
        for start in range(0, len(nonempty_texts), self.batch_size):
            batch = nonempty_texts[start : start + self.batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=_MAX_TOKENS,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                logits = self.model(**encoded).logits
            probabilities = torch.sigmoid(logits)
            rows.append((probabilities > _DECISION_THRESHOLD).cpu().numpy())
        if rows:
            result[nonempty_index] = np.vstack(rows).astype(np.int64)
        return result

    def cleanup(self) -> None:
        """Release model memory."""
        del self.model
        self.model = None
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()
