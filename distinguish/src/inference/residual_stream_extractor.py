"""Residual-stream embeddings from a causal LM's hidden states.

Each prompt is wrapped in the model's chat template; the hidden states at the
layer sitting at a chosen fraction of depth (75%) are pooled into one vector.

The DEFAULT pooling is the mean over ALL prompt token positions ("all_prompt"):
in our experiments this separates the prompt sets consistently better than the
paper's change-of-turn representation (~+3 C2ST accuracy points across models).
The paper's change-of-turn positions (the template tokens after the user
content, e.g. "<|im_end|>\\n<|im_start|>assistant" in the Qwen family) remain
available via positions_mode="change_of_turn" (mean or concat) for reproduction.
"""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common.logging_utils import log

_FALLBACK_SUFFIX_TOKENS = 4  # last-k positions if the template split fails
_SUFFIX_SENTINEL = "\x00sentinel\x00"  # never appears in real prompt text


def get_torch_device() -> str:
    """Best available torch device on this machine."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ResidualStreamExtractor:
    """Extracts change-of-turn residual-stream embeddings from a causal LM."""

    def __init__(
        self,
        model_name: str,
        layer_fraction: float = 0.75,
        positions_mode: str = "all_prompt",
        aggregation: str = "mean",
    ):
        # positions_mode: "change_of_turn" (template tokens after the user text)
        #   or "all_prompt" (every token position). aggregation: "mean" (average
        #   over the chosen positions) or "concat" (stack them into one long
        #   vector — only for change_of_turn, whose token count is fixed).
        self.model_name = model_name
        self.layer_fraction = layer_fraction
        self.positions_mode = positions_mode
        self.aggregation = aggregation
        self.device = get_torch_device()
        dtype = torch.float16 if self.device in ("cuda", "mps") else torch.float32

        log(f"Loading {model_name} on {self.device} for residual-stream extraction")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype
        ).to(self.device)
        self.model.eval()

        n_layers = self.model.config.num_hidden_layers
        # hidden_states[0] is the embedding layer; index i+1 is block i's output.
        self.layer_index = max(1, min(n_layers, round(layer_fraction * n_layers)))
        log(f"  using layer {self.layer_index}/{n_layers} (fraction {layer_fraction})")

        # The change-of-turn suffix is the template text after the user content.
        # Derive it once via a sentinel: searching each formatted prompt for the
        # user text is fragile (the text could coincide with template markup).
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": _SUFFIX_SENTINEL}],
            tokenize=False,
            add_generation_prompt=True,
        )
        sentinel_start = rendered.rfind(_SUFFIX_SENTINEL)
        sentinel_end = sentinel_start + len(_SUFFIX_SENTINEL)
        self._generation_suffix = rendered[sentinel_end:]
        # Template tokens before the user content (system / opening markup) — the
        # boundary the attribution content span begins at.
        self._prefix_template_len = len(
            self.tokenizer(
                rendered[:sentinel_start], add_special_tokens=False
            ).input_ids
        )
        # Fixed token count of the generation suffix, so "concat" produces a
        # constant-length vector (k * d_model) for every prompt.
        self._suffix_len = max(
            1,
            len(
                self.tokenizer(
                    self._generation_suffix, add_special_tokens=False
                ).input_ids
            ),
        )

    def extract(self, texts: list[str]) -> NDArray[np.float32]:
        """Embeddings of shape (len(texts), d_model)."""
        vectors = [
            self._extract_single(text)
            for text in tqdm(texts, desc="residual-stream", leave=False)
        ]
        return np.stack(vectors)

    def _extract_single(self, text: str) -> NDArray[np.float32]:
        formatted = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        input_ids = self.tokenizer(
            formatted, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(self.device)

        seq_len = input_ids.shape[1]
        with torch.inference_mode():
            outputs = self.model(input_ids, output_hidden_states=True)
        hidden = outputs.hidden_states[self.layer_index][0]  # (seq_len, d_model)

        if self.positions_mode == "all_prompt":
            # Pool over the whole prompt; concat is variable-length so it is not
            # offered here — averaging is the only fixed-size all-position vector.
            return hidden.mean(dim=0).float().cpu().numpy()

        if self.aggregation == "concat":
            # Exactly the last _suffix_len positions -> constant k*d vector.
            k = self._suffix_len
            window = hidden[max(0, seq_len - k) :]
            if window.shape[0] < k:  # pad short sequences by repeating the first
                window = torch.cat(
                    [window[:1].expand(k - window.shape[0], -1), window], dim=0
                )
            return window.reshape(-1).float().cpu().numpy()

        positions = self._change_of_turn_positions(formatted, seq_len)
        return hidden[positions].mean(dim=0).float().cpu().numpy()

    def extract_per_token(
        self, text: str
    ) -> tuple[list[str], NDArray[np.float32], tuple[int, int]]:
        """Per-token residuals for attribution (paper §3.3.5).

        Returns (token_strings, hidden of shape (T, d_model) at the probe layer,
        (content_start, content_end)) where the content span is the user-text
        tokens between the template prefix and the change-of-turn suffix. The
        whole-prompt mean of `hidden` equals what extract() pools, so a linear
        head w gives per-token contributions a_t = (1/T) w·h_t that sum exactly
        to the pooled probe score.
        """
        formatted = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        input_ids = self.tokenizer(
            formatted, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(self.device)
        seq_len = input_ids.shape[1]
        with torch.inference_mode():
            hidden = self.model(input_ids, output_hidden_states=True).hidden_states[
                self.layer_index
            ][0]
        tokens = [self.tokenizer.decode([tid]) for tid in input_ids[0].tolist()]
        suffix_positions = self._change_of_turn_positions(formatted, seq_len)
        content_start, content_end = self._content_span(formatted, seq_len)
        content_end = min(
            content_end, suffix_positions[0] if suffix_positions else seq_len
        )
        return tokens, hidden.float().cpu().numpy(), (content_start, content_end)

    def _content_span(self, formatted: str, seq_len: int) -> tuple[int, int]:
        """Token range of the user content (after the template prefix)."""
        suffix = self._generation_suffix
        if suffix and formatted.endswith(suffix):
            prefix = formatted[: len(formatted) - len(suffix)]
            # The template's system/opening tokens precede the user text; find
            # where the user turn starts by trimming the known prefix template.
            prefix_ids = self.tokenizer(prefix, add_special_tokens=False).input_ids
            start = min(self._prefix_template_len, len(prefix_ids))
            return start, len(prefix_ids)
        return 0, seq_len

    def _change_of_turn_positions(self, formatted: str, seq_len: int) -> list[int]:
        """Token positions of the template suffix after the user content."""
        suffix = self._generation_suffix
        if suffix and formatted.endswith(suffix):
            prefix = formatted[: len(formatted) - len(suffix)]
            prefix_len = len(self.tokenizer(prefix, add_special_tokens=False).input_ids)
            if 0 < prefix_len < seq_len:  # else tokens merged across the boundary
                return list(range(prefix_len, seq_len))
        return list(range(max(0, seq_len - _FALLBACK_SUFFIX_TOKENS), seq_len))

    def cleanup(self) -> None:
        """Release model memory."""
        del self.model
        self.model = None
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()
