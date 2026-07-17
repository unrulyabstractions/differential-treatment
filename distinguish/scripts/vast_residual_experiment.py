"""Large-model residual-stream C2ST — runs ON a vast.ai GPU box.

For each of the paper-scale models, extract the whole-prompt mean-pooled residual
stream at 75% depth and score target-vs-baseline separability with the linear
C2ST, on several datasets. Models load with device_map='auto' so a 70B shards
across GPUs; the HF cache is cleared between models to bound disk.

Run on the box AFTER at_setup, with .venv/bin/python (never `uv run` — see
cloud/at_setup.sh):
  HF_TOKEN=... .venv/bin/python scripts/vast_residual_experiment.py
"""

from __future__ import annotations

import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
# Closest real, accessible models to the paper's (Qwen3.6-27B / Llama3.3-70B /
# Gemma4-31B). gemma-2-27b extends our 2b->9b->27b ladder.
from src.common.dataset_tables import PromptDataset  # noqa: E402
from src.common.prompt_set_schema import qualified_author_ids  # noqa: E402
from src.distributional.c2st_linear import run_linear_c2st  # noqa: E402

MODELS = os.environ.get(
    "VAST_MODELS",
    "Qwen/Qwen2.5-32B-Instruct,google/gemma-2-27b-it,meta-llama/Llama-3.3-70B-Instruct",
).split(",")
# Ordered easy -> hard so the hard cases (blog ~0.6) are always reached. The
# within-target "null" split must land at ~0.5 — it is the proof the high-dim
# residual is not hallucinating signal (real cases separate, the null does not).
DATASETS = [
    "twitteraae",
    "reddit_l2",
    "pan17_variety",
    "blog_authorship",
    "prism",
    "thoughttrace",
    "wildchat",
    "null_control",
]
LAYER_FRACTION = 0.75
N_PERM = 10
OUT = REPO / "runs" / "vast_residual_results.json"
OUT.parent.mkdir(parents=True, exist_ok=True)


def load_pairs(name: str):
    if name == "null_control":
        # Split the twitteraae TARGET set in half by author (no real label) — a
        # true null: separability here can only be overfitting, so it must ~0.5.
        ds = PromptDataset.load(REPO / "data" / "twitteraae")
        t = ds.prompt_set("target")
        authors = np.array(qualified_author_ids(t, t)[: len(t.prompts)])
        uniq = sorted(set(authors))
        left = set(uniq[: len(uniq) // 2])
        labels = np.array([1 if a in left else 0 for a in authors])
        return t.texts, labels, list(authors)
    ds = PromptDataset.load(REPO / "data" / name)
    t, b = ds.prompt_set("target"), ds.prompt_set("baseline")
    return (
        t.texts + b.texts,
        np.array([1] * len(t.prompts) + [0] * len(b.prompts)),
        qualified_author_ids(t, b),
    )


def whole_prompt_residual(model, tok, layer_index, texts) -> np.ndarray:
    vecs = []
    dev = next(model.parameters()).device
    for i, text in enumerate(texts):
        formatted = tok.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        ids = tok(
            formatted, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(dev)
        with torch.inference_mode():
            h = model(ids, output_hidden_states=True).hidden_states[layer_index][0]
        vecs.append(h.mean(dim=0).float().cpu().numpy())
        if i % 200 == 0:
            print(f"    {i}/{len(texts)}", flush=True)
    return np.stack(vecs)


results = json.loads(OUT.read_text()) if OUT.exists() else {}
for model_name in MODELS:
    done = results.get(model_name, {})
    if "error" not in done and all(
        isinstance(done.get(ds), dict) for ds in DATASETS
    ):
        continue  # every dataset already computed for this model
    try:
        t0 = time.time()
        print(f"### loading {model_name}", flush=True)
        tok = AutoTokenizer.from_pretrained(model_name)
        # gemma-2 is bf16-native and overflows to NaN in fp16 (esp. the 27B);
        # load it in bf16. Qwen runs fine in fp16.
        dtype = torch.bfloat16 if "gemma" in model_name.lower() else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map="auto"
        )
        model.eval()
        n_layers = model.config.num_hidden_layers
        layer_index = max(1, min(n_layers, round(LAYER_FRACTION * n_layers)))
        print(f"### {model_name}: {n_layers} layers, using {layer_index}", flush=True)
        # Preserve any already-computed dataset cells (per-dataset resume); only
        # (re)write the metadata and clear a prior error marker.
        results[model_name] = {
            k: v for k, v in results.get(model_name, {}).items() if k != "error"
        }
        results[model_name].update(
            n_layers=n_layers,
            layer=layer_index,
            params_b=sum(p.numel() for p in model.parameters()) / 1e9,
        )
        for ds in DATASETS:
            if isinstance(results[model_name].get(ds), dict):
                continue  # per-dataset resume: skip cells already computed
            texts, labels, authors = load_pairs(ds)
            emb = whole_prompt_residual(model, tok, layer_index, texts)
            c2st = run_linear_c2st(emb, labels, authors, 5, N_PERM, np.random.default_rng(0))
            null_mean = float(np.mean(c2st.null_accuracies)) if c2st.null_accuracies else 0.5
            results[model_name][ds] = {
                "acc": round(float(c2st.accuracy), 4),  # held-out author-grouped CV
                "null_mean": round(null_mean, 4),  # label-permuted floor (~0.5 if real)
                "p_value": round(float(c2st.p_value), 4),
            }
            print(
                f"  {model_name.split('/')[-1]:26} {ds:14} acc={c2st.accuracy:.4f} "
                f"null={null_mean:.3f} p={c2st.p_value:.3f} dim={emb.shape[1]}",
                flush=True,
            )
            with open(OUT, "w") as fh:
                json.dump(results, fh, indent=1)
        print(f"### {model_name} done in {time.time() - t0:.0f}s", flush=True)
        del model, tok
        gc.collect()
        torch.cuda.empty_cache()
        # Bound disk: drop this model's weights before the next downloads.
        for cache in Path.home().glob(".cache/huggingface/hub/models--*"):
            if model_name.replace("/", "--") in cache.name:
                shutil.rmtree(cache, ignore_errors=True)
    except Exception as exc:  # keep going to the next model
        print(f"### {model_name} FAILED: {type(exc).__name__}: {exc}", flush=True)
        # Merge the error marker — never wholesale-replace the entry, which would
        # discard per-dataset cells already computed and defeat the resume logic.
        results.setdefault(model_name, {})["error"] = f"{type(exc).__name__}: {exc}"
        with open(OUT, "w") as fh:
            json.dump(results, fh, indent=1)
        gc.collect()
        torch.cuda.empty_cache()

print("ALL VAST RESIDUAL EXPERIMENTS DONE", flush=True)
print(json.dumps(results, indent=1), flush=True)
