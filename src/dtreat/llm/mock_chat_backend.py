"""Deterministic mock chat backend (realism level 0).

Model spec syntax (the whole spec travels in ChatRequest.model):
    mock:helper           — proposes the five case-study axes as JSON
    mock:target:biased    — planted differential behavior by community cues
    mock:target:null      — identical behavior for both communities
    mock:judge            — marker-matching judge (near-perfect)
    mock:judge:noisy      — judge with seeded 5% verdict flips

All randomness is derived from (model spec, prompt text, request seed), so any
call is reproducible in isolation regardless of execution order.
"""

from __future__ import annotations

import json
import re

from dtreat.common.judge_protocol import RESPONSE_END, RESPONSE_START
from dtreat.common.random_seed import rng_for

from .chat_backend_base import ChatBackend
from .chat_types import ChatRequest, ChatResult, ChatUsage
from .mock_behavior_profiles import (
    MOCK_AXES,
    MOCK_TARGET_PROFILES,
    detect_cued_community,
)

NOISY_JUDGE_FLIP_RATE = 0.05


class MockChatBackend(ChatBackend):
    """Dispatches on the mock model spec to play helper, target, or judge."""

    backend_name = "mock"

    def complete(self, request: ChatRequest) -> ChatResult:
        parts = request.model.split(":")
        if len(parts) < 2 or parts[0] != "mock":
            raise ValueError(f"Not a mock model spec: {request.model}")
        role, variant = parts[1], (parts[2] if len(parts) > 2 else None)

        user_text = "\n\n".join(m.content for m in request.non_system_messages())
        if role == "helper":
            text = self._helper_reply()
        elif role == "target":
            text = self._target_reply(request, user_text, variant or "biased")
        elif role == "judge":
            text = self._judge_reply(request, user_text, variant)
        else:
            raise ValueError(f"Unknown mock role: {role}")

        return ChatResult(
            text=text,
            model=request.model,
            backend=self.backend_name,
            usage=ChatUsage(
                input_tokens=len(user_text.split()),
                output_tokens=len(text.split()),
            ),
        )

    # ── helper ───────────────────────────────────────────────────────────

    def _helper_reply(self) -> str:
        proposals = [
            {
                "axis_id": axis.axis_id,
                "question": axis.question,
                "rationale": f"Mock rationale: communities may differ on '{axis.axis_id}'.",
            }
            for axis in MOCK_AXES
        ]
        return json.dumps(proposals, indent=2)

    # ── target ───────────────────────────────────────────────────────────

    def _target_reply(self, request: ChatRequest, prompt_text: str, profile_name: str) -> str:
        profile = MOCK_TARGET_PROFILES[profile_name]
        cued = detect_cued_community(prompt_text)
        rng = rng_for(request.model, prompt_text, request.seed or 0)
        sentences = ["Thanks for the details — here is what I would focus on."]
        for axis in MOCK_AXES:
            rates = profile.rates_for(axis.axis_id)
            rate = rates.rate_cued if cued else rates.rate_otherwise
            if rng.random() < rate:
                sentences.append(axis.realization)
        sentences.append("Happy to go deeper on any part of this.")
        return " ".join(sentences)

    # ── judge ────────────────────────────────────────────────────────────

    def _judge_reply(self, request: ChatRequest, judge_prompt: str, variant: str | None) -> str:
        response_text = _extract_between(judge_prompt, RESPONSE_START, RESPONSE_END)
        axis_ids = _extract_axis_ids(judge_prompt)
        flip_rate = NOISY_JUDGE_FLIP_RATE if variant == "noisy" else 0.0
        rng = rng_for(request.model, judge_prompt, request.seed or 0)

        verdicts: dict[str, str] = {}
        for axis_id in axis_ids:
            axis = next((a for a in MOCK_AXES if a.axis_id == axis_id), None)
            exhibited = bool(axis and axis.marker.lower() in response_text.lower())
            if flip_rate > 0 and rng.random() < flip_rate:
                exhibited = not exhibited
            verdicts[axis_id] = "YES" if exhibited else "NO"

        if len(axis_ids) == 1:
            return verdicts[axis_ids[0]]
        return json.dumps(verdicts, indent=2)


def _extract_between(text: str, start: str, end: str) -> str:
    """Slice out the judged response between sentinels ('' if absent)."""
    if start in text and end in text:
        return text.split(start, 1)[1].split(end, 1)[0].strip()
    return ""


def _extract_axis_ids(judge_prompt: str) -> list[str]:
    """Parse axis ids from the judge protocol's '- axis_id: question' lines."""
    section = judge_prompt.split(RESPONSE_START, 1)[0]
    return re.findall(r"^- ([a-z0-9_]+):", section, flags=re.MULTILINE)
