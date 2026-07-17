"""LLM-based extraction of the underlying instruction iota(x) (paper §3.1).

When prompt files carry no instruction annotations, an annotator LLM extracts
what each prompt is actually asking, in two passes:

1. per-prompt extraction of a short instruction phrase (community-blind:
   the annotator sees one prompt at a time and describes only the request);
2. canonicalization: the distinct phrases from BOTH communities are pooled
   and merged into shared snake_case instruction ids, so equivalent asks get
   the same id regardless of who asked.
"""

from __future__ import annotations

import re

from dtreat.common.experiment_config import ExperimentConfig
from dtreat.common.json_text_extraction import extract_first_json_object
from dtreat.llm.chat_client import ChatClient
from dtreat.llm.chat_types import ChatMessage
from dtreat.llm.parallel_chat_execution import ChatJob, execute_chat_jobs

from .prompt_set_schemas import CommunityPromptFile

ANNOTATOR_SYSTEM_PROMPT = (
    "You annotate user requests. Given one user message, state the underlying "
    "instruction — what is being asked — as a short verb-first phrase, "
    "ignoring tone, slang, personal details, and who is asking."
)

EXTRACTION_TEMPLATE = """User message:
{prompt_text}

Reply with ONLY the underlying instruction as a verb-first phrase of at most
6 words (e.g. "ask how to bulk up"). No punctuation, no explanations.
Instruction phrase:"""

CANONICALIZATION_TEMPLATE = """Group these instruction phrases so that phrases asking the same KIND of thing
share one canonical id. Phrases:
{phrase_lines}

Merge aggressively: use AT MOST {max_groups} distinct ids across all
{n_phrases} phrases. An id names the generic underlying request (like
"ask_bulking_advice", "ask_cutting_advice", "ask_supplement_advice",
"ask_plateau_help"), never specific details such as body parts, products,
numbers, or context. Phrases about gaining weight/muscle/bulking share one
id; phrases about losing fat/cutting/leaning out share one id; phrases about
supplements share one id. Most phrases MUST share an id with others.

Reply with ONLY a JSON object mapping EVERY phrase above to its canonical
snake_case id (at most 4 words)."""


def extract_instruction_phrases(
    client: ChatClient,
    prompts: list[tuple[str, str]],
    config: ExperimentConfig,
) -> dict[str, str]:
    """(prompt_id, text) pairs -> {prompt_id: instruction phrase}."""
    jobs = [
        ChatJob(
            job_id=prompt_id,
            request=client.build_request(
                [
                    ChatMessage("system", ANNOTATOR_SYSTEM_PROMPT),
                    ChatMessage("user", EXTRACTION_TEMPLATE.format(prompt_text=text)),
                ],
                temperature=0.0,
                max_tokens=30,
                seed=config.seed,
            ),
        )
        for prompt_id, text in prompts
    ]
    results, failures = execute_chat_jobs(
        client, jobs, max_workers=config.max_workers, description="extracting instructions"
    )
    if failures:
        failed_ids = ", ".join(f.job_id for f in failures[:5])
        raise RuntimeError(
            f"Instruction extraction failed for {len(failures)} prompts ({failed_ids}...). "
            "Every prompt needs an instruction for the comparability check."
        )
    return {
        prompt_id: _normalize_phrase(result.text) for prompt_id, result in results.items()
    }


def canonicalize_instruction_phrases(
    client: ChatClient, phrases: list[str], config: ExperimentConfig
) -> dict[str, str]:
    """Distinct phrases -> {phrase: canonical snake_case instruction id}."""
    distinct = sorted(set(phrases))
    phrase_lines = "\n".join(f"- {phrase}" for phrase in distinct)
    result = client.complete(
        client.build_request(
            [
                ChatMessage("system", ANNOTATOR_SYSTEM_PROMPT),
                ChatMessage(
                    "user",
                    CANONICALIZATION_TEMPLATE.format(
                        phrase_lines=phrase_lines,
                        n_phrases=len(distinct),
                        max_groups=config.max_instruction_groups,
                    ),
                ),
            ],
            temperature=0.0,
            max_tokens=1500,
            seed=config.seed,
        )
    )
    raw_mapping = extract_first_json_object(result.text) or {}
    mapping = {}
    for phrase in distinct:
        canonical = raw_mapping.get(phrase)
        # Unmapped phrases fall back to their own normalized form so no prompt
        # is silently dropped; the comparability check will surface mismatches.
        mapping[phrase] = _to_instruction_id(str(canonical) if canonical else phrase)
    return mapping


def annotate_prompt_sets(
    config: ExperimentConfig,
    target_set: CommunityPromptFile,
    baseline_set: CommunityPromptFile,
    client: ChatClient,
) -> None:
    """Extract + canonicalize instructions for both sets, in place.

    Both communities are pooled for canonicalization so equivalent asks share
    ids across sets — the whole point of the comparability assumption.
    """
    all_prompts = [
        (prompt.prompt_id, prompt.text)
        for prompt in target_set.prompts + baseline_set.prompts
    ]
    phrases_by_id = extract_instruction_phrases(client, all_prompts, config)
    mapping = canonicalize_instruction_phrases(
        client, list(phrases_by_id.values()), config
    )
    for prompt in target_set.prompts + baseline_set.prompts:
        phrase = phrases_by_id[prompt.prompt_id]
        prompt.instruction_phrase = phrase
        prompt.instruction_id = mapping[phrase]
        prompt.instruction_source = "extracted"


def _normalize_phrase(reply: str) -> str:
    phrase = reply.strip().splitlines()[0] if reply.strip() else "unknown request"
    phrase = re.sub(r"[^a-zA-Z0-9 ]", "", phrase).lower().strip()
    return phrase or "unknown request"


def _to_instruction_id(text: str) -> str:
    words = re.sub(r"[^a-z0-9 _]", "", text.lower()).split()
    return "_".join(words[:4]) or "unknown_request"
