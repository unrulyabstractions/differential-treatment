"""Literature retrieval for the `literature_rag` hypothesis condition.

Queries the arXiv API for abstracts relevant to the specific group pair and
deployment domain, so the helper is informed by retrieved research rather
than only the two bundled abstracts. Fails soft: on any network problem the
caller falls back to the bundled abstracts (never blocks a run).
"""

from __future__ import annotations

import re
import urllib.parse
import urllib.request

from dtreat.common.console_logging import log

ARXIV_API = "http://export.arxiv.org/api/query?search_query={query}&max_results={k}&sortBy=relevance"


def _domain_keyword(deployment_context: str) -> str:
    """A single load-bearing domain word from the deployment description."""
    stopwords = {"a", "an", "the", "to", "and", "of", "for", "about", "chat",
                 "consumer", "assistant", "giving", "adults", "asking"}
    for word in deployment_context.lower().replace(",", " ").split():
        if word not in stopwords and len(word) > 4:
            return word
    return "advice"


def retrieve_abstracts(
    target_community: str,
    baseline_community: str,
    deployment_context: str,
    max_abstracts: int = 4,
) -> str:
    """Top arXiv abstracts for '<pair> bias in language models <domain>'.

    Returns a formatted literature block, or "" when retrieval fails or
    finds nothing (caller falls back to bundled abstracts).
    """
    # quoted-phrase AND query; unquoted bag-of-words matches unrelated fields
    domain_keyword = _domain_keyword(deployment_context)
    query_text = (
        f'abs:"language model" AND abs:bias AND '
        f'(abs:"{target_community}" OR abs:"{domain_keyword}")'
    )
    url = ARXIV_API.format(query=urllib.parse.quote(query_text), k=max_abstracts)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "dtreat-research"})
        with urllib.request.urlopen(request, timeout=20) as response:
            feed = response.read().decode("utf-8", errors="replace")
    except Exception as error:
        log(f"  [note] literature retrieval failed ({type(error).__name__}); using bundled abstracts")
        return ""

    entries = re.findall(
        r"<title>(.*?)</title>.*?<summary>(.*?)</summary>", feed, flags=re.S
    )
    # first <title> is the feed's own; drop it if it leaked into matches
    blocks = []
    for title, summary in entries:
        title = " ".join(title.split())
        summary = " ".join(summary.split())[:900]
        if "arxiv" in title.lower() and "query" in title.lower():
            continue
        blocks.append(f"[{len(blocks) + 1}] {title}\n{summary}")
        if len(blocks) >= max_abstracts:
            break
    if not blocks:
        return ""
    return "Retrieved research abstracts (arXiv):\n\n" + "\n\n".join(blocks)
