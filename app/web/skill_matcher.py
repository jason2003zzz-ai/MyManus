import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class SkillDocument:
    id: str
    name: str
    summary: str
    content: str


@dataclass(frozen=True)
class SkillMatch:
    id: str
    name: str
    summary: str
    score: float
    retrieval_method: str = "tfidf"
    matched_terms: tuple[str, ...] = ()


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
    "you",
    "your",
    "请",
    "把",
    "给",
    "帮我",
    "需要",
    "这个",
    "那个",
    "里面",
    "一下",
}
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.+-]{1,}|[0-9]+|[\u4e00-\u9fff]+")
QUERY_TERM_ALIASES = {
    "搜索": ("检索", "查找", "联网", "资料"),
    "查询": ("检索", "查找", "联网", "资料"),
    "查找": ("检索", "搜索", "联网", "资料"),
    "research": ("search", "browse", "web"),
    "search": ("research", "browse", "web"),
    "lookup": ("search", "research", "web"),
}
RESEARCH_QUERY_INTENT_RE = re.compile(
    r"搜索|搜一下|搜一搜|查询|查找|检索|research|search|look\s*up|find",
    re.IGNORECASE,
)
NON_WEB_SEARCH_CONTEXT_RE = re.compile(
    r"邮件|邮箱|收件箱|发件人|gmail|mailbox|inbox|email",
    re.IGNORECASE,
)


def _cjk_ngrams(value: str) -> Iterable[str]:
    for size in (2, 3, 4):
        if len(value) < size:
            continue
        for index in range(0, len(value) - size + 1):
            yield value[index : index + size]


def tokenize(value: str) -> list[str]:
    tokens: list[str] = []
    for match in TOKEN_RE.finditer(value or ""):
        raw = match.group(0).strip().lower()
        if not raw or raw in STOPWORDS:
            continue
        if re.fullmatch(r"[\u4e00-\u9fff]+", raw):
            if 1 < len(raw) <= 8:
                tokens.append(raw)
            tokens.extend(token for token in _cjk_ngrams(raw) if token not in STOPWORDS)
        else:
            tokens.append(raw)
    return tokens


def query_tokens(value: str) -> list[str]:
    """Tokenize a task and add a small deterministic intent synonym set."""

    tokens = tokenize(value)
    expanded = list(tokens)
    non_web_search = bool(NON_WEB_SEARCH_CONTEXT_RE.search(value or ""))
    for token in tokens:
        if non_web_search and token in {"搜索", "查询", "查找"}:
            continue
        expanded.extend(QUERY_TERM_ALIASES.get(token, ()))
    if RESEARCH_QUERY_INTENT_RE.search(value or "") and not non_web_search:
        # Intent aliases receive a small extra weight so common colloquial forms
        # such as “搜一下” still cross the automatic Skill threshold.
        expanded.extend(("检索", "搜索", "查找", "联网", "资料") * 2)
    return expanded


def _weighted_document_tokens(document: SkillDocument) -> Counter[str]:
    tokens: Counter[str] = Counter()
    tokens.update(tokenize(document.name) * 4)
    tokens.update(tokenize(document.summary) * 3)
    tokens.update(tokenize(document.content))
    return tokens


def _tfidf(counter: Counter[str], idf: dict[str, float]) -> dict[str, float]:
    return {
        token: (1.0 + math.log(count)) * idf.get(token, 1.0)
        for token, count in counter.items()
        if count > 0
    }


def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(weight * right.get(token, 0.0) for token, weight in left.items())
    left_norm = math.sqrt(sum(weight * weight for weight in left.values()))
    right_norm = math.sqrt(sum(weight * weight for weight in right.values()))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def match_skills(
    query: str,
    documents: list[SkillDocument],
    *,
    exclude_ids: set[str] | None = None,
    top_k: int = 3,
    min_score: float = 0.08,
) -> list[SkillMatch]:
    """Select Skills with a small, deterministic TF-IDF matcher."""

    if not query.strip() or not documents or top_k <= 0:
        return []
    exclude_ids = exclude_ids or set()
    indexed = [
        (document, _weighted_document_tokens(document))
        for document in documents
        if document.id not in exclude_ids
    ]
    if not indexed:
        return []

    document_frequency: Counter[str] = Counter()
    for _, tokens in indexed:
        document_frequency.update(tokens.keys())
    total = len(indexed)
    idf = {
        token: math.log((total + 1.0) / (frequency + 0.5)) + 1.0
        for token, frequency in document_frequency.items()
    }
    query_vector = _tfidf(Counter(query_tokens(query)), idf)
    if not query_vector:
        return []

    matches: list[SkillMatch] = []
    for document, document_counter in indexed:
        document_vector = _tfidf(document_counter, idf)
        score = _cosine(query_vector, document_vector)
        matched_terms = tuple(
            token
            for token, _ in sorted(
                (
                    (token, query_vector[token] * document_vector.get(token, 0.0))
                    for token in query_vector.keys() & document_vector.keys()
                ),
                key=lambda item: item[1],
                reverse=True,
            )[:8]
        )
        if score >= min_score:
            matches.append(
                SkillMatch(
                    id=document.id,
                    name=document.name,
                    summary=document.summary,
                    score=round(score, 4),
                    matched_terms=matched_terms,
                )
            )

    matches.sort(key=lambda item: item.score, reverse=True)
    return matches[:top_k]
