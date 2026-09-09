from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SentencePart:
    index: int
    text: str


def prioritize_sentences(
    sentences: Iterable[str],
    *,
    seed: str,
    accept: Callable[[str], bool] | None = None,
) -> tuple[SentencePart, ...]:
    accepted = _accepted_parts(sentences, accept)
    contextual = _contextual_parts(accepted)
    return _ordered(contextual, seed) if contextual else _fallback(accepted)


def first_prioritized_sentence(
    sentences: Iterable[str],
    *,
    seed: str,
    accept: Callable[[str], bool] | None = None,
) -> SentencePart | None:
    """Return one preferred sentence without evaluating unnecessary candidates."""
    return _first_accepted(candidate_sentence_parts(sentences, seed=seed), accept)


def candidate_sentence_parts(
    sentences: Iterable[str],
    *,
    seed: str,
) -> tuple[SentencePart, ...]:
    """Order contextual sentences and retain the title only as a fallback."""
    parts = _sentence_parts(sentences)
    contextual = _ordered(_contextual_parts(parts), seed)
    return contextual + parts[:1]


def _sentence_parts(sentences: Iterable[str]) -> tuple[SentencePart, ...]:
    return tuple(SentencePart(index, sentence) for index, sentence in enumerate(_clean_sentences(sentences)))


def _first_accepted(
    parts: Iterable[SentencePart],
    accept: Callable[[str], bool] | None,
) -> SentencePart | None:
    for part in parts:
        if accept is None or accept(part.text):
            return part
    return None


def _accepted_parts(
    sentences: Iterable[str],
    accept: Callable[[str], bool] | None,
) -> tuple[SentencePart, ...]:
    return tuple(
        SentencePart(index, sentence)
        for index, sentence in enumerate(_clean_sentences(sentences))
        if accept is None or accept(sentence)
    )


def _clean_sentences(sentences: Iterable[str]) -> tuple[str, ...]:
    return tuple(sentence.strip() for sentence in sentences if sentence.strip())


def _contextual_parts(accepted: tuple[SentencePart, ...]) -> tuple[SentencePart, ...]:
    return tuple(part for part in accepted if part.index > 0)


def _ordered(parts: tuple[SentencePart, ...], seed: str) -> tuple[SentencePart, ...]:
    return tuple(sorted(parts, key=lambda part: (_rank(seed, part), part.index)))


def _fallback(accepted: tuple[SentencePart, ...]) -> tuple[SentencePart, ...]:
    if not accepted:
        return ()
    return (accepted[0],)


def _rank(seed: str, part: SentencePart) -> str:
    return hashlib.sha256(f"{seed}:{part.index}:{part.text}".encode()).hexdigest()
