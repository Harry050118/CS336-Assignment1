from __future__ import annotations

import os
from collections import Counter, defaultdict

import regex as re
from tqdm import tqdm

from cs336_basics.tokenizer.merge import (
    build_pair_heap,
    merge_pairs_with_heap_index,
    pop_most_frequent_pair,
)
from cs336_basics.tokenizer.utils import find_chunk_boundaries, string_to_bytes


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _iter_pretokens(text: str, special_tokens: list[str]) -> list[str]:
    if not text:
        return []

    if not special_tokens:
        return [m.group(0) for m in re.finditer(PAT, text)]

    escaped_specials = [re.escape(token) for token in sorted(set(special_tokens), key=len, reverse=True)]
    split_pat = "|".join(escaped_specials)

    pretokens: list[str] = []
    start = 0
    for match in re.finditer(split_pat, text):
        if match.start() > start:
            segment = text[start:match.start()]
            pretokens.extend(m.group(0) for m in re.finditer(PAT, segment))
        start = match.end()

    if start < len(text):
        segment = text[start:]
        pretokens.extend(m.group(0) for m in re.finditer(PAT, segment))

    return pretokens


def _build_word_counter(text: str, special_tokens: list[str]) -> Counter[tuple[int, ...]]:
    word_counter: Counter[tuple[int, ...]] = Counter()

    for pretoken in _iter_pretokens(text, special_tokens):
        token_ints = tuple(string_to_bytes(pretoken, return_int=True))
        if token_ints:
            word_counter[token_ints] += 1

    return word_counter


def _build_word_counter_from_file(
    input_path: str | os.PathLike,
    special_tokens: list[str],
) -> Counter[tuple[int, ...]]:
    split_special_token = special_tokens[0].encode("utf-8") if special_tokens else b"\n"
    desired_chunks = max(1, min(4, os.cpu_count() or 1))

    with open(input_path, "rb") as file:
        boundaries = find_chunk_boundaries(
            file=file,
            desired_num_chunks=desired_chunks,
            split_special_token=split_special_token,
        )

        word_counter: Counter[tuple[int, ...]] = Counter()
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            file.seek(start)
            text_chunk = file.read(end - start).decode("utf-8", errors="ignore")
            word_counter.update(_build_word_counter(text_chunk, special_tokens))

    return word_counter


def _build_pair_stats(
    word_counter: Counter[tuple[int, ...]],
) -> tuple[Counter[tuple[int, int]], dict[tuple[int, int], set[tuple[int, ...]]]]:
    pair_counter: Counter[tuple[int, int]] = Counter()
    pair_to_words: dict[tuple[int, int], set[tuple[int, ...]]] = defaultdict(set)

    for word, freq in word_counter.items():
        if len(word) < 2:
            continue
        for index in range(len(word) - 1):
            pair = (word[index], word[index + 1])
            pair_counter[pair] += freq
            pair_to_words[pair].add(word)

    return pair_counter, pair_to_words


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str] | None = None,
    show_progress: bool = False,
    progress_interval_pct: int = 10,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")

    special_tokens = special_tokens or []

    vocab: dict[int, bytes] = {index: bytes([index]) for index in range(256)}
    existing_tokens = set(vocab.values())

    for token in special_tokens:
        token_bytes = b"".join(string_to_bytes(token))
        if token_bytes not in existing_tokens:
            vocab[len(vocab)] = token_bytes
            existing_tokens.add(token_bytes)

    if len(vocab) >= vocab_size:
        return vocab, []

    word_counter = _build_word_counter_from_file(input_path, special_tokens)
    pair_counter, pair_to_words = _build_pair_stats(word_counter)

    merges: list[tuple[bytes, bytes]] = []
    pair_heap = build_pair_heap(pair_counter, vocab)
    merges_to_learn = vocab_size - len(vocab)
    safe_interval_pct = max(1, min(100, progress_interval_pct))
    next_progress_pct = safe_interval_pct

    merge_iter = range(merges_to_learn)
    if show_progress:
        merge_iter = tqdm(merge_iter, total=merges_to_learn, desc="train_bpe", unit="merge")

    for _ in merge_iter:
        if not pair_counter:
            break

        pair = pop_most_frequent_pair(pair_heap, pair_counter)
        token_1, token_2 = pair

        merge_bytes = (vocab[token_1], vocab[token_2])
        merges.append(merge_bytes)

        new_id = len(vocab)
        vocab[new_id] = merge_bytes[0] + merge_bytes[1]

        word_counter, pair_counter, pair_heap, pair_to_words = merge_pairs_with_heap_index(
            word_counter=word_counter,
            pair_counter=pair_counter,
            target_pair=pair,
            new_id=new_id,
            vocab=vocab,
            pair_heap=pair_heap,
            pair_to_words=pair_to_words,
        )

        if show_progress and merges_to_learn > 0:
            completed_merges = len(merges)
            completed_pct = int((completed_merges * 100) / merges_to_learn)
            while completed_pct >= next_progress_pct and next_progress_pct <= 100:
                print(
                    f"[train_bpe] progress: {next_progress_pct}% ({completed_merges}/{merges_to_learn} merges)",
                    flush=True,
                )
                next_progress_pct += safe_interval_pct

    if show_progress and merges_to_learn > 0 and next_progress_pct <= 100:
        completed_merges = len(merges)
        completed_pct = int((completed_merges * 100) / merges_to_learn)
        while completed_pct >= next_progress_pct and next_progress_pct <= 100:
            print(
                f"[train_bpe] progress: {next_progress_pct}% ({completed_merges}/{merges_to_learn} merges)",
                flush=True,
            )
            next_progress_pct += safe_interval_pct

    return vocab, merges


# Appended fast-train helpers

def _iter_pretokens_fast(text: str, special_tokens: list[str]):
    if not text:
        return

    if not special_tokens:
        for m in re.finditer(PAT, text):
            yield m.group(0)
        return

    escaped_specials = [re.escape(token) for token in sorted(set(special_tokens), key=len, reverse=True)]
    split_pat = "|".join(escaped_specials)

    start = 0
    for match in re.finditer(split_pat, text):
        if match.start() > start:
            segment = text[start:match.start()]
            for m in re.finditer(PAT, segment):
                yield m.group(0)
        start = match.end()

    if start < len(text):
        segment = text[start:]
        for m in re.finditer(PAT, segment):
            yield m.group(0)


def _build_word_counter_fast(text: str, special_tokens: list[str]) -> Counter[tuple[int, ...]]:
    word_counter: Counter[tuple[int, ...]] = Counter()

    for pretoken in _iter_pretokens_fast(text, special_tokens):
        token_ints = tuple(pretoken.encode("utf-8"))
        if token_ints:
            word_counter[token_ints] += 1

    return word_counter


def _build_word_counter_from_file_fast(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    show_progress: bool = False,
) -> Counter[tuple[int, ...]]:
    split_special_token = special_tokens[0].encode("utf-8") if special_tokens else b"\n"
    desired_chunks = max(16, min(64, (os.cpu_count() or 1) * 4))

    with open(input_path, "rb") as file:
        boundaries = find_chunk_boundaries(
            file=file,
            desired_num_chunks=desired_chunks,
            split_special_token=split_special_token,
        )

        chunk_ranges = list(zip(boundaries[:-1], boundaries[1:]))
        chunk_iter = chunk_ranges
        if show_progress:
            chunk_iter = tqdm(chunk_ranges, total=len(chunk_ranges), desc="build_word_counter_fast", unit="chunk", dynamic_ncols=True, mininterval=0.2)

        word_counter: Counter[tuple[int, ...]] = Counter()
        for start, end in chunk_iter:
            file.seek(start)
            text_chunk = file.read(end - start).decode("utf-8", errors="ignore")
            word_counter.update(_build_word_counter_fast(text_chunk, special_tokens))

    return word_counter


def train_bpe_fast(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str] | None = None,
    show_progress: bool = False,
    progress_interval_pct: int = 10,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")

    special_tokens = special_tokens or []

    vocab: dict[int, bytes] = {index: bytes([index]) for index in range(256)}
    existing_tokens = set(vocab.values())

    for token in special_tokens:
        token_bytes = token.encode("utf-8")
        if token_bytes not in existing_tokens:
            vocab[len(vocab)] = token_bytes
            existing_tokens.add(token_bytes)

    if len(vocab) >= vocab_size:
        return vocab, []

    word_counter = _build_word_counter_from_file_fast(
        input_path,
        special_tokens,
        show_progress=show_progress,
    )
    pair_counter, pair_to_words = _build_pair_stats(word_counter)

    merges: list[tuple[bytes, bytes]] = []
    pair_heap = build_pair_heap(pair_counter, vocab)
    merges_to_learn = vocab_size - len(vocab)
    safe_interval_pct = max(1, min(100, progress_interval_pct))
    next_progress_pct = safe_interval_pct

    merge_iter = range(merges_to_learn)
    if show_progress:
        merge_iter = tqdm(merge_iter, total=merges_to_learn, desc="train_bpe_fast", unit="merge", dynamic_ncols=True, mininterval=0.2)

    for _ in merge_iter:
        if not pair_counter:
            break

        pair = pop_most_frequent_pair(pair_heap, pair_counter)
        token_1, token_2 = pair

        merge_bytes = (vocab[token_1], vocab[token_2])
        merges.append(merge_bytes)

        new_id = len(vocab)
        vocab[new_id] = merge_bytes[0] + merge_bytes[1]

        word_counter, pair_counter, pair_heap, pair_to_words = merge_pairs_with_heap_index(
            word_counter=word_counter,
            pair_counter=pair_counter,
            target_pair=pair,
            new_id=new_id,
            vocab=vocab,
            pair_heap=pair_heap,
            pair_to_words=pair_to_words,
        )

        if show_progress and merges_to_learn > 0:
            completed_merges = len(merges)
            completed_pct = int((completed_merges * 100) / merges_to_learn)
            while completed_pct >= next_progress_pct and next_progress_pct <= 100:
                print(
                    f"[train_bpe_fast] progress: {next_progress_pct}% ({completed_merges}/{merges_to_learn} merges)",
                    flush=True,
                )
                next_progress_pct += safe_interval_pct

    if show_progress and merges_to_learn > 0 and next_progress_pct <= 100:
        completed_merges = len(merges)
        completed_pct = int((completed_merges * 100) / merges_to_learn)
        while completed_pct >= next_progress_pct and next_progress_pct <= 100:
            print(
                f"[train_bpe_fast] progress: {next_progress_pct}% ({completed_merges}/{merges_to_learn} merges)",
                flush=True,
            )
            next_progress_pct += safe_interval_pct

    return vocab, merges
