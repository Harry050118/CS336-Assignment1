from __future__ import annotations

import json
import os
import re as _re
from collections.abc import Iterable, Iterator
import regex as re
from cs336_basics.tokenizer.utils import string_to_bytes, utf8_bytes_to_string

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab: dict[int, bytes] = dict(vocab)
        self.merges = list(merges)

        self.special_tokens = special_tokens or []
        vocab_values = set(self.vocab.values())
        for token in self.special_tokens:
            token_bytes = b"".join(string_to_bytes(token))
            if token_bytes not in vocab_values:
                self.vocab[len(self.vocab)] = token_bytes
                vocab_values.add(token_bytes)

        self.token_to_id: dict[bytes, int] = {
            token_bytes: token_id for token_id, token_bytes in self.vocab.items()
        }
        self.merge_ranks: dict[tuple[bytes, bytes], int] = {
            pair: rank for rank, pair in enumerate(self.merges)
        }

        if self.special_tokens:
            escaped = [_re.escape(token) for token in sorted(set(self.special_tokens), key=len, reverse=True)]
            self._special_split_re = _re.compile("|".join(escaped))
        else:
            self._special_split_re = None

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> "BPETokenizer":
        with open(vocab_filepath, encoding="utf-8") as vocab_file:
            vocab_json = json.load(vocab_file)
            vocab = {int(token_id): token_str.encode("latin1") for token_str, token_id in vocab_json.items()}

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, encoding="utf-8") as merges_file:
            for line in merges_file:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                token_1, token_2 = line.split(" ", 1)
                merges.append((token_1.encode("latin1"), token_2.encode("latin1")))

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def _split_with_special_tokens(self, text: str) -> list[tuple[bool, str]]:
        if not text:
            return []
        if self._special_split_re is None:
            return [(False, text)]

        parts: list[tuple[bool, str]] = []
        start = 0
        for match in self._special_split_re.finditer(text):
            if match.start() > start:
                parts.append((False, text[start : match.start()]))
            parts.append((True, match.group(0)))
            start = match.end()

        if start < len(text):
            parts.append((False, text[start:]))

        return parts

    def _merge_pretoken_bytes(self, pretoken: str) -> list[bytes]:
        symbols = string_to_bytes(pretoken)
        if len(symbols) < 2:
            return symbols

        while True:
            best_rank: int | None = None
            best_pair: tuple[bytes, bytes] | None = None

            for idx in range(len(symbols) - 1):
                pair = (symbols[idx], symbols[idx + 1])
                rank = self.merge_ranks.get(pair)
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            merged: list[bytes] = []
            i = 0
            while i < len(symbols):
                if i + 1 < len(symbols) and symbols[i] == best_pair[0] and symbols[i + 1] == best_pair[1]:
                    merged.append(symbols[i] + symbols[i + 1])
                    i += 2
                else:
                    merged.append(symbols[i])
                    i += 1
            symbols = merged

        return symbols

    def encode(self, text: str) -> list[int]:
        if not text:
            return []

        ids: list[int] = []
        parts = self._split_with_special_tokens(text)

        for is_special, part in parts:
            if is_special:
                special_bytes = b"".join(string_to_bytes(part))
                token_id = self.token_to_id.get(special_bytes)
                if token_id is None:
                    raise ValueError(f"Unknown special token: {part}")
                ids.append(token_id)
                continue

            for match in re.finditer(PAT, part):
                pretoken = match.group(0)
                merged_symbols = self._merge_pretoken_bytes(pretoken)
                for symbol in merged_symbols:
                    token_id = self.token_to_id.get(symbol)
                    if token_id is None:
                        for byte in symbol:
                            ids.append(self.token_to_id[bytes([byte])])
                    else:
                        ids.append(token_id)

        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        token_chunks: list[bytes] = [self.vocab.get(token_id, b"\xef\xbf\xbd") for token_id in ids]

        try:
            return utf8_bytes_to_string(token_chunks)
        except UnicodeDecodeError:
            return b"".join(token_chunks).decode("utf-8", errors="replace")


Tokenizer = BPETokenizer
