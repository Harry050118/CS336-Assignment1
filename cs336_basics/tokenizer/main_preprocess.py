import json
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from cs336_basics.tokenizer.tokenizer import BPETokenizer
from cs336_basics.tokenizer.train_bpe import train_bpe_fast
from cs336_basics.tokenizer.utils import find_chunk_boundaries


TRAIN_TXT = Path('/mnt/dataset0/gjt/CS336/dataset/TinyStoriesV2-GPT4-train.txt')
VAL_TXT = Path('/mnt/dataset0/gjt/CS336/dataset/TinyStoriesV2-GPT4-valid.txt')
OUT_DIR = Path('/mnt/dataset0/gjt/CS336/dataset/TinyStoriesV2-GPT4-tokenizer')

VOCAB_PATH = OUT_DIR / 'vocab.json'
MERGES_PATH = OUT_DIR / 'merges.txt'
TRAIN_NPY_PATH = OUT_DIR / 'train.npy'
VAL_NPY_PATH = OUT_DIR / 'val.npy'

VOCAB_SIZE = 10000
SPECIAL_TOKENS = ['<|endoftext|>']
PROGRESS_INTERVAL_PCT = 5


def save_tokenizer_artifacts(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    vocab_path: Path,
    merges_path: Path,
) -> None:
    vocab_json = {token_bytes.decode('latin1'): token_id for token_id, token_bytes in vocab.items()}
    with vocab_path.open('w', encoding='utf-8') as f:
        json.dump(vocab_json, f, ensure_ascii=False, indent=2)

    with merges_path.open('w', encoding='utf-8') as f:
        for left, right in merges:
            f.write(f"{left.decode('latin1')} {right.decode('latin1')}\n")


def encode_txt_to_npy(tokenizer: BPETokenizer, txt_path: Path, npy_path: Path) -> None:
    split_special_token = b'<|endoftext|>'
    with open(txt_path, 'rb') as file:
        boundaries = find_chunk_boundaries(
            file=file,
            desired_num_chunks=max(4, min(16, (os.cpu_count() or 1))),
            split_special_token=split_special_token,
        )

    chunk_ranges = list(zip(boundaries[:-1], boundaries[1:]))
    all_token_ids = []

    with open(txt_path, 'rb') as file:
        for start, end in tqdm(chunk_ranges, desc=f'encode {txt_path.name}', unit='chunk'):
            file.seek(start)
            chunk_bytes = file.read(end - start)
            chunk_text = chunk_bytes.decode('utf-8', errors='ignore')
            all_token_ids.extend(tokenizer.encode(chunk_text))

    np.save(npy_path, np.asarray(all_token_ids, dtype=np.int32))
    print(f'Saved {npy_path} (num_tokens={len(all_token_ids)})')


def main() -> None:
    if not TRAIN_TXT.exists() or not VAL_TXT.exists():
        raise FileNotFoundError(
            'Input txt file not found. Expected:\n'
            f'  {TRAIN_TXT}\n'
            f'  {VAL_TXT}'
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f'Training tokenizer on {TRAIN_TXT} ...')
    initial_vocab_size = 256 + len(SPECIAL_TOKENS)
    estimated_merges = max(0, VOCAB_SIZE - initial_vocab_size)
    print(
        f'Training progress logs every {PROGRESS_INTERVAL_PCT}% '
        f'(target merges ~= {estimated_merges})'
    )
    vocab, merges = train_bpe_fast(
        input_path=TRAIN_TXT,
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        progress_interval_pct=PROGRESS_INTERVAL_PCT,
    )

    save_tokenizer_artifacts(vocab, merges, VOCAB_PATH, MERGES_PATH)
    print(f'Saved tokenizer artifacts: {VOCAB_PATH}, {MERGES_PATH}')

    tokenizer = BPETokenizer(vocab=vocab, merges=merges, special_tokens=SPECIAL_TOKENS)
    tasks = [
        (TRAIN_TXT, TRAIN_NPY_PATH),
        (VAL_TXT, VAL_NPY_PATH),
    ]
    for txt_path, npy_path in tqdm(tasks, desc='tokenize txt files', unit='file'):
        encode_txt_to_npy(tokenizer, txt_path, npy_path)

    print('Done.')
    print(f'Train tokens: {TRAIN_NPY_PATH}')
    print(f'Val tokens:   {VAL_NPY_PATH}')


if __name__ == '__main__':
    main()
