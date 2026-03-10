import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from cs336_basics.tokenizer.tokenizer import BPETokenizer
from cs336_basics.tokenizer.train_bpe import train_bpe_fast


TRAIN_TXT = Path('/mnt/dataset0/gjt/CS336/dataset/TinyStoriesV2-GPT4-train.txt')
VAL_TXT = Path('/mnt/dataset0/gjt/CS336/dataset/TinyStoriesV2-GPT4-valid.txt')
OUT_DIR = Path('/mnt/dataset0/gjt/CS336/dataset/TinyStoriesV2-GPT4-tokenizer')

VOCAB_PATH = OUT_DIR / 'vocab.json'
MERGES_PATH = OUT_DIR / 'merges.txt'
TRAIN_NPY_PATH = OUT_DIR / 'train.npy'
VAL_NPY_PATH = OUT_DIR / 'val.npy'

VOCAB_SIZE = 10000
SPECIAL_TOKENS = ['<|endoftext|>']


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
            f.write(f"{left.decode('latin1')} {right.decode('latin1')}\\n")


def encode_txt_to_npy(tokenizer: BPETokenizer, txt_path: Path, npy_path: Path) -> None:
    text = txt_path.read_text(encoding='utf-8', errors='ignore')
    token_ids = tokenizer.encode(text)
    np.save(npy_path, np.asarray(token_ids, dtype=np.int32))
    print(f'Saved {npy_path} (num_tokens={len(token_ids)})')


def main() -> None:
    if not TRAIN_TXT.exists() or not VAL_TXT.exists():
        raise FileNotFoundError(
            'Input txt file not found. Expected:\n'
            f'  {TRAIN_TXT}\n'
            f'  {VAL_TXT}'
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f'Training tokenizer on {TRAIN_TXT} ...')
    vocab, merges = train_bpe_fast(
        input_path=TRAIN_TXT,
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
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
