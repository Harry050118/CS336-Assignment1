import json
import os
import re
import time
from collections import defaultdict
from functools import wraps
from typing import BinaryIO, Callable

def print_color(content: str, color: str = "green"):
    print(f"[{color}]{content}[/{color}]")


# ─────────────────────────────────────────────
# 1. 文件分块
# ─────────────────────────────────────────────

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    把大文件切成若干块，每块边界对齐到 split_special_token。
    适合多进程并行训练时使用。
    实际返回的块数可能少于 desired_num_chunks（边界重叠时自动去重）。
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


# ─────────────────────────────────────────────
# 2. 字节 / 字符串 互转
# ─────────────────────────────────────────────

def string_to_bytes(s: str, return_int: bool = False) -> list[int] | list[bytes]:
    """
    字符串 → UTF-8 字节列表。
    return_int=False → [b'H', b'e', ...]
    return_int=True  → [72, 101, ...]
    """
    byte_array = s.encode("utf-8")
    return list(map(int, byte_array)) if return_int else [bytes([b]) for b in byte_array]


def utf8_bytes_to_string(byte_indices: list[bytes]) -> str:
    """
    字节列表 → UTF-8 字符串，是 string_to_bytes 的逆操作。
    """
    return b"".join(byte_indices).decode("utf-8")


# ─────────────────────────────────────────────
# 3. 语料预处理
# ─────────────────────────────────────────────

def preprocess_text(
    text: str,
    lowercase: bool = False,
    remove_urls: bool = True,
    remove_html: bool = True,
    normalize_whitespace: bool = True,
) -> str:
    """
    训练前清洗原始语料。
    - lowercase: 是否全部转小写
    - remove_urls: 是否移除 URL
    - remove_html: 是否移除 HTML 标签
    - normalize_whitespace: 是否合并多余空白
    """
    if remove_urls:
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
    if remove_html:
        text = re.sub(r"<[^>]+>", "", text)
    if lowercase:
        text = text.lower()
    if normalize_whitespace:
        # 统一换行符
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # 合并连续空格
        text = re.sub(r" +", " ", text)
        # 合并超过两个的连续换行
        text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ─────────────────────────────────────────────
# 4. 语料基本信息统计
# ─────────────────────────────────────────────

def corpus_stats(file: BinaryIO) -> dict:
    """
    训练前快速了解语料规模。
    返回文件大小、字符数、行数、唯一字符数等基本信息。
    """
    file.seek(0)
    content = file.read().decode("utf-8", errors="replace")
    file.seek(0)

    num_bytes = len(content.encode("utf-8"))
    lines     = content.split("\n")

    stats = {
        "file_size_mb":   round(num_bytes / 1024 / 1024, 2),
        "total_chars":    len(content),
        "total_lines":    len(lines),
        "non_empty_lines":sum(1 for l in lines if l.strip()),
        "total_words":    len(content.split()),
        "unique_chars":   len(set(content)),
        "unique_bytes":   len(set(content.encode("utf-8"))),
    }

    print("[CORPUS STATS]")
    for k, v in stats.items():
        print(f"  {k:<20} {v}")

    return stats


# ─────────────────────────────────────────────
# 5. 保存 / 加载 词表和合并规则
# ─────────────────────────────────────────────

def save_vocab_and_merges(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    output_dir: str | os.PathLike,
) -> None:
    """
    把训练结果持久化到磁盘：
    - vocab.json  : {token字符串: id}
    - merges.txt  : 每行一条合并规则
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    vocab_filepath  = os.path.join(output_dir, "vocab.json")
    merges_filepath = os.path.join(output_dir, "merges.txt")

    vocab_inv = {v.decode("latin1"): k for k, v in vocab.items()}
    with open(vocab_filepath, "w", encoding="utf-8") as vf:
        json.dump(vocab_inv, vf, ensure_ascii=False, indent=2)

    with open(merges_filepath, "w", encoding="utf-8") as mf:
        mf.write("#version: 0.2\n")
        for a, b in merges:
            mf.write(f"{a.decode('latin1')} {b.decode('latin1')}\n")

    print(f"[SAVE] vocab  → {vocab_filepath}")
    print(f"[SAVE] merges → {merges_filepath}")


def load_vocab_and_merges(
    output_dir: str | os.PathLike,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    从磁盘加载词表和合并规则，是 save_vocab_and_merges 的逆操作。
    返回 (vocab, merges)，格式与训练时完全一致。
    """
    vocab_filepath  = os.path.join(output_dir, "vocab.json")
    merges_filepath = os.path.join(output_dir, "merges.txt")

    assert os.path.exists(vocab_filepath),  f"找不到词表文件: {vocab_filepath}"
    assert os.path.exists(merges_filepath), f"找不到合并文件: {merges_filepath}"

    with open(vocab_filepath, "r", encoding="utf-8") as vf:
        vocab_inv = json.load(vf)
        vocab = {v: k.encode("latin1") for k, v in vocab_inv.items()}

    merges = []
    with open(merges_filepath, "r", encoding="utf-8") as mf:
        for line in mf:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            a, b = line.split(" ", 1)
            merges.append((a.encode("latin1"), b.encode("latin1")))

    print(f"[LOAD] vocab size  : {len(vocab)}")
    print(f"[LOAD] merges count: {len(merges)}")

    return vocab, merges


# ─────────────────────────────────────────────
# 6. encode / decode 一致性验证
# ─────────────────────────────────────────────

def validate_roundtrip(
    text: str,
    encode_fn: Callable[[str], list[int]],
    decode_fn: Callable[[list[int]], str],
    sample_size: int = 200,
) -> bool:
    """
    验证 encode → decode 后能否完整还原原文。
    训练完成后建议立即调用此函数检查结果正确性。
    """
    sample  = text[:sample_size]
    encoded = encode_fn(sample)
    decoded = decode_fn(encoded)

    if sample == decoded:
        print(f"✅ roundtrip 验证通过（样本长度: {len(sample)} 字符）")
        return True
    else:
        # 找到第一个不同的位置
        for i, (a, b) in enumerate(zip(sample, decoded)):
            if a != b:
                print(f"❌ 第 {i} 个字符不一致: 原文={repr(a)}, 还原={repr(b)}")
                break
        print(f"  原文  : {repr(sample[:50])}")
        print(f"  还原  : {repr(decoded[:50])}")
        return False


# ─────────────────────────────────────────────
# 7. 词表覆盖率分析
# ─────────────────────────────────────────────

def vocab_coverage(
    text: str,
    encode_fn: Callable[[str], list[int]],
) -> dict:
    """
    评估训练好的词表对语料的压缩效果。
    compression_ratio 越低说明词表压缩能力越强。
    """
    original_token_count = len(text.encode("utf-8"))  # 原始字节数
    encoded              = encode_fn(text)
    encoded_count        = len(encoded)
    compression          = encoded_count / original_token_count

    result = {
        "original_bytes":    original_token_count,
        "encoded_tokens":    encoded_count,
        "compression_ratio": f"{compression:.2%}",
        "avg_bytes_per_token": round(original_token_count / encoded_count, 2),
    }

    print("[VOCAB COVERAGE]")
    for k, v in result.items():
        print(f"  {k:<25} {v}")

    return result


# ─────────────────────────────────────────────
# 8. 进度条
# ─────────────────────────────────────────────

def progress_bar(iterable, total: int, desc: str = "Training"):
    """
    轻量级进度条，不依赖第三方库。
    用法：for item in progress_bar(my_list, total=len(my_list)):
    """
    start = time.time()
    for i, item in enumerate(iterable):
        yield item
        elapsed  = time.time() - start
        percent  = (i + 1) / total * 100
        filled   = int(percent // 5)
        bar      = "█" * filled + "░" * (20 - filled)
        eta      = (elapsed / (i + 1)) * (total - i - 1) if i > 0 else 0
        print(
            f"\r{desc} [{bar}] {percent:5.1f}%  "
            f"elapsed: {elapsed:5.1f}s  eta: {eta:5.1f}s",
            end="",
            flush=True,
        )
    print()  # 换行


# ─────────────────────────────────────────────
# 9. 计时装饰器
# ─────────────────────────────────────────────

def timeit(func):
    """
    给任意函数加上计时功能，训练时监控每步耗时。
    用法：@timeit
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start  = time.time()
        result = func(*args, **kwargs)
        end    = time.time()
        print(f"[TIME] {func.__name__} took {end - start:.2f}s")
        return result
    return wrapper

# ==================== Appended notebook support helpers (non-breaking) ====================
def measure_peak_rss_during(fn, interval: float = 0.05):
    import os
    import threading
    import time
    import psutil

    p = psutil.Process(os.getpid())
    peak = {"rss": p.memory_info().rss}
    stop = {"flag": False}

    def sampler():
        while not stop["flag"]:
            rss = p.memory_info().rss
            if rss > peak["rss"]:
                peak["rss"] = rss
            time.sleep(interval)

    t = threading.Thread(target=sampler, daemon=True)
    t.start()
    t0 = time.perf_counter()
    out = fn()
    secs = time.perf_counter() - t0
    stop["flag"] = True
    t.join(timeout=1.0)
    return out, secs, peak["rss"]


def longest_token(vocab: dict[int, bytes]) -> tuple[bytes, int, str]:
    tok = max(vocab.values(), key=len)
    return tok, len(tok), tok.decode("utf-8", errors="replace")


def make_profile_subset(
    input_path: str | os.PathLike,
    output_path: str | os.PathLike,
    max_bytes: int = 64 * 1024 * 1024,
) -> str:
    input_path = str(input_path)
    output_path = str(output_path)
    with open(input_path, "rb") as src, open(output_path, "wb") as dst:
        dst.write(src.read(max_bytes))
    return output_path
