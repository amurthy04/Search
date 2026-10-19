"""Read crawl shards and produce passages for embedding."""
from __future__ import annotations

import glob
import io
import re
from typing import Iterator

import orjson
import zstandard as zstd

_word_re = re.compile(r"\S+")


def iter_records(pattern: str) -> Iterator[dict]:
    """Stream records from zstd JSONL shards. A shard whose writer was killed mid-stream has a truncated tail:
    we stop at the first undecodable line/frame and move on (a warning goes to stderr)."""
    import sys
    for path in sorted(glob.glob(pattern)):
        n = 0
        try:
            with open(path, "rb") as fh:
                dctx = zstd.ZstdDecompressor()
                with dctx.stream_reader(fh) as reader:
                    for line in io.TextIOWrapper(reader, encoding="utf-8", errors="replace"):
                        if line.strip():
                            try:
                                rec = orjson.loads(line)
                            except orjson.JSONDecodeError:
                                print(f"[corpus] truncated record at end of {path} after {n} records", file=sys.stderr)
                                break
                            n += 1
                            yield rec
        except (zstd.ZstdError, OSError) as e:
            print(f"[corpus] {path}: {e!r} after {n} records", file=sys.stderr)


def chunk_text(text: str, words_per_chunk: int = 180, overlap: int = 30, max_chunks: int = 6) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(1, words_per_chunk - overlap)
    for start in range(0, len(words), step):
        chunk = words[start:start + words_per_chunk]
        if len(chunk) < 20 and chunks:  # tail too small; merge skip
            break
        chunks.append(" ".join(chunk))
        if len(chunks) >= max_chunks or start + words_per_chunk >= len(words):
            break
    return chunks


def iter_passages(pattern: str, words_per_chunk=180, overlap=30, max_chunks=6, dedupe=True):
    """Yield (doc_id, chunk_id, url, title, passage_text) across all shards."""
    seen_urls = set()
    doc_id = 0
    for rec in iter_records(pattern):
        url = rec["url"]
        if dedupe:
            if url in seen_urls:
                continue
            seen_urls.add(url)
        title = (rec.get("title") or "").strip()[:200]
        for ci, ch in enumerate(chunk_text(rec.get("text", ""), words_per_chunk, overlap, max_chunks)):
            yield doc_id, ci, url, title, (f"{title}\n{ch}" if title else ch)
        doc_id += 1
