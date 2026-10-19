"""Merge data/index/parts/*.{npy,meta,passages,offsets} into one emb prefix for index/query. Doc ids are re-based."""
import argparse, glob, io, os
import numpy as np, orjson, zstandard as zstd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="data/index/parts"); ap.add_argument("--out", default="data/index/emb")
    a = ap.parse_args()
    names = sorted(p[:-4] for p in glob.glob(os.path.join(a.parts, "*.npy")) if not p.endswith(".offsets.npy"))
    vecs = []; doc_base = 0; off_base = 0; offsets = [0]
    meta_fh = open(a.out + ".meta.jsonl.zst", "wb"); mw = zstd.ZstdCompressor(level=3).stream_writer(meta_fh)
    pfh = open(a.out + ".passages.bin", "wb")
    n = 0
    for p in names:
        v = np.load(p + ".npy"); vecs.append(v)
        max_d = -1
        with open(p + ".meta.jsonl.zst", "rb") as fh:
            with zstd.ZstdDecompressor().stream_reader(fh) as r:
                for line in io.TextIOWrapper(r, encoding="utf-8"):
                    if line.strip():
                        m = orjson.loads(line); max_d = max(max_d, m["d"]); m["d"] += doc_base
                        mw.write(orjson.dumps(m) + b"\n")
        doc_base += max_d + 1
        with open(p + ".passages.bin", "rb") as fh:
            data = fh.read(); pfh.write(data)
        off = np.load(p + ".offsets.npy")
        offsets.extend((off[1:] + off_base).tolist()); off_base += len(data)
        n += len(v)
    mw.close(); meta_fh.close(); pfh.close()
    np.save(a.out + ".offsets.npy", np.asarray(offsets, dtype=np.int64))
    np.save(a.out + ".npy", np.concatenate(vecs) if vecs else np.zeros((0, 384), np.float16))
    print(f"merged {len(names)} parts -> {n} passages, {doc_base} docs -> {a.out}.npy")


if __name__ == "__main__":
    main()
