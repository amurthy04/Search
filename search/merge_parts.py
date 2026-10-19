"""Merge data/index/parts/*.{npy,meta,passages,offsets} into one emb prefix for index/query. Doc ids are re-based."""
import argparse, glob, io, os
import numpy as np, orjson, zstandard as zstd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=["data/index/parts"]); ap.add_argument("--out", default="data/index/emb")
    ap.add_argument("--dedupe", action="store_true", help="drop passages whose exact text was already seen (syndicated copies)")
    a = ap.parse_args()
    import xxhash
    names = sorted(p[:-4] for d in a.parts for p in glob.glob(os.path.join(d, "*.npy")) if not p.endswith(".offsets.npy"))
    seen_hash = set(); dropped = 0
    vecs = []; doc_base = 0; offsets = [0]
    meta_fh = open(a.out + ".meta.jsonl.zst", "wb"); mw = zstd.ZstdCompressor(level=3).stream_writer(meta_fh)
    pfh = open(a.out + ".passages.bin", "wb")
    n = 0
    for p in names:
        v = np.load(p + ".npy")
        metas = []
        max_d = -1
        with open(p + ".meta.jsonl.zst", "rb") as fh:
            with zstd.ZstdDecompressor().stream_reader(fh) as r:
                for line in io.TextIOWrapper(r, encoding="utf-8"):
                    if line.strip():
                        m = orjson.loads(line); max_d = max(max_d, m["d"]); m["d"] += doc_base; metas.append(m)
        doc_base += max_d + 1
        with open(p + ".passages.bin", "rb") as fh:
            data = fh.read()
        off = np.load(p + ".offsets.npy")
        keep = np.ones(len(v), dtype=bool)
        if a.dedupe:
            for j in range(len(v)):
                h = xxhash.xxh64_intdigest(data[off[j]:off[j + 1]])
                if h in seen_hash:
                    keep[j] = False; dropped += 1
                else:
                    seen_hash.add(h)
        vecs.append(v[keep])
        for j in np.flatnonzero(keep):
            mw.write(orjson.dumps(metas[j]) + b"\n")
            chunk = data[off[j]:off[j + 1]]; pfh.write(chunk); offsets.append(offsets[-1] + len(chunk))
        n += int(keep.sum())
    mw.close(); meta_fh.close(); pfh.close()
    np.save(a.out + ".offsets.npy", np.asarray(offsets, dtype=np.int64))
    np.save(a.out + ".npy", np.concatenate(vecs) if vecs else np.zeros((0, 384), np.float16))
    print(f"merged {len(names)} parts -> {n} passages ({dropped} exact duplicates dropped), {doc_base} docs -> {a.out}.npy")


if __name__ == "__main__":
    main()
