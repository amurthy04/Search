"""One IVF-PQ index over every stored vector (Wikipedia rows + web passages), searched in parallel shards, with
exact re-scoring from the fp16 row files.

  uv run python -m search.ann build            # GPU k-means + assignment, PQ codes, shards -> data/ivf/
  uv run python -m search.ann build --nlist 16384 --m 192 --wiki-shards 8

Search path (one query):
  1. coarse: scores against the 16,384 centroids (one matvec), top `nprobe` lists
  2. scan: each shard scans those lists' 4-bit PQ codes (faiss fast-scan) on its own thread; faiss releases the
     GIL, so 8 shards take about as long as one
  3. re-score: the best `refine` candidates are scored exactly against their fp16 vectors, which stay on disk.
     Every candidate's page is requested at once (madvise WILLNEED) before any is read: random 768-byte reads
     one after another cost ~100 us each from the SSD, issued together ~10 us each.

Measured on NQ questions against brute force over all 42.7M Wikipedia vectors (bench/wiki_ann.py): recall is set
by the share of the index scanned (~1.5% for 95% of the true top 10) more than by the number of lists, and PQ192x4
(192 sub-vectors of 2 dimensions, 4 bits each: 96 B/vector) keeps the true top 10 inside the best ~100 candidates,
where PQ96x4 (48 B) does not.

Rows appended after a build (the hourly Wikipedia updates) are encoded into delta segments when the reader loads
or reloads, so writers never touch the index files. Rebuild once the delta grows large.
"""
from __future__ import annotations

import argparse
import json
import math
import mmap
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

IVF_DIR = "data/ivf"
DIM = 384
WIKI_F16 = "data/wiki/index/wiki.f16"
WIKI_STATE = "data/wiki/index/wiki.state.json"
WEB_EMB = "data/index/emb_all.npy"


def built_sources(d: str = IVF_DIR) -> set[str]:
    """Sources the index in `d` was built over (empty if there is none)."""
    mp = os.path.join(d, "meta.json")
    return set(json.load(open(mp))["sources"]) if os.path.exists(mp) else set()


def will_need(mm, ranges):
    """Request the pages of every (offset, length) range at once (MADV_WILLNEED), so random reads that follow are
    served together instead of one SSD trip after another. Ranges past the end of the mapping are clipped."""
    pg, size = mmap.PAGESIZE, len(mm)
    for off, length in ranges:
        end = min(off + length, size)
        if off < end:
            b = off - off % pg
            mm.madvise(mmap.MADV_WILLNEED, b, end - b)


def tail_merge(segs: list[tuple[int, int]], unit: int, fan_in: int = 4) -> int:
    """Size-tiered merging (like an LSM tree): the newest `fan_in` segments become one while they are on the same
    size level (level = log_fan_in(rows / unit)), so each row is re-encoded O(log n) times. Returns how many of the
    newest segments to merge now (0 or fan_in)."""
    if len(segs) < fan_in:
        return 0
    lv = {int(math.log(max(1.0, (b - a) / unit), fan_in)) for a, b in segs[-fan_in:]}
    return fan_in if len(lv) == 1 else 0


class Rows:
    """fp16 row file (raw, or a .npy) memory-mapped, with concurrent prefetch for random row reads."""

    def __init__(self, path: str, n: int | None = None):
        fd = os.open(path, os.O_RDONLY)
        try:
            self.mm = mmap.mmap(fd, 0, prot=mmap.PROT_READ)
        finally:
            os.close(fd)  # the mapping keeps the file open
        off = 0
        if path.endswith(".npy"):
            m = np.load(path, mmap_mode="r")
            assert m.dtype == np.float16 and m.shape[1] == DIM, f"{path}: {m.dtype} {m.shape}"
            off = m.offset; del m
        self.off = off
        rows = (len(self.mm) - off) // (DIM * 2)
        self.n = rows if n is None else min(n, rows)
        self.a = np.frombuffer(self.mm, dtype=np.float16, count=self.n * DIM, offset=off).reshape(self.n, DIM)

    def prefetch(self, ids: np.ndarray):
        will_need(self.mm, ((self.off + int(i) * DIM * 2, DIM * 2) for i in ids))

    def get(self, ids: np.ndarray) -> np.ndarray:
        return self.a[ids].astype(np.float32)


class Blob:
    """Read-only mmap of a byte file (passage texts) with concurrent prefetch of the ranges about to be read."""

    def __init__(self, path: str):
        self.mm = b""
        if os.path.getsize(path):
            fd = os.open(path, os.O_RDONLY)
            try:
                self.mm = mmap.mmap(fd, 0, prot=mmap.PROT_READ)
            finally:
                os.close(fd)

    def prefetch(self, ranges):
        """ranges: (offset, length) pairs about to be read."""
        if self.mm:
            will_need(self.mm, ranges)

    def read(self, off: int, length: int) -> bytes:
        return self.mm[off:off + length]


def _sources() -> dict[str, Rows]:
    """Every vector source that exists: Wikipedia rows (as far as its state file covers) and the web passages."""
    out = {}
    if os.path.exists(WIKI_STATE) and os.path.exists(WIKI_F16):
        out["wiki"] = Rows(WIKI_F16, json.load(open(WIKI_STATE))["n"])
    if os.path.exists(WEB_EMB):
        out["web"] = Rows(WEB_EMB)
    assert out, f"no vectors found ({WIKI_F16}, {WEB_EMB})"
    return out


class IVF:
    """Reader: coarse centroids, per-source shards (the build's, then delta segments for rows added since), one
    thread pool for the scans."""

    def __init__(self, d: str = IVF_DIR, nprobe: int = 384, threads: int = 14):
        import faiss
        self.faiss, self.d = faiss, d
        self.meta = json.load(open(os.path.join(d, "meta.json")))
        self.cent = np.ascontiguousarray(np.load(os.path.join(d, "centroids.npy")), dtype=np.float32)
        self.nlist = len(self.cent)
        self.shards: dict[str, tuple] = {
            src: tuple(faiss.read_index(os.path.join(d, f"{src}.shard{i:02d}.ivf")) for i in range(m["shards"]))
            for src, m in self.meta["sources"].items()}
        self.covered = {src: m["n"] for src, m in self.meta["sources"].items()}
        self.delta: dict[str, list] = {src: [] for src in self.shards}  # per source: [(lo, hi, index)] after the build
        self.pool = ThreadPoolExecutor(threads, thread_name_prefix="ivf")
        self._dlock = threading.Lock()     # swaps of a source's shard tuple
        self._xlock = threading.Lock()     # extend() calls, one at a time
        self._enc_lock = threading.Lock()  # first encode() loads the trained PQ
        self._proto = None
        self.nprobe = nprobe

    @property
    def nprobe(self) -> int:
        return self._nprobe

    @nprobe.setter
    def nprobe(self, v: int):
        # Fixed per shard and never changed by a search: faiss's search_preassigned wants the probe array exactly
        # nprobe wide, so a per-call nprobe would race between concurrent queries. Shards that hold only some of
        # the lists get a padded probe array instead (see search()).
        self._nprobe = v
        for shards in list(self.shards.values()):
            for idx in shards:
                idx.nprobe = v

    def coarse(self, q: np.ndarray):
        npb = self._nprobe
        s = self.cent @ q
        top = np.argpartition(-s, npb - 1)[:npb]
        top = top[np.argsort(-s[top])]
        return s[top][None, :].astype(np.float32), top[None, :].astype(np.int64)

    def search(self, q: np.ndarray, k: dict[str, int], extra=()):
        """q: (DIM,) float32. k: candidates wanted per source. extra: more callables to run on the pool alongside
        the shard scans (their results are returned in order). Returns ({source: (ids, approx scores)}, extras)."""
        npb = self._nprobe
        Dq, Iq = self.coarse(q)
        x = q[None, :]
        jobs = []
        for src, kk in k.items():
            m = self.meta["sources"].get(src, {})
            nb = m.get("shards", 0) if m.get("split") == "list" else 0
            for s, idx in enumerate(self.shards.get(src, ())):
                if s < nb:  # holds lists l with l % nb == s: its probes, padded to nprobe with -1 (skipped)
                    sel = np.flatnonzero(Iq[0] % nb == s)
                    if not len(sel):
                        continue
                    a = np.full((1, npb), -1, np.int64); a[0, :len(sel)] = Iq[0, sel]
                    # pad scores with the lowest real one: fast-scan scales its lookup tables by the score range,
                    # and this keeps the results identical to an exactly-sized call (checked on 200 queries)
                    d = np.full((1, npb), Dq[0, sel[-1]], np.float32); d[0, :len(sel)] = Dq[0, sel]
                else:  # a delta or live segment: every list
                    a, d = Iq, Dq
                jobs.append((src, self.pool.submit(self._scan, idx, x, kk, a, d)))
        ex = [self.pool.submit(f) for f in extra]
        out: dict[str, list] = {}
        for src, f in jobs:
            out.setdefault(src, []).append(f.result())
        res = {src: (np.zeros(0, np.int64), np.zeros(0, np.float32)) for src in k}
        for src, parts in out.items():
            D = np.concatenate([p[0] for p in parts]); I = np.concatenate([p[1] for p in parts])
            keep = I >= 0
            D, I = D[keep], I[keep]
            o = np.argsort(-D)[:k[src]]
            res[src] = (I[o], D[o])
        return res, [f.result() for f in ex]

    @staticmethod
    def _scan(idx, x, k, Iq, Dq):
        D, I = idx.search_preassigned(x, k, Iq, Dq)
        return D[0], I[0]

    def encode(self, x: np.ndarray, ids: np.ndarray):
        """A fast-scan shard holding vectors x (fp16 or fp32 rows) under ids, with this index's centroids and PQ."""
        faiss = self.faiss
        from faiss.contrib.ivf_tools import add_preassigned
        with self._enc_lock:
            if self._proto is None:
                self._quant = faiss.IndexFlatIP(DIM); self._quant.add(self.cent)
                self._placeholder = faiss.IndexFlatIP(DIM)
                self._proto = faiss.read_index(os.path.join(self.d, "proto_pq.ivf"))
        pq = faiss.clone_index(self._proto)
        pq.quantizer = self._quant  # residuals are taken against the real centroids
        pq.own_fields = False       # clone_index set it: freeing pq would otherwise free the shared quantizer
        ids = np.ascontiguousarray(ids, dtype=np.int64)
        for a in range(0, len(x), 65536):  # x may be fp16: converted a block at a time
            xb = np.ascontiguousarray(x[a:a + 65536], dtype=np.float32)
            asg = np.empty(len(xb), np.int64)
            for i in range(0, len(xb), 2048):
                asg[i:i + 2048] = np.argmax(xb[i:i + 2048] @ self.cent.T, axis=1)
            ib = ids[a:a + 65536].copy()  # kept alive: add_preassigned holds only a raw pointer
            add_preassigned(pq, xb, asg, ids=ib)
        fs = faiss.IndexIVFPQFastScan(pq)  # copies the codes; does not own its quantizer
        fs.referenced_objects = []         # faiss's wrapper keeps `pq` alive with it: a second copy of every code
        fs.quantizer = self._placeholder   # searched preassigned, so the centroids are not needed here
        fs.nprobe = self._nprobe
        return fs

    def extend(self, src: str, rows: Rows, n: int) -> int:
        """Make rows [covered, n) of `src` searchable (Wikipedia rows appended by the hourly updates). Rows new since
        the last call become one more delta segment, merged size-tiered (see tail_merge), and every segment is saved
        so the next start loads them instead of encoding again. Returns the number of rows added."""
        with self._xlock:
            segs = list(self.delta[src]) or self._load_delta(src, n)
            hi = segs[-1][1] if segs else self.covered[src]
            added = max(0, n - hi)
            if added:
                segs.append((hi, n, self.encode(rows.a[hi:n], np.arange(hi, n))))
                while m := tail_merge([(a, b) for a, b, _ in segs], 16384):
                    a, b = segs[-m][0], segs[-1][1]
                    segs[-m:] = [(a, b, self.encode(rows.a[a:b], np.arange(a, b)))]
                self._save_delta(src, segs)
            if segs != self.delta[src]:
                with self._dlock:
                    base = self.shards[src][:self.meta["sources"][src]["shards"]]
                    self.shards[src] = base + tuple(idx for _, _, idx in segs)
                    self.delta[src] = segs
            return added

    def _delta_meta(self, src: str) -> str:
        return os.path.join(self.d, f"{src}.delta.json")

    def _load_delta(self, src: str, n: int) -> list:
        mp = self._delta_meta(src)
        if not os.path.exists(mp):
            return []
        m = json.load(open(mp))
        if m.get("built") != self.meta.get("built"):
            return []  # another build's codes
        out, lo = [], self.covered[src]
        for seg in m.get("segs", ()):
            if seg["lo"] != lo or seg["hi"] > n or not os.path.exists(os.path.join(self.d, seg["file"])):
                break  # keep the contiguous prefix that still matches the row file
            idx = self.faiss.read_index(os.path.join(self.d, seg["file"])); idx.nprobe = self._nprobe
            out.append((seg["lo"], seg["hi"], idx)); lo = seg["hi"]
        return out

    def _save_delta(self, src: str, segs: list):
        meta = []
        for lo, hi, idx in segs:
            f = f"{src}.delta.{lo}-{hi}.ivf"
            path = os.path.join(self.d, f)
            if not os.path.exists(path):
                self.faiss.write_index(idx, path + ".tmp"); os.replace(path + ".tmp", path)
            meta.append({"lo": lo, "hi": hi, "file": f})
        mp = self._delta_meta(src)
        with open(mp + ".tmp", "w") as fh:
            json.dump({"built": self.meta.get("built"), "segs": meta}, fh)
        os.replace(mp + ".tmp", mp)
        keep = {m["file"] for m in meta}
        for f in os.listdir(self.d):
            if f.startswith(f"{src}.delta.") and f.endswith(".ivf") and f not in keep:
                os.remove(os.path.join(self.d, f))

    def add_segment(self, src: str, x: np.ndarray, ids: np.ndarray, replace: int = 0):
        """Append a shard for vectors added at runtime (source `src` need not exist in the build); `replace` drops
        that many of the source's newest shards first (a merge re-encodes them together)."""
        seg = self.encode(x, ids)
        with self._dlock:
            cur = self.shards.get(src, ())
            self.shards[src] = (cur[:len(cur) - replace] if replace else cur) + (seg,)


# ------------------------------------------------------------------------------------------------ build

def _kmeans(x: np.ndarray, nlist: int, iters: int, seed: int = 0) -> np.ndarray:
    """Spherical k-means on the GPU (MLX). x: (m, DIM) fp16, unit rows."""
    import mlx.core as mx
    rng = np.random.default_rng(seed)
    X = mx.array(x)
    C = mx.array(x[rng.choice(len(x), nlist, replace=False)])
    bs = max(1024, (1 << 29) // (nlist * 2))
    for it in range(iters):
        t = time.perf_counter()
        sums = mx.zeros((nlist, DIM), mx.float32); cnt = mx.zeros((nlist,), mx.float32)
        for i in range(0, len(x), bs):
            xb = X[i:i + bs]
            a = mx.argmax(xb @ C.T, axis=1)
            sums = sums.at[a].add(xb.astype(mx.float32)); cnt = cnt.at[a].add(1.0)
            mx.eval(sums, cnt)
        S = np.array(sums); cn = np.array(cnt)
        empty = np.flatnonzero(cn == 0)
        if len(empty):
            S[empty] = x[rng.choice(len(x), len(empty), replace=False)].astype(np.float32)
        S /= np.linalg.norm(S, axis=1, keepdims=True)
        C = mx.array(S.astype(np.float16))
        print(f"  k-means iter {it}: {len(empty)} empty, largest {int(cn.max())}, {time.perf_counter() - t:.1f}s", flush=True)
    return S.astype(np.float32)


def _assign(rows: Rows, cent: np.ndarray, lo: int = 0, hi: int | None = None) -> np.ndarray:
    """Nearest centroid of rows [lo, hi) on the GPU. Each block is evaluated before the next is queued: MLX is lazy,
    and a queue of (block x nlist) score matrices would all be alive at once."""
    import mlx.core as mx
    hi = rows.n if hi is None else hi
    C = mx.array(cent.astype(np.float16))
    bs = max(1024, (1 << 29) // (len(cent) * 2))
    out = np.empty(hi - lo, np.int64)
    for a0 in range(lo, hi, 1 << 20):
        xb = mx.array(np.ascontiguousarray(rows.a[a0:min(hi, a0 + (1 << 20))]))
        for i in range(0, xb.shape[0], bs):
            r = mx.argmax(xb[i:i + bs] @ C.T, axis=1); mx.eval(r)
            out[a0 - lo + i:a0 - lo + i + r.shape[0]] = np.array(r)
    return out


def cmd_build(a):
    import faiss
    from faiss.contrib.ivf_tools import add_preassigned
    srcs = _sources()
    tmp = a.out + ".tmp"
    os.makedirs(tmp, exist_ok=True)
    t0 = time.perf_counter()
    rng = np.random.default_rng(0)
    total = sum(r.n for r in srcs.values())
    m = min(total, a.nlist * a.per)
    sample = np.concatenate([r.a[np.sort(rng.choice(r.n, max(1, m * r.n // total), replace=False))] for r in srcs.values()])
    if a.centroids:
        cent = np.load(a.centroids).astype(np.float32); a.nlist = len(cent)
        print(f"centroids: {a.centroids} ({a.nlist} lists)", flush=True)
    else:
        print(f"k-means: {len(sample):,} sampled vectors -> {a.nlist} lists", flush=True)
        cent = _kmeans(sample, a.nlist, a.iters)
    np.save(os.path.join(tmp, "centroids.npy"), cent)
    quant = faiss.IndexFlatIP(DIM); quant.add(cent)
    proto = faiss.IndexIVFPQ(quant, DIM, a.nlist, a.m, 4, faiss.METRIC_INNER_PRODUCT)
    tr = sample[rng.choice(len(sample), min(len(sample), 1_000_000), replace=False)].astype(np.float32)
    proto.train(tr)  # coarse is fixed (quantizer full): trains the residual PQ only
    print(f"PQ{a.m}x4 trained ({time.perf_counter() - t0:.0f}s)", flush=True)
    placeholder = faiss.IndexFlatIP(DIM)  # shards are searched preassigned: no per-shard copy of the centroids
    p2 = faiss.clone_index(proto); p2.quantizer = placeholder; p2.own_fields = False  # placeholder is ours to free
    faiss.write_index(p2, os.path.join(tmp, "proto_pq.ivf"))
    meta = {"nlist": a.nlist, "m": a.m, "built": time.strftime("%Y-%m-%dT%H:%M:%S"), "sources": {}}
    for src, rows in srcs.items():
        # Shards split by list (shard s holds lists l with l % shards == s): a query's probes spread over the shards
        # and each shard scans whole lists. Splitting by row range instead hands every shard a 1/8 sliver of every
        # probed list, and fast-scan's per-list cost then dominates (1.0 vs 0.58 ms per shard at 1024 probes).
        nsh = a.wiki_shards if src == "wiki" else a.web_shards
        t = time.perf_counter()
        asg = _assign(rows, cent)
        print(f"{src}: {rows.n:,} rows assigned ({time.perf_counter() - t:.0f}s)", flush=True)
        t = time.perf_counter()
        shards = [faiss.clone_index(proto) for _ in range(nsh)]
        for b0 in range(0, rows.n, 1 << 20):  # one sequential pass over the rows feeds every shard
            b1 = min(rows.n, b0 + (1 << 20))
            x = rows.a[b0:b1].astype(np.float32); part = asg[b0:b1]
            for s, idx in enumerate(shards):
                sel = np.flatnonzero(part % nsh == s)
                ids = (sel + b0).astype(np.int64)  # must outlive the call: add_preassigned keeps only a raw pointer
                xs, ps = np.ascontiguousarray(x[sel]), np.ascontiguousarray(part[sel])
                add_preassigned(idx, xs, ps, ids=ids)
        print(f"{src}: encoded ({time.perf_counter() - t:.0f}s)", flush=True)
        for s in range(nsh):
            # IVFPQ + preassigned add, then converted: IndexIVF::add_core does not pack fast-scan codes correctly
            fs = faiss.IndexIVFPQFastScan(shards[s]); fs.quantizer = placeholder
            faiss.write_index(fs, os.path.join(tmp, f"{src}.shard{s:02d}.ivf"))
            print(f"{src} shard {s}: {shards[s].ntotal:,} rows", flush=True)
            shards[s] = None; del fs
        meta["sources"][src] = {"n": rows.n, "shards": nsh, "split": "list"}
    json.dump(meta, open(os.path.join(tmp, "meta.json"), "w"), indent=1)
    if os.path.exists(a.out):
        import shutil
        shutil.rmtree(a.out + ".old", ignore_errors=True); os.replace(a.out, a.out + ".old")
    os.replace(tmp, a.out)
    print(f"built {a.out} in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)


def main():
    import sys
    sys.modules.setdefault("torch", None)  # keep torch's OpenMP runtime out: faiss uses every core for the build
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--nlist", type=int, default=16384); b.add_argument("--m", type=int, default=192)
    b.add_argument("--per", type=int, default=64, help="k-means sample: vectors per list")
    b.add_argument("--iters", type=int, default=20); b.add_argument("--wiki-shards", type=int, default=8)
    b.add_argument("--web-shards", type=int, default=2)
    b.add_argument("--centroids", default="", help="use these centroids (.npy) instead of training new ones")
    b.add_argument("--out", default=IVF_DIR)
    a = ap.parse_args()
    {"build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    main()
