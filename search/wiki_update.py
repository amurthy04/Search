"""Keep the Wikipedia index current: embed only what changed since the dump (then since the last hour).

  python -m search.wiki_update run              # one cycle: poll changes, then fetch + diff + embed + append
  python -m search.wiki_update run --loop 3600  # hourly, forever (scripts/org.search.wikiupdate.plist runs this)
  python -m search.wiki_update status

Change feed (MediaWiki Action API, serial requests, maxlag=5): recentchanges (edits + creations in ns 0, latest
revision per page) and the move/delete logs, which catch articles entering ns 0 (Draft: -> article moves,
undeletions) and leaving it (moves out, deletions). Page ids/titles go into a queue; their *current* wikitext is
fetched only when applied, cleaned by the same code as the bulk build, and diffed per page against the passages
already indexed (xxh64 of the passage text): unchanged passages keep their pid, new ones are embedded and appended,
vanished ones are tombstoned in <prefix>.dead.u8 (search skips them). Pages never seen before (new articles, redirects
turned into articles) are all-new. Applying waits until the bulk build has indexed every dump shard, otherwise a
later bulk shard would re-add the dump's stale version on top.
"""
from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # online HF revision check segfaults once CoreML is loaded
os.environ.setdefault("OMP_NUM_THREADS", "1")  # libomp from faiss + torch in one process: >1 fails to init

import argparse
import atexit
import glob
import signal
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import orjson
import requests
import xxhash

from .wiki import PARTS, WIKI, _disamb, clean_wikitext, index_lock, legacy_ivf_sync, passages_for, repair, sync_pageids

API = "https://en.wikipedia.org/w/api.php"
UA = "SearchLocal/0.1 (personal on-device research index; https://github.com/amurthy04/Search)"
BACKFILL_FROM = "2026-10-01T00:00:00Z"  # dump revisions date from ~Oct 1-3; starting early only re-checks pages
BULK_SHARDS = 649  # wiki.00000-00648 = 259,358 dump streams / 400 streams per shard
LOG = "data/wiki/update.log"


def _now():
    return datetime.now(timezone.utc)


def _iso(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def _h(b: bytes) -> int:
    return xxhash.xxh64_intdigest(b) >> 1  # 63 bits: fits SQLite's signed INTEGER


class Api:
    def __init__(self):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": UA, "Accept-Encoding": "gzip"})
        self.calls = 0

    def get(self, **params) -> dict:
        params.update(format="json", formatversion=2, maxlag=5)
        for attempt in range(12):
            wait = min(120, 5 * 2 ** attempt)
            try:
                r = self.s.get(API, params=params, timeout=60)
                self.calls += 1
                retry_after = r.headers.get("Retry-After", "")
                if retry_after.isdigit():
                    wait = max(wait, int(retry_after))
                if r.status_code == 429 or r.status_code >= 500:
                    raise IOError(f"HTTP {r.status_code}")
                j = r.json()
                if j.get("error", {}).get("code") == "maxlag":
                    raise IOError("maxlag")
                if "error" in j:
                    raise RuntimeError(j["error"])
                return j
            except (IOError, requests.RequestException, ValueError) as e:
                print(f"[api] {e!r}; retry in {wait}s", flush=True)
                time.sleep(wait)
        raise IOError("API unavailable")


class Store:
    """pages.db: what is indexed per page (passage hash -> pid), the work queue and checkpoints."""

    def __init__(self, prefix: str):
        self.db = sqlite3.connect(os.path.join(os.path.dirname(prefix), "pages.db"))
        self.db.executescript("""
            PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;
            CREATE TABLE IF NOT EXISTS passages (pid INTEGER PRIMARY KEY, page INTEGER, h INTEGER);
            CREATE INDEX IF NOT EXISTS passages_page ON passages(page);
            CREATE TABLE IF NOT EXISTS pages (page INTEGER PRIMARY KEY, row INTEGER, title TEXT);
            CREATE INDEX IF NOT EXISTS pages_title ON pages(title);
            CREATE TABLE IF NOT EXISTS queue (page INTEGER PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS qtitle (title TEXT PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);""")

    def get(self, k, default=None):
        r = self.db.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone()
        return r[0] if r else default

    def put(self, k, v):
        self.db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (k, str(v)))


# ---------------------------------------------------------------- change feed -> queue

def poll(api: Api, st: Store, since: str | None = None) -> dict:
    end = _iso(_now() - timedelta(seconds=30))
    out = {"rc": 0, "moves": 0, "deletes": 0}
    start = since or st.get("rc_ckpt", BACKFILL_FROM)
    cont = {}
    while True:  # latest revision of every ns-0 page edited or created in the window
        j = api.get(action="query", list="recentchanges", rcnamespace=0, rctype="edit|new", rcprop="title|ids|timestamp",
                    rclimit="max", rcdir="newer", rcstart=start, rcend=end, rctoponly=1, **cont)
        rows = j["query"]["recentchanges"]
        st.db.executemany("INSERT OR IGNORE INTO queue VALUES (?)", [(r["pageid"],) for r in rows])
        out["rc"] += len(rows)
        out.setdefault("rc_new", 0); out["rc_new"] += sum(r["type"] == "new" for r in rows)
        if rows and not since:  # a --since test run must not move the real checkpoints
            st.put("rc_ckpt", rows[-1]["timestamp"])  # resume point if interrupted (re-queueing is idempotent)
        st.db.commit()
        if "continue" not in j:
            break
        cont = {"rccontinue": j["continue"]["rccontinue"], "continue": j["continue"]["continue"]}
    if not since:
        st.put("rc_ckpt", end)
    for letype in ("move", "delete"):  # all namespaces: Draft:/User: -> article moves have a non-zero source ns
        start = since or st.get(f"log_{letype}", BACKFILL_FROM)
        cont = {}
        while True:
            j = api.get(action="query", list="logevents", letype=letype, leprop="title|ids|type|details|timestamp",
                        lelimit="max", ledir="newer", lestart=start, leend=end, **cont)
            pages, titles = [], []
            for e in j["query"]["logevents"]:
                if e.get("actionhidden") or "title" not in e:
                    continue
                if letype == "move":
                    tgt_ns = e.get("params", {}).get("target_ns")
                    if e["ns"] == 0 or tgt_ns == 0:  # moved into, out of, or within the article space
                        if e.get("logpage"):
                            pages.append(e["logpage"])
                        if tgt_ns == 0:
                            titles.append(e["params"]["target_title"])
                        if e["ns"] == 0:
                            titles.append(e["title"])
                        out["moves"] += 1
                elif e["ns"] == 0:  # delete / restore / revision-delete: refetch by title to see what is there now
                    titles.append(e["title"])
                    if e.get("logpage"):
                        pages.append(e["logpage"])
                    out["deletes"] += 1
            st.db.executemany("INSERT OR IGNORE INTO queue VALUES (?)", [(p,) for p in pages])
            st.db.executemany("INSERT OR IGNORE INTO qtitle VALUES (?)", [(t,) for t in titles])
            if j["query"]["logevents"] and not since:
                st.put(f"log_{letype}", j["query"]["logevents"][-1]["timestamp"])
            st.db.commit()
            if "continue" not in j:
                break
            cont = {"lecontinue": j["continue"]["lecontinue"], "continue": j["continue"]["continue"]}
        if not since:
            st.put(f"log_{letype}", end)
    st.db.commit()
    return out


# ---------------------------------------------------------------- index -> store

def _load_state(prefix: str) -> dict:
    return orjson.loads(open(prefix + ".state.json", "rb").read())


def bulk_done(prefix: str, shards: int) -> bool:
    if not os.path.exists(prefix + ".state.json"):
        return False
    have = set(_load_state(prefix)["parts"])
    return all(f"wiki.{i:05d}" in have for i in range(shards))


def seed(st: Store, prefix: str):
    """Record (page, passage hash) for every indexed pid not yet in the store: the bulk build on first use, then
    whatever `search.wiki index` (Wikidata cards: negative page ids, skipped) or our own commits appended."""
    s = _load_state(prefix)
    n, upto = s["n"], int(st.get("seeded_upto", 0))
    if upto >= n:
        return
    doc = np.memmap(prefix + ".doc.i32", dtype=np.int32, mode="r", shape=(n,))
    pageid = np.fromfile(prefix + ".pageid.i64", np.int64)
    titles = open(prefix + ".titles.txt", encoding="utf-8").read().split("\n")
    poff = np.load(prefix + ".poff.npy")
    pas = np.memmap(prefix + ".passages.bin", dtype=np.uint8, mode="r")
    dead = np.fromfile(prefix + ".dead.u8", np.uint8) if os.path.exists(prefix + ".dead.u8") else np.zeros(0, np.uint8)
    t0 = time.time()
    for a in range(upto, n, 500_000):
        b = min(n, a + 500_000)
        rows = doc[a:b]; pg = pageid[rows]
        live = [(i, int(pg[i - a])) for i in range(a, b) if pg[i - a] > 0 and not (i < len(dead) and dead[i])]
        st.db.executemany("INSERT OR REPLACE INTO passages VALUES (?,?,?)",
                          [(i, p, _h(pas[poff[i]:poff[i + 1]].tobytes())) for i, p in live])
        last = {}
        for i, p in live:
            last[p] = int(doc[i])
        st.db.executemany("INSERT OR REPLACE INTO pages VALUES (?,?,?)", [(p, r, titles[r]) for p, r in last.items()])
        st.put("seeded_upto", b); st.db.commit()
        if n - upto > 1_000_000:
            print(f"[seed] {b:,}/{n:,} pids ({time.time()-t0:.0f}s)", flush=True)


# ---------------------------------------------------------------- queue -> fetch -> diff -> embed -> append

def fetch(api: Api, pageids: list[int], titles: list[str]) -> list[dict]:
    """Current state of each page: {page, title, ns, gone, text}. gone = missing, redirect or not an article."""
    out = []
    for key, vals in (("pageids", pageids), ("titles", titles)):
        for i in range(0, len(vals), 50):
            batch = vals[i:i + 50]
            got, cont = {}, {}
            while True:  # long batches come back in several pieces (rvcontinue)
                j = api.get(action="query", prop="revisions|info", rvprop="content|ids|timestamp", rvslots="main",
                            **{key: "|".join(map(str, batch))}, **cont)
                for p in j.get("query", {}).get("pages", []):
                    g = got.setdefault(p.get("pageid") or ("t", p["title"]), dict(p))
                    if p.get("revisions"):
                        g["revisions"] = p["revisions"]
                if "continue" not in j:
                    break
                cont = {k: v for k, v in j["continue"].items()}
            for p in got.values():
                rev = (p.get("revisions") or [{}])[0]
                text = rev.get("slots", {}).get("main", {}).get("content")
                gone = bool(p.get("missing") or p.get("invalid") or p.get("redirect") or p.get("ns") != 0 or text is None)
                out.append({"page": p.get("pageid") or 0, "title": p.get("title", ""), "gone": gone, "text": text})
    return out


def plan(st: Store, pages: list[dict]) -> dict:
    """Diff each page's fresh passages against the indexed ones."""
    adds, dead, removed, kept, seen = [], set(), set(), 0, set()
    for f in pages:
        key = f["page"] or ("t", f["title"])  # pages fetched by title that no longer exist all have page 0
        if key in seen:
            continue
        seen.add(key)
        known = st.db.execute("SELECT row, title FROM pages WHERE page=?", (f["page"],)).fetchone() if f["page"] else None
        if f["gone"] or _disamb.search(f["text"][:4000]) or f["title"].endswith("(disambiguation)"):
            ids = [f["page"]] if f["page"] else [r[0] for r in st.db.execute("SELECT page FROM pages WHERE title=?", (f["title"],))]
            for p in ids:
                pids = [r[0] for r in st.db.execute("SELECT pid FROM passages WHERE page=?", (p,))]
                dead.update(pids)
                if pids:
                    removed.add(p)
            continue
        old: dict[int, list[int]] = {}
        for pid, h in st.db.execute("SELECT pid, h FROM passages WHERE page=?", (f["page"],)):
            old.setdefault(h, []).append(pid)
        row = known[0] if known and known[1] == f["title"] else None  # moved page -> new title row
        texts = passages_for(f["title"], clean_wikitext(f["text"]))
        new_h = set()
        for x in texts:
            h = _h(x.encode("utf-8"))
            if h in new_h:
                continue
            new_h.add(h)
            if old.get(h):
                old[h].pop(); kept += 1
            else:
                adds.append({"page": f["page"], "title": f["title"], "row": row, "text": x})
        for pids in old.values():
            dead.update(pids)
        if not texts:
            removed.add(f["page"])
    return {"adds": adds, "dead": dead, "removed": removed, "kept": kept, "pages": len(seen)}


def commit(prefix: str, parts: str, p: dict, vecs: np.ndarray) -> int:
    """Append new passages + tombstones in the same layout `search.wiki index` writes (state file last)."""
    from .ann import IVF_DIR
    # With the shared IVF (search.ann), readers index rows appended after its build themselves, so only the row
    # files change here; the legacy single-file .ivf is maintained only when that index does not exist.
    legacy = not os.path.exists(os.path.join(IVF_DIR, "meta.json"))
    lock = index_lock(prefix)
    try:
        s = _load_state(prefix)
        repair(prefix, s)
        sync_pageids(prefix, parts, s)
        n0 = s["n"]
        index = None
        if legacy:
            import faiss
            index = faiss.read_index(prefix + ".ivf")
            legacy_ivf_sync(index, prefix, n0)
        new_rows, docs, offs = {}, [], []
        poff = np.load(prefix + ".poff.npy")
        end = int(poff[-1])
        with open(prefix + ".titles.txt", "ab") as tf, open(prefix + ".pageid.i64", "ab") as gf, \
                open(prefix + ".passages.bin", "ab") as pf:
            for a in p["adds"]:
                row = a["row"]
                if row is None:
                    if a["page"] not in new_rows:
                        new_rows[a["page"]] = s["pages"]; s["pages"] += 1
                        tf.write(a["title"].encode() + b"\n"); gf.write(np.int64(a["page"]).tobytes())
                    row = new_rows[a["page"]]
                b = a["text"].encode("utf-8"); pf.write(b); end += len(b); offs.append(end); docs.append(row)
        if len(vecs):
            if index is not None:
                index.add(vecs.astype(np.float32))
            with open(prefix + ".f16", "ab") as vf, open(prefix + ".doc.i32", "ab") as df:
                vf.write(vecs.astype(np.float16).tobytes()); df.write(np.asarray(docs, np.int32).tobytes())
            np.save(prefix + ".poff.tmp.npy", np.concatenate([poff, np.asarray(offs, np.int64)]))
            os.replace(prefix + ".poff.tmp.npy", prefix + ".poff.npy")
        n = n0 + len(vecs)
        dp = prefix + ".dead.u8"
        dead = np.zeros(n, np.uint8)
        if os.path.exists(dp):
            old = np.fromfile(dp, np.uint8)[:n]; dead[:len(old)] = old
        if p["dead"]:
            dead[np.fromiter(p["dead"], np.int64)] = 1
        # The tombstones go to dead.u8.<update> first and replace dead.u8 only after the state file names that
        # update: a commit killed before its state write must not hide the old passages (repair() rolls back).
        k = s.get("updates", 0) + 1
        pend = f"{dp}.{k}"
        dead.tofile(pend + ".tmp"); os.replace(pend + ".tmp", pend)
        if len(vecs) and index is not None:
            faiss.write_index(index, prefix + ".ivf.tmp"); os.replace(prefix + ".ivf.tmp", prefix + ".ivf")
        s["n"] = n; s["updates"] = k
        open(prefix + ".state.json.tmp", "wb").write(orjson.dumps(s)); os.replace(prefix + ".state.json.tmp", prefix + ".state.json")
        os.replace(pend, dp)
        return n0
    finally:
        lock.close()


_emb = None


def apply(api: Api, st: Store, a) -> dict:
    global _emb
    if _emb is None:  # CoreML must load before faiss is imported in this process
        from .embed import make_embedder
        _emb = make_embedder("coreml", "BAAI/bge-small-en-v1.5", max_seq=256, batch_size=16)
    seed(st, a.prefix)
    tot = {"pages": 0, "kept": 0, "added": 0, "tombstoned": 0, "removed_pages": 0, "fetch_s": 0.0, "embed_s": 0.0,
           "commit_s": 0.0, "new_pages": []}
    while True:
        k = a.commit_pages if not a.limit else min(a.commit_pages, a.limit - tot["pages"])
        if k <= 0:
            break
        # pages we already index first (their old passages are what turns stale), then new ones
        ids = [r[0] for r in st.db.execute("SELECT q.page FROM queue q LEFT JOIN pages p ON p.page = q.page "
                                           "ORDER BY p.page IS NULL, q.page DESC LIMIT ?", (k,))]
        titles = [r[0] for r in st.db.execute("SELECT title FROM qtitle LIMIT ?", (max(0, k - len(ids)),))]
        if not ids and not titles:
            break
        t0 = time.time(); pages = fetch(api, ids, titles); tot["fetch_s"] += time.time() - t0
        known = {r[0] for r in st.db.execute(f"SELECT page FROM pages WHERE page IN ({','.join('?' * len(ids))})", ids)} if ids else set()
        p = plan(st, pages)
        t0 = time.time()
        vecs = _emb.encode([x["text"] for x in p["adds"]]) if p["adds"] else np.zeros((0, 384), np.float32)
        tot["embed_s"] += time.time() - t0
        t0 = time.time()
        commit(a.prefix, a.parts, p, vecs)
        if p["dead"]:
            st.db.executemany("DELETE FROM passages WHERE pid=?", [(i,) for i in p["dead"]])
        st.db.executemany("DELETE FROM pages WHERE page=?", [(i,) for i in p["removed"]])
        st.db.executemany("DELETE FROM queue WHERE page=?", [(i,) for i in ids])
        st.db.executemany("DELETE FROM qtitle WHERE title=?", [(t,) for t in titles])
        st.db.commit()
        seed(st, a.prefix)  # records the pids just appended
        tot["commit_s"] += time.time() - t0
        new = sorted({x["title"] for x in p["adds"] if x["page"] not in known and x["row"] is None and x["page"]})
        tot["new_pages"] += new
        tot["pages"] += p["pages"]; tot["kept"] += p["kept"]; tot["added"] += len(p["adds"])
        tot["tombstoned"] += len(p["dead"]); tot["removed_pages"] += len(p["removed"])
        print(f"[apply] {tot['pages']:,} pages: +{tot['added']:,} passages, {tot['kept']:,} kept, "
              f"{tot['tombstoned']:,} tombstoned", flush=True)
    tot["new_page_count"] = len(tot["new_pages"]); tot["new_pages"] = tot["new_pages"][:20]
    return tot


def cycle(a) -> dict:
    api, st = Api(), Store(a.prefix)
    t0 = time.time()
    rec = {"t": _iso(_now()), **poll(api, st, a.since), "poll_s": round(time.time() - t0, 1)}
    rec["queued"] = st.db.execute("SELECT (SELECT count(*) FROM queue) + (SELECT count(*) FROM qtitle)").fetchone()[0]
    if not bulk_done(a.prefix, a.bulk_shards):
        rec["status"] = "waiting for bulk build"
    else:
        rec.update(apply(api, st, a)); rec["status"] = "applied"
    rec["api_calls"] = api.calls; rec["cycle_s"] = round(time.time() - t0, 1)
    rec = {k: round(v, 1) if isinstance(v, float) else v for k, v in rec.items()}
    st.put("last_cycle", orjson.dumps(rec).decode()); st.db.commit()
    with open(a.log, "ab") as f:
        f.write(orjson.dumps(rec) + b"\n")
    print(orjson.dumps(rec).decode(), flush=True)
    return rec


def cmd_run(a):
    os.makedirs("data/run", exist_ok=True)
    for f in glob.glob("data/run/wiki_update.*.pid"):  # left by a run that aborted (atexit never ran)
        try:
            os.kill(int(open(f).read()), 0)
        except (ValueError, OSError):
            os.remove(f)
    pidfile = f"data/run/wiki_update.{os.getpid()}.pid"  # search.chat pauses us while the LLM generates
    open(pidfile, "w").write(str(os.getpid()))
    atexit.register(lambda: os.path.exists(pidfile) and os.remove(pidfile))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    while True:
        t0 = time.time()
        cycle(a)
        if not a.loop:
            break
        time.sleep(max(60.0, a.loop - (time.time() - t0)))


def cmd_status(a):
    st = Store(a.prefix)
    q = st.db.execute("SELECT (SELECT count(*) FROM queue), (SELECT count(*) FROM qtitle), (SELECT count(*) FROM pages), "
                      "(SELECT count(*) FROM passages)").fetchone()
    dp = a.prefix + ".dead.u8"
    dead = int(np.fromfile(dp, np.uint8).sum()) if os.path.exists(dp) else 0
    print(orjson.dumps({"rc_ckpt": st.get("rc_ckpt"), "log_move": st.get("log_move"), "log_delete": st.get("log_delete"),
                        "queued_pages": q[0], "queued_titles": q[1], "tracked_pages": q[2], "tracked_passages": q[3],
                        "seeded_upto": st.get("seeded_upto"), "tombstones": dead,
                        "bulk_done": bulk_done(a.prefix, a.bulk_shards),
                        "last_cycle": orjson.loads(st.get("last_cycle", "null"))}, option=orjson.OPT_INDENT_2).decode())


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "status"):
        p = sub.add_parser(name)
        p.add_argument("--prefix", default=WIKI); p.add_argument("--parts", default=PARTS)
        p.add_argument("--bulk-shards", type=int, default=BULK_SHARDS)
        if name == "run":
            p.add_argument("--loop", type=float, default=0, help="seconds between cycles (0 = one cycle)")
            p.add_argument("--limit", type=int, default=0, help="max pages applied per cycle (testing)")
            p.add_argument("--commit-pages", type=int, default=20_000, help="pages per index commit")
            p.add_argument("--since", default=None, help="ISO time to poll from instead of the checkpoints (testing)")
            p.add_argument("--log", default=LOG)
    a = ap.parse_args()
    {"run": cmd_run, "status": cmd_status}[a.cmd](a)


if __name__ == "__main__":
    main()
