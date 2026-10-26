"""Wikidata facts for every English Wikipedia article, as one short "fact card" per article, in SQLite.

  python -m search.wikidata filter     # truthy NT dump (43.5 GB bz2, ~500 GB of text) -> facts.nt + labels.nt (ripgrep)
  python -m search.wikidata build      # + enwiki page_props (page id -> Q id) -> data/wiki/wikidata.db, and one
                                       # card passage per article -> data/wiki/shards/wiki.10000+ (embedded + indexed
                                       # with Wikipedia by search.wiki embed/index)
  python -m search.wikidata card "Nvidia"

The "truthy" dump holds only each statement's best-ranked value as one line `<Q> <P> <value> .`, so filtering is
a regex pass: parallel bz2 decode (indexed_bzip2, all cores) piped through ripgrep, keeping English labels,
descriptions and aliases, and 89 whitelisted properties. Units and qualifiers are not in the truthy dump, so only
unit-free quantities are rendered (population, employees, ...). Cards are attached to Wikipedia hits at answer time,
and aliases give an exact-name entity lookup for the query.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from contextlib import contextmanager

import numpy as np
import orjson
import xxhash

from .query import in_sorted

RAW = "data/wiki/raw"
NT = RAW + "/wikidata-truthy.nt.bz2"
FACTS = "data/wiki/wd_facts.nt"
LABELS = "data/wiki/wd_labels.nt"
PAGE_PROPS = RAW + "/enwiki-page_props.sql.gz"
DB = "data/wiki/wikidata.db"

# property -> label used in the card. Order = order in the card.
PROPS = {
    "P31": "instance of", "P279": "subclass of", "P1448": "official name", "P1813": "short name",
    "P571": "founded", "P576": "dissolved", "P569": "born", "P19": "birthplace", "P570": "died", "P20": "place of death",
    "P27": "citizenship", "P106": "occupation", "P39": "position held", "P69": "educated at", "P108": "employer",
    "P101": "field", "P800": "notable work", "P166": "award", "P26": "spouse", "P22": "father", "P25": "mother",
    "P40": "child", "P3373": "sibling", "P102": "political party", "P140": "religion", "P184": "doctoral advisor",
    "P112": "founded by", "P169": "CEO", "P488": "chairperson", "P1037": "director/manager", "P159": "headquarters",
    "P740": "formed in", "P452": "industry", "P1128": "employees", "P127": "owned by", "P749": "parent organization",
    "P355": "subsidiary", "P414": "stock exchange", "P249": "ticker", "P856": "website",
    "P178": "developer", "P176": "manufacturer", "P277": "programming language", "P306": "operating system",
    "P348": "version", "P275": "license", "P50": "author", "P57": "director", "P58": "screenwriter", "P162": "producer",
    "P161": "cast", "P175": "performer", "P86": "composer", "P264": "record label", "P136": "genre", "P577": "published",
    "P495": "country of origin", "P407": "language", "P123": "publisher",
    "P17": "country", "P131": "located in", "P30": "continent", "P36": "capital", "P1376": "capital of",
    "P35": "head of state", "P6": "head of government", "P37": "official language", "P38": "currency",
    "P1082": "population", "P47": "borders", "P206": "next to body of water", "P625": "coordinates",
    "P641": "sport", "P54": "team", "P413": "position", "P118": "league", "P115": "home venue", "P286": "head coach",
    "P580": "start", "P582": "end", "P585": "date", "P276": "location", "P710": "participant", "P1346": "winner",
    "P361": "part of", "P463": "member of", "P737": "influenced by", "P1303": "instrument", "P1412": "languages spoken",
}
MAX_VALUES = {"P161": 8, "P166": 6, "P40": 6, "P355": 6, "P47": 10, "P1412": 4, "P106": 5, "P39": 5, "P800": 5,
              "P710": 8, "P463": 5, "P737": 5, "P136": 4, "P69": 4, "P108": 5}

_E = r"<http://www\.wikidata\.org/entity/Q\d+>"
LABEL_RE = (_E + r" <(?:http://www\.w3\.org/2000/01/rdf-schema#label|http://schema\.org/description|"
            r"http://www\.w3\.org/2004/02/skos/core#altLabel)> \".*\"@en \.$")
FACT_RE = _E + r" <http://www\.wikidata\.org/prop/direct/(?:" + "|".join(PROPS) + r")> "


def rg_bin() -> str:
    for p in ("/opt/homebrew/bin/rg", "/usr/local/bin/rg"):
        if os.path.exists(p):
            return p
    sys.exit("ripgrep (rg) not found: brew install ripgrep")


def cmd_filter(a):
    """One parallel decode feeding two ripgrep filters (labels, facts)."""
    import indexed_bzip2
    rg = rg_bin()
    t0 = time.time()
    outs = [open(LABELS + ".tmp", "wb"), open(FACTS + ".tmp", "wb")]
    procs = [subprocess.Popen([rg, "--no-config", "-N", "--no-unicode", "-e", pat], stdin=subprocess.PIPE, stdout=o,
                              bufsize=1 << 22) for pat, o in ((LABEL_RE, outs[0]), (FACT_RE, outs[1]))]
    n = 0
    with indexed_bzip2.open(a.nt, parallelization=a.threads) as f:
        carry = b""
        while True:
            buf = f.read(64 << 20)
            if not buf:
                break
            buf = carry + buf
            cut = buf.rfind(b"\n") + 1  # hand rg whole lines only
            carry = buf[cut:]
            chunk = buf[:cut]
            for p in procs:
                p.stdin.write(chunk)
            n += len(chunk)
            if n // (20 << 30) != (n - len(chunk)) // (20 << 30):
                el = time.time() - t0
                print(f"{n/1e9:.0f} GB decoded, {n/1e6/el:.0f} MB/s", flush=True)
    for p in procs:
        p.stdin.close(); p.wait()
    for o in outs:
        o.close()
    os.replace(LABELS + ".tmp", LABELS); os.replace(FACTS + ".tmp", FACTS)
    print(f"done: {n/1e9:.1f} GB in {(time.time()-t0)/60:.1f} min; labels {os.path.getsize(LABELS)/1e9:.1f} GB, "
          f"facts {os.path.getsize(FACTS)/1e9:.1f} GB", flush=True)


_pp_row = re.compile(rb"\((\d+),'wikibase_item','Q(\d+)',")


def enwiki_qids() -> dict[int, tuple[str, int]]:
    """Q id -> (article title, wikitext bytes) for every article we extracted (bytes = popularity proxy)."""
    pages = {}
    for f in glob.glob("data/wiki/shards/wiki.*.pages.jsonl"):
        for line in open(f, "rb"):
            r = orjson.loads(line); pages[r["id"]] = (r["t"], r["b"])
    out = {}
    with gzip.open(PAGE_PROPS, "rb") as f:
        for line in f:  # rows of one INSERT can span many lines
            if b"wikibase_item" in line:
                for m in _pp_row.finditer(line):
                    pid = int(m.group(1))
                    if pid in pages:
                        out[int(m.group(2))] = pages[pid]
    return out


_q = re.compile(rb"^<http://www\.wikidata\.org/entity/Q(\d+)> <([^>]+)> (.*) \.$")
_uesc = re.compile(r"\\u([0-9A-Fa-f]{4})|\\U([0-9A-Fa-f]{8})")
_vq = re.compile(rb"^<http://www\.wikidata\.org/entity/Q(\d+)>$")


def _literal(v: bytes) -> str:
    """NT literal -> display text: dates -> 'June 28, 1971' / '1971', numbers -> '1,234', points -> 'lat, lon'."""
    s = v.decode("utf-8", "replace")
    if s.startswith("<"):
        return s[1:-1]  # URL (website)
    m = re.match(r'^"(.*)"(?:\^\^<([^>]+)>|@[\w-]+)?$', s)
    if not m:
        return s
    val, typ = m.group(1), m.group(2) or ""
    val = _uesc.sub(lambda u: chr(int(u.group(1) or u.group(2), 16)), val).replace('\\"', '"').replace("\\\\", "\\")
    if typ.endswith("#dateTime"):
        d = re.match(r"^([+-]?\d+)-(\d\d)-(\d\d)", val)
        if d:
            y, mo, da = int(d.group(1)), int(d.group(2)), int(d.group(3))
            ys = f"{-y} BC" if y < 0 else str(y)
            if mo == 1 and da == 1 or mo == 0:
                return ys
            from .wiki import _MONTHS
            return f"{_MONTHS[mo - 1]} {da}, {ys}" if da else f"{_MONTHS[mo - 1]} {ys}"
    if typ.endswith("#decimal"):
        try:
            x = float(val)
            return f"{int(x):,}" if x == int(x) else f"{x:,.2f}"
        except ValueError:
            return val
    if typ.endswith("#wktLiteral"):
        p = re.match(r"Point\(([-\d.]+) ([-\d.]+)\)", val)
        if p:
            return f"{float(p.group(2)):.3f}, {float(p.group(1)):.3f}"
    return val


def cmd_build(a):
    t0 = time.time()
    qids = enwiki_qids()
    print(f"{len(qids):,} articles have a Wikidata item ({time.time()-t0:.0f}s)", flush=True)
    facts: dict[int, list] = {}
    need = set(qids)
    pfx = b"http://www.wikidata.org/prop/direct/"
    with open(FACTS, "rb") as f:
        for line in f:
            m = _q.match(line.rstrip(b"\n"))
            if not m:
                continue
            s = int(m.group(1))
            if s not in qids:
                continue
            p = m.group(2)[len(pfx):].decode()
            o = m.group(3)
            vq = _vq.match(o)
            if vq:
                o = int(vq.group(1)); need.add(o)
            elif o.startswith(b'"') and b'"@' in o and not o.endswith(b'"@en'):
                continue  # monolingual text in another language (official name in German, ...)
            facts.setdefault(s, []).append((p, o))
    print(f"facts for {len(facts):,} items, {len(need):,} labels needed ({time.time()-t0:.0f}s)", flush=True)
    label, desc, alias = {}, {}, {}
    with open(LABELS, "rb") as f:
        for line in f:
            m = _q.match(line.rstrip(b"\n"))
            if not m:
                continue
            q = int(m.group(1))
            if q not in need:
                continue
            kind = m.group(2)
            text = _literal(m.group(3))
            if kind.endswith(b"#label"):
                label[q] = text
            elif kind.endswith(b"description"):
                desc[q] = text
            elif q in qids:
                alias.setdefault(q, []).append(text)
    print(f"labels {len(label):,}, descriptions {len(desc):,} ({time.time()-t0:.0f}s)", flush=True)
    order = {p: i for i, p in enumerate(PROPS)}
    if os.path.exists(DB + ".tmp"):
        os.remove(DB + ".tmp")
    db = sqlite3.connect(DB + ".tmp")
    db.execute("PRAGMA journal_mode=OFF"); db.execute("PRAGMA synchronous=OFF")
    db.execute("CREATE TABLE cards (title TEXT PRIMARY KEY, qid INTEGER, label TEXT, card TEXT, bytes INTEGER)")
    db.execute("CREATE TABLE names (name TEXT, title TEXT, bytes INTEGER)")
    rows, names = [], []
    for q, (title, nbytes) in qids.items():
        lab = label.get(q, title)
        parts = []
        by_p: dict[str, list[str]] = {}
        for p, o in facts.get(q, ()):
            v = label.get(o) if isinstance(o, int) else _literal(o)
            if v and v not in by_p.setdefault(p, []):
                by_p[p].append(v)
        for p in sorted(by_p, key=order.__getitem__):
            vals = by_p[p][:MAX_VALUES.get(p, 3)]
            parts.append(f"{PROPS[p]}: {', '.join(vals)}")
        card = f"{lab}" + (f" ({desc[q]})" if q in desc else "") + (". " + "; ".join(parts) if parts else "")
        rows.append((title, q, lab, card, nbytes))
        for nm in {lab, title, *alias.get(q, ())}:
            if 2 <= len(nm) <= 80:
                names.append((nm.lower(), title, nbytes))
        if len(rows) >= 200_000:
            db.executemany("INSERT OR IGNORE INTO cards VALUES (?,?,?,?,?)", rows); rows.clear()
            db.executemany("INSERT INTO names VALUES (?,?,?)", names); names.clear()
    db.executemany("INSERT OR IGNORE INTO cards VALUES (?,?,?,?,?)", rows)
    db.executemany("INSERT INTO names VALUES (?,?,?)", names)
    db.execute("CREATE INDEX names_name ON names(name)")
    db.commit(); db.close()
    os.replace(DB + ".tmp", DB)
    Cards.name_hashes(DB)
    write_card_shards()
    print(f"wrote {DB} ({os.path.getsize(DB)/1e9:.1f} GB) in {(time.time()-t0)/60:.1f} min", flush=True)


CARD_SHARD0 = 10000  # card shards sit after Wikipedia's (wiki.00000-00648) in the same dir, embedder and index
CARDS_PER_SHARD = 200_000


def wd_url(qid: int) -> str:
    return f"https://www.wikidata.org/wiki/Q{qid}"


def write_card_shards(out: str = "data/wiki/shards"):
    """Every fact card becomes a passage in the Wikipedia index (same encoder, same IVF), so a question like
    "companies headquartered in Santa Clara founded in 1993" can match the card itself, not only the article."""
    import zstandard as zstd
    db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    n = 0; fh = w = None
    for title, qid, card in db.execute("SELECT title, qid, card FROM cards WHERE card LIKE '%: %' ORDER BY rowid"):
        if n % CARDS_PER_SHARD == 0:
            if w:
                w.close(); fh.close(); os.replace(path + ".tmp", path)
            path = os.path.join(out, f"wiki.{CARD_SHARD0 + n // CARDS_PER_SHARD:05d}.jsonl.zst")
            fh = open(path + ".tmp", "wb"); w = zstd.ZstdCompressor(level=3).stream_writer(fh)
        w.write(orjson.dumps({"d": qid, "c": 0, "u": wd_url(qid), "t": title, "x": f"{title} — Wikidata facts\n{card}"}) + b"\n")
        n += 1
    if w:
        w.close(); fh.close(); os.replace(path + ".tmp", path)
    print(f"wrote {n:,} card passages to {out}/wiki.{CARD_SHARD0}+", flush=True)


CARD_ROWS = DB + ".rows"  # card text by Wikipedia title row: <CARD_ROWS>.bin + <CARD_ROWS>.off.npy


def build_card_rows(titles_path: str = "data/wiki/index/wiki.titles.txt", out: str = CARD_ROWS):
    """The card of every title row of the Wikipedia index (line i of wiki.titles.txt), as one blob + offsets: a
    card is then one read that can be requested ahead of time, instead of a B-tree walk in the 3 GB SQLite file.
    Rows appended by later updates fall back to SQLite (Cards.cards_for)."""
    db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    t0 = time.time()
    off = [0]
    with open(titles_path, "rb") as tf, open(out + ".bin.tmp", "wb") as bf:
        batch = []

        def flush():
            got = dict(db.execute(f"SELECT title, card FROM cards WHERE title IN ({','.join('?' * len(batch))})", batch))
            for t in batch:
                c = got.get(t)
                if c:
                    b = c.encode(); bf.write(b); off.append(off[-1] + len(b))
                else:
                    off.append(off[-1])
            batch.clear()
        for line in tf:
            batch.append(line[:-1].decode("utf-8"))
            if len(batch) == 500:
                flush()
        if batch:
            flush()
    np.save(out + ".off.tmp.npy", np.asarray(off, np.uint64))
    with open(out + ".json.tmp", "w") as f:
        json.dump(_card_rows_key(titles_path, os.path.getsize(titles_path)), f)
    os.replace(out + ".bin.tmp", out + ".bin"); os.replace(out + ".off.tmp.npy", out + ".off.npy")
    os.replace(out + ".json.tmp", out + ".json")
    print(f"card rows: {len(off) - 1:,} title rows, {off[-1] / 1e9:.2f} GB in {time.time() - t0:.0f}s", flush=True)


def _card_rows_key(titles_path: str, size: int, db: str = DB) -> dict:
    """What a card-row store was built from: the Wikidata DB, and the titles file's first `size` bytes (rows only ever
    get appended, so a rebuilt or reordered index shows up as a different hash of that prefix)."""
    with open(titles_path, "rb") as f:
        head = f.read(min(size, 1 << 20))
        f.seek(max(0, size - (1 << 20))); tail = f.read(min(size, 1 << 20))
    return {"db_mtime": os.path.getmtime(db), "titles_bytes": size,
            "titles_hash": xxhash.xxh3_64_hexdigest(head + tail)}


class Cards:
    """Read side used by the chat server: card for a Wikipedia title, and exact-name entity lookup on the query."""

    def __init__(self, path: str = DB, titles: str = "data/wiki/index/wiki.titles.txt"):
        self.ok, self.path = os.path.exists(path), path
        self._free: list[sqlite3.Connection] = []  # idle read-only connections (see _db)
        self.rows_off = None
        if not self.ok:
            return
        self.names_h = self.name_hashes(path)
        rows = path + ".rows"
        if os.path.exists(rows + ".json") and os.path.exists(titles):
            key = json.load(open(rows + ".json"))
            if os.path.getsize(titles) >= key["titles_bytes"] and _card_rows_key(titles, key["titles_bytes"], path) == key:
                from .ann import Blob
                self.rows_off = np.load(rows + ".off.npy")
                self.rows_blob = Blob(rows + ".bin")
            else:  # built for another index or DB: SQLite lookups until `search.wikidata card-rows` is rerun
                print("[wikidata] card-row store is stale, using SQLite", flush=True)

    def cards_for(self, rows: list[int], titles: list[str]) -> list[str | None]:
        """Card of each (Wikipedia title row, title): from the row store, all reads requested at once; rows past
        the store (articles added after it was built) from SQLite."""
        off = self.rows_off
        if off is None:
            return [self.card(t) for t in titles]
        inside = [d is not None and 0 <= d < len(off) - 1 for d in rows]
        self.rows_blob.prefetch((int(off[d]), int(off[d + 1] - off[d])) for d, ok in zip(rows, inside) if ok)
        out = []
        for d, t, ok in zip(rows, titles, inside):
            if ok:
                b = self.rows_blob.read(int(off[d]), int(off[d + 1] - off[d]))
                out.append(b.decode("utf-8") if b else None)
            else:
                out.append(self.card(t))
        return out

    @contextmanager
    def _db(self):
        """A read-only connection to itself for the duration: a shared one serializes every lookup on its mutex, and
        one per thread is opened cold on every request (the HTTP server runs each request on a new thread). Idle
        connections are kept, so their page caches stay warm."""
        try:
            c = self._free.pop()
        except IndexError:
            c = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, check_same_thread=False)
        try:
            yield c
        finally:
            self._free.append(c)

    @staticmethod
    def name_hashes(path: str = DB) -> np.ndarray:
        """Sorted 64-bit hashes of every name/alias (12M, 95 MB), cached next to the DB: link() asks SQLite only
        about query n-grams that are names, instead of one B-tree lookup per n-gram (~20 per question)."""
        hp = path + ".names.u64"
        if not os.path.exists(hp) or os.path.getmtime(hp) < os.path.getmtime(path):
            db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            h = np.fromiter((xxhash.xxh3_64_intdigest(r[0].encode()) for r in db.execute("SELECT DISTINCT name FROM names")),
                            np.uint64)
            h.sort(); h.tofile(hp + ".tmp"); os.replace(hp + ".tmp", hp)
        return np.fromfile(hp, np.uint64)

    def get(self, title: str) -> tuple[int, str] | None:
        if not self.ok:
            return None
        with self._db() as db:
            return db.execute("SELECT qid, card FROM cards WHERE title=?", (title,)).fetchone()

    def card(self, title: str) -> str | None:
        if not self.ok:
            return None
        with self._db() as db:
            r = db.execute("SELECT card FROM cards WHERE title=?", (title,)).fetchone()
        return r[0] if r else None

    def link(self, query: str, max_n: int = 6, k: int = 2) -> list[str]:
        """Longest query n-grams that are an article's name or alias -> titles (most-written-about article wins)."""
        if not self.ok:
            return []
        raw = [w.strip(".'") for w in re.findall(r"[\w'.&+-]+", query)]
        words = [w.lower() for w in raw]
        cand = []  # (i, n, gram), longest first, left to right
        for n in range(min(max_n, len(words)), 0, -1):
            for i in range(len(words) - n + 1):
                gram = " ".join(words[i:i + n])
                if n == 1 and (len(gram) < 3 or gram in _STOP or not raw[i][:1].isupper()):
                    continue  # one word links only when the user capitalized it ("Apple" yes, "apple pie" no)
                if n > 1 and (words[i] in _STOP or words[i + n - 1] in _STOP or words[i] in ("a", "an", "of", "in")):
                    continue  # "the capital" is not a reference to the newspaper The Capital
                cand.append((i, n, gram))
        if not cand:
            return []
        is_name = in_sorted(self.names_h, np.fromiter((xxhash.xxh3_64_intdigest(g.encode()) for _, _, g in cand),
                                                      np.uint64, len(cand)))
        found, covered = [], set()
        with self._db() as db:
            for (i, n, gram), hit in zip(cand, is_name):
                if not hit or any(j in covered for j in range(i, i + n)):
                    continue
                r = db.execute("SELECT title FROM names WHERE name=? ORDER BY bytes DESC LIMIT 1", (gram,)).fetchone()
                if r and r[0] not in found:
                    found.append(r[0]); covered.update(range(i, i + n))
        return found[:k]


_STOP = set("a an of in on at to for and or by as it its i me my you your we our what when where which who whom whose why how does did do is are was were the this that these those with "
            "from into about there their they them have has had been being will would could should than then also "
            "today news latest tell explain give show find list best most more less many much some time year".split())


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("filter"); f.add_argument("--nt", default=NT); f.add_argument("--threads", type=int, default=12)
    sub.add_parser("build")
    sub.add_parser("card-rows")
    c = sub.add_parser("card"); c.add_argument("query")
    a = ap.parse_args()
    if a.cmd == "card":
        cards = Cards()
        for t in cards.link(a.query) or [a.query]:
            print(t, "->", cards.card(t))
        return
    {"filter": cmd_filter, "build": cmd_build, "card-rows": lambda a: build_card_rows()}[a.cmd](a)


if __name__ == "__main__":
    main()
