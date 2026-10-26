"""English Wikipedia -> passages -> vectors, from the official multistream XML dump (no crawling).

  python -m search.wiki extract --workers 14     # dump -> data/wiki/shards/*.jsonl.zst (rerun to pick up newly downloaded streams)
  python -m search.wiki embed                     # shards -> data/wiki/parts (CoreML GPU), resumable, waits for new shards
  python -m search.wiki show "Anarchism"          # print the passages of one article (cleaner check)

The multistream dump is a concatenation of independent bz2 streams of 100 pages each; the index file gives each
stream's byte offset, so workers decode disjoint ranges in parallel. Wikitext is cleaned with regexes + brace matching:
templates that carry prose (dates, unit conversions, language/flag names, lists) are rendered, everything else
(references, infoboxes, navboxes, tables, files, categories) is dropped; infobox facts come from Wikidata instead.
Each section is packed into ~180-word passages headed "Title — Section".
"""
from __future__ import annotations

import argparse
import bz2
import glob
import html
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import orjson
import zstandard as zstd

RAW = "data/wiki/raw"
DUMP = RAW + "/enwiki-pages-articles-multistream.xml.bz2"
INDEX = RAW + "/enwiki-pages-articles-multistream-index.txt.bz2"
SHARDS = "data/wiki/shards"
PARTS = "data/wiki/parts"
STREAMS_PER_SHARD = 400  # 40k pages -> ~150k passages per shard

# ---------------------------------------------------------------- wikitext -> text

_comment = re.compile(r"<!--.*?-->", re.S)
_drop_tags = re.compile(r"<(ref|gallery|math|chem|ce|syntaxhighlight|source|timeline|imagemap|score|graph|mapframe|"
                        r"templatedata|templatestyles|hiero|inputbox|categorytree|references)\b[^>]*?(?:/>|>.*?</\1\s*>)", re.S | re.I)
_self_ref = re.compile(r"<ref\b[^>]*/>", re.I)
_br = re.compile(r"<br\s*/?>", re.I)
_tag = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]*\b[^>]*>")
_tmpl = re.compile(r"\{\{([^{}]*)\}\}")
_table = re.compile(r"\{\|(?:(?!\{\|).)*?\|\}", re.S)
_bold = re.compile(r"'{2,}")
_ext_link = re.compile(r"\[(?:https?:)?//[^\s\]]+(?:\s+([^\]]*))?\]")
_magic = re.compile(r"__[A-Z]+__")
_heading = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$", re.M)
_ws = re.compile(r"[ \t ]+")
_paren_junk = re.compile(r"\(\s*[,;]?\s*\)|\(\s*;\s*|\s+([,.;:])")
_num = re.compile(r"^-?\d{1,4}$")

SKIP_SECTIONS = {"references", "external links", "see also", "notes", "further reading", "bibliography", "sources",
                 "citations", "footnotes", "works cited", "gallery", "notes and references", "references and notes",
                 "explanatory notes", "general and cited sources", "cited sources", "general sources", "external sources",
                 "selected bibliography", "discography", "filmography", "publications", "awards and nominations"}
_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
           "November", "December"]
_RANGE = {"to", "-", "–", "and", "or", "by", "x", "×", "+", "&"}
_KEEP_FIRST = {"nowrap", "nobr", "small", "smaller", "big", "sic", "abbr", "angbr", "mvar", "math", "var", "nts",
               "ill", "interlanguage link", "illm", "transl", "transliteration", "smallcaps", "sc", "small caps",
               "visible anchor", "vanchor", "anchor", "linktext", "keypress", "code", "tt", "samp", "lang-rtl",
               "bigger", "resize", "lc", "uc", "nihongo", "nihongo2", "korean", "zh", "chinese", "sortname", "marriage",
               "flag", "flagcountry", "flagu", "flagdeco", "country", "unichar", "noitalic", "not a typo", "proper name",
               "ship", "sclass", "usssubs", "hms", "uss", "sic hide", "as written", "spaces", "j", "ns", "hs", "gaps"}
_LIST = {"ubl", "unbulleted list", "plainlist", "plain list", "hlist", "flatlist", "flat list", "bulleted list",
         "cslist", "enum", "ordered list", "collapsible list"}
_DATE = {"birth date", "birth date and age", "bda", "death date", "death date and age", "dda", "start date",
         "end date", "start date and age", "end date and age", "dts", "date", "film date", "release date",
         "death year and age", "birth year and age", "birth-date", "death-date", "birth based on age as of date",
         "launch date", "opening date"}


def _fmt_date(parts):
    nums = [p.strip() for p in parts if _num.match(p.strip())][:3]
    if not nums:
        return ""
    y = nums[0]
    if len(nums) >= 3 and 1 <= int(nums[1]) <= 12:
        return f"{_MONTHS[int(nums[1]) - 1]} {int(nums[2])}, {y}"
    if len(nums) == 2 and 1 <= int(nums[1]) <= 12:
        return f"{_MONTHS[int(nums[1]) - 1]} {y}"
    return y


def _render(m: re.Match) -> str:
    body = m.group(1)
    parts = body.split("|")
    name = parts[0].strip().lower().replace("_", " ")
    pos = [p for p in parts[1:] if "=" not in p.split("[[")[0]]  # positional args (ignore key=value)
    if name in _DATE or name.startswith("birth date") or name.startswith("death date"):
        return _fmt_date(pos)
    if name in ("convert", "cvt", "convert/2"):
        if len(pos) >= 4 and pos[1].strip() in _RANGE:
            return f"{pos[0].strip()} {pos[1].strip()} {pos[2].strip()} {pos[3].strip()}"
        return " ".join(p.strip() for p in pos[:2])
    if name.startswith("lang-"):
        return pos[0].strip() if pos else ""
    if name in ("lang", "langx", "script", "nihongo3") and len(pos) >= 2:
        return pos[1].strip()
    if name in _KEEP_FIRST:
        return pos[0].strip() if pos else ""
    if name in _LIST:
        return ", ".join(p.strip() for p in pos if p.strip())
    if name in ("circa", "c.", "ca", "c"):
        return "c. " + pos[0].strip() if pos else "c."
    if name in ("us$", "usd", "us dollar", "a$", "ca$", "€", "£", "¥", "₹", "inr", "eur", "gbp", "jpy", "aud", "cad"):
        sym = {"usd": "US$", "us dollar": "US$", "inr": "₹", "eur": "€", "gbp": "£", "jpy": "¥", "aud": "A$", "cad": "CA$"}.get(name, parts[0].strip())
        return sym + (pos[0].strip() if pos else "")
    if name in ("frac", "sfrac", "fraction"):
        if len(pos) == 3:
            return f"{pos[0].strip()} {pos[1].strip()}/{pos[2].strip()}"
        if len(pos) == 2:
            return f"{pos[0].strip()}/{pos[1].strip()}"
        return f"1/{pos[0].strip()}" if pos else ""
    if name == "val" and pos:
        return pos[0].strip()
    if name in ("snd", "spaced ndash", "spaced en dash", "dash", "ndash"):
        return " – "
    if name in ("mdash", "em dash"):
        return "—"
    if name in ("'", "' \"", "\" '", "''", "apostrophe"):
        return "'"
    if name in ("·", "dot", "middot", "•", "bull"):
        return " · "
    if name in ("quote", "blockquote", "cquote", "quotation", "quote box"):
        return pos[0].strip() if pos else ""
    return ""


def _strip_links(s: str) -> str:
    """[[A|B]] -> B, [[A]] -> A; drop [[File:..]] / [[Category:..]] / interwiki (nested links in captions handled)."""
    out = []
    i = 0
    n = len(s)
    while True:
        j = s.find("[[", i)
        if j < 0:
            out.append(s[i:]); break
        out.append(s[i:j])
        # find the matching ]] with nesting
        depth = 0; k = j
        while k < n - 1:
            if s.startswith("[[", k):
                depth += 1; k += 2
            elif s.startswith("]]", k):
                depth -= 1; k += 2
                if depth == 0:
                    break
            else:
                k += 1
        else:
            out.append(s[j:]); break
        inner = s[j + 2:k - 2]
        ns = inner.split(":", 1)[0].strip().lower() if ":" in inner.split("|", 1)[0] else ""
        if ns in ("file", "image", "category", "media") or (ns and len(ns) <= 3 and ns.isalpha() and "|" not in inner):
            pass  # drop files/categories/interwiki
        else:
            if "[[" in inner:
                inner = _strip_links(inner)
            text = inner.split("|")[-1] if "|" in inner else inner
            if ns in ("wikt", "wiktionary", "w", "s", "wikisource") and "|" not in inner:
                text = inner.split(":", 1)[1]
            out.append(text)
        i = k
    return "".join(out)


def clean_wikitext(t: str) -> list[tuple[str, str]]:
    """Wikitext -> [(section heading, plain text)] with the lead section heading ''."""
    t = html.unescape(t) if "&" in t else t  # XML entities were decoded by the parser; this handles &nbsp; etc.
    t = _comment.sub("", t)
    t = _drop_tags.sub("", t)
    t = _self_ref.sub("", t)
    for _ in range(12):  # innermost-first template rendering
        t2 = _tmpl.sub(_render, t)
        if t2 == t:
            break
        t = t2
    t = re.sub(r"\{\{|\}\}", "", t)  # unbalanced leftovers
    for _ in range(4):
        t2 = _table.sub("", t)
        if t2 == t:
            break
        t = t2
    t = _strip_links(t)
    t = _ext_link.sub(lambda m: m.group(1) or "", t)
    t = _br.sub(" ", t)
    t = _tag.sub("", t)
    t = _bold.sub("", t)
    t = _magic.sub("", t)
    sections = []
    pos = 0; head = ""
    for m in _heading.finditer(t):
        sections.append((head, t[pos:m.start()]))
        head = m.group(2).strip(); pos = m.end()
    sections.append((head, t[pos:]))
    out = []
    skip_level = False
    for head, body in sections:
        hl = head.lower().strip(" :")
        if hl in SKIP_SECTIONS:
            continue
        paras = []
        for line in body.split("\n"):
            line = line.strip()
            if not line or line[0] in "|!{}" or line.startswith("----"):
                continue
            line = line.lstrip("*#:;").strip()
            line = _ws.sub(" ", line)
            line = _paren_junk.sub(lambda m: m.group(1) or "", line).strip()
            if len(line) < 3:
                continue
            paras.append(line)
        if paras:
            out.append((head, paras))
    return out


def passages_for(title: str, sections, words: int = 180) -> list[str]:
    """Pack paragraphs of each section into ~`words`-word passages; never mix sections."""
    out = []
    for head, paras in sections:
        hdr = f"{title} — {head}" if head else title
        cur, n = [], 0
        def emit():
            if cur and n >= 12:
                out.append(hdr + "\n" + " ".join(cur))
        for p in paras:
            w = p.split()
            if cur and n + len(w) > words:
                emit(); cur, n = [], 0
            if len(w) > words:  # long paragraph: split with a small overlap
                for s in range(0, len(w), words - 20):
                    piece = w[s:s + words]
                    if len(piece) < 20 and s:
                        break
                    cur.append(" ".join(piece)); n += len(piece)
                    if n >= words:
                        emit(); cur, n = [], 0
                continue
            cur.append(p); n += len(w)
        emit()
    return out


# ---------------------------------------------------------------- dump -> shards

_page = re.compile(rb"<page>(.*?)</page>", re.S)
_title = re.compile(rb"<title>(.*?)</title>")
_ns = re.compile(rb"<ns>(\d+)</ns>")
_id = re.compile(rb"<id>(\d+)</id>")
_text = re.compile(rb"<text[^>]*>(.*?)</text>", re.S)
_disamb = re.compile(r"\{\{\s*(disambiguation|disambig|dab|hndis|geodis|set index|surname|given name|sia)\b", re.I)


def _xml_unescape(b: bytes) -> str:
    s = b.decode("utf-8", "replace")
    return s.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"').replace("&#039;", "'").replace("&amp;", "&")


def url_for(title: str) -> str:
    return "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")


def iter_pages(raw: bytes):
    for m in _page.finditer(raw):
        p = m.group(1)
        ns = _ns.search(p)
        if not ns or ns.group(1) != b"0" or b"<redirect" in p:
            continue
        tm = _text.search(p)
        if not tm:
            continue
        yield int(_id.search(p).group(1)), _xml_unescape(_title.search(p).group(1)), _xml_unescape(tm.group(1))


def _extract_shard(dump: str, ranges: list[tuple[int, int]], out: str) -> dict:
    t0 = time.perf_counter()
    npages = npass = 0
    tmp = out + ".tmp"
    with open(dump, "rb") as f, open(tmp, "wb") as fh, open(out.replace(".jsonl.zst", ".pages.jsonl"), "wb") as ph:
        w = zstd.ZstdCompressor(level=3).stream_writer(fh)
        for a, b in ranges:
            f.seek(a)
            raw = bz2.decompress(f.read(b - a))
            for pid, title, text in iter_pages(raw):
                if _disamb.search(text[:4000]) or title.endswith("(disambiguation)"):
                    continue
                ps = passages_for(title, clean_wikitext(text))
                ph.write(orjson.dumps({"id": pid, "t": title, "b": len(text), "p": len(ps)}) + b"\n")
                npages += 1
                for ci, x in enumerate(ps):
                    w.write(orjson.dumps({"d": pid, "c": ci, "u": url_for(title), "t": title, "x": x}) + b"\n")
                    npass += 1
        w.close()
    os.replace(tmp, out)
    return {"shard": os.path.basename(out), "pages": npages, "passages": npass, "s": time.perf_counter() - t0}


def stream_offsets(index_path: str) -> list[int]:
    offs = []
    last = -1
    with bz2.open(index_path, "rt", encoding="utf-8") as f:
        for line in f:
            o = int(line.split(":", 1)[0])
            if o != last:
                offs.append(o); last = o
    return offs


def cmd_extract(a):
    os.makedirs(a.out, exist_ok=True)
    offs = stream_offsets(a.index)
    size = os.path.getsize(a.dump)
    complete = a.complete or size >= a.expected_size
    ends = offs[1:] + [size if complete else -1]
    streams = [(o, e) for o, e in zip(offs, ends) if e != -1 and e <= size]
    print(f"{len(offs)} streams in index, {len(streams)} available ({size/1e9:.1f} GB on disk)", flush=True)
    jobs = []
    for si in range(0, len(offs), STREAMS_PER_SHARD):
        group = list(zip(offs[si:si + STREAMS_PER_SHARD], ends[si:si + STREAMS_PER_SHARD]))
        if any(e == -1 or e > size for _, e in group):
            break  # not fully downloaded yet; shards stay in order
        out = os.path.join(a.out, f"wiki.{si // STREAMS_PER_SHARD:05d}.jsonl.zst")
        if not os.path.exists(out):
            jobs.append((group, out))
    if a.limit:
        jobs = jobs[:a.limit]
    print(f"{len(jobs)} shards to extract", flush=True)
    t0 = time.time(); tp = tq = 0
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(_extract_shard, a.dump, g, o) for g, o in jobs]
        for fu in as_completed(futs):
            r = fu.result(); tp += r["pages"]; tq += r["passages"]
            el = time.time() - t0
            print(orjson.dumps({**r, "total_pages": tp, "total_passages": tq, "pages_per_s": round(tp / el)}).decode(), flush=True)


def cmd_show(a):
    offs = stream_offsets(a.index)
    with bz2.open(a.index, "rt", encoding="utf-8") as f:
        for line in f:
            o, pid, title = line.rstrip("\n").split(":", 2)
            if title == a.title:
                break
        else:
            sys.exit("not found")
    o = int(o); e = offs[offs.index(o) + 1]
    with open(a.dump, "rb") as f:
        f.seek(o); raw = bz2.decompress(f.read(e - o))
    for p, t, text in iter_pages(raw):
        if t == a.title:
            t0 = time.perf_counter(); ps = passages_for(t, clean_wikitext(text)); dt = time.perf_counter() - t0
            for x in ps[:a.n]:
                print(x, "\n")
            print(f"{len(ps)} passages from {len(text)} chars of wikitext in {dt*1e3:.1f} ms")


# ---------------------------------------------------------------- shards -> vectors

def _read_shard(path):
    with open(path, "rb") as fh, zstd.ZstdDecompressor().stream_reader(fh) as r:
        import io
        for line in io.TextIOWrapper(r, encoding="utf-8"):
            if line.strip():
                yield orjson.loads(line)


def cmd_embed(a):
    """Embed each shard into a part (same layout as search.embed_watch, so merge/index tools apply). Pauses itself
    while the chat server's LLM is generating (it SIGSTOPs pids listed in data/run/*.pid)."""
    from .embed import make_embedder
    os.makedirs(a.parts, exist_ok=True); os.makedirs("data/run", exist_ok=True)
    emb = make_embedder({"gpu": "coreml", "ane": "coreml_ane"}[a.device], "BAAI/bge-small-en-v1.5", max_seq=256,
                        batch_size=a.batch_size)
    pidfile = f"data/run/wiki_embed.{os.getpid()}.pid"  # registered only once the model is up (removed in finally)
    total = 0; t_start = time.time()
    def claim(prefix) -> bool:
        """Several embedders (GPU, ANE) share the shard list; a .claim file created with O_EXCL owns a shard."""
        c = prefix + ".claim"
        try:
            os.kill(int(open(c).read()), 0)  # live owner -> skip
            return False
        except (FileNotFoundError, ValueError):
            pass
        except OSError:
            os.remove(c)  # owner died
        try:
            fd = os.open(c, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        os.write(fd, str(os.getpid()).encode()); os.close(fd)
        return True

    try:
        open(pidfile, "w").write(str(os.getpid()))
        while True:
            s = None
            for cand in sorted(glob.glob(os.path.join(a.shards, "wiki.*.jsonl.zst"))):
                pre = os.path.join(a.parts, os.path.basename(cand).replace(".jsonl.zst", ""))
                if not os.path.exists(pre + ".npy") and claim(pre):
                    s = cand; break
            if s is None:
                if not a.wait:
                    break
                time.sleep(30); continue
            prefix = os.path.join(a.parts, os.path.basename(s).replace(".jsonl.zst", ""))
            t0 = time.perf_counter()
            recs = list(_read_shard(s))
            texts = [r["x"] for r in recs]
            order = np.argsort([len(x) for x in texts])
            vecs = np.empty((len(texts), 384), np.float16)
            for i in range(0, len(order), 4096):
                idx = order[i:i + 4096]
                vecs[idx] = emb.encode([texts[j] for j in idx]).astype(np.float16)
            with open(prefix + ".meta.jsonl.zst", "wb") as fh:
                w = zstd.ZstdCompressor(level=3).stream_writer(fh)
                for r in recs:
                    w.write(orjson.dumps({"d": r["d"], "c": r["c"], "u": r["u"], "t": r["t"]}) + b"\n")
                w.close()
            offsets = [0]
            with open(prefix + ".passages.bin", "wb") as fh:
                for x in texts:
                    b = x.encode("utf-8"); fh.write(b); offsets.append(offsets[-1] + len(b))
            np.save(prefix + ".offsets.npy", np.asarray(offsets, dtype=np.int64))
            np.save(prefix + ".tmp.npy", vecs); os.replace(prefix + ".tmp.npy", prefix + ".npy")  # .npy last = part ready
            os.remove(prefix + ".claim")
            dt = time.perf_counter() - t0; total += len(texts)
            print(orjson.dumps({"shard": os.path.basename(s), "dev": a.device, "passages": len(texts), "per_s": round(len(texts) / dt),
                                "total": total, "elapsed_min": round((time.time() - t_start) / 60, 1)}).decode(), flush=True)
    finally:
        if os.path.exists(pidfile):
            os.remove(pidfile)


# ---------------------------------------------------------------- parts -> index (incremental)
#
# 25M x 384 fp16 vectors are 19 GB, too big for RAM next to everything else, and HNSW over them would need ~40 GB.
# IVF-PQ fast-scan keeps a compressed copy in RAM (98 B/vector); the top `refine` candidates are re-scored exactly
# against the fp16 vectors, read from an mmap of the file on disk. New parts are appended without retraining.
# On the first 786k passages: PQ alone recall@10 0.77, + refine 100 -> 0.974 at 0.4 ms (PQ96: 0.93, PQ48: 0.77).

WIKI = "data/wiki/index/wiki"


def _state_path(prefix: str = WIKI):
    return prefix + ".state.json"


def index_lock(prefix: str = WIKI):
    """Exclusive lock held around every index write (search.wiki index and search.wiki_update)."""
    import fcntl
    f = open(os.path.join(os.path.dirname(prefix), ".lock"), "a")
    fcntl.flock(f, fcntl.LOCK_EX)
    return f


def _part_pageids(part: str) -> list[int]:
    """Page id of each title row a part adds, in order (Wikidata cards stored as -qid so they never collide)."""
    out, last = [], None
    import io
    with open(part + ".meta.jsonl.zst", "rb") as fh, zstd.ZstdDecompressor().stream_reader(fh) as r:
        for line in io.TextIOWrapper(r, encoding="utf-8"):
            m = orjson.loads(line)
            if m["d"] != last:
                out.append(-m["d"] if "wikidata.org" in m["u"] else m["d"]); last = m["d"]
    return out


def sync_pageids(prefix: str, parts_dir: str, st: dict):
    """<prefix>.pageid.i64 maps title row -> page id. Indexes built before it existed get it rebuilt from the parts
    (valid while only `index` has written rows, i.e. before search.wiki_update first applies)."""
    path = prefix + ".pageid.i64"
    have = os.path.getsize(path) // 8 if os.path.exists(path) else 0
    if have == st["pages"]:
        return
    if have > st["pages"]:  # a run died after appending rows but before writing its state
        os.truncate(path, st["pages"] * 8); return
    assert have == 0 and not st.get("updates"), f"{path}: {have} rows vs {st['pages']} title rows"
    ids = [i for p in st["parts"] for i in _part_pageids(os.path.join(parts_dir, p))]
    assert len(ids) == st["pages"], f"rebuilt {len(ids)} page ids for {st['pages']} title rows"
    np.asarray(ids, np.int64).tofile(path + ".tmp"); os.replace(path + ".tmp", path)


def repair(prefix: str, st: dict):
    """Cut the append-only files back to what the state file covers: a writer that died after appending but before
    writing its state would otherwise leave rows that shift every later passage/title (state is always written last)."""
    n, pages = st["n"], st["pages"]
    # tombstones prepared by search.wiki_update.commit: apply them if the state names their update, else drop them
    for f in glob.glob(prefix + ".dead.u8.*"):
        k = f.rsplit(".", 1)[1]
        if k.isdigit() and int(k) == st.get("updates") and os.path.getsize(f) == n:
            os.replace(f, prefix + ".dead.u8")
        else:
            os.remove(f)
    for ext, size in ((".f16", n * 768), (".doc.i32", n * 4), (".pageid.i64", pages * 8)):
        if os.path.exists(prefix + ext) and os.path.getsize(prefix + ext) > size:
            os.truncate(prefix + ext, size)
    if os.path.exists(prefix + ".poff.npy"):
        poff = np.load(prefix + ".poff.npy")
        if len(poff) > n + 1:
            poff = poff[:n + 1]; np.save(prefix + ".poff.tmp.npy", poff); os.replace(prefix + ".poff.tmp.npy", prefix + ".poff.npy")
        if os.path.getsize(prefix + ".passages.bin") > poff[-1]:
            os.truncate(prefix + ".passages.bin", int(poff[-1]))
    tp = prefix + ".titles.txt"
    if os.path.exists(tp):
        b = open(tp, "rb").read()
        if b.count(b"\n") > pages:
            cut = 0
            for _ in range(pages):
                cut = b.index(b"\n", cut) + 1
            os.truncate(tp, cut)


def legacy_ivf_sync(index, prefix: str, n: int):
    """Bring the legacy single-file .ivf to exactly n rows: drop rows of a writer that died before its state write,
    and add rows appended while the shared IVF (search.ann) was in use and nothing maintained this file."""
    import faiss
    if index.ntotal > n:
        index.remove_ids(faiss.IDSelectorRange(n, index.ntotal))
    if index.ntotal < n:
        vecs = np.memmap(prefix + ".f16", dtype=np.float16, mode="r", shape=(n, 384))
        for a in range(index.ntotal, n, 1 << 20):
            index.add(np.asarray(vecs[a:min(n, a + (1 << 20))], dtype=np.float32))
    assert index.ntotal == n, f"index has {index.ntotal} vectors, state says {n}"


def cmd_index(a):
    lock = index_lock(getattr(a, "prefix", WIKI))
    try:
        _index(a)
    finally:
        lock.close()


def _index(a):
    import faiss
    WIKI = getattr(a, "prefix", globals()["WIKI"])
    os.makedirs(os.path.dirname(WIKI), exist_ok=True)
    st = orjson.loads(open(_state_path(WIKI), "rb").read()) if os.path.exists(_state_path(WIKI)) else {"parts": [], "n": 0, "pages": 0}
    repair(WIKI, st)
    sync_pageids(WIKI, a.parts, st)
    ready = sorted(p[:-4] for p in glob.glob(os.path.join(a.parts, "wiki.*.npy"))
                   if not p.endswith((".offsets.npy", ".tmp.npy")))
    new = [p for p in ready if os.path.basename(p) not in st["parts"]]
    from .ann import built_sources
    # the shared IVF (search.ann) indexes appended rows at read time; without it, the legacy single-file .ivf
    legacy = "wiki" not in built_sources()
    if not new and not (legacy and st["n"] and not os.path.exists(WIKI + ".ivf")):
        print("index up to date:", st["n"], "passages"); return
    t0 = time.time()
    if not legacy:
        index = None  # only the row files grow
    elif os.path.exists(WIKI + ".ivf"):
        index = faiss.read_index(WIKI + ".ivf")
        legacy_ivf_sync(index, WIKI, st["n"])
    else:  # train the coarse clusters + PQ codebooks on the rows so far, or the first parts
        if st["n"]:
            rows = np.memmap(WIKI + ".f16", dtype=np.float16, mode="r", shape=(st["n"], 384))
            sample = rows[np.sort(np.random.default_rng(0).choice(st["n"], min(st["n"], a.nlist * 60), replace=False))]
        else:
            sample = np.concatenate([np.load(p + ".npy", mmap_mode="r")[::3] for p in new[:4]])
        sample = sample.astype(np.float32)[np.random.default_rng(0).permutation(len(sample))[:a.nlist * 60]]
        index = faiss.index_factory(384, f"IVF{a.nlist},PQ{a.m}x4fs", faiss.METRIC_INNER_PRODUCT)
        print(f"training IVF{a.nlist},PQ{a.m}x4fs on {len(sample):,} vectors", flush=True)
        index.train(sample)
        print(f"trained in {time.time()-t0:.0f}s", flush=True)
        legacy_ivf_sync(index, WIKI, st["n"])  # rows written while the shared IVF covered them (ids 0..n-1)
    poffs = [np.load(WIKI + ".poff.npy") if os.path.exists(WIKI + ".poff.npy") else np.zeros(1, np.int64)]
    with open(WIKI + ".f16", "ab") as vf, open(WIKI + ".passages.bin", "ab") as pf, open(WIKI + ".doc.i32", "ab") as df, \
            open(WIKI + ".titles.txt", "ab") as tf, open(WIKI + ".pageid.i64", "ab") as gf:
        for p in new:
            v = np.load(p + ".npy")
            if index is not None:
                index.add(v.astype(np.float32))
            vf.write(v.tobytes())
            off = np.load(p + ".offsets.npy")
            pf.write(open(p + ".passages.bin", "rb").read())
            poffs.append(off[1:] + poffs[-1][-1])
            rows, last = [], None
            with open(p + ".meta.jsonl.zst", "rb") as fh, zstd.ZstdDecompressor().stream_reader(fh) as r:
                import io
                for line in io.TextIOWrapper(r, encoding="utf-8"):
                    m = orjson.loads(line)
                    if m["d"] != last:
                        tf.write(m["t"].encode() + b"\n"); st["pages"] += 1; last = m["d"]
                        gf.write(np.int64(-m["d"] if "wikidata.org" in m["u"] else m["d"]).tobytes())
                    rows.append(st["pages"] - 1)
            df.write(np.asarray(rows, np.int32).tobytes())
            st["n"] += len(v); st["parts"].append(os.path.basename(p))
            print(f"+{os.path.basename(p)}: {len(v):,} -> {st['n']:,} passages", flush=True)
    np.save(WIKI + ".poff.tmp.npy", np.concatenate(poffs)); os.replace(WIKI + ".poff.tmp.npy", WIKI + ".poff.npy")
    if index is not None:
        faiss.write_index(index, WIKI + ".ivf.tmp"); os.replace(WIKI + ".ivf.tmp", WIKI + ".ivf")
    open(_state_path(WIKI) + ".tmp", "wb").write(orjson.dumps(st)); os.replace(_state_path(WIKI) + ".tmp", _state_path(WIKI))
    print(f"index: {st['n']:,} passages, {st['pages']:,} articles ({time.time()-t0:.0f}s)", flush=True)


class _Grow:
    """An append-only array kept in RAM. Items are only ever added past the end, into spare room of the same buffer, so
    the view of the first items an older snapshot holds never changes, and a reload copies only what is new."""

    def __init__(self, dtype):
        self.buf, self.n = np.empty(0, dtype), 0

    def extend(self, x) -> np.ndarray:
        m = self.n + len(x)
        if m > len(self.buf):
            nb = np.empty(m + m // 16 + 4096, self.buf.dtype); nb[:self.n] = self.buf[:self.n]; self.buf = nb
        self.buf[self.n:m] = x
        self.n = m
        return self.buf[:m]


class Titles:
    """Title row -> title: titles.txt as one byte buffer plus the offset of each row's newline (13M Python strings
    would hold ~1 GB; this is its 290 MB)."""

    def __init__(self, blob: np.ndarray, ends: np.ndarray):
        self.blob, self.ends = blob, ends

    def __len__(self):
        return len(self.ends)

    def __getitem__(self, d: int) -> str:
        a = int(self.ends[d - 1]) + 1 if d else 0
        return self.blob[a:int(self.ends[d])].tobytes().decode("utf-8")


class WikiIndex:
    """Read side. `reload()` picks up passages appended by `search.wiki index` / `search.wiki_update` (the state
    file is written last); everything a search touches is swapped in as one tuple, `s`. A search takes it once and
    passes it to each call (the `s` arguments), so a concurrent reload never mixes old and new.

    With `ivf` (a search.ann.IVF built over Wikipedia), candidates come from the shared sharded index and rows
    appended since its build are encoded into its delta segments on reload; without it, from the single legacy
    `.ivf` file."""

    def __init__(self, prefix: str = WIKI, nprobe: int = 64, refine: int = 100, ivf=None):
        if ivf is not None and "wiki" not in ivf.covered:
            ivf = None
        self.prefix, self.nprobe, self.refine, self.ivf = prefix, nprobe, refine, ivf
        self.s = None; self.mtime = 0.0; self._ram = None
        self.reload()

    @property
    def n(self) -> int:
        return self.s[0] if self.s else 0

    def _mtime(self) -> float:
        # dead.u8 is replaced right after the state file (search.wiki_update.commit), so watch both
        dp = self.prefix + ".dead.u8"
        return max(os.path.getmtime(self.prefix + ".state.json"), os.path.getmtime(dp) if os.path.exists(dp) else 0.0)

    def _in_ram(self, n: int, pages: int):
        """doc, poff, pageid and titles up to the state file, in RAM (~900 MB): every query reads a few dozen random
        entries of each, and as mmaps under memory pressure each of those reads is a separate ~50-100 us trip to the
        SSD. The files only grow, so a reload reads just the rows added since the last one."""
        p = self.prefix
        key = os.stat(p + ".doc.i32").st_ino  # a rebuilt index has new files
        r = self._ram
        if r is None or r["key"] != key or r["doc"].n > n or r["ends"].n > pages:
            r = self._ram = {"key": key, "doc": _Grow(np.int32), "poff": _Grow(np.int64), "pageid": _Grow(np.int64),
                             "blob": _Grow(np.uint8), "ends": _Grow(np.int64)}

        def grow(g, path, count, npy=False):
            if count > g.n:
                src = np.load(path, mmap_mode="r") if npy else np.memmap(path, dtype=g.buf.dtype, mode="r")
                g.extend(src[g.n:count])
            return g.buf[:g.n]
        doc = grow(r["doc"], p + ".doc.i32", n)
        poff = grow(r["poff"], p + ".poff.npy", n + 1, npy=True)
        pg = p + ".pageid.i64"  # title row -> page id; Wikidata card rows hold -qid (their hits link to Wikidata)
        pageid = grow(r["pageid"], pg, min(pages, os.path.getsize(pg) // 8)) if os.path.exists(pg) else np.zeros(0, np.int64)
        blob, ends = r["blob"], r["ends"]
        if pages > ends.n:  # newlines of the rows added since, found in the bytes after the last known row
            start = int(ends.buf[ends.n - 1]) + 1 if ends.n else 0
            tail = np.memmap(p + ".titles.txt", dtype=np.uint8, mode="r")[start:]
            nl = np.flatnonzero(tail == 10)[:pages - ends.n]
            ends.extend(nl + start)
            blob.extend(tail[:int(nl[-1]) + 1] if len(nl) else tail[:0])
        return doc, poff, pageid, Titles(blob.buf[:blob.n], ends.buf[:ends.n])

    def reload(self) -> bool:
        from .ann import Blob, Rows
        sp = self.prefix + ".state.json"
        if not os.path.exists(sp) or self._mtime() == self.mtime:
            return False
        mtime = self._mtime()
        st = orjson.loads(open(sp, "rb").read())
        n = st["n"]
        rows = Rows(self.prefix + ".f16", n)
        index = None
        if self.ivf is not None:
            self.ivf.extend("wiki", rows, n)
        elif os.path.exists(self.prefix + ".ivf"):
            import faiss
            index = faiss.read_index(self.prefix + ".ivf")
            index.nprobe = self.nprobe
        elif not self.s:
            print("[wiki] no index over the Wikipedia rows: run `search.ann build` or `search.wiki index`", flush=True)
        doc, poff, pageid, titles = self._in_ram(n, st["pages"])
        pas = Blob(self.prefix + ".passages.bin")
        dp = self.prefix + ".dead.u8"  # 1 = passage replaced or removed by search.wiki_update
        dead = np.fromfile(dp, np.uint8) if os.path.exists(dp) else np.zeros(0, np.uint8)
        self.s = (n, index, rows, doc, poff, pas, titles, dead, pageid)
        self.mtime = mtime
        return True

    def candidates(self, qv: np.ndarray, s=None) -> np.ndarray:
        """Row ids of the best `refine` passages by compressed-code score."""
        s = s or self.s
        if self.ivf is not None:
            return self.ivf.search(qv, {"wiki": self.refine})[0]["wiki"][0]
        if not s or s[1] is None:
            return np.zeros(0, np.int64)
        _, ids = s[1].search(qv.reshape(1, -1).astype(np.float32), self.refine)
        return ids[0]

    def live_ids(self, ids: np.ndarray, s=None) -> np.ndarray:
        """Drop ids past the state file (an index can be one writer run ahead of it) and tombstoned rows."""
        s = s or self.s
        if not s:
            return np.zeros(0, np.int64)
        n, dead = s[0], s[7]
        ids = ids[(ids >= 0) & (ids < n)]
        if len(dead):
            ids = ids[(ids >= len(dead)) | (dead[np.minimum(ids, len(dead) - 1)] == 0)]
        return np.sort(ids)

    def prefetch(self, ids: np.ndarray, s=None):
        s = s or self.s
        if s:
            s[2].prefetch(ids)

    def hits(self, qv: np.ndarray, ids: np.ndarray, k: int = 8, prefetched: bool = False, s=None) -> list[dict]:
        """Exact re-score of candidate rows on their fp16 vectors, best passage per article."""
        s = s or self.s
        if not s or not len(ids):
            return []
        _, _, rows, doc, _, _, titles, _, pageid = s
        if not prefetched:
            rows.prefetch(ids)
        sc = rows.get(ids) @ qv.astype(np.float32)
        out, seen = [], set()
        for j in np.argsort(-sc):
            i = int(ids[j]); d = int(doc[i])
            if d in seen:
                continue
            seen.add(d)
            t = titles[d]
            g = int(pageid[d]) if d < len(pageid) else 0
            out.append({"score": float(sc[j]), "pid": i, "title": t, "doc": d,
                        "url": f"https://www.wikidata.org/wiki/Q{-g}" if g < 0 else url_for(t)})
            if len(out) >= k:
                break
        return out

    def search(self, qv: np.ndarray, k: int = 8, s=None) -> list[dict]:
        """Approximate top-`refine` from the compressed codes, exact re-score on fp16, one passage per article."""
        s = s or self.s
        if not s:
            return []
        return self.hits(qv, self.live_ids(self.candidates(qv, s), s), k, s=s)

    def prefetch_passages(self, pids):
        if self.s:
            poff, pas = self.s[4], self.s[5]
            pas.prefetch((int(poff[p]), int(poff[p + 1] - poff[p])) for p in pids)

    def passage(self, pid: int, max_chars: int = 700) -> str:
        poff, pas = self.s[4], self.s[5]
        a, b = int(poff[pid]), int(poff[pid + 1])
        return pas.read(a, min(b, a + max_chars * 4) - a).decode("utf-8", "ignore")[:max_chars + 200]


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--dump", default=DUMP); e.add_argument("--index", default=INDEX); e.add_argument("--out", default=SHARDS)
    e.add_argument("--workers", type=int, default=14); e.add_argument("--limit", type=int, default=0)
    e.add_argument("--expected-size", type=int, default=26917225724); e.add_argument("--complete", action="store_true")
    s = sub.add_parser("show")
    s.add_argument("title"); s.add_argument("-n", type=int, default=6)
    s.add_argument("--dump", default=DUMP); s.add_argument("--index", default=INDEX)
    m = sub.add_parser("embed")
    m.add_argument("--shards", default=SHARDS); m.add_argument("--parts", default=PARTS)
    m.add_argument("--batch-size", type=int, default=16); m.add_argument("--device", default="gpu")
    m.add_argument("--wait", action="store_true", help="keep waiting for new shards (while extract is still running)")
    i = sub.add_parser("index")
    i.add_argument("--parts", default=PARTS); i.add_argument("--nlist", type=int, default=16384); i.add_argument("--m", type=int, default=192)
    i.add_argument("--prefix", default=WIKI)
    a = ap.parse_args()
    {"extract": cmd_extract, "show": cmd_show, "embed": cmd_embed, "index": cmd_index}[a.cmd](a)


if __name__ == "__main__":
    main()
