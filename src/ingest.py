"""
Ingest the PostgreSQL docs: crawl -> parse -> chunk -> JSONL.

Usage (run from the repo root):
    pip install requests beautifulsoup4 tiktoken
    python src/ingest.py download      # crawl HTML into data/raw/ (cached, resumable)
    python src/ingest.py chunk         # parse + chunk into data/chunks/chunks.jsonl
    python src/ingest.py sample -n 20  # print random chunks for your eyeball check

Each output line is one chunk:
    chunk_id, url (with #anchor when known), page_title, section_path,
    context (deduped breadcrumb; prepend to text when embedding), text, n_tokens
"""

import argparse
import functools
import json
import random
import re
import statistics
import time
from collections import deque
from pathlib import Path
from urllib.parse import urldefrag, urljoin
from transformers import AutoTokenizer

import requests
import tiktoken
from bs4 import BeautifulSoup, NavigableString

# --------------------------------------------------------------------------- config
PG_VERSION = "17"
BASE_URL = f"https://www.postgresql.org/docs/{PG_VERSION}/"

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_PATH = ROOT / "data" / "chunks" / "chunks.jsonl"

# Token budgets. bge-small has a 512-token limit on ITS tokenizer; we count with
# tiktoken (cl100k) as a proxy, so leave headroom.
MAX_TOKENS = 380
OVERLAP_TOKENS = 60
MIN_TOKENS = 25  # drop fragments smaller than this (usually heading-only sections)

# Pages that are noise for Q&A (index, release notes, legal boilerplate).
SKIP_PAGES = re.compile(r"^(bookindex|release(-.*)?|legalnotice)\.html$")

HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5}
ADMONITIONS = {"note", "warning", "caution", "tip", "important"}
BLOCK_CONTAINERS = {"p", "div", "li", "ul", "ol", "blockquote", "dd", "dt", "section"}


# --------------------------------------------------------------------------- download
def fetch(session: requests.Session, url: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=30)
            r.raise_for_status()
            r.encoding = "utf-8"
            return r.text
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def download(delay: float = 0.3) -> None:
    """BFS from the docs index, staying inside /docs/<version>/. Cached on disk."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "askpg-portfolio-project (educational, polite crawler)"

    start = BASE_URL + "index.html"
    queue, seen = deque([start]), {start}
    fetched = 0

    while queue:
        url = queue.popleft()
        path = RAW_DIR / url[len(BASE_URL):]
        if path.exists():
            html = path.read_text(encoding="utf-8")
        else:
            html = fetch(session, url)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(html, encoding="utf-8")
            fetched += 1
            if fetched % 50 == 0:
                print(f"  fetched {fetched} pages, {len(queue)} queued")
            time.sleep(delay)

        for a in BeautifulSoup(html, "html.parser").find_all("a", href=True):
            link, _ = urldefrag(urljoin(url, a["href"]))
            if (
                link.startswith(BASE_URL)
                and link.endswith(".html")
                and link not in seen
                and not SKIP_PAGES.match(link[len(BASE_URL):])
            ):
                seen.add(link)
                queue.append(link)

    print(f"Done. {len(seen)} pages in {RAW_DIR} ({fetched} newly downloaded)")


# --------------------------------------------------------------------------- parsing
def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\u200b", "")).strip()


def table_to_text(table) -> str:
    rows = []
    for tr in table.find_all("tr"):
        cells = [clean(c.get_text(" ")) for c in tr.find_all(["th", "td"])]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def dl_to_blocks(dl, out: list) -> None:
    """<dt>term</dt><dd>definition</dd> pairs -> one block each.
    Config parameters (work_mem, etc.) are documented this way, so keep term + body together."""
    terms: list[str] = []
    for child in dl.find_all(["dt", "dd"], recursive=False):
        if child.name == "dt":
            terms.append(clean(child.get_text()))
        else:
            sub: list = []
            walk(child, sub)
            body = "\n".join(e[1] for e in sub if e[0] == "text")
            head = " / ".join(terms)
            out.append(("text", f"{head}\n{body}" if head else body))
            terms = []


def walk(node, out: list) -> None:
    """Flatten a DOM subtree into ordered events:
       ("heading", level, title, anchor_id) or ("text", block_string)."""
    buf: list[str] = []

    def flush() -> None:
        t = clean("".join(buf))
        buf.clear()
        if t:
            out.append(("text", t))

    for child in node.children:
        if isinstance(child, NavigableString):
            buf.append(str(child))
            continue

        name = child.name
        classes = set(child.get("class", []))

        if name in HEADINGS:
            flush()
            out.append(("heading", HEADINGS[name], clean(child.get_text()),
                        child.get("data-anchor") or None))
        elif name == "pre":
            flush()
            code = child.get_text().replace("\u200b", "").strip("\n")
            if code.strip():
                out.append(("text", code))
        elif name == "table":
            flush()
            txt = table_to_text(child)
            if txt:
                out.append(("text", txt))
        elif name == "dl":
            flush()
            dl_to_blocks(child, out)
        elif name == "div" and classes & ADMONITIONS:
            # "Note"/"Warning" boxes contain an <h3>; don't let it split the section.
            flush()
            label = sorted(classes & ADMONITIONS)[0].capitalize()
            sub: list = []
            walk(child, sub)
            body = "\n".join(e[1] for e in sub if e[0] == "text" and e[1].strip().lower() != label.lower())
            if body:
                out.append(("text", f"{label}: {body}"))
        elif name in ("ul", "ol"):
            flush()
            for i, li in enumerate(child.find_all("li", recursive=False), 1):
                sub_events: list = []
                walk(li, sub_events)
                marker = f"{i}. " if name == "ol" else "- "
                first = next((k for k, e in enumerate(sub_events) if e[0] == "text"), None)
                if first is not None:
                    sub_events[first] = ("text", marker + sub_events[first][1])
                out.extend(sub_events)
        elif name in BLOCK_CONTAINERS:
            flush()
            walk(child, out)
        else:
            buf.append(child.get_text())  # inline: <code>, <a>, <em>, <span> ...
    flush()

def heading_anchor(h, root):
    """Section anchor for a heading tag. The PG docs HTML has used several layouts, so try each:
    an <a id> inside the heading, an id on the heading itself, the permalink (<a class="id_link"
    href="#ID">), then the nearest enclosing element that has an id (<div class="sect2" id="ID">)."""
    a = h.find("a", id=True)
    if a:
        return a["id"]
    if h.get("id"):
        return h["id"]
    link = h.find("a", href=re.compile(r"^#."))
    if link:
        return link["href"][1:]
    for p in h.parents:
        if p is root:
            break
        if p.get("id"):
            return p["id"]
    return None

def parse_page(html: str):
    """Return (page_title, sections). Each section: {path, anchor, blocks}."""
    soup = BeautifulSoup(html, "html.parser")

    title_tag = soup.title.get_text() if soup.title else ""
    page_title = re.sub(r"^PostgreSQL:\s*Documentation:\s*\d+:\s*", "", clean(title_tag))

    root = soup.find("div", id="docContent") or soup.body
    if root is None:
        return page_title, []
    for h in root.find_all(list(HEADINGS)):  # before a.id_link is removed below: its href is one anchor source
        anchor = heading_anchor(h, root)
        if anchor:
            h["data-anchor"] = anchor
    for selector in (".navheader", ".navfooter", ".toc", "script", "style", "a.id_link"):
        for tag in root.select(selector):
            tag.decompose()
    for a in root.find_all("a", string="#"):  # permalink markers, in case the class name differs
        a.decompose()
    for br in root.find_all("br"):  # otherwise "sqlname<br>sqllen" glues into "sqlnamesqllen"
        br.replace_with("\n")

    events: list = []
    walk(root, events)

    sections, stack = [], []  # stack: [(level, title)]
    cur = {"path": [], "anchor": None, "blocks": []}
    for ev in events:
        if ev[0] == "heading":
            _, level, title, anchor = ev
            if cur["blocks"]:
                sections.append(cur)
            stack = [s for s in stack if s[0] < level] + [(level, title)]
            cur = {"path": [t for _, t in stack], "anchor": anchor, "blocks": []}
        else:
            cur["blocks"].append(ev[1])
    if cur["blocks"]:
        sections.append(cur)
    return page_title, sections


# --------------------------------------------------------------------------- chunking
@functools.lru_cache(maxsize=1)
def _tok():
    # model_max_length raised only to silence the "longer than 512" warning; we count, not encode for the model
    return AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5", model_max_length=10**6)

def ntok(text: str) -> int:
    return len(_tok()(text, add_special_tokens=False)["input_ids"])


def split_long(block: str, max_tokens: int) -> list[str]:
    """Split an oversized block on lines (code) or sentences (prose); hard-split as last resort."""
    if ntok(block) <= max_tokens:
        return [block]

    if "\n" in block:
        units, sep = block.split("\n"), "\n"
    else:
        units, sep = re.split(r"(?<=[.!?])\s+", block), " "

    pieces, cur, cur_tok = [], [], 0
    for u in units:
        t = ntok(u)
        if t > max_tokens:  # single giant line: cut by tokens
            if cur:
                pieces.append(sep.join(cur))
                cur, cur_tok = [], 0
            offs = _tok()(u, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
            for i in range(0, len(offs), max_tokens):
                j = min(i + max_tokens, len(offs)) - 1
                pieces.append(u[offs[i][0]:offs[j][1]])
            continue
        if cur and cur_tok + t > max_tokens:
            pieces.append(sep.join(cur))
            cur, cur_tok = [], 0
        cur.append(u)
        cur_tok += t
    if cur:
        pieces.append(sep.join(cur))
    return pieces


LEADIN_RE = re.compile(r"^(Table|Figure|Example) [\w.-]+\. ")


def is_leadin(u: str) -> bool:
    """Short caption / 'as follows:' line that only makes sense next to the block after it."""
    return ntok(u) < 60 and (u.rstrip().endswith(":") or bool(LEADIN_RE.match(u)))


def chunk_blocks(blocks: list[str]) -> list[str]:
    """Greedy-pack blocks into <= MAX_TOKENS chunks, carrying ~OVERLAP_TOKENS of tail into the next."""
    units = [u for b in blocks for u in split_long(b, MAX_TOKENS)]
    chunks, cur, cur_tok = [], [], 0
    for u in units:
        t = ntok(u)
        if cur and cur_tok + t > MAX_TOKENS:
            held = []  # don't strand a caption/lead-in at the end of a chunk
            while len(cur) > 1 and is_leadin(cur[-1]):
                held.insert(0, cur.pop())
            chunks.append("\n\n".join(cur))
            carry, carry_tok = [], 0
            for prev in reversed(cur):
                pt = ntok(prev)
                if carry_tok + pt > OVERLAP_TOKENS:
                    break
                carry.insert(0, prev)
                carry_tok += pt
            cur = carry + held
            cur_tok = sum(ntok(x) for x in cur)
        cur.append(u)
        cur_tok += t
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks

SKIP_SECTIONS = {"see also", "author", "authors"}  # pure cross-references / credits: skipped on purpose, and counted


def bare_title(t: str) -> str:
    """'F.29.3. Author' -> 'author'"""
    return re.sub(r"^(?:[A-Z]\.)?[\d.]+\s+", "", t).strip().lower()


def merge_small_sections(sections: list[dict]):
    """Fold sections under MIN_TOKENS into a neighbour on the same page instead of dropping them.
    Small sections are prepended to the next big section (a purpose line or Synopsis belongs with
    what follows); small sections at the end of a page are appended to the previous one.
    Every output section carries `covers`: the (anchor, path) of every original section it contains,
    so the eval's coverage check can still find a gold section after merging."""
    out, pending, skipped = [], [], []

    def labelled(sec: dict) -> str:
        label = sec["path"][-1] if sec["path"] else ""
        body = "\n".join(sec["blocks"])
        return body if (not label or label in body) else f"{label}\n{body}"

    for sec in sections:
        if sec["path"] and bare_title(sec["path"][-1]) in SKIP_SECTIONS:
            skipped.append(sec)
            continue
        if ntok("\n\n".join(sec["blocks"])) < MIN_TOKENS:
            pending.append(sec)
            continue
        out.append({
            "path": sec["path"],
            "anchor": sec["anchor"],
            "blocks": [labelled(p) for p in pending] + sec["blocks"],
            "covers": [(p["anchor"], p["path"]) for p in pending] + [(sec["anchor"], sec["path"])],
        })
        pending = []

    if pending:  # small sections at the end of the page
        tail = [labelled(p) for p in pending]
        cov = [(p["anchor"], p["path"]) for p in pending]
        if out:
            out[-1]["blocks"].extend(tail)
            out[-1]["covers"].extend(cov)
        else:  # the whole page is tiny: keep it as one section
            out.append({"path": pending[0]["path"], "anchor": pending[0]["anchor"],
                        "blocks": tail, "covers": cov})
    return out, skipped

def chunk_all() -> None:
    files = sorted(RAW_DIR.rglob("*.html"))
    if not files:
        raise SystemExit("No HTML in data/raw/. Run: python src/ingest.py download")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dropped_path = OUT_PATH.with_name("dropped.jsonl")
    skipped_path = OUT_PATH.with_name("skipped.jsonl")
    sizes, dropped, n_skipped, n_merged = [], 0, 0, 0
    with OUT_PATH.open("w", encoding="utf-8") as f, \
         dropped_path.open("w", encoding="utf-8") as fd, \
         skipped_path.open("w", encoding="utf-8") as fs:
        for path in files:
            rel = path.relative_to(RAW_DIR).as_posix()
            url = BASE_URL + rel
            page_title, sections = parse_page(path.read_text(encoding="utf-8"))
            sections, skipped = merge_small_sections(sections)
            for s in skipped:
                n_skipped += 1
                fs.write(json.dumps({"page": path.stem, "section_path": s["path"],
                                     "text": "\n".join(s["blocks"])[:300]}, ensure_ascii=False) + "\n")
            idx = 0
            for sec in sections:
                n_merged += len(sec["covers"]) - 1
                for text in chunk_blocks(sec["blocks"]):
                    n = ntok(text)
                    if n < MIN_TOKENS:
                        dropped += 1
                        fd.write(json.dumps({"page": path.stem, "section_path": sec["path"],
                                             "n_tokens": n, "text": text}, ensure_ascii=False) + "\n")
                        continue
                    rec = {
                        "chunk_id": f"{path.stem}:{idx:04d}",
                        "url": url + (f"#{sec['anchor']}" if sec["anchor"] else ""),
                        "page_title": page_title,
                        "section_path": sec["path"],
                        "covers": [{"anchor": a, "section_path": p} for a, p in sec["covers"]],
                        "context": " > ".join(dict.fromkeys([page_title, *sec["path"]])),
                        "text": text,
                        "n_tokens": n,
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    sizes.append(n)
                    idx += 1

    print(f"Pages parsed:        {len(files)}")
    print(f"Chunks written:      {len(sizes)}  -> {OUT_PATH}")
    print(f"Small sections merged into a neighbour: {n_merged}")
    print(f"Skipped on purpose (See Also / Author): {n_skipped}  -> {skipped_path}")
    print(f"Dropped (<{MIN_TOKENS} tokens after merging): {dropped}  -> {dropped_path}")
    if sizes:
        print(f"Tokens/chunk:        min {min(sizes)}, median {int(statistics.median(sizes))}, max {max(sizes)}")

# --------------------------------------------------------------------------- inspection
def sample(n: int, seed: int = 0) -> None:
    if not OUT_PATH.exists():
        raise SystemExit("Run `python src/ingest.py chunk` first.")
    rows = [json.loads(line) for line in OUT_PATH.open(encoding="utf-8")]
    random.Random(seed).shuffle(rows)
    for r in rows[:n]:
        print("=" * 80)
        print(f"{r['chunk_id']}  |  {r['n_tokens']} tokens  |  {r['url']}")
        print(f"{r['page_title']}  >  {' > '.join(r['section_path'])}")
        print("-" * 80)
        print(r["text"])
    print("=" * 80)


# --------------------------------------------------------------------------- cli
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download")
    sub.add_parser("chunk")
    s = sub.add_parser("sample")
    s.add_argument("-n", type=int, default=20)
    args = ap.parse_args()

    if args.cmd == "download":
        download()
    elif args.cmd == "chunk":
        chunk_all()
    else:
        sample(args.n)