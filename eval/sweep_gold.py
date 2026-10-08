r"""
Label sweep: for every ANSWERABLE question, list NON-gold sections whose chunks share distinctive terms with the
question's reference_answer, so you can read them and decide whether they also state the answer (rule 2).

Usage (from the repo root, cmd):
    python eval\sweep_gold.py
    python eval\sweep_gold.py --top 6 --min-terms 2 --max-df 0.02
    python eval\sweep_gold.py --ids fact-03 fact-07          # only some questions
    python eval\sweep_gold.py > eval\sweep_d3.txt             # save it to read calmly

Method (a lexical heuristic, NOT a judge):
  1. Distinctive terms = tokens of the reference_answer (lowercase, snake_case kept whole, stopwords and
     tokens under 3 characters dropped) that appear in at most --max-df of all chunks (default 2%).
     Common words ("memory", "table") are dropped; identifiers and rare words ("work_mem", "hostssl") stay.
  2. Every chunk that is not in a gold section is scored by the summed IDF of the distinctive terms it contains
     and must contain at least --min-terms of them.
  3. Chunks are grouped by section (page + host section heading); the best chunk of each section is shown with the
     matched terms and a snippet. Top --top sections per question.
  4. For reference, the best GOLD chunk's term coverage is printed, so you can see what a strong match looks like.

It does NOT use the retriever or the reranker: labelling from rankings would tilt the eval toward that mode.

Limits (read before trusting the output):
  - It only finds sections that share WORDS with the reference answer. A section that states the answer in other
    words (for example "take a long time" vs "much slower") is invisible to it. "Nothing listed" is not proof that
    no other section answers the question.
  - A listed section is a CANDIDATE. Read the printed text (or the chunk) and add it as gold only if a sentence in
    it really states the answer. Sharing terms is not the same as answering.
  - Reference answers for some questions are short, so there are few distinctive terms; the header line shows how
    many were used.
  - Unanswerable questions are skipped on purpose.
Untested on the real files when written (syntax-checked only).
"""
import argparse
import math
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_eval as ev  # noqa: E402  (provides load_dataset, load_chunks, chunk_keys, gold_key)

TOKEN_RE = re.compile(r"[a-z0-9_]+")
STOP = frozenset(
    "a an and are as at be by for from how i in is it of on or that the this to was what when where which who why "
    "with does do can not no yes its it's but if than then also may must might such these those there their they "
    "into out over under about between each other more most some any all only both either one two three use used "
    "uses using set sets default value values time times way ways need needs needed".split()
)


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def distinctive_terms(reference: str) -> list[str]:
    seen, out = set(), []
    for t in tokens(reference):
        if len(t) >= 3 and t not in STOP and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def snippet(text: str, matched: list[str], width: int = 230) -> str:
    low = text.lower()
    best = min((low.find(t) for t in matched if low.find(t) >= 0), default=0)
    start = max(0, best - width // 3)
    s = text[start:start + width].replace("\n", " ")
    return ("..." if start else "") + s + ("..." if start + width < len(text) else "")


def main(top: int, min_terms: int, max_df: float, ids: list | None) -> None:
    dataset = [q for q in ev.load_dataset() if q["type"] != "unanswerable"]
    if ids:
        unknown = [i for i in ids if i not in {q["id"] for q in dataset}]
        if unknown:
            raise SystemExit(f"Unknown or unanswerable id(s): {unknown}")
        dataset = [q for q in dataset if q["id"] in ids]
    chunks = ev.load_chunks()
    n = len(chunks)

    chunk_tokens = [set(tokens(c["context"] + " " + c["text"])) for c in chunks]
    df = Counter(t for ts in chunk_tokens for t in ts)
    keys = [ev.chunk_keys(c) for c in chunks]

    for q in dataset:
        gold = {ev.gold_key(g) for g in q["gold"]}
        terms_all = distinctive_terms(q.get("reference_answer", ""))
        terms = [t for t in terms_all if 0 < df[t] <= max_df * n]
        absent = [t for t in terms_all if df[t] == 0]
        idf = {t: math.log(n / df[t]) for t in terms}

        print("\n" + "=" * 100)
        print(f"{q['id']} ({q['type']}): {q['question']}")
        print("  gold: " + "; ".join(f"{g['page']} / {g.get('heading') or g.get('anchor')}" for g in q["gold"]))
        print(f"  distinctive terms used ({len(terms)}): {terms}")
        if absent:
            print(f"  terms absent from the corpus (ignored): {absent}")
        if len(terms) < min_terms:
            print(f"  too few distinctive terms (< {min_terms}); nothing to sweep for this question.")
            continue

        scored = []  # (score, matched, chunk index, is_gold)
        for i, ts in enumerate(chunk_tokens):
            matched = [t for t in terms if t in ts]
            if len(matched) >= min_terms:
                scored.append((sum(idf[t] for t in matched), matched, i, bool(keys[i] & gold)))

        gold_hits = [s for s in scored if s[3]]
        if gold_hits:
            g = max(gold_hits, key=lambda s: s[0])
            print(f"  best gold chunk: {chunks[g[2]]['chunk_id']} matches {len(g[1])}/{len(terms)} terms")
        else:
            print(f"  no gold chunk matches {min_terms}+ of the terms: the reference answer is worded unlike the "
                  f"gold text, so read the candidates with extra care")

        cands, seen_sections = [], set()
        for score, matched, i, is_gold in sorted(scored, key=lambda s: s[0], reverse=True):
            if is_gold:
                continue
            c = chunks[i]
            sec = (c["chunk_id"].split(":")[0], tuple(c["section_path"]))
            if sec in seen_sections:
                continue
            seen_sections.add(sec)
            cands.append((score, matched, c))
            if len(cands) >= top:
                break

        if not cands:
            print("  no non-gold section shares enough terms.")
            continue
        for rank, (score, matched, c) in enumerate(cands, 1):
            print(f"\n  [{rank}] {c['chunk_id']}  score {score:.1f}  {len(matched)}/{len(terms)} terms: {matched}")
            print(f"      {c['context']}")
            print(f"      label if it answers: {{\"page\": \"{c['chunk_id'].split(':')[0]}\", "
                  f"\"heading\": {c['section_path'][-1]!r}}}")
            print(f"      {snippet(c['text'], matched)}")

    print("\n" + "=" * 100)
    print("Read each candidate. Add a section as gold only if a sentence in it states the answer.")
    print("Note the label heading above is the HOST section of the best chunk; if that chunk merged small sections,"
          " check its `covers` before labelling.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top", type=int, default=5, help="candidate sections per question")
    ap.add_argument("--min-terms", type=int, default=2, help="distinctive terms a chunk must contain")
    ap.add_argument("--max-df", type=float, default=0.02, help="drop terms that appear in more than this fraction of chunks")
    ap.add_argument("--ids", nargs="+", help="only these question ids")
    a = ap.parse_args()
    main(a.top, a.min_terms, a.max_df, a.ids)
