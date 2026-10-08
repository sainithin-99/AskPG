r"""
Print the text of every chunk that covers a given section, so you can read the sentences before labelling.

Usage (from the repo root, cmd):
    set PYTHONUTF8=1
    python eval\show_section.py sql-explain:id-1.9.3.148.8 runtime-config-short:RUNTIME-CONFIG-SHORT > eval\read_d3.txt

Each argument is page:anchor, copied from the output of `python eval\run_eval.py find ... --text`.
A chunk is printed if (page, anchor) is in its `covers` (the same rule the eval uses), so a chunk that merged
several small sections is printed under each of them, and you can see the whole chunk, not just the host section.
Untested when written.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_eval as ev  # noqa: E402  (provides load_chunks, chunk_keys)


def main(targets: list[str]) -> None:
    chunks = ev.load_chunks()
    for t in targets:
        if ":" not in t:
            print(f"SKIP {t!r}: expected page:anchor")
            continue
        page, anchor = t.split(":", 1)
        hits = [c for c in chunks if (page, anchor) in ev.chunk_keys(c)]
        print("=" * 100)
        print(f"{page} / {anchor}: {len(hits)} chunk(s)")
        if not hits:
            print("  no chunk covers this (check the anchor spelling)")
        for c in hits:
            print("-" * 100)
            print(f"{c['chunk_id']}  {c['n_tokens']} tok  {c['context']}")
            print(c["text"])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python eval\\show_section.py page:anchor [page:anchor ...]")
    main(sys.argv[1:])
