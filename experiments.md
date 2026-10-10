# AskPG experiment log

Rules for this file
- Every run gets a row with numbers. Every finding says whether it is VERIFIED (from saved results), HYPOTHESIS (untested) or RETRACTED.
- Bug fixes define the baseline and are not experiments. Diagnostic findings are logged as findings; their fixes are logged as separate experiments.
- One change per experiment. Re-run all four modes after any label change and give the run a new label-version prefix (d1, d2, d3 ...). Rows from different label versions are NEVER compared as progress.
- Anything marked FILL IN is a value I do not have yet.

---------------------------------------------------------------------------

## 1. Environment and pinned versions

| Item | Value |
|---|---|
| OS / shell | Windows, Command Prompt (cmd) |
| Python | 3.10 |
| Project path | E:\Projects\AskPG |
| RAM | 16 GB (about 92% used earlier, Docker/WSL2 likely the cause) |
| GPU | NVIDIA GTX 1660 Ti, 6 GB VRAM, driver 610.74 (CUDA 13.3 supported) |
| torch | 2.10.0+cu128, CUDA verified (device prints cuda:0 at index time) |
| Precision | fp32 everywhere (the 16-series has no useful fp16 / tensor cores) |
| Vector DB | Qdrant, image qdrant/qdrant:v1.19.1, digest sha256:12364fe851b9f17356fc88189fc06d1b521262e04659ec7345975b00c9246a10 (running container `qdrant` confirmed to use it), qdrant-client 1.19.1, localhost:6333 |
| Embedder | BAAI/bge-small-en-v1.5 (384-dim, cosine, max_seq_length 512), query-side instruction prefix only |
| Keyword search | bm25s (in memory), built from chunks.jsonl at start-up |
| Reranker | BAAI/bge-reranker-base, CrossEncoder, fp32, max length 512 |
| Docs corpus | PostgreSQL 17 HTML docs, crawled from https://www.postgresql.org/docs/17/ (crawl date: FILL IN) |
| Other packages | sentence-transformers, transformers, bm25s, openai, python-dotenv, requests, beautifulsoup4 (exact pip versions: FILL IN from `pip freeze`) |
| Git commit of each run | FILL IN (add `git rev-parse --short HEAD` to every run row from now on) |

Harmless warnings: HF_TOKEN "unauthenticated" notice; "position_ids UNEXPECTED" load notes.

## 2. Pipeline configuration (current baseline)

| Stage | Setting |
|---|---|
| Chunk budget | MAX_TOKENS=400, OVERLAP_TOKENS=60, MIN_TOKENS=25, counted with the BERT tokenizer of the embedder (not tiktoken) |
| Chunk size result | 6489 chunks, tokens min 25 / median 276 / max 472; 0 truncated by the embedder |
| Embedded text | breadcrumb (`context`) + newline + body |
| Index | collection askpg_pg17, 6489 points, deterministic uuid5 point ids |
| BM25 tokenizer | lowercase, snake_case kept whole, small stopword list, no stemming, SPLIT_IDENTIFIERS=False |
| Candidates per retriever | N_CANDIDATES=50 |
| Fusion | Reciprocal Rank Fusion, RRF_K=60 |
| Reranker input | top RERANK_TOP=30 fused candidates, pair = (query, breadcrumb + text), RERANK_MAX_LEN=512 |
| Reranker score | sigmoid of the logit (0-1, saturates near 1) |
| k to the LLM | DEFAULT_K=5 |
| LLM | any OpenAI-compatible endpoint; default openai/gpt-oss-20b on NVIDIA build.nvidia.com; temperature 0; max tokens 1500 |
| Citation enforcement | well-formedness only (refusal sentence or at least one citation, all [n] in 1..k, one corrective retry). NOT faithfulness |
| Refusal | LLM only. No retrieval gate built |

## 3. Index / chunking experiment log

| Version | What changed | Chunks | Dropped | Truncated at 512 | Index time | Notes |
|---|---|---|---|---|---|---|
| v0 | tiktoken budget 380, CPU | 6008 | n/a | 83 silently truncated | 807 s | wrong tokenizer for the embedder |
| v1 | BERT-tokenizer budget 400, GPU | 6499 | 1019 | 0 | 104 s | tokenizer fix; many drops of small sections |
| v2 | merge small sections into a neighbour; skip See Also / Author(s) | 6489 | 33 | 0 | 90 s | 762 merged, 243 skipped (211 See Also, 21 Author, 11 Authors; all verified legitimate); `covers` field added |
| anchor fix | heading_anchor() tries several HTML layouts | 6489 | n/a | n/a | n/a | chunks with '#anchor' in the url: 0 -> 6489/6489 |
| v3 | chunk_blocks fix (lead-in stays with next block; short tail appended to previous chunk) + first chunk of a merged section takes covers[0] anchor/breadcrumb | 6489 | 13 (true loss 2) | 0 | 81 s | tokens 25 / 276 / 472; verified on sql-rollback (chunk 0000 url ends #SQL-ROLLBACK with context "ROLLBACK"; chunk 0002 is "ROLLBACK > Examples") |

The 13 v3 drops were read: 2 are real losses (index copyright line, internals part intro: junk); 11 are single-sentence table/view intros whose text is present inside a kept chunk.

Known accepted weaknesses: some chunks start mid-flow; many short chunks on SQL command pages; table cells flattened to text; a short fact appended to the end of a long chunk is exposed to reranker truncation (COMMENT); anchors are per section, not per parameter; v3 gate-order flaw (size check runs before held lead-ins are popped, so a tiny leftover chunk can be emitted, e.g. sql-comment:0006 is fully contained in 0007; cosmetic, fix only if a re-chunk happens anyway).

## 4. Evaluation dataset and label versions

Dataset: eval/dataset.jsonl, 30 questions: 12 factual (including the four known-answer regression cases: COMMENT standard, ROLLBACK syntax, work_mem, ALTER TABLE SET LOGGED), 5 multi_hop (ALL golds needed), 4 ambiguous (ANY gold is enough), 9 unanswerable (7 are near-misses: MySQL replication, SQL Server Always On, system-versioned temporal tables, ClickHouse comparison, Kubernetes operator, PostgreSQL hosting provider, uuidv7() which is not in 17). 21 answerable + 9 unanswerable. 1 question = 4.8 points of a success@k metric.

Labels were drafted by Claude from memory. A label is trusted only after it passes eval\check_gold.py, the run_eval coverage check, and a read of the sentence (eval\show_section.py).

| Version | Definition | Gold sections | When / how |
|---|---|---|---|
| d1 | original labels, with the amb-03 heading fix to '19.8.2. When to Log' | FILL IN (d1 count) | before any run |
| d2 | d1 + multi-03 first gold -> '5.12.2.1. Example'; multi-05 first gold -> '18.9.1. Basic Setup'; amb-03 + {auto-explain, AUTO-EXPLAIN} and {pgstatstatements, PGSTATSTATEMENTS} | 35 | 7 Oct 2026, eval\relabel_d2.py. Made AFTER seeing the vector failures only. Backup: eval\dataset_pre_d2.jsonl |
| d3 | d2 + 7 added gold sections (below) | 42 | 7 Oct 2026, eval\relabel_d3.py. Backup of d2: eval\dataset_pre_d3.jsonl. The script refuses to run twice |

d3 additions (each sentence read in the docs text):
- fact-05 (VACUUM FULL vs plain VACUUM): + sql-vacuum `Description`, routine-vacuuming `24.1.1. Vacuuming Basics`, routine-vacuuming `24.1.2. Recovering Disk Space`. Existing gold: sql-vacuum `Parameters`. Evidence: sql-vacuum:0001 (rewrites the table, much slower, ACCESS EXCLUSIVE); routine-vacuuming:0001 (reclaims more space, much more slowly, ACCESS EXCLUSIVE); routine-vacuuming:0002 (plain VACUUM does not return space to the OS; VACUUM FULL writes a new file, can take a long time, needs extra disk).
- amb-02 (back up a database): + {backup, BACKUP} (backup:0000 names the three approaches; thin but valid).
- fact-09 (EXPLAIN ANALYZE): + sql-explain `Description` ("The ANALYZE option causes the statement to be actually executed, not only planned..."), + sql-explain `Parameters` (ANALYZE: "Carry out the command and show actual run times and other statistics"; only chunk 0003 of that section states it, limit (a) applies).
- amb-01 (speed up queries on a large table): + ddl-partitioning `5.12.1. Overview` ("Query performance can be improved dramatically in certain situations"). The existing parent golds (5.12, 14.1) stay.
- multi-04: no change.

d3 disclosure
- d3 labels were widened after reading the d2 rerank results, which favours the reranker. The fact-05 and amb-02 additions came from reading the docs after the d2 failures. The fact-09 and amb-01 candidates came from Claude's memory and were verified by reading the sentence.
- Effect of the fact-05 widening alone: the reranker's question-level net moved from +1 to +2 vs hybrid and from -1 to 0 vs vector. This is a label effect, not an improvement.
- Of the new fact-05 sections the reranker preferred sql-vacuum Description and 24.1.2 over Parameters, but demoted 24.1.1 (hybrid 2 -> rerank 7).
- The sweep (eval\sweep_gold.py) confirmed NO new section by itself.
- Dev-set relabelling is CLOSED after d3.

Sweep outcome per question (four classes, never "swept all 21")
- (a) Read and rejected: fact-03 (`-S` lives in the ~15-option postgres General Purpose section, any chunk of it would hit; 19.18 Short Options is a lookup table); fact-06 (`-N` says "chosen automatically by initdb"; kernel-resources has formulas, no default); fact-07 (24.1 intro only points to 24.1.6); fact-11 (52.9 describes the view); fact-01 (control, no additions); earlier rejections: fact-02, fact-04, fact-08, fact-10, fact-12, multi-01, multi-02, multi-04, multi-05, amb-03, amb-04 (candidates shared terms or were subsections / lookup tables / other commands). BORDERLINE and excluded by rule: fact-07 / 19.10 (the autovacuum_naptime entry says the daemon issues VACUUM and ANALYZE as needed) and fact-11 / 19.2 (hba_file is the host-based authentication file).
- (b) Confirmed: fact-05, fact-09, amb-01, amb-02 (BACKUP is thin).
- (c) Sweep too thin (0-1 usable terms), checked by `find --text` only: fact-01, fact-03, fact-06, fact-11, multi-03. Also weak: fact-07, fact-09, amb-01, amb-03, amb-04.
- (d) multi_hop alternatives skipped by design (rule 7): 29.2.2 / 29.2 / 29.8 for multi-02; 18.8 Encryption Options (hostssl) for multi-05; sql-createtable Parameters for multi-03. Logged under limit (b).

Labelling rules in force: (1) leaf-level, disjoint sections; matching is strict (a chunk hits only if the gold section is in its own `covers`); (2) judge labels from the docs, never from retrieval lists; (3) freeze before running modes, version any later change, re-run all four modes; (4) hold out about 25% as an untouched TEST split once the set reaches 50-100 questions; (5) a section is gold only if a SENTENCE states the answer (lookup tables, shared terms and partial/parameter-entry sentences do not qualify; borderline = exclude and log); (6) a label list is "confirmed to answer", never "complete"; (7) multi_hop: add a section only for a third needed piece, never an alternative for one half.

## 5. Run registry (eval/results/*.json)

| Result file | Labels | Retriever | Notes |
|---|---|---|---|
| v3-vector | d1 | vector | first vector run on the v3 index |
| d1-bm25 | d1 | bm25 | the `name` field inside the file says d2 (wrong); run on d1 labels |
| d1-hybrid | d1 | hybrid | same `name` field issue |
| d2-vector | d2 | vector | |
| d2-bm25 | d2 | bm25 | |
| d2-hybrid | d2 | hybrid | |
| d2-hybrid_rerank | d2 | hybrid_rerank | |
| d3-vector | d3 | vector | 7 Oct 2026 |
| d3-bm25 | d3 | bm25 | 7 Oct 2026 |
| d3-hybrid | d3 | hybrid | 7 Oct 2026 |
| d3-hybrid_rerank | d3 | hybrid_rerank | 7 Oct 2026 |

All runs: 30 questions (21 answerable, 9 unanswerable), k=10, single process, warm, one query at a time, index = v3 (6489 chunks). Latency is local only (no LLM calls).

## 6. Retrieval results

### 6.1 d3 labels (CURRENT, frozen)

| mode | success@1 | success@3 | success@5 | success@10 | MRR | gold_recall@5 | distinct sections top 5 | p50 / p95 total ms |
|---|---|---|---|---|---|---|---|---|
| vector | 0.476 | 0.714 | 0.810 | 0.905 | 0.616 | 0.762 | 3.857 | 34 / 53 |
| bm25 | 0.429 | 0.571 | 0.667 | 0.857 | 0.537 | 0.639 | 4.333 | 0.2 / 0.4 |
| hybrid (RRF) | 0.476 | 0.619 | 0.714 | 0.952 | 0.586 | 0.710 | 4.048 | 37 / 57 |
| hybrid_rerank | 0.619 | 0.714 | 0.810 | 0.905 | 0.695 | 0.714 | 4.238 | 1215 / 1524 |

Rows for the README table (as printed by run_eval): d3-vector 0.476 / 0.810 / 0.905 / 0.616 / 34 / 53; d3-bm25 0.429 / 0.667 / 0.857 / 0.537 / 0 / 0; d3-hybrid 0.476 / 0.714 / 0.952 / 0.586 / 37 / 57; d3-hybrid_rerank 0.619 / 0.810 / 0.905 / 0.695 / 1215 / 1524 (columns: success@1 / success@5 / success@10 / MRR / p50 ms / p95 ms).

Questions passing @5 (of 21): vector 17, bm25 14, hybrid 15, hybrid_rerank 17. Rank-1 hits: 10 / 9 / 10 / 13.

Per type (success@1 / @3 / @5 / @10 / MRR). multi_hop@1 = 0 is structural (ALL golds needed): read multi_hop at @5/@10 (n=5). gold_recall@5 is misleading for ambiguous (ANY is enough): read success@k.

| type (n) | vector | bm25 | hybrid | hybrid_rerank |
|---|---|---|---|---|
| factual (12) | 0.750 / 1.000 / 1.000 / 1.000 / 0.861 | 0.583 / 0.750 / 0.833 / 0.917 / 0.683 | 0.667 / 0.833 / 0.917 / 1.000 / 0.767 | 0.917 / 0.917 / 0.917 / 0.917 / 0.917 |
| multi_hop (5) | 0.000 / 0.000 / 0.200 / 0.600 / 0.104 | 0.000 / 0.200 / 0.400 / 0.800 / 0.189 | 0.000 / 0.200 / 0.400 / 0.800 / 0.173 | 0.000 / 0.200 / 0.400 / 0.800 / 0.179 |
| ambiguous (4) | 0.250 / 0.750 / 1.000 / 1.000 / 0.521 | 0.500 / 0.500 / 0.500 / 0.750 / 0.536 | 0.500 / 0.500 / 0.500 / 1.000 / 0.556 | 0.500 / 0.750 / 1.000 / 1.000 / 0.675 |

Questions failing @5 (gold ranks, one per gold section):
- vector: multi-01 [7,3]; multi-02 [None,5]; multi-04 [8,2]; multi-05 [2,None]
- bm25: fact-02 [6]; fact-10 [None]; multi-02 [7,3]; multi-03 [10,1]; multi-05 [3,None]; amb-01 [7,None,None,None]; amb-04 [None,None,None]
- hybrid: fact-10 [8]; multi-02 [6,3]; multi-04 [6,2]; multi-05 [1,None]; amb-01 [8,None,None,None]; amb-04 [None,None,10]
- hybrid_rerank: fact-01 [None]; multi-01 [7,1]; multi-04 [2,6]; multi-05 [4,None]

Latency d3 (ms, p50 / p95 / max):

| stage | vector | bm25 | hybrid | hybrid_rerank |
|---|---|---|---|---|
| embed | 18.1 / 23.8 / 24.2 | n/a | 15.7 / 24.4 / 26.3 | 16.8 / 21.7 / 32.0 |
| vector search | 13.2 / 37.0 / 37.5 | n/a | 19.9 / 38.7 / 40.3 | 10.5 / 31.6 / 34.8 |
| bm25 | n/a | 0.2 / 0.4 / 12.2 | 0.4 / 0.5 / 0.6 | 0.4 / 0.9 / 11.6 |
| fusion | n/a | n/a | 0.1 / 0.1 / 0.1 | 0.1 / 0.1 / 0.1 |
| rerank | n/a | n/a | n/a | 1180.8 / 1487.2 / 1512.5 |
| total | 33.8 / 53.4 / 58.3 | 0.2 / 0.4 / 12.2 | 37.4 / 57.2 / 64.5 | 1214.7 / 1524.2 / 1565.7 |

Rerank is about 97% of local time; hybrid_rerank costs about 33x hybrid. 30 queries, so p95 is roughly the second-largest value (indicative only).

### 6.2 d2 labels (superseded; do NOT compare with d3 as progress)

| mode | success@1 | success@3 | success@5 | success@10 | MRR | p50 / p95 total ms |
|---|---|---|---|---|---|---|
| vector | 0.381 | 0.714 | 0.810 | 0.905 | 0.552 | 27 / 59 |
| bm25 | 0.381 | 0.524 | 0.619 | 0.857 | 0.496 | 0.2 / 0.4 |
| hybrid (RRF) | 0.429 | 0.571 | 0.714 | 0.952 | 0.550 | 36 / 60 |
| hybrid_rerank | 0.571 | 0.667 | 0.762 | 0.905 | 0.655 | 1298 / 1634 (rerank p50 1237, p95 1605) |

Questions passing @5 (of 21): vector 17, bm25 13, hybrid 15, hybrid_rerank 16. Rank-1 hits: 8 / 8 / 9 / 12. Per type at @5 (vector / bm25 / hybrid / rerank): factual 1.00 / 0.75 / 0.92 / 0.83; multi_hop 0.20 / 0.40 / 0.40 / 0.40; ambiguous 1.00 / 0.50 / 0.50 / 1.00.

### 6.3 d1 labels (superseded)

| mode | success@1 | success@3 | success@5 | success@10 | MRR |
|---|---|---|---|---|---|
| vector | 0.381 | 0.667 | 0.714 | 0.810 | 0.517 |
| bm25 | 0.333 | 0.476 | 0.571 | 0.762 | 0.443 |
| hybrid (RRF) | 0.381 | 0.476 | 0.619 | 0.857 | 0.487 |

The d1 -> d2 -> d3 changes in bm25 and vector (bm25 @5: 0.571 -> 0.619 -> 0.667; vector @5: 0.714 -> 0.810 -> 0.810) came from label edits alone, with no retriever change.

## 7. Reranker harm / lift analysis (eval\compare_runs.py, k=5)

HARM = pass in A, fail in B. LIFT = fail in A, pass in B. Question level is the headline number; section level counts redundant gold sections of questions that still pass (fact-05 twice and amb-02 in d3). The exposure flag means "at least one chunk of the gold section has a query+chunk pair over 512 XLM-R tokens"; it over-flags big sections, so read "exposed chunks n/m".

### 7.1 d3 (CURRENT)

| comparison | question level | section level harm | section level lift | harm exposed / not exposed | lift exposed / not exposed |
|---|---|---|---|---|---|
| hybrid -> hybrid_rerank | pass 15 -> 17, net +2; LIFT 4 (fact-10, multi-02, amb-01, amb-04); HARM 2 (fact-01, multi-01); both fail: multi-04, multi-05 | 6 of 25 (24%) | 6 of 17 (35%) | 1 of 3 / 5 of 22 | 0 of 2 / 6 of 15 |
| vector -> hybrid_rerank | pass 17 -> 17, net 0; LIFT 1 (multi-02); HARM 1 (fact-01); both fail: multi-01, multi-04, multi-05 | 5 of 27 (19%) | 3 of 15 (20%) | 2 of 4 / 3 of 23 | 0 of 1 / 3 of 14 |

Changed gold sections vs hybrid (A rank -> B rank, max pair length in XLM-R tokens, exposed chunks):
- HARM fact-01 Compatibility 2 -> None, 561 tok, 2/3 exposed
- HARM fact-05 Parameters 4 -> 6, 488 tok, 0/7
- HARM fact-05 24.1.1 Vacuuming Basics 2 -> 7, 385 tok, 0/1
- HARM multi-01 13.2.1 Read Committed 5 -> 7, 488 tok, 0/5
- HARM multi-04 25.3 Continuous Archiving 2 -> 6, 441 tok, 0/2
- HARM amb-02 25.1 SQL Dump 1 -> 6, 408 tok, 0/2
- LIFT fact-10 19.11.1 Statement Behavior 8 -> 1; multi-02 29.12 Quick Setup 6 -> 2; multi-04 25.1 SQL Dump 6 -> 2; amb-01 11.1 Introduction 8 -> 5; amb-02 BACKUP 9 -> 4; amb-04 20.1 pg_hba.conf None -> 1

Changed gold sections vs vector (A rank -> B rank, max pair length in XLM-R tokens, exposed chunks):
- HARM fact-01 Compatibility 2 -> None, 561 tok, 2/3
- HARM fact-05 Parameters 3 -> 6, 488 tok, 0/7
- HARM fact-05 24.1.1 Vacuuming Basics 2 -> 7, 385 tok, 0/1
- HARM multi-04 25.3 Continuous Archiving 2 -> 6, 441 tok, 0/2
- HARM amb-04 5.9 Row Security Policies 4 -> None, 528 tok, 1/14
- LIFT multi-02 29.12 Quick Setup None -> 2; multi-04 25.1 SQL Dump 8 -> 2; amb-04 20.1 pg_hba.conf 9 -> 1

### 7.2 d2 (superseded)

| comparison | question level | section harm | section lift | harm exposed / not exposed |
|---|---|---|---|---|
| hybrid -> hybrid_rerank | LIFT 4 (fact-10, multi-02, amb-01, amb-04); HARM 3 (fact-01, fact-05, multi-01); net +1 | 5 of 21 (24%) | 5 of 14 (36%) | 1 of 3 / 4 of 18 |
| vector -> hybrid_rerank | LIFT 1 (multi-02); HARM 2 (fact-01, fact-05); net -1 | 4 of 22 (18%) | 3 of 13 (23%) | 2 of 4 / 2 of 18 |

## 8. Findings

Verified (from saved results)
1. Labels moved the metrics more than the code (section 6.3).
2. The reranker is a PRECISION gain, not a recall gain: rank-1 hits 13 vs 10 (vector), MRR 0.695 vs 0.616, factual @1 0.917 vs 0.750; but @5 is 17 vs 17 and @10 equals vector (hybrid has the best @10, 0.952). Cost: about 33x hybrid's latency. At k=5 sources to the LLM, @5 decides whether the answer is in the prompt.
3. Reranker net vs hybrid is +2 (at the noise boundary: 1 question = 4.8 points); vs vector it is 0. Factual @5: reranker 11/12 vs vector 12/12 (fact-01).
4. Truncation explains a MINORITY of harms: the only question-level truncation harm is fact-01 (2 of 3 chunks exposed, pair up to 561 tokens). Section harm among exposed sections: 1 of 3 vs 5 of 22 (hybrid), 2 of 4 vs 3 of 23 (vector). Counts too small to prove anything; the direction fits.
5. Hybrid vs vector on d3: tie at @1 (0.476), worse at @3 and @5 (0.714 vs 0.810), better at @10 (0.952 vs 0.905), MRR 0.586 vs 0.616. Vector fails @5 only on multi_hop; hybrid also fails fact-10, amb-01, amb-04.
6. fact-05 (d2 harm) was a LABEL LIMIT, not a reranker error; fixed by d3 (passes in all four modes, so it no longer discriminates).
7. multi-01 is a real near-tie harm (reranker scores 0.966-0.998 among SET TRANSACTION, 13.4 application-level pages, 13.2.3 Serializable, two chunks from one page).
8. multi-04 on d3 is a SWAP between its two golds (hybrid SQL Dump 6 / continuous archiving 2; rerank 2 / 6; vector 8 / 2): both modes fail by one rank. Cause is a VOCABULARY GAP (below), not truncation, not a label defect.
9. multi-05 is a real retrieval miss in every mode: the hostssl text is in pg_hba 20.1 chunks 0002/0003/0011; vector ranks 66/23, bm25 14/149, hybrid 33/41, chunk 0002 outside the top 200 of both; hybrid ranks beyond 30 are invisible to the reranker.
10. multi_hop is weak everywhere (@5: 0.2 vector, 0.4 others; n=5).
11. COMMENT (fact-01) fully diagnosed: the answer sentence is in sql-comment:0008 at char 1636 of 1675, in the cut tail of the reranker pair; ranks vector 2, bm25 3, hybrid 2, hybrid_rerank 22/30; predict on the full chunk 0.0199, last 300 characters only 0.996.
12. Reranker truncation exposure: 159 of 6489 chunks (2.5%) are over 480 XLM-R tokens before the query is added (largest chunk: 472 BERT = 689 XLM-R). A reranker score cannot reveal it.
13. Reranker score scale: predict returns sigmoid outputs (0.000-1.000), saturating (all five work_mem hits 0.958-0.978; top 8 of multi-01 0.966-0.998).
14. Scores from bge-small sit in a narrow band (about 0.63-0.88).
15. glossary:0013 really contains the Rollback definition, so there is NO glossary experiment.

multi-04 control experiment (7 Oct 2026, ONE question, reading aid only, not evidence for a pipeline change). Reranker scores for continuous-archiving:0000 / 0001 (0001 holds the pg_dump contrast):

| query | 0000 | 0001 |
|---|---|---|
| full original question | 0.060 | 0.136 |
| what is continuous archiving? | 0.851 | 0.896 |
| how does continuous archiving differ from pg_dump? | 0.170 | 0.950 |
| ... from a logical dump? | 0.055 | 0.863 |
| ... from an SQL dump? | 0.156 | 0.301 |
| SQL dump versus continuous archiving | 0.210 | 0.385 |

Reading: chunk 0001 never uses the words "SQL dump" (it says pg_dump / logical), and the same comparison phrasing scores 0.86-0.95 when it uses those words. So the weak spot is vocabulary, not multi-part structure; query decomposition is NOT supported as the fix. The first version of this diagnostic changed two variables at once (aspect split AND the term pg_dump); only the control separated them.

Hypotheses (untested)
- RRF gives the weaker BM25 an equal vote, which is why hybrid does not beat vector at @5. Do not tune on 21 questions.
- A cross-encoder scores chunks against the whole multi-part query, so overview chunks that mention both halves beat chunks that answer one half well. Not supported for multi-04; open for multi-01 and multi-05.
- Larger RERANK_TOP alone will not fix multi-05 (not predicted to help).

Retracted
- "Chunk 0000 of continuous-archiving is not self-contained" (it scores 0.851 on its own topic).
- "Query decomposition fixes multi-04" (the control contradicts it).
- The earlier "net +1 vs hybrid / -1 vs vector" and the d2 harm counts, replaced by d3 (a label effect, see section 4).

## 9. Refusal calibration (retrieval-only; unchanged by relabelling)

Top-1 score, answerable vs unanswerable (d3):

| mode | answerable min / median / max | unanswerable min / median / max |
|---|---|---|
| vector (cosine) | 0.704 / 0.787 / 0.883 | 0.490 / 0.735 / 0.785 |
| bm25 | 3.759 / 6.582 / 18.905 | 2.048 / 4.675 / 5.902 |
| hybrid (RRF) | 0.024 / 0.033 / 0.033 | 0.016 / 0.028 / 0.031 |
| hybrid_rerank (sigmoid) | 0.328 / 0.989 / 1.000 | 0.000 / 0.310 / 0.977 |

Cosine, BM25 and RRF do not separate the two sets. Reranker top-1: lowest answerable amb-04 0.328, amb-01 0.562, fact-10 0.585, fact-01 0.705, multi-05 0.820, multi-04 0.860. Unanswerable: World Cup 0.000, capital of France 0.022, SQL Server 0.056, uuidv7 0.087, Kubernetes 0.310, ClickHouse 0.440, hosting provider 0.666, temporal tables 0.681, MySQL replication 0.977.

Threshold sweep on hybrid_rerank top-1 (refuse when top-1 < t; 21 answerable, 9 unanswerable):

| t | answerable wrongly refused | unanswerable let through |
|---|---|---|
| 0.05 | 0 of 21 | 7 of 9 |
| 0.10 | 0 of 21 | 5 of 9 |
| 0.20 | 0 of 21 | 5 of 9 |
| 0.30 | 0 of 21 | 5 of 9 |
| 0.50 | 1 of 21 | 3 of 9 |
| 0.70 | 3 of 21 | 1 of 9 |
| 0.90 | 7 of 21 | 1 of 9 |
| 0.95 | 8 of 21 | 1 of 9 |
| 0.98 | 10 of 21 | 0 of 9 |

Conclusion: a threshold of 0.70 would wrongly refuse 3 broad answerable questions (amb-04, amb-01, fact-10) and let MySQL replication through; the gap between 0.681 and 0.705 is 0.024 on 9 unanswerable questions, which cannot calibrate anything. Illustrative only (tiny n, same data used to read the sweep, sigmoid saturates). Do NOT build a gate before the LLM's own refusal behaviour on the unanswerable set is measured; if the LLM already refuses them, a gate adds only false refusals. Recalibrate on any chunker or reranker change; try raw logits.

LLM refusal probes (6 Oct 2026, openai/gpt-oss-20b, hybrid_rerank, k=5, n=3, NOT an evaluation): World Cup, MySQL replication and hosting provider were all status=refused (MySQL: LLM 8.9 s, 1250 in / 125 out tokens; hosting: 3.0 s, 525 / 91).

## 10. LLM provider notes

Provider: NVIDIA build.nvidia.com free API (https://integrate.api.nvidia.com/v1). Probed 30 Sep - 1 Oct 2026, not re-probed since; re-probe with src\probe_llm.py before generation runs and record the exact model id and date for every run.
- Working default: openai/gpt-oss-20b (about 2.8 s on the first test; 3.0-8.9 s on the 6 Oct refusal probes, 91-125 output tokens): LLM latency is highly variable, report p50/p95.
- Also callable: nvidia/nemotron-3-nano-omni-30b-a3b-reasoning (ok, about 12 s); nvidia/nemotron-3-super-120b-a12b (good answers, cites with the full-width bracket, handled by normalization).
- Avoid: nemotron-3-ultra-550b and nemotron-3.5-lightning (reasoning leaks into the answer, slow), glm-5.3-flash (empty, slow); kimi-k3, deepseek-v4.1-flash, glm-5.3 timed out.
- Listed by /models but not callable by this account (HTTP 404): llama-3.1-nemotron-70b/51b, mistral and others. Every model that answered is a reasoning model.
- Limits: about 40 requests/minute (429s), limited free credits, no per-token billing. Latency includes the network and must be reported separately from local numbers. Reasoning models spend max_tokens on hidden thinking (generate.py raises a clear error on an empty answer).
- Cost: LLM_PRICE_IN_PER_M and LLM_PRICE_OUT_PER_M are currently 0, so cost prints $0. Before the generation eval set a STATED reference price and record its source here: price in = FILL IN USD per 1M tokens, price out = FILL IN USD per 1M tokens, source = FILL IN (URL and date). Cost in the write-up is an estimate from token counts, not a bill.
- Local fallback: Ollama (llama3.1:8b / llama3.2:3b); a 6 GB GPU shared with the reranker may spill to CPU.

## 11. Security incident and checklist

Incident: the NVIDIA key was committed in a template file named env.example (no leading dot; the real .env was never committed) and GitGuardian flagged it. Remediation given: revoke and regenerate the key, rename to .env.example with a placeholder that does not start with the key prefix, new .gitignore, delete .git and re-init, force-push (or recreate the repo), enable secret scanning and push protection, mark the alert resolved.

Checklist (tick only after verifying each step yourself):
- [ ] old key revoked in the NVIDIA console
- [ ] new key only in the local .env (never in chat, code or the template)
- [ ] .env.example holds a placeholder that does not start with a real-looking key prefix
- [ ] .gitignore contains .env, data/, qdrant_storage/, __pycache__/, *.pyc, .venv/
- [ ] `git status` read before the commit
- [ ] `git grep --cached -n "nvapi-"` prints nothing
- [ ] repo history shows one clean commit
- [ ] secret scanning and push protection enabled on the GitHub repo
- [ ] GitGuardian alert marked resolved

## 12. Known limitations (README material)

1. Section-level hit matching: every chunk of a section carries the same `covers`, so a hit means "a chunk of the right section was retrieved", not "the chunk with the answer"; success@1 can be overstated where one section holds many items. Big multi-item sections are not added as gold for one item (fact-03 `-S`).
2. Section-level labels punish a retriever that finds a DIFFERENT valid source (fact-05 and amb-02 were such cases; alternative sources for one half of a multi_hop question are the same limit). The generation-level answer-correctness judge must arbitrate.
3. Parent-level golds are penalised under strict matching (amb-01 5.12 and 14.1, amb-03 14.1); a leaf was added for amb-01 only.
4. Label noise and small n: 21 answerable questions, 1 question = 4.8 points; differences under about 2 questions are noise; multi_hop n=5. Labels were widened after seeing results (d2, d3) and the widening favoured the reranker.
5. Anchors are per section, not per parameter: a work_mem citation links to the "Memory" section, a glossary hit links to #GLOSSARY.
6. Citation enforcement checks well-formedness, NOT that the cited source supports the claim (faithfulness needs the LLM judge). `uncited_sentences` is a noisy proxy.
7. Reranker truncation: 512 XLM-R tokens cut the END of long chunks (2.5% of chunks exposed); a reranker score cannot reveal it.
8. Vocabulary gaps (SQL dump vs pg_dump) are a retrieval problem; decomposition does not fix them.
9. Refusal rests on the LLM alone; no retrieval gate; reranker scores separate unanswerable questions only partially.
10. Hosted-model drift and availability; free-tier rate limits and credits; LLM latency includes the network.
11. Latency numbers are local, single process, warm, one query at a time, on a GTX 1660 Ti; a deployed version will likely run on CPU.
12. A lexical sweep is blind on short reference answers; a label list is "confirmed to answer", not "complete".

## 13. Planned experiments (each one change, logged on the larger dev set; none started)

| # | Experiment | Motivation / evidence | What to measure | Cost to watch |
|---|---|---|---|---|
| E1 | Query expansion for vocabulary gaps (SQL dump -> pg_dump / logical dump) | multi-04 control | multi_hop and ambiguous success@5, per-question lift/harm | extra latency, risk of drift |
| E2 | Query decomposition for multi-aspect questions | multi-05, multi-01 (NOT supported for multi-04) | multi_hop success@5/@10 | extra retriever calls, LLM calls |
| E3 | RERANK_TOP depth (30 -> 50 / 100) | multi-05 gold at hybrid rank 33-41 invisible to the reranker | multi_hop success@5, rerank latency | rerank time scales with depth |
| E4 | Crowding / diversity (two chunks from one page) | multi-01 | distinct sections in top 5, success@5 | possible precision loss |
| E5 | Why hybrid under-performs vector at @5 (RRF weights, fewer BM25 candidates) | d3 hybrid @5 0.714 vs vector 0.810 | success@5 / @10 per type | none expected |
| E6 | Reranker-truncation fixes, each separate: (1) budget chunks by the larger of the two tokenizers (re-baseline); (2) score overlapping windows of long chunks and take the max (only the ~2.5% exposed); (3) blend reranker score with RRF rank | fact-01 (test case), 159 exposed chunks | fact-01 and exposed-section harm rate; MUST NOT raise RERANK_MAX_LEN | re-index or extra rerank calls |
| E7 | Raw-logit reranker score for refusal thresholds | sigmoid saturation | answerable vs unanswerable separation | none |
| E8 | Merge already-kept small sections (sql-altertable crowding) | crowding on command pages | success@k | re-chunk, re-baseline |
| E9 | v3 gate-order fix in chunk_blocks (size check after popping held lead-ins) | sql-comment:0006 duplicate | chunk count, dropped count | re-chunk, re-baseline |
| E10 | SPLIT_IDENTIFIERS=True in the BM25 tokenizer | "work mem" vs work_mem | bm25 and hybrid success@k | none |
| E11 | LLM comparison (gpt-oss-20b vs others), refusal behaviour on the unanswerable set incl. near-misses, closed-book baseline | decides whether a refusal gate is worth building | faithfulness, answer correctness, refusal precision/recall, tokens, cost, latency p50/p95 | rate limit, credits |

Not yet written: generation metrics in run_eval.py (faithfulness, answer correctness, closed-book baseline); tracing and p50/p95 aggregation across many queries; CI regression gate.

## 14. Template for every new entry

```
### E<number>: <one-line name>   (<date>)
Label version: d<n>   Index version: v<n>   Git commit: <short hash>   Result file: eval/results/<name>.json
Change (ONE thing): <what was changed, with the exact parameter values before and after>
Hypothesis / why: <one or two sentences, say if it is a hypothesis>
Command: <the exact command run>
Results (same columns as section 6.1): success@1 / @3 / @5 / @10 | MRR | p50 / p95 total ms | per type
Question-level LIFT / HARM vs the baseline run (eval\compare_runs.py): <ids>
Cost: latency delta, token / cost delta, index time or size delta
Verdict: <keep / drop / inconclusive (n too small)>   Caveats: <label limits, noise, anything changed at the same time>
```

## 15. Generation run g1 (8 Oct 2026)

Source tags: [FILE] = recomputed from eval/results/gen-g1.json or audit_g1.txt; [NOTES] = from my notes of console output (snippet and judge JSON files not re-read when this was written); [READING] = my own reading, one reader.

### 15.1 Setup

| Item | Value |
|---|---|
| Run | gen-g1, meta time 2026-10-08T11:51:26 |
| Retrieval | hybrid_rerank, k=5, labels d3, index v3 (6489 chunks) |
| Generator | openai/gpt-oss-20b via https://integrate.api.nvidia.com/v1, temperature 0, max tokens 1500, min gap 1.6 s |
| Judge | nvidia/nemotron-3-super-120b-a12b, temperature 0, prompt md5 b0f970773ac5, JUDGE_MAX_TOKENS=8000 |
| Dataset md5 | c14aa3a16bd39712d3a6fb5351cd473c |
| chunks_md5 | NOT in g1 meta. Measured at snippet-check time: db65e9d099d2929c2a40c540210b5acc (the judge and audit read chunks.jsonl as it is now) |
| git commit | FILL IN (g1 meta has none) |
| Price | LLM_PRICE_* = 0, so cost prints $0. Reference price: FILL IN with source and date |

### 15.2 Row

| run | model | answerable ok | false refusals | unanswerable refused | llm_error | p50 / p95 total ms | p50 / p95 llm ms | tokens in/out | est. cost |
|---|---|---|---|---|---|---|---|---|---|
| gen-g1 | openai/gpt-oss-20b | 20/21 | 1 | 9/9 | 0 | 9579 / 26684 | 8163 / 25094 | 42764 / 8986 | $0 (price unset) |

[FILE] Status: 20 ok, 10 refused (9 unanswerable + fact-01), 0 citation_failed, 0 llm_error, 0 retries. By type: factual 11 ok / 1 refused, multi_hop 5 ok, ambiguous 4 ok, unanswerable 9 refused.

[FILE] Latency (ms, p50 / p95 / max): retrieval 1342 / 1578 / 1590 (rerank p50 1266); llm 8163 / 25094 / 32340; total 9579 / 26684 / 33919. n=30, so p95 is about the second-largest value. LLM time is about 85% of p50 request time and includes the network. It tracks output length: 1.9 s for a refusal, 32 s for a 745-token answer. completion_tokens include hidden reasoning (a one-sentence answer shows 162).

### 15.3 gold_in_prompt split (end-to-end diagnostic)

[FILE] gold_in_prompt is True for 17 of 21 answerable questions (all status=ok). This equals d3 hybrid_rerank success@5 = 0.810, as expected: same retriever, same k.

| question | gold_in_prompt | outcome | reading |
|---|---|---|---|
| fact-01 (COMMENT) | False | REFUSED | retrieval failure: reranker truncation pushed the gold chunk out of the top 5; the refusal is faithful to the prompt |
| multi-01 | False | ok | answered from incomplete evidence, did not say which part was unsupported |
| multi-04 | False | ok | same; 6 of 9 claims not grounded per the judge, yet judged "correct" and "complete" |
| multi-05 | False | ok | same |

The only false refusal is a retrieval error. The mean `uncited_sentences` proxy (2.65) is not a quality number.

### 15.4 Refusal behaviour

[FILE] 9 of 9 unanswerable questions refused with the exact refusal sentence, including the 7 near-misses (MySQL replication, SQL Server, temporal tables, ClickHouse, Kubernetes, hosting provider, uuidv7). Decision stays: no retrieval gate. A 0.70 reranker threshold would wrongly refuse 3 of 21 answerable questions. Caveat: n=9, one model, one run, temperature 0.

### 15.5 Snippet check (rule 4, deterministic) [NOTES]

26 snippets in 10 of the 20 ok answers: 13 supported, 12 flagged (10 UNSUPPORTED + 2 PARTIAL), 1 trivial. My classification of the 12 [READING, not all verified]:
- 1 wrong in a way that matters: amb-04 `CREATE VIEW ... SECURITY BARRIER` is invalid syntax. The valid form is in rules-privileges:0003, which the generator never saw, and the cited chunk 0002 shows that view as the insecure example.
- Several invented illustrative examples: fact-10 (both), multi-05 hostssl line, amb-01 `idx_col`, amb-03 `log_min_duration = 1000`, amb-04 `REVOKE/GRANT CONNECT ... app_user`.
- Edited docs examples: multi-03 parent table, amb-01 INCLUDE, amb-02 pg_basebackup without `-h mydbserver`.
- Checker gaps: fact-02 `AND NO CHAIN` and fact-04 `ALTER TABLE ... SET LOGGED` (docs use `[ NO ]` and `{ LOGGED | UNLOGGED }`). Correction: fact-02 is not purely a checker artifact, because the sentence around that snippet is wrong against the source (see 15.6).
- Only multi-02's snippets were fully verbatim. 7 of 20 ok answers contain a non-verbatim snippet. This is a count, not an error rate.

Open decision: is rule 4 (verbatim snippets only) the behaviour I want? Known checker gaps (bracket/brace syntax, config one-liners under 8 characters, one changed word fails a line) are NOT fixed. Fixing them after seeing results would be "snippet check v2" and needs a g1 re-check.

### 15.6 LLM judge [NOTES]

20 ok answers, 0 judge errors, 109 claims: 89 supported, 20 not_supported, 0 contradicted (supported rate 0.817). 12 of 20 answers fully supported. Correctness 18 correct / 2 partial; completeness 14 complete / 6 partial. Faithfulness means "stated by the shown sources", i.e. grounding, not truth. Correctness vs the reference is lenient (multi-04 is "correct" and "complete" with 6 of 9 claims ungrounded), so always report both. "contradicted" is not a reliable label; the headline is "not supported by the shown sources". Validation on 7 claims I chose myself: 7/7 match (a smoke test, not an agreement statistic). The "24 claims supported only by an uncited source" figure is unusable (many answers cite once per paragraph).

### 15.7 Judge audit (audit_g1.txt, 79 claims, ONE reader)

Sample: all 20 flagged claims in 8 answers + 4 seeded random fully-supported answers (fact-02, fact-06, fact-08, multi-02). Not a population sample. [FILE: the claim lists below match the audit file; the totals are from my reading.]

- Flagged claims: about 13 right or defensible, 4 false positives, 3 borderline.
  - False positives: multi-03 x3 ("declare the table with a partitioning method...", "virtual constraint / btree index per partition" which [4] and [5] support, "these rules ensure uniqueness..." which [1] states); multi-04 "smaller because it does not dump indexes" (backup-file:0002 [3] says it for indexes; WAL is not mentioned).
  - Borderline: multi-04 "physical backup ... WAL" and "hot backup"; amb-01 closing summary sentence.
- False negatives (judge "supported", I disagree):
  - Clear: fact-02 "`AND CHAIN` (or `AND NO CHAIN`) starts a new transaction" (source: otherwise no new transaction); multi-03 "ancestor tables" (source: descendant); multi-03 "must reference the raw key columns only" (source: must include all key columns).
  - Leniency / inconsistency: invented snippets in fact-10 (`statement_timeout = 300000`), multi-05 (hostssl 192.168.1.0/24) and amb-01 (`idx_col`) marked supported while the fact-10 `5min` snippet was flagged; amb-04 "INSERT, UPDATE, DELETE" not in the sources; amb-02 pg_basebackup "supported" on a non-verbatim evidence quote; amb-02 "three supported methods" lists base backup in place of continuous archiving, unflagged.
- Agreement: 72 of 79 (91%) counting only clear errors; 64 of 79 (81%) counting borderline and leniency cases. One of the 4 random fully-supported answers (fact-02) hid a real error.
- Reading: the judge matches topic words and misses direction and qualifier distortions; errors go both ways and roughly cancel in 0.817, but individual verdicts are unreliable. Use it for relative comparison with the same judge and prompt, audit the claims that change between runs, never present 0.817 as accuracy.

### 15.8 Corrections to earlier statements

- multi-04 PITR "contradicted" was too strong: "cannot be used for PITR" in continuous-archiving:0027 is about standalone hot backups. Accurate: the speed, size and hot-backup claims come from a passage about a different thing (misattribution); the PITR claim is not in the shown chunks (probably true elsewhere: unverified).
- "smaller because pg_dump does not dump indexes" IS in the shown chunks (backup-file:0002); only the WAL part is not.
- multi-03's closing paragraph is supported by ddl-partitioning:0013; its "ancestor" claim is wrong.
- amb-04: the secure-view form is in a chunk the generator never saw, so "not_supported" is defensible and "contradicted" depended on that unseen chunk.
- 8 Oct upload advice was partly wrong: the project copies of generate.py (v2) and dataset.jsonl (d3) were current, no duplicates existed.

### 15.9 Limitations specific to g1

1. One run, one generator, temperature 0, n=21 answerable: no variance estimate. 1 question = 4.8 points.
2. The judge is a single-model instrument with measured disagreement (81-91% on an enriched sample read by one person).
3. g1 meta lacks chunks_md5 and git_commit (patch to run_gen_eval.py pending confirmation); the judge and audit read the current chunks.jsonl.
4. No closed-book baseline yet, so "retrieval adds value" is still unmeasured.
5. Citation enforcement is well-formedness only; status=ok is not quality.
6. multi_hop answers with incomplete evidence are not flagged by the model (multi-01, 04, 05).
7. Rule 4 behaviour is undecided; the snippet check has known gaps.
8. LLM latency includes the network and varies 2-32 s with output length.

### 15.10 To fill in / update elsewhere in this file

- Section 10: add the 8 Oct probe (nemotron-3-super-120b-a12b OK, gpt-oss-20b OK, nemotron-3-nano-omni-30b-a3b-reasoning HTTP 503). Update "not re-probed since 1 Oct".
- Section 13: replace "Not yet written: generation metrics ..." with: tracing and p50/p95 done (tested by g1); judge and snippet check written; still not written: closed-book baseline, CI gate.
- Section 1: git commit, pip freeze, crawl date, d1 gold count (FILL INs).