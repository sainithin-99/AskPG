# AskPG experiment log

Rules for this file
- Only numbers that were actually measured. Hypotheses are labelled as such.
- Bug fixes define the baseline and are not experiments. Diagnostic findings are logged as findings; their fixes are logged as experiments, one change at a time.
- Nothing in "Pre-eval observations" is a result. Each row is a single query (n=1). Retrieval/generation gains are only claimed from eval runs (section "Eval runs"), which do not exist yet.
- Every run records: date, index version, exact model id (hosted models change), and whether latency includes the network.

## Environment
| Item | Value |
|---|---|
| OS / shell | Windows, Command Prompt (cmd) |
| Python | 3.10 |
| GPU | NVIDIA GTX 1660 Ti, 6 GB, driver 610.74, torch 2.10.0+cu128, device cuda:0 |
| RAM | 16 GB (high Docker/WSL2 usage observed) |
| Qdrant server | 1.19.1, image `qdrant/qdrant:v1.19.1`, digest `sha256:12364fe851b9f17356fc88189fc06d1b521262e04659ec7345975b00c9246a10` (running container confirmed to use this image id `sha256:dd57172b...`) |
| qdrant-client | 1.19.1 |
| Embedder | BAAI/bge-small-en-v1.5, fp32, query prefix only |
| Reranker | BAAI/bge-reranker-base (CrossEncoder, fp32, max_length 512), scores are sigmoid outputs (0-1) |
| LLM (default) | openai/gpt-oss-20b via NVIDIA build API (free tier, ~40 RPM) |
| Corpus | PostgreSQL 17 docs (17.11 at crawl time), 1127 pages |

## Chunking and index versions
| Version | What changed | Chunks | Dropped | Truncated (embedder) | Tokens/chunk (min / median / max) | Index time |
|---|---|---|---|---|---|---|
| v0 | tiktoken budget 380, CPU | 6008 | not recorded | 83 (1.4%) at 512 | n/a | 807 s (7 chunks/s) |
| v1 | BERT tokenizer budget 400, GPU | 6499 | 1019 | 0 | n/a | 104 s |
| v2 | small sections merged into a neighbour (762 merged, 243 skipped on purpose) | 6489 | 33 | 0 | n/a | 90 s |
| anchor fix | section anchors in chunk URLs (not an experiment) | 6489 | 33 | 0 | unchanged | n/a |
| v3 | chunk_blocks absorbs short lead-ins/tails; merged chunk takes URL and breadcrumb of covers[0] (bug fix) | 6489 | 13 | 0 | 25 / 276 / 472 | 81 s (~80 chunks/s) |

v3 verification (6 Oct 2026): 6489/6489 chunk URLs carry an anchor; sql-rollback:0000 now ends in `#SQL-ROLLBACK` with context `ROLLBACK` (was `#id-1.9.3.167.6`, `ROLLBACK > Parameters`); skipped.jsonl holds only See Also (211), Author (21), Authors (11).

### v1 drop audit
994 of 1019 drops were whole sections missing from the index (25 were leftovers of kept sections); about 92% came from reference pages (sql 611, spi 129, app 76, ecpg 75, contrib 50). A sample of 25: 7 junk (See Also / author), 10 substantive content, 8 one-line purpose statements. Small sample, wide uncertainty. This motivated v2.

## Findings (diagnostics, not experiments)

### F1. A small drop count hid a lost answer (v2)
Reading the 33 v2 drops showed one was the only chunk containing "There is no COMMENT command in the SQL standard". v2 protected whole sections but not chunks; chunk_blocks could still emit a sub-25-token chunk. v3 fixed it ("no COMMENT command" count 0 -> 1 in chunks.jsonl). Lesson: read the drops; the count is not evidence they are harmless.

### F2. The 13 v3 drops: true loss is 2 fragments
11 of the 13 drops are single-sentence intros to a table or view (catalog pg_index, pg_partitioned_table, pg_trigger, event-trigger-matrix, hstore F.17.2, libpq-example, two monitoring-stats views, multibyte, pgbench, spgist). Checked directly: the first 60 characters of each of these 11 appear inside a kept chunk. The cause is in my v3 gate (size checked before popping the held lead-in, which leaves a tiny leftover chunk; the intro text is carried into the next chunk by the overlap). The other 2 (index copyright line, internals part intro) are junk and not kept. Not fixed: cosmetic. If a later experiment forces a re-chunk, check the size after popping the held lead-ins.

### F3. The reranker silently truncates long chunks (diagnosed, NOT fixed)
- Query: "is the COMMENT command part of the SQL standard?". The answer sentence is in `sql-comment:0008` (396 BERT tokens, breadcrumb "COMMENT > Examples"), at character 1636 of 1675, i.e. the very end of the chunk.
- Ranks of that chunk: vector 2/200, bm25 3/200, hybrid 2/81, hybrid_rerank 22/30. The first stage finds it; the reranker demotes it.
- Cause: the query + chunk pair is cut at 512 XLM-R tokens (the chunk alone is 525). Decoding the truncated input ends mid-"COMMENT ON VIEW ..." and does not contain the sentence.
- CrossEncoder.predict scores: full chunk 0.0199; last 300 characters only (with breadcrumb) 0.996; the rank-1 chunk for this query scored 0.7052.
- Caveat: the tail-only input also removes the examples, so truncation and content dilution are not perfectly isolated.
- Scale: the XLM-R tokenizer counts much more than the BERT one on this corpus (largest chunk: 472 BERT = 689 XLM-R tokens). 159 of 6489 chunks (2.5%) exceed 480 XLM-R tokens before the query is added (approximate threshold).
- A reranker score cannot reveal this (the truncated chunk looked "clearly irrelevant").
- Candidate fixes, each a separate experiment after the baseline eval: (1) budget chunks by the larger of the two tokenizers (re-baseline); (2) score overlapping windows of long chunks and take the max; (3) blend the reranker score with the RRF rank.

### F4. work_mem demotion is NOT truncation
- Query: "what does work_mem control?". Definition chunk `runtime-config-resource:0005` (265 BERT tokens; the phrase "Sets the base maximum amount of memory" is at character 19 of 1236, so the start of the chunk survives truncation).
- Ranks: vector 2, bm25 2, hybrid 1, hybrid_rerank 5 of 30. Reranker scores of the top 5 are all 0.958-0.978; rank 1 is `hash_mem_multiplier` (0.9782, mentions work_mem in its text), the definition chunk scores 0.9578.
- Cause unknown. Not investigated further from one query.

### F5. Reranker scores are sigmoid outputs and saturate
Observed range 0.02-0.9955. Ranking is unaffected (monotonic), but separation is weak: ROLLBACK correct chunk 0.87 vs irrelevant 0.52-0.61; all five work_mem hits 0.958-0.978; the MySQL probe below has a top-1 of 0.9775. Raw logits (if the installed sentence-transformers `predict` accepts `activation_fn`) are a candidate for refusal-threshold calibration.

## Pre-eval observations (single queries, NOT results)
| Query | Index | First stage | After hybrid_rerank | Note |
|---|---|---|---|---|
| what is the syntax of ROLLBACK? | v2 | vector top-1 and bm25 top-1 = Compatibility/Examples chunk; synopsis chunk vector rank 3, not in bm25 top 5 | synopsis chunk rank 1 (0.8733) | reranker helped. glossary:0013 (rank 3) legitimately contains the Rollback definition |
| is the COMMENT command part of the SQL standard? | v3 | hybrid rank 2 | rank 22 of 30 | F3, reranker harm via truncation |
| what does work_mem control? | v3 | hybrid rank 1 | rank 5 | F4, reranker harm, cause unknown |
| ALTER TABLE SET LOGGED | v2 (UNLOGGED check on v3 chunks) | n/a | rank 1 = sql-altertable:0016 | 4 of 5 hits from one page (crowding); rank 5 is ALTER SEQUENCE. Rerun on v3 before relying on it |

Two of these (ROLLBACK, ALTER TABLE) were run on the v2 index and not re-run after v3; the eval will cover them.

## Refusal probes (pre-eval, n=3, model openai/gpt-oss-20b, 6 Oct 2026, hybrid_rerank k=5)
| Query | Status | Top-1 reranker score | Retrieved sources | Tokens in/out | LLM ms |
|---|---|---|---|---|---|
| who won the 2022 World Cup? | refused | not recorded | pure noise | not recorded | not recorded |
| how do I set up MySQL replication? | refused | 0.9775 (logical-replication-quick-setup:0000), then 0.6026, 0.5914 | PostgreSQL logical replication pages | 1250 / 125 | 8880 |
| what is the best managed PostgreSQL hosting provider? | refused | not recorded | noise (Solaris notes, sourcerepo, external admin tools, JIT, history) | 525 / 91 | 3045 |

Reading: the LLM refused all three, including the topically close MySQL case. But the MySQL top-1 reranker score (0.9775) is as high as answerable queries (0.87-0.9955), so a reranker-score threshold cannot be assumed to separate these; refusal currently rests on the LLM alone. Retrieval always returns k results. The eval's unanswerable set must include near-miss questions (wrong product, similar topic), not only obvious noise. Cost shows $0 because LLM_PRICE_* are 0: set a stated reference price before eval runs (cost is an estimate from token counts, not a bill).

## LLM provider findings (30 Sep - 1 Oct 2026, not re-probed)
- NVIDIA /models lists models the account cannot call (llama-3.1-nemotron-70b/51b, mistral and others returned 404). Every model that answered is a reasoning model.
- Works: openai/gpt-oss-20b (default; 1 attempt, ~2.8 s and 179 output tokens on the first test; 3.0-8.9 s on the 6 Oct refusals), nvidia/nemotron-3-nano-omni-30b-a3b-reasoning (ok, ~12 s), nvidia/nemotron-3-super-120b-a12b (good answers; cites with full-width brackets, handled by normalization).
- Avoid: nemotron-3-ultra-550b and nemotron-3.5-lightning (reasoning leaks into the answer, slow), glm-5.3-flash (empty, slow). kimi-k3, deepseek-v4.1-flash, glm-5.3 timed out.
- Re-probe with `python src\probe_llm.py <substrings>` before eval runs and record model id + date per run. Free-tier limit ~40 requests/minute.

## Eval runs
No eval dataset exists yet. Table to fill, one row per run, one change at a time.

| Run name | Index | Mode | Model | n | success@1 | success@3 | success@5 | MRR | gold_recall@5 | distinct sections@5 | retrieval p50 / p95 ms | Reranker harm / lift |
|---|---|---|---|---|---|---|---|---|---|---|---|---|

Planned analyses: reranker harm rate (gold in hybrid top 5 but not in hybrid_rerank top 5) and lift rate, split by whether the gold chunk is exposed to reranker truncation (XLM-R pair length > 512); closed-book baseline; faithfulness and answer correctness (LLM judge); refusal behaviour on unanswerable questions including near-misses.

Known limits to repeat in the README: hits are matched at section level via `covers` (a wrong chunk of the right section counts, so success@1 can be overstated); anchors are per section, not per parameter or glossary term; citation enforcement checks well-formedness, not faithfulness.

## Operational lessons
- Secret leak: the NVIDIA API key was committed in a template file named `env.example` (no leading dot; the real `.env` was never committed) and flagged by GitGuardian. Order that worked: revoke the key first, then clean the history, then prevent (secret scanning + push protection, `.env.example` with a placeholder that does not look like a key, `.gitignore` for `.env`). Checklist to tick: [ ] old key revoked, [ ] new key only in local `.env`, [ ] `git grep --cached -n "nvapi-"` prints nothing, [ ] repo history is one clean commit, [ ] push protection on.
- Qdrant down shows up as `WinError 10061` / "Failed to obtain server version": start Docker Desktop, `docker start qdrant`, check the collection has 6489 points.
- Check chunk budgets against every model in the pipeline (embedder and reranker tokenizers disagree on SQL-heavy text).

## Open items
- Fix the retrieve.py docstring: "sigmoid of the logit", not "logit".
- Record top-1 reranker scores for the World Cup and hosting probes.
- Start the eval (label 30+ questions, then run vector, bm25, hybrid, hybrid_rerank one at a time).
- Make generate() return a status instead of raising on an empty LLM answer (needed for batch runs).