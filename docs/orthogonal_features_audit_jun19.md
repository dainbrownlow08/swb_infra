# Audit: newest Tannen feature extractors (Jun 19, 2026)

Deep audit of the four newest feature extractors on branch
`audit-fixes-interactional-extractors`:

- `src/swb_extract/features/laughter.py`
- `src/swb_extract/features/topic_label.py`
- `src/swb_extract/features/personal_focus_score.py`
- `src/swb_extract/features/mutual_revelation_flag.py`

**Method:** read all four extractors + their tests (45 pass) + the supporting
`turn_gap`/`manifest` infrastructure + `scripts/build_tannen_features.py`; ran
the **actual 214,204-row output files** to measure real distributions; and
spot-checked the highest-risk extractor against transcripts.

---

## Trustworthiness scores (results derived from each extractor)

| Extractor | Score | Verdict |
|---|---|---|
| `topic_label` | **9/10** | Deterministic LDC table join; anchored by a verified mapping. Trust it. |
| `laughter` | **8/10** | Faithful count of on-disk annotations; output matches the survey exactly. Caveats are downstream, not code. |
| `personal_focus_score` (raw count columns) | **5/10** | Usable only as *pooled* aggregates; construct validity unvalidated. |
| `personal_focus_score` (per-utterance score) | **2/10** | Degenerate — do not use per-utterance. |
| `mutual_revelation_flag` | **3/10** | Loose operationalization; spot-checked precision ~30–40%. |

---

## `topic_label.py` — 9/10, trustworthy

Pure deterministic join: `conv_tab.csv` (call→ivi_no) → `topic_tab.csv`
(ivi_no→description). Test anchors call 2001 → `CLOTHING AND DRESS` / ivi 303,
and UNK handling (call 3178). It's a conversation-level covariate, correctly
documented as control/stratification, not a PCA input.

**Only risk:** hardcoded column indices (`CONV_COL_IVI=4`, etc.). If LDC ships a
different layout it silently shifts; the anchor test catches gross drift.
`_strip_csv_field` double-strips quotes after `csv.reader` already dequotes —
harmless but redundant.

## `laughter.py` — 8/10, trustworthy as an annotation count

Output **reproduces the docstring's survey exactly**: 12,402 laughter / 10,275
laughed-word / 8,381 noise / 4,009 vocalized-noise. That exact match is a strong
correctness signal. 7.5% of utterances carry any laughter; `other`=255
(negligible). Logic is simple, deterministic, well-tested.

What it measures is **annotated** laughter — the real caveats, *not code defects*:

- MS-State laughter/noise annotation has known inter-transcriber inconsistency.
  The number is a faithful count of a noisy label.
- Counts are **raw**, not rate-normalized. Any downstream use must divide by
  utterance count or duration, or it becomes a talkativeness proxy (same class
  of artifact as the backchannel/length issue in the PCA).
- **Not wired into the canonical table** — `build_tannen_features.py`'s
  `TANNEN_FEATURE_CSVS` lists only topic/personal-focus/mutual-revelation.
  Laughter is built but never merged.

## `personal_focus_score.py` — raw counts 5/10, per-utterance score 2/10

The redesign (raw counts always defined + gated ratio) is the right direction,
but the **measured output is still degenerate at utterance level**:

- **71.5% of score cells empty** (worse than the original 54.7% — the
  `MIN_HITS_FOR_SCORE=3` gate is stricter).
- **54.7% of utterances have zero topical hits at all.**
- Of the 28.5% that *are* scored, **48.6% are still exactly 0.0 or 1.0** —
  saturation persists despite the gate.

Deeper validity problem in the counting itself. The "hit count" is
**category-activations, not word matches**, and Empath's neural expansion makes
one word light up many categories:

- `compute_counts("i loved my family")` → **personal=11** for a 4-word utterance.
- `"war"` activates **both** `death` (PERSONAL) and `war` (IMPERSONAL) — a single
  word feeds numerator *and* denominator, so the personal/impersonal split is not
  a clean partition at the word level.

Consequences: `MIN_HITS_FOR_SCORE=3` is trivially cleared by one emotional
content word (so the gate filters less than it looks), and the ratio is driven by
which *clusters* a word triggers. The PERSONAL/IMPERSONAL/UNDECIDED assignment of
the 194 Empath categories is also a hand-made, **unvalidated** mapping
(defensible but contestable: `achievement`→personal, `money`→impersonal, etc.).

**Verdict:** trust the raw count columns *only* as ratio-of-sums pooled to
speaker/conversation/topic (as the docstring itself advises). Never trust the
per-utterance score column.

## `mutual_revelation_flag.py` — 3/10, the weakest

Infrastructure is clean (shares `turn_gap`'s chronological index, spaCy cached
per worker, only computes needed keys). The **operationalization is the
problem.** "Personal anecdote" = ≥5 tokens + a first-person pronoun + ≥1 VBD/VBN
tag. Flag = two adjacent different-speaker turns both passing that.

Spot-checked 12 of the 3,061 flagged pairs (1.43% rate) against transcripts. Real
precision looks like **~30–40%**. Clear false positives:

- *"...school out there in Lubbock"* / **"i enjoyed talking to you"** — closing
  pleasantry, flagged because it has first-person + past tense + 5 tokens.
- *"we've been talking for five minutes"* / **"i appreciate the call i enjoyed
  talking"** — meta-conversational closing, both sides.
- *"...they have gone metric"* / **"even they have switched... we're not on the
  cutting edge"** — impersonal topic (metric system), flagged on incidental
  past-participle + pronoun.

Root causes:

1. **Past-tense verb + first-person is a weak anecdote proxy** — catches
   opinions, closings, meta-talk, and impersonal-topic turns.
2. **No reciprocity/topical-relatedness check.** Tannen's construct is that one
   personal disclosure *elicits a similar one*; the flag only checks adjacency +
   both-personal.
3. **spaCy POS tagging on lowercased, unpunctuated, disfluent SWBD text** is
   materially less accurate than on the newswire it was trained on.
4. **Segmentation-fragile.** "Previous" is the immediate chronological utterance
   (incl. backchannels, incl. mid-turn fragments) — *not* a merged turn. A
   backchannel or an utterance split breaks or fabricates adjacency. It also
   inherits the un-FTO'd turn index (AUDIT §3 fix 1 not applied here).

---

## How to make them more reliable

**Cheap / high-value:**

1. **Normalize laughter** to a rate (per utterance/min per side) before any
   analysis; wire `laughter.csv` into `build_tannen_features.py`. Then validate
   annotated laughter coverage against NXT gold on the 642-conv subset.
2. **Drop the per-utterance personal-focus score from any model.** Emit and use
   only pooled ratio-of-sums at speaker/conversation/topic level. Add a
   min-analyzed-tokens floor per aggregate.
3. **Pin Empath category drift** — the loud-fail assert is good; add a frozen
   snapshot of each Tannen category's word list to the repo so silent lexicon
   updates can't move the construct under you.

**Validation (converts "computed" into "trustworthy"):**

4. **Hand-label precision/recall for `mutual_revelation_flag`** on a stratified
   ~100-pair sample — there is currently *zero* ground truth on its ~30–40%
   precision. Same for `personal_focus` against human topic-personalness ratings
   on a sample.
5. **Validate `personal_focus` category mapping** against NXT/gold or human
   ratings; report inter-category leakage (the `war`→death+war problem). Consider
   replacing Empath with a smaller audited lexicon or an LLM-judge
   topic-personalness score on aggregated turns.

**Redesign for `mutual_revelation_flag`** (if kept):

6. Operate on **merged turns excluding backchannels** (the FTO turn unit from
   AUDIT §3 fix 1), not raw utterances.
7. Tighten the anecdote test: require past-tense **+ a personal-topic signal**
   (reuse `personal_focus` PERSONAL hits ≥ threshold), excluding closing/meta
   phrases.
8. Add a **reciprocity gate** (lexical/topical overlap or embedding similarity
   between the two turns) so it matches Tannen's "elicits a similar statement,"
   not mere adjacency.

---

## Bottom line

`topic_label` and `laughter` produce trustworthy numbers today (with downstream
normalization for laughter). `personal_focus_score` is trustworthy only as a
pooled aggregate — never per-utterance. `mutual_revelation_flag` should be
treated as **unvalidated and probably low-precision** until a sample is
hand-labeled; do not put it in a published table yet.

Recommended first step: build the hand-labeling harness for mutual-revelation
precision — cheapest way to convert the riskiest extractor's trust score from
"guess" to "measured."
