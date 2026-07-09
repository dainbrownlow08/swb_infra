# Audit — July 2026 Paper-Submission Minimum

> **Execution layer over `docs/AUDIT.md`.** Read its **Pipeline Plan P0–P1** and **Continuum
> Validation Plan Q0–Q5** first — full specs live there and are not repeated; this file holds
> only scope, deltas, task order, and gates. Where sequencing differs, this file governs.
> Written 2026-07-08. Rev 2026-07-09a: involvement panel + tiered question validation.
> **Rev 2026-07-09b: the NXT gold suite folded in (Delta 7) — and, per Dain's directive,
> time/development cost is no longer a scoping criterion; scope is set by critique-removal
> only.** Steps/tasks renumbered again; this revision supersedes 09a's numbering and its
> shed-order (obsolete). Rev 2026-07-09c: **§4A6 taxometrics reinstated** (Dain) — §4A now in
> scope in full; taxometrics = T12 / NB07 Step 20; downstream steps/tasks shifted +1.
>
> **Binding rules:** every analysis lands in **NB07** (`analysis/07_final.ipynb`) as appended
> numbered Steps per P1's convention (markdown header + code cells ending in one printed
> verdict line, inserted before the Conclusion; Conclusion reconciled at every landing; fixed
> seeds; B and runtime printed). Non-analysis code outside NB07: `src/swb_extract/
> stats_modality.py`, `src/swb_extract/taxometrics.py`, `src/swb_extract/nxt.py`,
> `src/swb_extract/features/overlap_split.py` (+ tests; the extractor follows
> `docs/PIPELINE.md`). Q5's do-not list and the integrity
> clause apply verbatim: decision rules and admission bars are fixed in advance; a discordant
> result is recorded and investigated in place, never tuned away.

## Scope

**In:** §4A **in full** — A1–A8, incl. A6 taxometrics (reinstated 7/9c) · pooled personal focus (Pipeline Step-11 analysis
half) · optional RT mini-audit · **the NXT gold suite (Delta 7)**: gold backchannel validation
of the 38-token allowlist (§2.8, §4C12a), gold question validation **and** classifier
(§4C12b, §3-minor), gold-only involvement behaviors panel + gold axis, disfluency
de-conflation of repetition (§4C12c), cooperative-vs-obstructive overlap split (§4E-a),
DAMSL-trained bc classifier corpus-wide (§4C12a). Env verified: diptest 0.11.0, sklearn 1.8.0,
scipy 1.17.1 (`jf_skew_t`) — sklearn suffices for the classifiers; zero new dependencies.

**Out** (decision record):

- **Pipeline Plan C2/C3** (frozen-builder banners, loader dtype hygiene): repo hygiene, not
  battery-load-bearing.
- **§4B/§4C11/§4D** (ICC, CFA, mixed models, accommodation, mismatch→quality): the
  positive-story program, not the modality claim.
- **Not rescued by gold** (exclusions with citable reasons): `mutual_revelation_flag` — no
  disclosure labels exist; measured ~30–40% spot-check precision (Jun-19 audit) is the stated
  exclusion reason. Personal-focus **construct validity** — no personalness labels; stays
  Empath-pooled with the §4C12 caveat. Voice quality (§4E-d) — no annotation. Prosody layers
  (`accent`/`breaks`, 150-file subset) → post-submission validation of rising-terminal/marked
  shifts. `topic_label` (9/10) still unused (§4D16 out of scope).

## Deltas vs the AUDIT.md plans (this file wins where they differ)

1. **NB07 step numbers** (rev 7/9b): trust evidence → **11** · gold alignment + validation →
   **12** · repetition de-conflation → **13** · overlap-split validity → **14** · classifiers →
   **15** · F_int + involvement panel → **16** · gold involvement panel + gold axis → **17** ·
   modality battery → **18** · clusterability → **19** · taxometrics → **20** · power → **21**
   · multiverse → **22** · summary → **23**. Later Pipeline-Plan steps take 24+.
2. **F_int primary set unchanged** (identity preservation for the NB05-F3 match): `FTO Sec`
   (in-window), `Overlap Duration Sec`, `Overlap Count`, `Overlap Onset Flag`, `Latching
   Flag`, Within Pause ×4, `Repetitions In Previous Utterance`, laughter rate (laughs per 100
   utterances over **all** utterances, inline). New involvement variables (panel, coop/obstr
   split, gold behaviors) enter via the panel refit, the gold panel, and the battery's added
   axes — never by silently redefining F_int. Battery F_int rows ride the primary.
3. **Trust adjudication = battery + panel inputs** (supersedes Q0's "no FEATURES.md changes"):
   C1 Tier 1 (11 live features; SUSPECT repetitions resolved as C(n,2)-by-design + boundary
   unit test) + Tier 2 (FTO helpers) + Tier-4 subset (`Latching Flag`, Overlap ×3, Within
   Pause ×4) + Tier-3 laughter mini-check + PF pooling evidence (A1 acceptance: pooled null%
   ≪ 71.5%, no 0/1 spike, min-hits sensitivity {10,30,100}, ~10-utt Empath spot-check;
   promote the hit columns, `Personal Focus Score` → **Deprecated**) + RT mini-audit if
   `rt_rising_share` kept. **New columns now do enter the registry**: the two overlap-split
   columns (Delta 7d) are added as WIP and adjudicated by Step 14.
4. **Bounded reruns:** utterance-level battery rows at **B=99** (pre-authorized, printed note).
5. **Involvement panel** (caller-level, corpus-wide): `PF_ratio` (pooled, ≥30 hits, sensitivity
   printed) · `laughs_per_100utt` · `question_rate` — best validated corpus-wide variant per
   Steps 12/15 verdicts (classifier if it clears its bar, else heuristic-with-measured-P/R,
   else excluded with gold quoted descriptively) · `obstructive_overlap_share` (Delta 7d, after
   Step-14 adjudication) · optional `rt_rising_share`. Slots: Step 16 sensitivity FA refit ·
   Step 19 third matrix vol+int+panel · Step 22 feature_set arm. Claim-scoping language
   regardless of outcomes: rebuttal unconditional; positive claim scoped to measured domains;
   limitations name what remains unmeasured (voice quality, validated disclosure, narrative
   beyond quotation) — the pre-gold-suite limitation list shrinks accordingly.
6. **Question validation, all tiers authorized** (time gate removed): Tier 1 inventory + gold
   rate; Tier 2 alignment P/R + miss decomposition (by gold tag AND by leading-discourse-marker
   counterfactual) + (`Question Flag` OR `Rising Terminal Flag`) recall; Tier 3 marker-prefix
   extractor fix if Tier 2 shows fixable structure (re-extract, re-validate; never tune blind);
   Tier 4 classifier = Step 15. Citations fixed: Jurafsky, Shriberg & Biasca 1997 (SWBD-DAMSL
   manual, TR 97-02); Stolcke et al. 2000 (*Comp. Linguistics* 26(3)); Calhoun et al. 2010
   (*LREC J.* 44(4) — the on-disk NXT resource); Shriberg et al. 1998 (*Lang. & Speech*
   41(3–4) — prosody marks declarative questions; the RT bridge).
7. **The NXT gold suite** (added 7/9b; inventory verified on disk 2026-07-09 — all layers
   full-coverage: `dialAct`/`disfluency`/`terminals`/`turns`/`syntax`/`phonwords` = 1,284
   sides, 642 conversations). One parser (`nxt.py`) feeds everything; each item below is a tag
   filter + groupby on top of it. **Tag membership is computed by parsing base tag +
   decorations** (`qy^g^t`, `sd(^q)^t`, `ba,fe` occur) — never exact-string match; the full
   inventory is printed (Step 12) and every membership decision recorded **before** any P/R or
   rate ships (pre-registration discipline).
   - **7a — gold backchannels** (`b` 18,335; decide {`b`} vs {`b`,`bh`,`bk`} membership from
     the coders manual before computing): allowlist-38 precision/recall vs gold; gold-bc and
     classifier-bc join the multiverse `bc_def` axis. *Removes the strongest remaining attack
     on the rebuttal: "your central correction rides on a hand-made allowlist" (§2.8).*
   - **7b — gold questions** (q-family incl. `qy^d` 669 declarative, `qh` 303 rhetorical,
     `qy^g`/`^g` tag): validation per Delta 6; gold rate is the subset panel variable.
   - **7c — gold-only involvement behaviors** (no heuristic equivalents exist): collaborative
     completion `^2` (324 + suffix forms) · mirror/allo-repetition `b^m` + `^m` decorations
     (≈600) · echo-question backchannel `bh` (632) · appreciations `ba` (2,488) · quotation /
     constructed dialogue `^q` + `(^q)` (≈1,400 — Tannen's signature involvement strategy;
     covers the narrative dimension §4E-g) · gold bc rate (listener activity; sign-ambiguity
     noted, irrelevant to modality tests). → **Step 17 gold involvement panel**: side-level
     (n=1,284) rates per 100 utterances + subset-restricted corpus features (FTO, overlap,
     coop/obstr share, laughter) → Horn's + varimax FA → **gold involvement axis** → battery
     rows (dip/Silverman/BLRT/fit-family) at side level, caller-level sensitivity (callers
     with ≥1 gold side; n printed). *Removes "you measured the acoustic-temporal shell, not
     involvement": the axis is built from human-annotated involvement behaviors.*
   - **7d — cooperative vs obstructive overlap** (§4E-a; the one new extractor,
     `features/overlap_split.py` + tests + 2 WIP registry rows + `swb-extract table` rebuild):
     on merged FTO turns, an overlap is **obstructive** if the overlapped speaker's turn
     terminates within W s of overlap onset and the overlapper holds the floor; **cooperative**
     otherwise; backchannel-only overlaps cooperative by definition; W fixed in advance
     (1.0 s, recorded). Columns: `Cooperative Overlap Count`, `Obstructive Overlap Count`.
     **Step 14 gold validity check**: overlapping gold `b` events → expected ~all cooperative;
     `+` continuations (9,401) → floor-retention agreement rate; `+`-across-interjection also
     doubles as a free check of `fto.py`'s turn-merging. *Removes "you can't tell supportive
     overlap from interruption" — the most Tannen-diagnostic missing distinction.*
   - **7e — disfluency de-conflation of repetition** (`disfluency/` full coverage): fraction
     of `Repetitions In Current/Previous Utterance` counts falling inside gold repair spans;
     repairs-excluded repetition variant on the subset; correlate with mirror rate (7c) —
     rhetorical vs disfluent repetition separated. *Removes "your repetition feature is mostly
     stuttering" (§4C12c) — it hits two of vol11's eleven features.*
   - **7f — DAMSL-trained classifiers corpus-wide** (§4C12a/b): backchannel + question
     classifiers (sklearn; lexical + timing features; grouped CV by conversation — no
     leakage; seeded; CV P/R/F1 printed). **Admission bars fixed now:** bc classifier joins
     `bc_def` if CV F1 ≥ 0.85; question classifier becomes the panel's `question_rate` if CV
     F1 ≥ 0.70. In-notebook (analysis-side definitions, like the allowlist) — no new table
     columns. *Removes "your gold analyses cover only 26% of the corpus."*

## Tasks — strict order

A task is **done** when its gate passes, FEATURES.md/AUDIT.md reflect it (project `CLAUDE.md`
rule), and a checkpoint commit is proposed to Dain.

- [x] **T0 — checkpoint commit.** _Done 2026-07-09, `fa7b635`._ The 6/29 reorg is still uncommitted; commit everything on
  `audit-fixes-interactional-extractors` (incl. this file). _Gate:_ clean `git status`.
- [x] **T1 — `stats_modality.py` + tests**, Q1 spec verbatim. _Gate:_ pytest green
  (planted structure detected, planted null passed). _Done 2026-07-09: 15 known-answer tests,
  3.5 s (`python3 -m pytest tests/test_stats_modality.py`)._
- [x] **T2 — Phase-0 refactor** (Q2): `build_unit_table` + `run_pca` in-notebook. _Gate:_
  identical headline numbers (487 callers, PC1 42.5%, K=2, dip p .994/.993, ΔBIC(2−1)=+11).
  _Done 2026-07-09: full rerun, 22/22 printed-output checks identical (incl. loadings + FDR
  table); Conclusion carries the reconciliation block + the P1 step convention._
- [ ] **T3 — NB07 Step 11: trust evidence** per Delta 3 + capstone self-check in Step 10.
  _Gate:_ checks pass; registry rows moved; counts/date bumped.
- [ ] **T4 — `nxt.py` + `tests/test_nxt.py`** (truncated-real-file fixtures; parses dialAct +
  terminals + disfluency) **+ NB07 Step 12: gold alignment & validation** — inventory print,
  ≥50%-overlap DA→utterance matching (match rate printed, expect >90%), 7a allowlist P/R,
  7b question P/R per Delta 6 Tiers 1–3. _Gate:_ verdict lines = gold-bc membership decision,
  allowlist P/R, question admit/exclude; FEATURES.md Question/Echo notes carry measured P/R.
- [ ] **T5 — NB07 Step 13: repetition de-conflation** (7e). _Gate:_ repair-overlap fraction +
  de-conflated variant + mirror correlation printed; repetition FEATURES.md notes updated.
- [ ] **T6 — `features/overlap_split.py` + tests + registry rows + `swb-extract table` rebuild
  + NB07 Step 14: overlap-split validity** (7d gold checks). _Gate:_ full-notebook rerun clean
  under the stale guard with headline numbers unmoved (new columns don't touch vol11); gold
  agreement rates printed; the two columns adjudicated.
- [ ] **T7 — NB07 Step 15: classifiers** (7f). _Gate:_ grouped-CV P/R/F1 printed; admission
  bars applied as fixed; corpus-wide predictions materialized in-notebook.
- [ ] **T8 — NB07 Step 16: F_int (primary, Delta 2) + panel sensitivity refit (Delta 5).**
  _Gate:_ both loading tables + variance shares; "no matching factor" recorded as a finding.
- [ ] **T9 — NB07 Step 17: gold involvement panel + gold axis** (7c). _Gate:_ FA loadings,
  axis identified, battery rows appended at side level (+ caller sensitivity, n printed).
- [ ] **T10 — NB07 Step 18: the modality battery** per Q2 Step 16's matrix — {PC1, PC2
  caller · PC1 side · PC1 utterance (10-feature + turn-initial sensitivity) · F_int caller} ×
  {dip · Silverman · BLRT · fit-family (not at utterance level)}; shared BATTERY list; no FDR
  (state why). **Side-level fit-family = the ΔBIC −110 closer — verdict recorded in
  "formally explained (or not) as skew-fitting" terms.** _Gate:_ every cell has a row; B +
  runtimes printed.
- [ ] **T11 — NB07 Step 19: multivariate clusterability** on **four** matrices — vol11,
  vol+interactional, vol+int+panel (caller) and the gold panel (side): dip-on-distances,
  Hopkins vs N(0, Σ̂), gap k=1..6. _Gate:_ verdict line per matrix; BATTERY rows.
- [ ] **T12 — `src/swb_extract/taxometrics.py` + `tests/test_taxometrics.py` + NB07 Step 20:
  taxometrics** (§4A6, reinstated 7/9c). Module per Q1 spec verbatim: `gen_data` (Ruscio &
  Kaczetow 2008 comparison-data generator), `mambac`, `maxeig`, `lmode`, `ccfi`; end-to-end
  known-answer tests (dimensional-generated → CCFI < 0.45; taxonic at the paper's parameters →
  CCFI > 0.55; seeded). Step 20 per Q2 Step 18: 4–6 caller-level indicators by |PC1 loading|
  (sign-aligned, |r| > 0.85 dedup, correlation matrix + validity note printed);
  MAMBAC/MAXEIG/L-Mode; CCFI per procedure + mean vs B=100 dimensional + B=100 taxonic
  comparison sets; base-rate sensitivity π ∈ {0.30, 0.45, 0.55} + the GMM(2)-weight estimate;
  the observed-curve-over-envelopes figure (the publishable exhibit); optional RTaxometrics
  CSV export (non-blocking). **Plus the gold-panel variant:** the same battery on side-level
  gold involvement indicators (n=1,284) — taxometrics on human-annotated involvement
  behaviors, the novel exhibit. _Gate:_ pytest green incl. known-answer tests; CCFI verdicts
  per the Ruscio bands recorded per procedure, both indicator sets.
- [ ] **T13 — NB07 Step 21: recovery/power** (Q2 Step 19: Arm T at n=487 and n=4,876, B=200;
  Arm D false alarms; n=1,284 covered by interpolation in prose; **taxometric sub-arm
  reinstated** — taxonic `gen_data` 5-indicator sets at the paper's base rate → CCFI on a
  B=25 subset, runtime printed). _Gate:_ the "we had the power ≈X / false-alarm ≈Y" sentence,
  incl. CCFI detection and false-alarm rates.
- [ ] **T14 — NB07 Step 22: multiverse** — grid now `bc_def` ∈ {allowlist38, token≤2, union,
  none, classifier (if admitted), gold-subset} × transform 2 × unit 2 × min_utt 3 ×
  feature_set {vol11, vol11+pitch3, vol+interactional, vol+int+panel} = **up to 288 specs**
  (gold-subset arms carry reduced n — printed per spec). _Gate:_ specification-curve figure;
  "X of N specs unimodal" headline; non-unimodal specs enumerated and diagnosed.
- [ ] **T15 — NB07 Step 23 + close-out:** BATTERY summary table (A1's ask, now incl. the gold
  axis); Conclusion: formal-battery block (**dimensional, not normal**), the taxometric CCFI
  verdicts (both indicator sets), claim-scoping + shrunken limitations paragraph, gold-suite
  numbers (allowlist P/R, question P/R, classifier CVs, coop/obstr validity) with the Delta-6
  citations. Bookkeeping: §4A1–8, §3.3, §3-minor, §4C12a/b/c, §4E-a status lines;
  §2.1/§2.2/§2.8/§5.2 cross-refs; dashboard recount; Verdict addendum; final top-to-bottom
  execution. _Gate:_ Q4's acceptance checklist **in full** (A6 line included), plus every
  Delta-7 verdict recorded.

## Sequencing notes (dependency order, not time)

T0 → T1/T2 (independent) → T3 → T4 → {T5, T6, T7 in any order — all consume T4's alignment} →
T8 (needs T3 PF + T6 columns + T7 question verdict) → T9 (needs T4/T6/T7) → T10–T15 in order
(T12's Step 20 consumes T2's caller table + T9's gold panel; its module + tests, like T1's,
have no data dependency and can be built at any time).
The only table rebuild on the path is T6's; the only re-extraction is Tier 3's
`question_flags` (transcript-based) — still **no audio re-extraction anywhere**. Utterance-level
bootstraps stay at B=99 so every landing's full rerun remains cheap.
