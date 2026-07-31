# NB08 Measurement Plan — the measurement-first restructure (2026-07-29)

The execution plan for reorganizing the analysis around **measurement**: five empirical
combined-space PCA coordinates + one continuous theory-derived Tannen HI/HC score per caller,
with the type question demoted to a secondary test and the battery/multiverse mass demoted to
appendix citations. Companion to `docs/AUDIT.md` (the canonical audit + evidence ledger);
the completed submission-battery plan this superseded as "current phase" is archived at
`docs/archive/Audit_July_2026_Paper_Submission_Min.md`. Terminology: **"CPCA" = the combined trusted-space PCA** —
NB07 Step 24's analysis (26 features, 487 callers, Horn K=5), promoted from exploratory to
central.

Cell indices below refer to `analysis/07_final.ipynb` at HEAD `1e36206` (78 cells) — renamed
`analysis/07_demoted_in_favor_of_5D.ipynb` on 2026-07-29 with one close-out cell appended
(→ 79 cells; indices 0–77 unchanged); steps are stable names, indices shift on edit.

---

## 0. Decision summary

**D1 (recommended): build a new `analysis/08_measurement.ipynb`; freeze NB07 as the
audit/battery record.** Not an in-place restructure of 07, because:

1. NB07's recorded outputs + reconciliation trail *are* the evidence base — AUDIT.md cites
   "NB07 Step N" dozens of times, and reordering cells without a full re-run (~55 min) makes
   every recorded output lie about its position.
2. The new notebook is a different genre (measurement instrument, paper-shaped) than 07
   (audit-response log organized by T0–T15). Mixing them degrades both.
3. The demoted material becomes **compact cited summaries** of 07's recorded numbers instead
   of 40+ minutes of appendix compute — NB08 targets a ~15–25 min clean run vs 07's ~55.
4. Risk: moving ~78 cells of a 55-min-to-regenerate artifact via nbformat is the maximum-risk
   edit; writing a fresh notebook that *transplants* the needed cells is the same reuse with
   none of the corruption exposure.

NB07 gets exactly one edit: a closing reconciliation note (markdown append, no re-run) saying
it is closed as the battery record and superseded as the living notebook by NB08. _(Done
2026-07-29, together with the rename to `07_demoted_in_favor_of_5D.ipynb`.)_

**Honesty framing (fixed now, per the paper-chronology rules).** The construct map (S6) is
*theory-cited* — every direction anchored to Tannen 2005 Ch.7 via `docs/tannen_feature_map.md`
— and *fixed before the score is computed*. It is **not** pre-registered relative to the data:
Step 24's loadings were seen first (2026-07-22 commit). NB08 says exactly that, in those
words. Never "a priori constructs"; the defense is that directions come from the published
catalogue with page citations, not from the loadings.

---

## 1. NB07 inventory — one line per step, with disposition

| Step (cells) | One-line outcome | → NB08 |
|---|---|---|
| Header (0) | Frames 07 as the living audit notebook on the canonical guarded table | Rewrite: S1 opens with the measurement goal, no two-style framing |
| 1 (2–3) | Canonical table via `load_features_table`, sides→callers via `call_con_tab` (214,204 utts → 487 callers ≥20 substantive) | **Reuse verbatim** → S2 |
| 1b (4–5) | FTO reproduces the canonical turn-taking shape (median +0.140 s, 37.7% overlap) where `Turn Gap` gave −0.49 s | Transplant → Appendix A; S2 keeps one sentence |
| 2 (6–7) | Backchannel allowlist flag + `build_unit_table` caller aggregation machinery | **Reuse verbatim** → S2 |
| 3 (8–9) | vol11 volume PCA (+ the shared `run_pca` runner) | Machinery reused; vol11 PCA itself → one-paragraph "exploratory precursor" mention in S4 |
| 4 (10–11) | Horn's parallel analysis keeps K=2 on vol11 (PC1 42.5%) | Horn machinery reused in S4; K=2 result cited as precursor |
| 5 (12–13) | Both volume PCs unimodal: dip p .994/.993, single KDE mode, ΔBIC(2−1)=+11 | Instruments reused in S9; results cited → Appendix C |
| 6 (14–15) | Histograms + fitted normal: unimodal but skewed — dimensional, not normal | Same |
| 7 (16–17) | Retained PCs are not backchannel/length artifacts (r −.07/+.04; PC2 −.058/−.015) | Adapt → run on the 5 combined PCs + Tannen score (S8/App A) |
| 8 (18–19) | Welch+FDR demographics: gender robust both axes (d −.41/−.27); education demoted to suggestive (p_fdr .085); region/generation null | Adapt → S9b on score + 5 PCs |
| 8b (20–21) | The old education p=.014 was the n=6 unknown bucket under pooled variance (2×2 decomposition) | Transplant → Appendix F (data quality) |
| 9 (22–23) | Group-density overlay figure: shifts along the continuum, heavy overlap | Adapt (`overlay(tbl=…)`) → S9b |
| 11 (24–28) | Trust adjudication: 8 hard invariants at 0 violations (after re-extracting stale latching); registry 45 Trusted | Cite in S3 table; PF pooled-estimator lines transplanted into S2's panel-variable cell |
| 12 (29–33) | NXT gold: DA match 99.2%; allowlist backchannels P .842/R .917/F1 .878; Question Flag P .553 → excluded; gold q rate 7.83% | Cite in S3; **12a gold parse transplanted** for S8's external validation |
| 13 (34–36) | Repetition de-conflation: 24.5%/14.2% repair-attributable; self- vs allo-repetition r≈0 | Cite in S3 (caveat column) |
| 14 (37–40) | Coop/obstructive overlap split gold-validated (98.1%, 73.2%) → Trusted; 31.7% of 74,550 events obstructive | Cite in S3 |
| 15 (41–43) | DAMSL classifiers: backchannel F1 .888 admitted; question F1 .681 < .70 not admitted | Cite in S3 |
| 16 (44–46) | F_int identified (overlap +.84/+.97, FTO −.74); panel refit stable, PF/laughs/rising attach to engagement | Superseded by the combined space; cited as precursor; its 11-variable list feeds the 26-feature build |
| 17 (47–50) | Gold involvement axis from 6 human-annotated behaviors (n=1,085 sides): dimensional-with-skew (skew_t beats GMM2 by ΔBIC 18) | **17a/17b transplanted** for S8's external validation; battery rows cited → App C |
| 18 (51–54) | Formal battery, 6 axis×level cells: dip/Silverman unimodal 10/12; every BLRT k=2 resolved by fit-family as skew-fitting (incl. the §2.2 site, +36) | Battery functions transplanted for S9; results cited → App C |
| 19 (55–58) | Four-matrix clusterability: gap k̂=1, dist-dip unimodal everywhere; Hopkins elevation bounded to shape via copula null | Machinery reused in S9; results cited → App D |
| 20 (59–62) | Taxometrics: caller CCFI mean .151 → decisively dimensional (π-robust); gold indicators .460 ambiguous (indicator validity r .09) | Cite → App D |
| 21 (63–66) | Power: fit-family/CCFI detect the paper's mixture at 100% and false-alarm at 0%; dip underpowered (10%); BLRT's 76% skew-alarm absorbed by the pairing | Cite → App D |
| 22 (67–69) | Multiverse: 288/288 specifications unimodal PC1 | Cite → App E |
| 23 (70–71) | BATTERY single table + close-out verdicts | Cite → App C (table pointer) |
| 24 (72–74) | **Combined trusted-space PCA: 26 features, Horn K=5 (73.4% cum.), volume/pausing/loudness separate axes, pitch fuses with overlap on PC1; demographic overlays on all 5** | **Promoted: 24a → S4 core; 24b → S9b** |
| 10 + Conclusion (75–77) | Recorded summary + reconciled conclusion + submission close-out | Stays in NB07 (the record); NB08 writes its own S10 |

---

## 2. NB08 section-by-section build spec

### S1 — State the measurement goal *(new prose)*
Every speaker gets (a) five empirical combined-PCA coordinates describing observed
conversational behavior, (b) one continuous Tannen HI/HC score derived from those
coordinates. No two-style opener, no modality opener. One paragraph on data provenance (the
canonical guarded table) and one on the honesty framing from §0.
**Check:** none (prose).

### S2 — Unit of analysis *(reuse: cells 1, 3, 7; new: one compact panel-variable cell)*
Load + caller roll-up + `build_unit_table`/allowlist machinery verbatim. Add one new cell
deriving the caller-level panel variables the 26-feature space needs from Trusted table
columns: `PF_ratio` (pooled ≥30-hit estimator — lift the few lines from Step 11),
`rt_rising_share` (pooled defined-only share over **all** callers on the 2026-07-29
semitone-era flag, with `rt_defined_share` coverage carried as a companion diagnostic —
the ≥30-defined-tails subset (n=428) is an S8 robustness re-run, not eligibility; see
`analysis/validate_rising_terminal.py`), `laughs_per_100utt` (all utterances),
`obstructive_overlap_share`. Brief prose: caller/side resolution, FTO window, min-20 floor,
487/493 retained, missingness (pitch caller-means NaN-skip ~8% of utterances; 487/487 callers
complete on the 26). Diagnostics (FTO shape, Step 7 artifact check) → Appendix A.
**Check:** printed n == 487; panel-variable cell asserts value ranges (shares ∈ [0,1], no NaN).

### S3 — The trusted feature space *(new compact table; numbers cited from NB07)*
One table, one row per behavioral domain — Volume & rate · Turn timing (FTO) · Overlap &
latching · Pausing & repetition · Pitch & loudness · Involvement markers — listing features,
trust status (render programmatically from `docs/FEATURES.md` via
`swb_extract.registry`), gold validation evidence (allowlist P .842/R .917; overlap checks
98.1%/73.2%; repair fractions 24.5%/14.2%; DA match 99.2%), and caveats. A second short
**exclusions** table: Question Flag (P .553; classifier F1 .681 < bar), per-utterance
Personal Focus Score (deprecated, 71.5% null), `mutual_revelation_flag` (~30–40% precision),
`Turn Gap` (broken → FTO), Echo/MGQ (WIP). Full per-step verdicts → Appendix B pointers.
**Check:** registry render lists exactly the 26 features S4 consumes.

### S4 — The five-dimensional style space *(adapt cell 73 = Step 24a)*
The central analysis. Primary spec unchanged: caller / allowlist38 / no transform / min 20 /
trusted-26 (winsor+log1p robustness cited from the multiverse). Report standardization,
Horn's retention (**assert K=5** — recorded eigs [5.92, 5.40, 3.51, 2.64, 1.66] vs rand95;
the assert fires if a table rebuild shifts the geometry), variance per PC (22.7 / 20.7 /
13.5 / 10.1 / 6.4 = 73.4% cum.), the full 26×5 loading matrix, and caller scores.
_(2026-07-29: the rising-terminal **semitone redesign** re-extracted the flag and rebuilt
the table before W5 — deliberately, so S4 records its reference values once, on the fixed
measurement. Step 24's recorded eigs/loadings are therefore the **Hz-era precursor
reference**: W5 re-derives and records fresh values, keeps the K=5 expectation as the
assert, and reports the Hz-era vs semitone-era comparison — in particular whether PC1's
pitch–rising fusion survives the register-invariant detector.)_ Close with
one paragraph: the vol11 K=2 solution (Steps 3–6) as the exploratory precursor subspace.
Resolve 24a's upstream globals so the cell runs from S2's outputs alone.
**Check:** Horn assert; variance row sums match; loadings reproduce the recorded Step 24
values to rounding.

### S5 — Interpret the five components *(new exhibits)*
Per PC: loading bar plot, strongest ± features, one cautious behavioral sentence (no PC is
labeled "HI" or "HC"), score histogram. Draft labels to adjudicate from the recorded
loadings: PC1 pitch+overlap+rising vs pauses/repetition/fillers ("expressive-overlapping vs
hesitant"); PC2 length/repetition/loudness general-intensity; PC3 within-turn pausing vs
rate; PC4 loudness block; PC5 pitch+FTO vs overlap contrast. Representative caller profiles:
z-profile bars for callers at score percentiles {5, 25, 50, 75, 95} (chosen rule-based, not
cherry-picked).
**Check:** each plot's top-loading feature matches the printed loading table.

### S6 — The Tannen construct map *(new frozen cell + doc update)*
The full 26-row table, fixed **before** S7 executes, every direction citing
`docs/tannen_feature_map.md` (Ch.7 dims; PDF pages). Draft to adjudicate at the walkthrough
(⚑ = contested, decide at W7):

| Feature | Domain | Dir | Tannen anchor | In *t*? |
|---|---|---|---|---|
| loudness mean/std/range | Loudness | HI | dim 2a; marked amplitude shifts (p. 62, 202) | yes |
| pitch std / pitch range | Pitch | HI | dim 2b; marked pitch shifts (p. 62, 202) | yes ⚑ (Hz-scale sex confound → drop-pitch variant in S8) |
| pitch mean | Pitch | — | absolute F0 = anatomy/sex proxy; theory names *shifts* | **no** ⚑ |
| word_rate / syllable_rate | Rate | HI | dim 5c faster rate (p. 61, 202) | yes |
| FTO Sec | Turn timing | HC (+) | dim 3a/5b; avoiding interturn pauses = HI, so +FTO = spacing | yes |
| Latching Flag | Turn timing | HI | latching as canonical involvement (p. 119) | yes |
| Overlap Duration/Count/Onset | Overlap | HI | Ch.4 cooperative simultaneity (p. 113–122) | yes |
| obstructive_overlap_share | Overlap | — | construct-contested: Tannen's HI overlap is *cooperative*; no unambiguous direction | **no** ⚑ |
| Within Pause Total/Count/Rate/Max | Pausing | HC | dim 2c within-turn pauses | yes |
| Repetitions In Current | Repetition | HI | dim 6a/5d floor-getting repetition | yes ⚑ (24.5% repair-attributable, Step 13 caveat) |
| Repetitions In Previous | Repetition | HI | dim 6 incorporating other's offer (p. 117) | yes (14.2% caveat) |
| Filler Words per Second | Pausing | — | not a named Tannen feature; filled-pause vs discourse-marker readings conflict (map Part 3 §2) | **no** ⚑ |
| token_count | Volume | — | utterance-length covariate, no clean Ch.7 direction; artifact-check history | **no** ⚑ |
| Pronouns per Second | Involvement | — | weak dim-1 proxy; PF_ratio is the sanctioned dim-1 measure | **no** ⚑ |
| PF_ratio | Involvement | HI | dim 1 personal focus of topic | yes |
| laughs_per_100utt | Involvement | HI | dim 9 laughter | yes |
| rt_rising_share | Involvement | HI | expressive/questioning intonation (dims 3d/4) | yes ⚑ (MNAR caveat; weakest anchor — adjudicate) |

Draft tally: 21 in / 5 out. Also refresh `docs/tannen_feature_map.md` Part 2's stale status
column (overlap/laughter/pauses now built) while we're in the file.
**Check:** the map cell hard-codes the frozen table; S7 reads directions from it, nowhere
else.

### S7 — Project Tannen into the space *(new; the core new computation)*
Let Z (487×26) be the standardized matrix, L (26×5) the unit-norm loading matrix, S = ZL the
scores, t ∈ ℝ²⁶ the map's ±1/0 vector (primary: unit per-feature weights; ⚑ domain-balanced
1/n_domain weights as an S8 sensitivity, since 4 pause features vs 1 latching implicitly
weight domains).

- **w = Lᵀt** — the theory direction in PC coordinates; report w and the captured share
  ‖Lᵀt‖²/‖t‖² (how much of the theory vector lies inside the retained subspace — small would
  itself be a reportable finding).
- **Score** uᵢ = Sᵢ·w = zᵢ(LLᵀ)t, standardized over callers; orientation check
  corr(score, Overlap Count) > 0, else flip and record. Percentile = rank/(n+1).
- **Contributions:** per-component cᵢₖ = Sᵢₖwₖ; per-feature fᵢⱼ = zᵢⱼ(LLᵀt)ⱼ; both sum to uᵢ.
- **Key property (assert in a check cell):** the score is invariant to any within-subspace
  rotation of L (LLᵀ is a projector) — verify numerically with a random 5×5 orthogonal Q.
  This is the answer to "PC axes are arbitrary": the score depends on the *subspace*, not the
  basis, so K=5 is the only geometry choice it inherits.
- **Comparison estimator for S8:** the raw theory contrast rᵢ = zᵢ·t/‖t‖ (no PCA
  restriction); corr(u, r) quantifies what the subspace restriction does.
- **Export** (paths ⚑ D8, after checking `docs/PIPELINE.md` conventions):
  `utterances_v2/derived/caller_style_scores.csv` — caller_no, n_utt, PC1–5, tannen_score,
  tannen_pct, c₁–c₅, f₁–f₂₆ — plus `caller_style_loadings.csv` (26×5 L, t, w). Analysis
  *outputs*, not feature columns; no FEATURES.md registration, note in PIPELINE.md.

**Check:** the rotation-invariance assert + score computed two ways (S·w vs Z·(LLᵀt)) agree
to 1e-10.

### S8 — Validate the score *(new)*
1. **Directional agreement:** sign(corr(score, feature)) vs map direction for all included
   features; report k/21 and diagnose disagreements (word_rate loads −.17 on PC1, so rate is
   the likely dissident — the theory direction is *not* PC1, which is the point).
2. **Stability:** caller bootstrap B=500 — refit PCA (K fixed at 5), recompute w and scores,
   correlate with primary → mean r + CI. No Procrustes needed (rotation invariance).
3. **Map variants:** drop-pitch, domain-balanced weights, each ⚑ row flipped/included,
   raw-contrast r, plus the rt coverage-robustness variant (score recomputed on the
   ≥30-defined-tails callers, n=428) — a small correlation matrix of score variants.
4. **Leverage:** leave-one-feature-out score deltas (max |Δr|); contribution-share
   distribution (no single feature should dominate).
5. **External:** the gold involvement axis (transplanted 12a + 17a/17b chain, minus battery
   rows): corr(caller score, caller-mean gold-axis score, n=320) **and** against the six raw
   gold behavior rates — the clean external part, since rt/PF/laughs also sit inside the
   score (overlap stated).
6. **Incremental over PC1:** w's component weights; corr(gold behaviors, score) vs vs
   combined-PC1 alone vs vol-PC1; ΔR². The comparison is "theory-guided 5-D projection vs
   PC1," not "one cluster vs two."
7. *(⚑ D9, recommend defer)* split-half/ICC across a caller's ~9 sides — §4B10's next-paper
   reliability line; out of scope here.

**Check:** each subsection prints one verdict line; the bootstrap cell asserts B and seed.

### S9 — The secondary type question *(machinery transplanted from Steps 5/18/19)*
Only now. On the **score**: dip, KDE modes (Step-5 convention), Silverman B=999, BLRT B=999,
fit-family BIC (seeds 0, B printed — battery conventions). On the **26-feature space** (⚑ D7:
the actual space; the 5-D score matrix is its linear image): dist-dip, Hopkins under the
copula null, gap k=1..6; optional k=2 bootstrap-ARI stability (cheap; gives the summary its
"no stable two-group partition" sentence directly). Three-sentence summary: heavily
overlapping continuum; no two modes in the score; no stable 2-partition. Everything else
cited → Appendices C–E. This section supports the metric; it does not dominate.
**Check:** verdict lines in the recorded battery format.

### S9b — Demographics as secondary description *(adapt cells 19/23/74)*
After scoring, per the demote table: Welch + BH-FDR over the family 4 demographics × (5 PCs +
score) = 24 tests (family stated in advance), `overlay(tbl=…)` figures for the score and any
significant PC, unknown-education exclusion carried. Interpretation caveat printed: gender on
the score partially rides the Hz-scale pitch block — read alongside S8's drop-pitch variant.
Education artifact detail → Appendix F.
**Check:** FDR family size printed; Step 24b overlays reproduce.

### S10 — Close with the measurement claim *(new prose)*
What was measured, the instrument delivered (coordinates + score + exports), scope and
limitations (inherited from the close-out's shrunken list), and one paragraph on where the
type evidence lives (S9 + appendices + NB07). No rebuttal framing.

### Appendices *(compact; cited numbers, minimal compute)*
- **A** Sample/preprocessing diagnostics: FTO validity (transplant 1b), artifact check on the
  new PCs (adapt Step 7) — artifact columns now include **per-caller RT defined-rate**
  (ascertainment coverage) alongside backchannel share and length: corr(PC_k, coverage) ≈ 0
  is the gate for the rising-terminal MNAR channel (2026-07-29 redesign).
- **B** Feature-validation record: per-step verdict paragraphs citing NB07 Steps 11–15.
- **C** Continuity/battery record: Step 23 table citation + Steps 5/6/17/18 headlines.
- **D** Clusterability/taxometrics/power: Steps 19/20/21 headlines (CCFI .151; 100%/0%).
- **E** Multiverse: 288/288 + spec-curve pointer.
- **F** Education data quality: transplant 8b (cheap, self-contained).

---

## 3. Mechanics, dependencies, runtime

- **Transplant discipline (memory rules):** back up before every nbformat batch; JSON-diff
  against the backup after; smoke-test each new late cell with stubbed globals before the
  real run; background `nbconvert`, never piped.
- **Dependency resolution at W5:** cell 73 (24a) consumes globals built across Steps 11/16
  (panel variables, F_int variable list). S2's new panel-variable cell replaces them; trace
  73's free variables and satisfy each from S2/S4 only.
- **Gold chain (W9):** 12a parse (cell 30) → 17a panel (48) → 17b axis (49), battery rows
  stripped. Self-contained NXT parse ~3–5 min; no NB07 re-run needed anywhere in this plan.
- **Battery functions (W10):** locate the dip/Silverman/BLRT/fit-family definitions in the
  Step 17–18 cells and transplant the defs only.
- **Runtime budget:** load+rollup ~1 min · PCA+Horn seconds · projection instant · bootstrap
  1–2 min · gold chain 3–5 min · score battery 3–8 min · 26-feature clusterability 2–4 min ·
  demographics seconds → **~15–25 min** clean run.

## 4. Bookkeeping (finish-line tasks)

1. ✅ NB07 closing note (done 2026-07-29, with the rename).
2. `docs/AUDIT.md`: preamble pointer to this plan (done with this commit); at completion —
   living-notebook designation moves to NB08; §4C11 gets a note (the theory projection
   *addresses* the "is there a single HI–HC axis" worry by construction — we measure a theory
   contrast inside a 5-D space rather than asserting unidimensionality; CFA proper stays ⬜);
   dashboard recount.
3. `docs/PIPELINE.md`: note the two exported score CSVs as analysis outputs.
4. `docs/tannen_feature_map.md`: Part 2 status refresh (with S6).
5. Memory: update the project memory with the NB08 phase.

## 5. Walkthrough order

| W | Builds | Sections | Ends with |
|---|---|---|---|
| W1 ✅ | D1/D2 decided 2026-07-29; D3–D9 adjudicated at their walkthroughs | — | this doc updated |
| W2 ✅ | NB08 skeleton: headers + S1 prose (2026-07-29) | S1 | nbconvert smoke pass green |
| W3 | Machinery transplant + run | S2 | n=487 printed |
| W4 | Trusted-space + exclusions tables | S3 | registry render = 26 |
| W5 | Combined PCA, deps resolved | S4 | Horn K=5 assert green |
| W6 | Component exhibits | S5 | plots match loadings |
| W7 | Construct map frozen (⚑ rows adjudicated) | S6 | map cell + doc updated |
| W8 | Projection + score + exports | S7 | invariance assert green |
| W9 | Validations incl. gold chain | S8 | verdict lines |
| W10 | Type question on score/space | S9 | battery-format verdicts |
| W11 | Demographics | S9b | FDR family printed |
| W12 | Close + appendices | S10, A–F | full clean run, timed |
| W13 | Bookkeeping | — | AUDIT/PIPELINE/map/memory updated |

## 6. Open decisions

- **D1** ✅ decided 2026-07-29: new NB08; NB07 frozen — renamed
  `analysis/07_demoted_in_favor_of_5D.ipynb`, close-out cell appended (its one edit).
- **D2** ✅ decided 2026-07-29: `analysis/08_measurement.ipynb` (skeleton created, smoke-passed).
- **D3** Construct-map ⚑ rows (pitch mean, obstructive share, fillers, token_count,
  pronouns out; rt_rising in; repetition caveats) — adjudicate at W7.
- **D4** Theory-vector weighting — unit per-feature primary, domain-balanced as sensitivity.
- **D5** Demographics placement (S9b after the type question) + the 24-test FDR family.
- **D6** External validation via transplanted gold chain (required — score×gold is a new
  number, not citable from NB07).
- **D7** S9 multivariate matrix = the 26-feature space; include the k=2 bootstrap-ARI.
- **D8** Export paths/names (after PIPELINE.md check).
- **D9** ICC/split-half — defer to the §4B10 reliability line (next paper).
