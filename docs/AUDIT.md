# Audit — canonical v2

> **Living document, compressed 2026-07-31.** The full first-generation audit (764 lines,
> including the two completed execution plans) is archived verbatim at
> `docs/archive/AUDIT_2026-06_v1_full.md`; this v2 keeps the same **§1–§5 numbering** so
> code-docstring citations ("AUDIT.md §3 fix 1", "§4A6", "§4C12", "§4E-a") still resolve.
> **Legend:** ✅ done (implemented *and in use*) · 🟡 partial (built but not wired in, or done
> at one level not the rigorous one) · ⬜ not started. The ✅-vs-🟡 line is the discipline that
> matters: most work stalls at "built ≠ in use."
>
> **Current phase:** NB08 measurement (`docs/NB08_MEASUREMENT_PLAN.md`, W3+ pending).
> **Repo focus (2026-07-31 compression):** `analysis/08_measurement.ipynb` is the living
> notebook; `analysis/07_demoted_in_favor_of_5D.ipynb` is its frozen evidence record; NB00–06
> live in `analysis/archive/`. Planned later step: once NB08 stands alone, NB07 also moves to
> the archive.

## Verdict

The paper's "two styles" result is unrecoverable (§1). The corrected re-analysis supports a
**unimodal, dimensional** structure — never claim "normal" — and that claim is now formally
supported at every level: dip + Silverman + BLRT + fit-family at caller/side/utterance, the
side-level ΔBIC −110 formally explained as skew-fitting (skewnorm beats GMM(2) by 36),
four-matrix clusterability at gap k̂ = 1, taxometrics at CCFI .151 (decisively dimensional,
π-robust), recovery power 100% detection / 0% false alarm on the deciding pair, 288/288
multiverse specifications unimodal, and an NXT-gold suite measuring every heuristic the
analysis rides. The current work (NB08) turns this from a defended null into a measurement:
five combined-PCA style coordinates + one theory-derived Tannen HI/HC score per caller. What
remains for "airtight" is the positive-story program (§4B10 reliability, §4C11 CFA, §4D) and
external replication (§4F) — the next paper.

## Evidence ledger — the NB07 record

`analysis/07_demoted_in_favor_of_5D.ipynb` is frozen (closed 2026-07-29, never re-run,
~55 min); its recorded outputs are the citable evidence. One line per landing:

| NB07 Step | Result |
|---|---|
| 1–4 | Canonical table via guarded loader; 214,204 utts → 487 callers (≥20 substantive); vol11 PCA, Horn K=2, PC1 42.5% — reproduces NB06 to rounding |
| 1b | FTO distribution canonical: median +0.140 s, 37.7% overlap (Heldner & Edlund shape) |
| 5–6 | Volume PCs unimodal: dip p .994/.993, ΔBIC(2−1) = +11; skewed, not normal |
| 7 | PCs are not backchannel/length artifacts (r ≤ .07) |
| 8/8b | Welch + BH-FDR demographics: gender robust (d −.41/−.27); education demoted to suggestive — the old p=.014 was the n=6 unknown-education bucket under pooled variance |
| 11 | Trust adjudication: 8 hard invariants at 0 violations corpus-wide (caught + fixed a stale latching vintage, 1,333 rows); pooled personal-focus sanctioned; RT missingness audited (30.3% MNAR, length-dependent) |
| 12 | NXT gold: DA match 99.2% (52,890 labelled utts); backchannel allowlist P .842 / R .917 / F1 .878; Question Flag P .553 / R .236 → **excluded**; gold q-rate 7.83% |
| 13 | Repetition de-conflation: 24.5% (current) / 14.2% (previous) repair-attributable; self- vs allo-repetition r ≈ 0 → involvement repetition = gold `^m` mirror |
| 14 | Cooperative/obstructive overlap split gold-validated (gold-`b` events 98.1% cooperative; `+`-continuation floor retention 73.2%) → Trusted; 31.7% of 74,550 events obstructive |
| 15 | Classifiers: backchannel CV F1 .888 admitted (multiverse bc_def axis); question F1 .681 < .70 bar → not admitted |
| 16 | F_int identified (overlap +.84/+.97, FTO −.74); PF_ratio, laughs, rising share load on engagement |
| 17 | Gold involvement axis (6 human-annotated behaviors, n=1,085 sides): dimensional-with-skew |
| 18 | Formal battery (6 axis×level cells): dip/Silverman unimodal; every BLRT k=2 resolved as skew-fitting — incl. the §2.2 site: side-level skewnorm beats GMM(2) by ΔBIC 36 |
| 19 | Clusterability, 4 matrices: gap k̂ = 1, distance-dip unimodal; Hopkins elevation bounded to non-Gaussian shape via copula null, not clusters |
| 20 | Taxometrics (§4A6): caller CCFI mean **.151** → dimensional, robust to π ∈ {.30,.45,.55}; gold-indicator variant .460 ambiguous on low indicator validity (r̄ .09) |
| 21 | Power: fit-family + CCFI detect the paper's claimed mixture at 100% / false-alarm 0% at our n; dip underpowered (10%); BLRT's 76% skew-alarm is why it is paired with fit-family |
| 22–23 | Multiverse: **288/288 specifications unimodal**; single BATTERY table recorded |
| 24 | Combined trusted-space PCA: 26 features, Horn **K=5** (73.4% cum.) — volume/pausing/loudness separate axes, pitch fuses with overlap on PC1. **Hz-era reference values** — W5 re-records on the semitone-era table |

**2026-07-29 rising-terminal semitone redesign** (§3.3, gold-validated in
`analysis/validate_rising_terminal.py`): flag now thresholds `Terminal ST Slope` ≥ 3.07 st/s
(the old 30 Hz/s at the 169 Hz median register); r(share, pitch) .485 → .315; null set +
Hz column bit-identical; known-groups qy 2.48× / AUC .721 PASS, `^d` 1.65× failed-and-
adjudicated on record; MGQ recomposed 2,436 → 2,406. Registry: 46 Trusted.

## 1. Legacy autopsy (settled — basis for everything above)

The published result fails at four independent levels; full detail in the archived v1 §1.
Headlines: PC1 was a length/timing axis (turn gap −0.78, token count −0.59); the legacy
reader pooled ~6 unrelated calls per "conversation" (hence −4 s turn gaps); the unified
matrix was a positional cbind in three different row orders (misattributing essentially every
row); silhouette-on-1D and cluster-then-t-test were circular; the HC/HI label file came from
code that no longer exists. The bimodality was row-normalization arithmetic + backchannels —
a property of no speaker.

## 2. Notebook soft spots (reviewer-facing; all closed or tracked)

| # | Soft spot | Status |
|---|---|---|
| 2.1 | No formal multimodality test | ✅ dip + Silverman (+BLRT + fit-family) everywhere (Step 18) |
| 2.2 | Side-level BIC picked k=2 (ΔBIC −110) | ✅ formally explained as skew-fitting (Step 18) |
| 2.3 | Pseudoreplication (sides vs 543 callers) | 🟡 caller dedup + FDR done; mixed-effects `(1|caller)` still ⬜ (§4B9) |
| 2.4 | Gender–F3 claim means-only | ⬜ superseded in practice by Step 24b overlays; formal F3 test never run |
| 2.5 | NYC-overlap claim rode broken Turn Gap | 🟡 volume axes null per-caller; FTO-wired overlap rerun ⬜ (§4D17) |
| 2.6 | No topic control | ⬜ `Topic Label` built + registered, never joined (§4D16) |
| 2.7 | No reliability accounting | ⬜ ICC/split-half = §4B10, deferred to next paper (NB08 D9) |
| 2.8 | Allowlist/transform/aggregation sensitivity | ✅ measured vs gold + swept in the 288-spec multiverse |

## 3. Pipeline fixes + trust state

All five fixes **✅ built, in use, and gold-adjudicated** (details: ledger above; registry:
`docs/FEATURES.md`, 46 Trusted / 6 WIP / 5 Deprecated, parsed by `registry.py` and enforced
by `load_features_table`):

1. ✅ **Turn Gap → FTO** (`fto.py`): merged turns, backchannel-predecessor exclusion, word-tight bounds. Legacy `Turn Gap` Deprecated (replication only).
2. ✅ **One canonical table** (`features_table.py`, `swb-extract table`): zip-merge with per-row key assertions (the §1 row-scramble defense) + stale/registry guards at load. NB07/NB08 read only through the loader; `merge_test.csv` frozen for NB00–06 replication.
3. ✅ **Degenerate features**: personal focus → pooled raw hits (per-utterance score Deprecated); rising terminal → word-anchored tail, missingness preserved (never fillna(0)), **semitone threshold 2026-07-29**. The sole remaining NaN→0 lives in the bannered frozen `scripts/build_merge_test.py`.
4. ✅ **Machine-gun pitch term**: per-side P75 baseline, not population median. Columns stay WIP (intent gate rides the excluded Question Flag; re-extract-before-use note stands).
5. ✅ **Laughter counted before stripping** (5 columns, reconciled corpus-wide, Trusted; `laughs_per_100utt` in the panel).

Minor line, still open: ⬜ `pyproject.toml` declares only `empath`+`spacy` — librosa, numpy,
nltk, textstat, pandas undeclared; ⬜ loader flag-dtype coercion (`Int64`) + tests for the
registry/loader guards.

**2026-07-30 extractor audit — open hardening/caveat items** (full risk table in that
session's report; both repos green: local 308 passed / 13 skipped, all 13 = empath not
installed):

- 🟡 **Rate↔pause shared denominator** (risk 7): word/syllable/per-second rates divide by trans-span duration, mechanically anticoupling them with the Within-Pause family — part of Step 24's PC3 is arithmetic. NB08 sensitivity: articulation-rate variant via `phonation ≈ duration − Within Pause Total` from existing Trusted columns.
- 🟡 **Loudness silence dilution** (risk 6): RMS over the padded slice mixes amplitude with speech density; dim-2a mapping partially confounded with pausing. Needs a FEATURES.md caveat + (only if load-bearing in S6) a word-tight variant.
- 🟡 **CSV cache vintage class** (risk 6, bit once — latching): caches key on (row, header); an algorithm change under an unchanged schema silently reuses stale rows. Fix: version/params stamp per feature CSV, before W5 re-records.
- ⬜ **ST-slope artifact bound** (risk 5): the Hz column has the ±1000 Hz/s convention; the ST column the flag thresholds has none. Record one before W5.
- ⬜ Small (risk ≤3): `float("nan")` passes the tolerant parsers (add `math.isfinite` before Fisher §4F); empath not installed in the test env (13 Trusted-extractor tests silently skip); `_ensure_cmudict` disables SSL verification globally; two "overlap" definitions across `overlap.py` (word-level) vs `overlap_split.py` (turn-span) deserve a registry cross-note.

## 4. Avenues

**A. Formal dimensionality battery — ✅ complete** (A1 dip · A2 Silverman · A3 BLRT · A4
skew-family BIC · A5 multivariate clusterability · A6 taxometrics · A7 recovery power ·
A8 multiverse). Results in the ledger; machinery in `src/swb_extract/stats_modality.py` +
`taxometrics.py` (known-answer tested).

**B. Style as a trait** — 🟡 B9 caller dedup + Welch/FDR done, mixed-effects `(1|caller)` ⬜;
⬜ **B10 ICC/split-half trait stability** — the highest-value unpursued analysis (543 callers
× ~9 calls; also disattenuates every effect size). Deferred to the reliability paper (NB08 D9).

**C. Construct validity** — ✅ C12 gold suite complete (ledger Steps 12–15). 🟡 C11
single-axis question: NB07 Step 24 (K=5, split factors) is exploratory evidence against a
unidimensional HI–HC axis; **NB08's theory projection addresses this by construction**
(a theory contrast measured inside a 5-D space, not an assertion of unidimensionality);
semopy CFA proper stays ⬜. ⬜ C13 perceptual anchoring (rater study).

**D. Interactional/dyadic analyses** — ⬜ D14 accommodation/entrainment; ⬜ D15 style
mismatch → `rating_tab.csv` call quality (the most direct test of Tannen's actual thesis —
most exciting unpursued avenue); ⬜ D16 topic join (built, never joined); ✅ D17 FTO grounded
in the turn-taking literature (Step 1b).

**E. Feature gaps** — ✅ (a) overlap split · ✅ (b) laughter · ⬜ (c) filled-pause vs
discourse-marker split (opposite signs currently cancel in the filler rate) · ⬜ (d) voice
quality (parselmouth/eGeMAPS) · ⬜ (e) marked-shift dynamics (slope/reset/contour vs static
moments) · ⬜ (f) speaker-level aggregates (unblocked) · 🟡 (g) narrative block — gold
quotation rate measured; #25–28 unbuilt.

**F. External replication** — ⬜ Fisher (same genre, 10×), CallHome/CallFriend (familiar
dyads — the boundary-condition test), CANDOR (outcomes).

## 5. Order of work

1. ✅ Fixes (§3) → 2. ✅ Battery (§4A) → **3. ▶ NB08 measurement phase — current**
(`docs/NB08_MEASUREMENT_PLAN.md`: W3 machinery transplant next; W5 re-records Step 24 on the
semitone-era table; W7 freezes the construct map; W13 bookkeeping) → 4. ⬜ Positive story
(B10 ICC, C11 CFA, D14/D15/D16, mixed models) → 5. ⬜ Replication (§4F). Framing rule
throughout: claim **unimodal/dimensional**, never "normal."

## Doc map (after the 2026-07-31 compression)

| Doc | Role |
|---|---|
| `docs/AUDIT.md` | this file — canonical state: verdict, evidence ledger, trust, roadmap |
| `docs/FEATURES.md` | the trust registry — machine-parsed; moving a row between sections IS the trust workflow |
| `docs/PIPELINE.md` | data-layer contract + add-a-feature/recompute loop |
| `docs/NB08_MEASUREMENT_PLAN.md` | current-phase execution plan (archive at phase end) |
| `docs/tannen_feature_map.md` | theory reference (Tannen Ch.7 ↔ features; S6 cites it; Part 2 refresh due at W7) |
| `docs/archive/` | superseded docs, verbatim (see its README) |
| `analysis/` | `08_measurement.ipynb` living · `07_demoted_in_favor_of_5D.ipynb` frozen record · `validate_rising_terminal.py` RT evidence · `archive/` NB00–06 |
