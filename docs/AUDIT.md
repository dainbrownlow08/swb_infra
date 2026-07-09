# Audit: Switchboard conversational-styles repo

> **Living document.** Status markers track what has been done against the audit.
> _Last reviewed: 2026-06-26 — through `analysis/06_caller_level_volume_brizan_revisions_plus`
> and the Jun-9/10 extractor set. §3.2 corrected 2026-06-29: the canonical-table builder
> (`features_table.py` + `swb_extract table`) was already built; the marker had understated the fix as "no builder exists." §3.2 updated 2026-06-30: the spine cutover has **begun** — `analysis/07_final.ipynb` is the first notebook reading the canonical table via `load_features_table` (see the §3.2 status below). §3.5 corrected 2026-07-02: laughter **is** merged — the five columns ship in the Jun-29 `features_table.csv` rebuild and are registered in `docs/FEATURES.md`; the "not yet merged" in its status was stale, and the residual gap is analysis use. **Pipeline Plan 7/2/2026** appended at the bottom of this doc: the execution plan for taking every §3 item to ✅ in NB07. **Continuum Validation Plan 7/2/2026** appended the same day: the §4A battery build-out (dip / Silverman / BLRT / skew-fit / multivariate clusterability / taxometrics / recovery power / multiverse) for NB07. **Paper-Submission Minimum 7/8/2026** (rev 7/9b): `docs/Audit_July_2026_Paper_Submission_Min.md` — the execution plan for the submission push (time cost dropped as a scoping criterion 7/9): **§4A in full, A6 taxometrics reinstated 7/9**; minimal trust adjudication; an involvement panel (pooled personal focus, laughter, validated question rate, overlap split); and the **NXT gold suite** — gold backchannel validation of the allowlist (§2.8/§4C12a), gold+classifier questions (§4C12b), gold-only involvement behaviors (collab completion / mirror / echo-q / appreciation / quotation) with a gold involvement axis in the battery, disfluency de-conflation of repetition (§4C12c), and the cooperative-vs-obstructive overlap split (§4E-a). Where sequencing differs, it governs._
>
> **Legend:** ✅ done (implemented and in use) · 🟡 partial (built but not wired into analysis,
> or done at one level but not the rigorous version) · ⬜ not started.
> §1 is the legacy-paper autopsy — established findings, not action items, so it carries no markers.

## Progress dashboard

**~12 done · ~9 partial · ~23 not started.** _(7/9: §4A1 → ✅ all levels; §4A5 → ✅ four-matrix clusterability (gap k=1 everywhere; Hopkins nuance recorded); §4A2–4 → ✅ — the battery ran (Steps 17–18): dip+Silverman unimodal at caller/side, BLRT k=2 rejections all resolved by fit-family as skew-fitting, the side-level ΔBIC −110 formally closed (skewnorm over GMM2 by 36); §4C12 ⬜→🟡 — gold layer + classifiers measured; §4E-a ⬜→✅ — overlap split Trusted on gold checks; §4E-g ⬜→🟡 — gold quotation rate in the Step-17 panel.)_

The shape of the work, not just the count:

- **The extractor pipeline is largely corrected** — FTO, laughter, personal-focus,
  rising-terminal, and the machine-gun pitch term are all rebuilt and unit-tested (§3). **Most
  fixes are not yet wired into the notebooks**: NB00–05 still load the raw `Turn Gap` and the
  backchannel-stripped feature CSVs — but **NB06 now adopts FTO** (Jun-26), the first fix to
  cross from "built" to "in use" (§3.1). "Built" ≠ "in use" — hence the still-large 🟡 band.
- **NB06 delivered the first real distributional rigor** at the *corrected caller-level unit*:
  Hartigan's dip test, Horn's parallel analysis, dropped silhouette, FDR-corrected demographics
  (§4A1, §2.8, §4B9-partial), and (Jun-26) swapped the broken `Turn Gap` for **FTO** with an
  in-notebook distribution check (§3.1, §17). With the corrected feature, parallel analysis now
  retains **K=2** (PC3 was a borderline keeper under the old `Turn Gap`); the unimodal-continuum
  finding and the robust gender/education effects are unchanged. It explicitly defers the
  "airtight" battery (Silverman / bootstrap-LRT / skew-fit / taxometrics).
- **Still entirely ahead:** the airtight unimodality battery (§4A2–8), the whole "positive story"
  — trait stability/ICC, construct validity/CFA, mixed-effects demographics, accommodation,
  style-mismatch → call quality (§4B10, §4C, §4D) — and external replication (§4F). This matches
  the audit's own §5 ordering: fixes → battery → positive story → features → replication.

## Verdict

Your correction arc is sound, and the evidence behind it is stronger than your notebooks currently claim. The paper's "two styles" result fails at four independent levels — any one of which would invalidate it — and your backchannel diagnosis is the correct explanation for the one surviving bimodality. However, your continuum conclusion currently rests on **KDE peak-counting with hand-picked bandwidths**, and at speaker level **BIC actually selected k=2** (ΔBIC = −110) — you dismissed this by magnitude comparison, but a hostile reviewer will seize on it. The single most important thing you have not yet done is convert "I didn't find bimodality" into formal, positive statistical evidence for dimensionality. The good news: the tools for that exist, your corpus has untapped gold annotations sitting on disk that enable them, and there are several theory-driven analyses (trait stability, accommodation, style-mismatch → call quality) that turn the unimodal result from a disappointment into a foundation.

---

## 1. The paper and the legacy code: what the audit established

_These are established diagnoses of the original paper and legacy code — findings, not action
items — recorded as the evidentiary basis for the re-analysis. No status markers apply._

The paper's intended arc was reasonable — RQ1: is there evidence for HI/HC patterns at scale; RQ2: what feature profiles distinguish styles; RQ3: how do styles distribute over demographics. The execution fails at every layer:

**Visible in the paper itself:**
- PC1 (~66% variance) loads almost entirely on turn gap (−0.78) and token count (−0.59) — it's a length/timing axis, not a style axis.
- Silhouette analysis to choose GMM component count is improper for 1-D continuous data (it favors any split; k=2 nearly always wins on skewed unimodal data).
- The Welch's t-tests are circular: clusters were derived from PC1, then the same features were tested between clusters. Significance is guaranteed by construction; d = 0.02–0.06 confirms there's no real separation.
- Pitch features were dropped *because* they "create far less well defined conversational style groups" — selecting features to manufacture structure.
- Mean turn gap of **−3.9 s in both groups** is physically implausible and went unexamined.

**What the legacy code audit adds (much of this you may not have known):**
- The −4 s turn gaps now have a root cause: the legacy reader used the **3-digit directory prefix as the conversation ID**, pooling ~6 unrelated phone calls, sorting their utterances by timestamp, and chaining them as predecessors. Turn gap — the paper's top-loading feature — was computed across different conversations.
- `aligned_acoustic_linguistic.csv`, the matrix the "does the bimodal still hold" check ran on, is a **positional cbind of three tables in three different row orders** (one a different length, silently truncating 77,682 rows). Essentially every row attributes acoustic and linguistic features to the wrong utterance — including gluing dev-set male audio features to test-set female demographics. The producing notebook printed the contradiction in its own output and saved anyway.
- A deeper, earlier misalignment: wavs were cut against original transcripts (line numbers including `[silence]`/`[noise]` lines) but transcripts were fetched by line number from *cleaned* files — so even the 138k-row "unified" matrix pairs linguistic features with the wrong audio.
- k=2 was never supported by the team's own model-selection scans (their elbow/silhouette runs picked 3, 4, and 14 in different notebooks). The HC/HI label file was produced by code that **no longer exists in the repo**, from turn gaps that had suffered a seconds→nanoseconds timestamp corruption, with "interrupted" mapping to HC in one row and HI in the next.
- Supporting stats were broken or biased: a t-test cell that filters a numeric column against the string `'Female'` (always empty), per-utterance OLS with no speaker clustering, an EDA that deleted **all** zero-repetition and zero-pause rows before analysis, and two incompatible acoustic feature definitions (piptrack vs pyin; resampled vs native) shipping under identical column names.

Bottom line for your PI conversation: the published result is not weakened — it is unrecoverable. The bimodality survived the row-scrambling precisely because it was never a property of any speaker; it was a property of the row-normalization arithmetic plus backchannels.

## 2. Your notebooks: what holds, what's soft

The progression (replicate → fix scaling → autopsy the clusters → speaker level → interactional factors) is methodical and the headline findings are supported by printed outputs: the replication reproduced the paper's structure (silhouette k=2 = 0.744, Turn Gap +0.905 under row-norm); the "HC" cluster was **98% ≤2-token utterances** topped by *yeah/um-hum/uh-huh*; removing the 25.2% backchannels left PC0 unimodal at every bandwidth; speaker-level and 19-feature interactional analyses stayed unimodal; varimax cleanly isolated an interactional-engagement factor (F3: overlap onset +0.87, overlap duration +0.79, turn gap −0.56).

The soft spots a reviewer will find:

1. 🟡 **No formal multimodality test anywhere.** Unimodality claims rest on KDE mode counts with ad-hoc prominence thresholds that drift between notebooks (1% in NB3/NB4, 5% in NB5).
   ↳ **Status:** Hartigan's dip test *added* in NB06 (cell 11: PC1/PC2/PC3 dip p = 0.96/0.93/0.99). The "no test anywhere" soft spot is downgraded, not closed — Silverman + bootstrap-LRT (the accepted pair / airtight version) are still missing → §4A2–3.
2. 🟡 **The speaker-level BIC picked k=2** (15,756 vs 15,866 for k=1). You argued ΔBIC −110 ≪ the backchannel split's −53,000, which is fair rhetoric but not a test. A 2-Gaussian fit beating 1 Gaussian is *expected* for any skewed unimodal distribution — but you never fit the one-component skewed alternative that would prove it.
   ↳ **Status:** Caller-level dedup in NB06 flips the result to **ΔBIC(2−1) = +10, k=1 preferred** (the k=2 was a side-level pseudoreplication artifact) — empirically defused. The skew-normal-vs-2-Gaussian fit that *formally* explains it (§4A4, "closes your biggest vulnerability") is still ⬜.
3. 🟡 **Pseudoreplication everywhere.** "Speaker" = conversation-side (`sw####A/B`), but `call_con_tab.csv` shows 4,876 sides come from only **543 callers (~9 calls each)**. Every demographic test (gender d = −0.330, NYC +0.83, education) treats ~9 dependent observations per person as independent. None of this is acknowledged in the notebooks.
   ↳ **Status:** Unit fixed to the caller via `call_con_tab.csv` in NB04/05/06 (n ≈ 487–493), and NB06 adds Benjamini-Hochberg FDR across the demographic family. The remaining gap — mixed-effects models with `(1|caller)` (§4B9) — is ⬜, so dependence *within* a caller's calls is still unmodelled.
4. ⬜ **The gender–F3 claim** (one of NB5's three headline conclusions) is supported by group means only — no test, no effect size, unlike the NYC comparisons.
   ↳ **Status:** Unchanged. NB06 tests gender on the *volume* PCs (d + p + FDR: PC1 d=−0.47, etc.) but never on the interactional **F3** factor, which lives only in NB05 and is still means-only (female +0.25 vs male −0.29).
5. 🟡 **The NYC-isn't-overlap finding is premature**, because it depends on Turn Gap and Latching, both of which are currently broken (see §3). The overlap features themselves are sound (word-alignment based), but F3 mixes them with the broken timing features.
   ↳ **Status:** NB06 reframes NYC as **not significant per-caller** on the volume axes (region ANOVA p_fdr 0.36–0.73). But the F3/overlap analysis (NB05) still rides the raw `Turn Gap`; the overlap-specific claim needs an FTO-wired rerun (§3.1 / §4D17).
6. ⬜ **No topic control** — `topic_label.py` exists but is never joined; Switchboard topics are assigned, and topic is a plausible confound for both region and gender effects.
   ↳ **Status:** Unchanged. `topic_label.py` + `topic_label.csv` are built, but no notebook joins topic. → §4D16.
7. ⬜ **No reliability accounting** — median 36 substantive utterances per side means speaker profiles are noisy; PC variance and all effect sizes are attenuated by an unquantified amount.
   ↳ **Status:** Unchanged. No ICC / split-half / disattenuation anywhere. → §4B10.
8. **Smaller items:** the 38-token backchannel allowlist is copy-pasted across three notebooks with no sensitivity check; NB4/NB5 skip the winsorize+log1p step NB3 Part C established; no parallel analysis for component retention; 17,053 pitch-null rows silently dropped; `tannen_features.csv` — the file named for the initiative — is never read by any notebook.
   ↳ **Status:** ✅ **parallel analysis** now done (Horn's, NB06 cell 9, retains K=3). Still ⬜: winsorize+log1p (skipped in NB4/5/6), pitch-null handling (undocumented), 38-token allowlist sensitivity, and `tannen_features.csv` is still never read by a notebook.

## 3. Pipeline: fix these before the next analysis round

The Jun 9 extractors are well-built (clean word-alignment infrastructure, every extractor unit-tested, no join fan-out, zero duplicate keys). Five things need fixing, in order:

1. ✅ **Turn Gap semantics.** It's computed against the chronological predecessor *including backchannels*, so 60.3% of gaps are negative and the median "gap" is −0.49 s. This poisons Latching (1.15% positives — near-constant) and the machine-gun composite (its "fast follow ≤0.5 s" criterion is satisfied by every negative gap). Redefine as **floor-transfer offset (FTO)**: merge consecutive same-speaker utterances into turns, exclude backchannel-only predecessors, and measure gap at genuine floor transfers (Heldner & Edlund 2010 is the standard reference — SWBD FTOs should center ~+200 ms, which would also situate your data in the turn-taking literature).
   ↳ **Status:** ✅ **Wired into NB06** (2026-06-26). `src/swb_extract/features/fto.py` (+`test_fto.py`, 12 passing tests) implements merged turns, backchannel-predecessor exclusion, and gap-at-floor-transfer; **regenerated fresh** (the Jun-10 CSV was stale vs the Jun-14 algorithm — 2,703 rows / 1.26% changed on rebuild). NB06 replaces `Turn Gap` with `FTO Sec` (restricted to a [−2, +2] s response window), recomputes end-to-end, and validates the distribution (median **+0.140 s**, 37.7% overlap — §17). `latching_flag.py`/`machine_gun_question.py` are rewired to FTO at the extractor level. **NB00–05 still load the raw `Turn Gap`** (`turn_gap.py` kept on purpose for paper replication). _Step-11 adjudication (2026-07-09) caught a stale pre-rebuild vintage in `latching_flag.csv` — 3 invariant violations visible, 1,333/214,204 rows corrected on re-extraction (latch rate 17.8→17.9%); `machine_gun_question.csv` is the other FTO-derived CSV and carries the same risk (flagged in its FEATURES.md note — re-extract before first use). The mtime stale-guard cannot see cross-CSV vintage drift; Step 11's invariants are the working defense._
2. 🟡 **One canonical table.** `tannen_features.csv` (May 9) has only topic/personal-focus/mutual-revelation; `merge_test.csv` (Jun 9, 39 cols) has everything else; and no script in the repo builds merge_test's original base columns. Write one version-controlled builder producing a single table.
   ↳ **Status:** 🟡 **Builder ✅ built + CLI-wired; analysis not cut over.** Fix #2's deliverable exists: `src/swb_extract/features_table.py` (+ the `swb_extract table` command) builds a single from-source table — a pure function of `manifest.csv` + every `features/*.csv`, rebuilt in seconds as a **zip-merge with per-row key assertions** (any misalignment / duplicate / missing / trailing row is a hard error naming the file+row — the structural defense against the §1 row-scramble). All 26 feature CSVs are inputs, so a rebuild folds in FTO, laughter, the corrected machine-gun, gated personal-focus, and topic with **no separate merge step**. What keeps it 🟡: **nothing reads `features_table.csv`** — analysis still loads `merge_test.csv` (7 refs) + the Jun-25 `paper_aligned_*_PCA` derivatives, and the on-disk `features_table.csv` is **stale** (Jun 10 — predates both the Jun-14 builder and the FTO/laughter/personal-focus fixes) and git-untracked. `build_tannen_features.py` still builds only the dim-1–2 slice (5 cols); `merge_test.csv` stays **frozen by design** for paper replication and is *replaced* on cutover, not rebuilt. The two builder bugs — `build_merge_test.py:102` `fillna(0)` on Rising Terminal, and its **population-median** machine-gun composite (line 107; the §3.4-"fixed" bug, still live here) — are **merge_test-only** and disappear on cutover (`features_table.py` preserves missingness and ingests the corrected `machine_gun_question.csv`). Remaining: the **cutover** — regenerate `features_table.csv`, repoint NB06+/the §4A battery onto it, retire/freeze `merge_test.csv`, fold the tannen track — deferred to the **end of the fixes block, before the battery** (§5) so the airtight stats never run on the buggy frozen table.
   ↳ **Reorg (2026-06-29):** canonical table **regenerated** fresh → `utterances_v2/derived/features_table.csv` (214,204×56, key-assertions clean across all 26 extractors). Added the missing **trust axis** and **stale-data guard**: a living trust document `docs/FEATURES.md` (every column bucketed **Trusted**/**WIP**/**Deprecated**, parsed by `src/swb_extract/registry.py`), `src/swb_extract/analysis.py::load_features_table` (the single loader every trustworthy notebook imports — refuses a table older than its inputs, and refuses unregistered columns), and `docs/PIPELINE.md` (add-feature + recompute procedure + data-layer contract). Stale duplicate `features_table.csv`, `merge_test_backup.csv`, and the unread `tannen_features.csv` quarantined to `_archive/`. **Still 🟡:** the spine cutover (repoint NB06+ off `merge_test`/`paper_aligned` onto the loader) and the per-feature **trust walkthrough** (initial buckets set 2026-06-29 in `docs/FEATURES.md`: 17 Trusted, 33 WIP, 4 Deprecated).
   ↳ **Cutover begun (2026-06-30):** `analysis/07_final.ipynb` — a faithful rebuild of NB06 — is the **first notebook to read the canonical table through `load_features_table`** (`include="provisional"`), replacing NB06's frozen `paper_aligned_standardized_PCA.csv`. Executed clean on the fresh table (214,204 utts → 493 callers → 487 retained ≥20 substantive), so the stale-data + registry guards now gate the analysis line and **"nothing reads `features_table.csv`" no longer holds**. NB06 never used the file's precomputed `pca_*`/`gmm_*` columns (it recomputes its own StandardScaler+PCA), so nothing is lost by the loader path; the result **reproduces NB06 to within rounding** (K=2, PC1 42.5%, dip p 0.994/0.993, ΔBIC(2−1)=+11; Gender PC1 d=−0.41 p_fdr 6.1e-5, Education PC1 p_fdr 0.036, Region/Generation null — 3 of 8 survive) — itself the first check that the integration is faithful. **Still 🟡** (done at one level, not the rigorous version): NB01–03 stay frozen on `merge_test`/`paper_aligned` **by design** (paper replication); the §4A battery is not yet on the loader; `merge_test.csv` is not yet retired; and the per-feature **trust walkthrough** in `docs/FEATURES.md` remains initial buckets (33 WIP unconfirmed). NB07 is the **living** notebook — audit concerns get appended to it and re-reconciled after each `swb-extract table`.
3. 🟡 **Degenerate features.** `personal_focus_score`: 54.7% NaN and 64.6% of defined values exactly 0.0 or 1.0 — Empath saturates on short utterances; aggregate to speaker level over content words, or replace with a pronoun+lexicon score over a minimum-token window. `rising_terminal`: 68.5% missing **not at random** (short/unvoiced tails), and `build_merge_test.py` fills NaN→0, conflating "unmeasurable" with "not rising."
   ↳ **Status:** Both extractor halves fixed and now **adjudicated in NB07 Step 11 (2026-07-09, submission-plan T3)**: pooled personal focus is the sanctioned estimator with printed evidence (caller-level PF_ratio at ≥30 pooled hits — 100% coverage of the 487 callers, mean .619 ± .129, zero 0/1 saturation vs the per-utterance score's 71.5% null / 48.6% saturation, recomputed inline as the contrast) and the degenerate per-utterance `Personal Focus Score` is **Deprecated** in the registry; rising-terminal missingness audited — 30.3% null, MNAR but **voicing/length-dependent** (null RISES with utterance length, 21%→35% for ≤2→≥6 tokens, contra the docstring's short-tails guess), slope clip ±1000 Hz/s recorded, both RT columns Trusted with never-fillna(0) notes. _In use 2026-07-09 (NB07 Step 16, T8): `PF_ratio` (floor 30, 100% coverage) and `rt_rising_share` (defined-only mean, MNAR caveat carried) are involvement-panel variables — both load ≈+0.35 on the engagement factor._ Residual for ✅: the C2 FROZEN banner on `scripts/build_merge_test.py`, whose line-102 `fillna(0)` is the sole remaining NaN→0 conflation (replication-only).
4. ✅ **The machine-gun composite's pitch term** uses the whole-population median — i.e., it's substantially a female-speaker indicator. Use speaker-relative pitch (z within speaker).
   ↳ **Status:** `machine_gun_question.py` now builds a per-conversation-**side** pitch baseline (`HIGH_PITCH_PCTL=75`, `MIN_SIDE_PITCH_N=10`) and gates on speaker-relative pitch, not the population median.
5. 🟡 **Laughter** (Tannen dim 9) is still stripped, not counted, by every extractor — your own feature map flagged the one-pass fix (count brackets, then strip). It remains the only dimension with raw signal on disk and zero columns.
   ↳ **Status:** `laughter.py` (+`test_laughter.py`) now counts `[laughter]`/`[laughter-word]`/`[noise]`/`[vocalized-noise]` brackets before stripping. _Corrected 2026-07-02:_ the five laughter columns (`Laughter Count`, `Laughed Word Count`, `Noise Count`, `Vocalized Noise Count`, `Other Bracket Count`) **are merged** — they ship in the Jun-29 `features_table.csv` rebuild and are registered (WIP) in `docs/FEATURES.md`; the earlier "not yet merged" (via `orthogonal_features_audit_jun19.md`, which predates the rebuild) was stale. Remaining gap: **no notebook analyzes them.** Code ✅, merge ✅, in-use ⬜ → Pipeline Plan 7/2/2026, Workstream B (NB07 Step 13). _Update 2026-07-09 (Step 11, T3): all five columns reconciled corpus-wide against the bracket-preserving Transcript (0 mismatches; 7.5% any-laughter incidence, matching the Jun-19 survey) and promoted to Trusted; **in use 2026-07-09 (NB07 Step 16, T8): `laughs_per_100utt` (all-utterance rate) is an F_int variable and a panel variable — it loads +0.32 on the engagement factor (with overlap onset/duration and PF_ratio), i.e. dim 9 is now in the analysis.**_

(Minor: ⬜ `librosa`, `numpy`, `textstat`, `nltk`, `pandas` still undeclared in `pyproject.toml` — only `empath`+`spacy` are listed; ⬜ merged flags serialize as floats; ✅ the question-flag rate anomaly is validated and explained (2026-07-09, NB07 Step 12: gold question rate **7.83%** vs flag 3.34%; flag precision .553 / recall .236, with declarative, tag, and marker-prefixed questions accounting for the misses; flag **excluded** from analyses — the classifier route is §4C12 / submission-plan T7).)

## 4. Avenues you have not pursued

This is the core of your ask. Organized from "makes the current claim rigorous" to "builds the future program." Tools are open-source and Python-first per your constraints; R noted only where the canonical implementation is R-only.

### A. Make "unimodal continuum" a formal statistical claim

1. ✅ **Hartigan's dip test** (`pip install diptest`) on PC0/PC1/F3 at utterance, side, and caller level — the standard test of unimodality, replacing KDE eyeballing. Report dip statistic + bootstrap p for every axis in a single table.
   ↳ **Status:** ✅ Done 2026-07-09 (NB07 Steps 17–18, T9–T10): dip rows in the shared BATTERY ledger for volume PC1/PC2 + F_int at caller level, PC1 at side level (n=3,705), PC1-10-feature at utterance level (n=158,640; + turn-initial sensitivity n=76,207), and the gold involvement axis at side (n=1,085) + caller (n=320) — every cell unimodal (dip p ≥ .98; utterance-level p is out-of-table above n=72k, caveat printed).
2. ✅ **Silverman's bandwidth test** (critical-bandwidth bootstrap) as the complementary mode test — dip and Silverman together are the accepted pair.
   ↳ **Status:** ✅ Run 2026-07-09 (NB07 Steps 17–18): B=999 (B=99 at utterance level, pre-authorized), unimodal at every caller/side cell (p .36–.80); **rejects at utterance level (p=0)** — investigated in place per Q5: h_crit ≈1.3–1.5 SD with non-monotone mode counts = density-floor flickers from the 0.01% far tail (|z|>5, quadratic repetition counts); a 0.1% winsorize collapses every moderate-bandwidth mode count to 1. The REJECT rows stand in BATTERY; Step 22's winsor+log1p arm is the formal sensitivity.
3. ✅ **Parametric bootstrap likelihood-ratio test for k=1 vs k=2** (simulate from the fitted 1-component model, refit both, build the LR null — ~30 lines with sklearn; `mclustBootstrapLRT` in R if you want the citable canonical version). This is the direct, correct answer to "but BIC picked k=2."
   ↳ **Status:** ✅ Run 2026-07-09 (NB07 Steps 17–18, B=999 / B=99 utterance): BLRT rejects k=1-Gaussian in favor of k=2 in all 6 non-gold cells and both gold cells — **and every rejection is resolved by §4A4's fit-family comparison as skew-fitting** (a single skewed component beats GMM(2) on BIC wherever fit-family runs). The pair of results together is the airtight form: not one Gaussian, but one *skewed* component rather than two Gaussians.
4. ✅ **Fit a single skew-normal / skew-t / lognormal against the 2-Gaussian mixture** (scipy `skewnorm`/`skewt` MLE; `mixsmsn` in R for skewed mixtures). If one skewed component beats two Gaussians on BIC — very likely given PC0 skew +0.49 — the k=2 BIC result is formally explained as skew-fitting, not types. *This single analysis closes your biggest vulnerability.*
   ↳ **Status:** ✅ Run 2026-07-09 (NB07 Steps 17–18). **The §2.2 closer landed: at side level a single skewnorm beats the 2-Gaussian mixture by ΔBIC 36 — "BIC picked k=2" is formally explained as skew-fitting, not types.** Same resolution at every other fit-family cell: caller PC1 lognorm +18, PC2 jf_skew_t +12, F_int lognorm +24, gold axis side jf_skew_t +18 / caller skewnorm +13 (ΔBIC = BIC(GMM2) − BIC(best single family), positive = single wins).
5. ✅ **Multivariate clusterability, not just 1-D projections.** Unimodal PCs don't preclude structure elsewhere in the 19-dim space. Run the dip test on pairwise distances, the Hopkins statistic, and the gap statistic against a proper null (PCA-shaped Gaussian reference). "No clusters in the full feature space" is a much stronger claim than "no clusters on PC0."
   ↳ **Status:** ✅ Run 2026-07-09 (NB07 Step 19, T11) on four matrices — vol11 (caller), vol+interactional (caller), vol+int+panel (caller), gold panel (side): **gap statistic chooses k=1 and pairwise-distance dip is unimodal (p≈1.0) on all four**. Hopkins rejects the correlated-Gaussian N(0,Σ̂) null everywhere (H .71–.82) — investigated in place (Q5): that null tests Gaussian *shape*, and a second, shape-matched calibration (Gaussian copula with empirical marginals) absorbs most of the elevation (vol+int p=.26, gold p=.11) but leaves vol11 (p=.046) and vol+int+panel (p=.000) elevated → higher-order dependence beyond pairwise correlations (co-occurring extremes), **not** discrete clusters (gap/dip-dist exclude those directly). Recorded as a standing, bounded discordance; Step 20 taxometrics is the pre-registered arbiter for exactly this question.
6. ⬜ **Taxometric analysis — the centerpiece I'd recommend.** Types-vs-continuum is a solved methodological problem in psychopathology research: Meehl's MAXEIG/MAMBAC/L-Mode with the **Comparison Curve Fit Index** (CCFI < 0.45 ⇒ dimensional; `RTaxometrics` in R; MAMBAC is simple enough to hand-roll in Python). Nobody in the conversational-style literature has applied taxometrics. With 543 callers (or 3,595 sides) and your indicator set, this is feasible and would be a publishable methods contribution on its own: "Conversational style is dimensional, not taxonic."
   ↳ **Status:** Not started — named as deferred in NB06's recorded conclusion ("the airtight version still needs the dip/BLRT/taxometric battery"). _Briefly deferred from the submission plan on time cost (2026-07-08); **reinstated 2026-07-09** when time cost was dropped as a scoping criterion — in scope as Paper-Submission Min **T12** (`taxometrics.py` + NB07 Step 20), including a side-level gold-involvement-indicator variant (taxometrics on human-annotated involvement behaviors — no prior art in the conversational-style literature)._
7. ⬜ **Recovery simulation (power analysis for types).** Simulate data with the paper's own claimed structure (means −0.31/+0.36, SDs 0.13/0.31, weights 0.55/0.45), push it through your corrected pipeline, show dip/BLRT/taxometrics all detect it. This converts your null into "we had the power to see two styles; they are not there."
8. ⬜ **Multiverse / specification-curve analysis.** Your conclusion currently depends on one backchannel definition (38-token allowlist), one transform (winsorize+log1p), one aggregation (mean over ≥20 utterances). Run the grid — backchannel definition (allowlist / token-count threshold / DAMSL-trained classifier), scaling, feature subsets, with/without pitch, level — and report the fraction of specifications yielding unimodality. This is what "exhaustive" looks like to a reviewer.

### B. Fix the unit of analysis; test style as a *trait*

9. 🟡 **Caller-level deduplication** via `call_con_tab.csv` → caller_no. Rerun the distributional battery at true speaker level (543 callers). All demographic claims need re-estimation as **mixed-effects models**: `feature ~ sex + region + generation + education + topic + (1|caller)`, with FDR correction (statsmodels `MixedLM` / `pymer4`). This will also properly adjudicate NYC-vs-rest (currently confoundable with gender/generation composition) and give the gender–F3 claim its missing test.
   ↳ **Status:** Caller dedup done (NB04/05/06) and FDR added (NB06). The mixed-effects models themselves — `MixedLM` with `(1|caller)` and topic as a covariate — are ⬜; NB06 uses Welch t / one-way ANOVA on caller means.
10. ⬜ **ICC / split-half trait stability — the highest-value unpursued analysis in the repo.** Each caller has ~9 conversations: compute ICC of style scores across a caller's calls, and split-half (odd/even utterances) reliability within calls. If style position is stable across conversations with different strangers on different topics, you have *positive* evidence for style as an individual-difference trait distributed along a continuum — exactly the foundation Tannen's continuum reading needs. Low ICC is equally informative (style is situational/dyadic). Reliability estimates also let you disattenuate every effect size you report.

### C. Construct validity — is there an "involvement" axis at all?

11. ⬜ **Confirmatory factor analysis** (`semopy`). Tannen predicts the HI markers covary: short FTOs, latching, overlap, speed, loudness, questions, personal focus. Your NB5 varimax already shows volume, loudness, rate, and interaction splitting into *separate* factors — i.e., a single HI–HC axis may not exist as a unidimensional construct. Test a hierarchical model (general involvement factor over the four group factors) vs orthogonal factors and report fit. Either result reshapes the paper: a continuum needs an axis, and right now your best candidate is F3, not PC0.
    ↳ **Status:** Not started. NB05 has *exploratory* varimax (F0 loudness / F1 rate / F2 overlap / F3 interactional) — suggestive, but no confirmatory model fit / comparison.
12. 🟡 **Validate features against the gold annotations already on disk.** `corpus/nxt_switchboard_ann/xml/` contains NXT Switchboard: **dialog acts for 642 conversations (1,284 sides) including gold backchannel tags**, gold disfluency annotation, and accent/break (prosody) layers for a subset (150 accent files; 451 accents/breaks files in `corpus/annotated_files/`). Use it to (a) measure precision/recall of your 38-token backchannel allowlist, then train a small classifier and apply corpus-wide; (b) validate Question Flag (your 3.26% vs ~6–7% expected — likely missing declarative questions, which rising-terminal should catch once its missingness is fixed); (c) split repetition into **disfluent repair vs rhetorical repetition** — Tannen means the latter, and your current counters conflate them with disfluency, which is a real confound for the "repetition = involvement" reading.
    ↳ **Status:** 🟡 In progress (2026-07-09, submission-plan T4): `src/swb_extract/nxt.py` (+10 tests, truncated-real fixtures) parses dialAct/terminals/disfluency; NB07 Step 12 aligns gold DAs to our utterances (99.2% match rate; 544 conversations ∩ manifest; 52,890 labelled utterances). Measured: **(a) allowlist-38 vs gold backchannels P .842 / R .917** (F1 .878 — the §2.8 defense); **(b) Question Flag P .553 / R .236** → excluded from analyses (pre-registered bar), gold q-rate 7.83% vs 3.34%. _Classifiers landed 2026-07-09 (NB07 Step 15, T7): leak-free grouped-CV logistic pipelines (BOS-marked uni+bigrams + 20 Trusted timing columns) — **backchannel CV F1 .888** (P .824 / R .964) clears its pre-registered .85 bar and joins the multiverse `bc_def` axis; **question CV F1 .681** misses its .70 bar by .019 → NOT admitted (recorded, not tuned): the involvement panel carries no question_rate, and the gold rate 7.83% is the citable number. Corpus-wide masks are in-notebook definitions; no table columns._ _((c) done 2026-07-09, NB07 Step 13: **24.5%** of current-utterance / 14.2% of previous-utterance pair-repetitions attributable to gold repair material — quantified, with de-conflated variants computed and extractor↔gold sanity r=.978; repetition-vs-mirror r≈0 → self- and allo-repetition are distinct constructs, so the involvement-relevant repetition variable is the gold `^m` mirror rate.)_
13. ⬜ **Perceptual anchoring.** Tannen's construct is ultimately a hearer judgment. Have 2–3 raters (lab members) rate a stratified sample of ~100 conversation excerpts on involvement (and naturalness); correlate with your composite scores. Without this, the continuum is a continuum of *something measured*; with it, it's a continuum of *perceived style*.

### D. Interactional and dyadic analyses — where Tannen's theory actually lives

14. ⬜ **Accommodation / entrainment.** Tannen's framework is about what happens *between* speakers. Levitan & Hirschberg's proximity/convergence/synchrony measures were developed on precisely this kind of data — compute whether partners converge on rate, loudness, FTO, overlap over the call, and whether B's style position depends on A's. "Style is a continuum along which speakers *move toward their partners*" is a positive, theory-rich result.
15. ⬜ **Style mismatch → conversation quality.** `tables/rating_tab.csv` has per-call **DIFFICULTY, TOPICALITY, NATURALNESS** ratings for all 2,438 calls (column semantics confirmed against the DAMSL `.utt` headers). Tannen's central claim is that style *clash* — not style itself — causes interactional trouble. Test: |style_A − style_B| (and dyad mean) predicting rated difficulty/naturalness, mixed model with caller random effects. This requires no types whatsoever and is the most direct test of Tannen's actual thesis anyone could run at scale. I'd call this the most exciting unpursued avenue in the repo.
    ↳ **Status:** Not started. `rating_tab.csv` is on disk but no notebook joins it.
16. ⬜ **Topic as within-speaker manipulation.** Join `topic_label` (built, never used). Topics differ in personal focus (the prompts in `topic_tab.csv` are classifiable); each caller appears across multiple topics. Test whether speakers shift toward involvement on personal topics *within caller* — Tannen predicts yes; it also doubles as the topic control for all demographic claims.
    ↳ **Status:** Not started — `topic_label.csv` is built but still never joined (same gap as §2.6).
17. ✅ **Ground turn-gap distributions in the turn-taking literature** (Levinson & Torreira 2015; Heldner & Edlund 2010): after the FTO fix, your gap distribution should reproduce the canonical unimodal ~+200 ms shape with an overlap tail — a free external-validity check that also retroactively explains the paper's −4 s nonsense.
    ↳ **Status:** ✅ Done in NB06 **Step 1b** (2026-06-26). The regenerated FTO distribution reproduces the canonical shape — median **+0.140 s**, **37.7%** overlap, a single mode just after zero, 91% within [−1, +2] s — plotted against the Heldner & Edlund ~+200 ms reference. Retroactively explains the legacy −4 s nonsense.

### E. Feature gaps still worth closing (from your map, re-prioritized)

Highest theory-per-effort first:

- ✅ **(a) Cooperative vs obstructive overlap** — the split your `overlap.py` explicitly deferred; operationalize as overlap after which the original speaker retains vs loses the floor — this is *the* HI-diagnostic distinction.
  ↳ **Status:** ✅ Done 2026-07-09 (submission plan T6). `src/swb_extract/features/overlap_split.py` (+11 unit tests) classifies every overlap event on the merged FTO turn walk (`fto.build_turn_events`, factored out so FTO and the split share one state machine): backchannel-only → cooperative by definition; contained interjection → cooperative; floor-taking overlap → obstructive iff the holder ceded within the pre-registered W=1.0 s. Corpus-wide: 74,550 events, 31.7% obstructive. **In the table and adjudicated Trusted** on NB07 Step 14's pre-registered gold checks: overlapping gold-`b` events 98.1% cooperative (bar 90); `+`-continuation floor retention across intervening talk 73.2% (bar 70) — the latter doubling as the promised gold check of `fto.py`'s turn-merging. `obstructive_overlap_share` feeds the involvement panel (Step 16).
- ✅ **(b) Laughter counter** (map #17, trivial, dim 9 currently empty). _(Done: `laughter.py` — see §3.5; note it is built but not yet merged into the analysis tables.)_
- ⬜ **(c) Filled-pause vs discourse-marker split** (map #2 — the two halves of your current filler rate have *opposite* theoretical signs and are cancelling). _(`filler_word_rate.py` still one bucket.)_
- ⬜ **(d) Voice quality** — jitter/shimmer/HNR/H1–H2 via `praat-parselmouth` or openSMILE eGeMAPS (both open-source; dim 2d has zero columns).
- ⬜ **(e) Marked-shift dynamics** — pitch slope per syllable, reset counts, contour entropy instead of static moments (your map's own closing argument); validate against the NXT accent/break gold subset. _(`pitch.py` still static moments only.)_
- ⬜ **(f) Speaker-level aggregates** (map #29–33: silence ratio, overlap-initiation rate, persistence, narrative share, humor rate) — cheap groupbys once FTO and laughter exist. _(FTO + laughter now exist, so this is unblocked.)_
- 🟡 **(g) The Ch.5 narrative block** (#25–28) last — heaviest lift, save for after the distributional paper.
  ↳ **Status:** 🟡 The quotation / constructed-dialogue slice is now measured (2026-07-09, NB07 Step 17, T9): gold `^q`/`(^q)` rate (1,280 timed events) is a gold-involvement-panel variable — Tannen's signature narrative-involvement device, human-annotated. The full #25–28 feature block (story detection, narrative share, etc.) remains unbuilt.

### F. External validity

18. ⬜ **Fisher** (same genre, ~10× larger) to replicate unimodality; **CallHome/CallFriend** for the critical contrast — *familiar* dyads. The paper itself conceded strangers may suppress involvement; if intimates shift toward HI but the distribution stays unimodal, the continuum is robust *and* you've explained the suppression; if bimodality appears with intimates, you've found a real boundary condition. Either outcome strengthens the work. **CANDOR** (free for research) adds modern data with post-conversation outcome surveys — a replication target for the mismatch→quality analysis.

## 5. Suggested order of work

1. 🟡 **Fixes** (days): FTO turn gap; one canonical table; backchannel classifier validated on NXT dialAct; caller dedup; laughter counter; filled-pause split.
   ↳ FTO ✅ built / ✅ wired in NB06 · canonical table 🟡 (builder ✅ built + CLI-wired; **cutover begun — NB07 reads the loader**; §4A battery + `merge_test` retirement + trust walkthrough remain) · caller dedup ✅ · laughter ✅ built / ⬜ not merged · backchannel-vs-NXT ⬜ · filled-pause split ⬜.
2. 🟡 **The unimodality battery** (1–2 weeks): dip + Silverman + bootstrap-LRT + skew-fit comparison + multivariate clusterability + recovery simulation + multiverse grid. This is the defensible core of the "continuum" paper.
   ↳ dip ✅ + parallel analysis ✅ (NB06) · Silverman / bootstrap-LRT / skew-fit 🟡 (`stats_modality.py` built + tested 7/9; runs land in NB07 Step 18) · multivariate / recovery / multiverse ⬜ · taxometrics ⬜ (reinstated 7/9 → T12).
3. ⬜ **The positive story** (2–4 weeks): ICC trait stability; CFA of the involvement construct; mixed-model demographics; accommodation; mismatch → rating_tab quality.
   ↳ Nothing started; this is the largest untouched block and the audit's highest-value avenues (§4B10, §4C11, §4D15).
4. 🟡 **Feature expansion** (ongoing, dimension-at-a-time per your preference): overlap split, voice quality, marked shifts, speaker aggregates.
   ↳ laughter ✅ · **overlap split ✅ (T6, in the table + Trusted)** · voice quality / marked shifts / speaker aggregates ⬜ (the last is now unblocked).
5. ⬜ **Replication** (later): Fisher / CallHome / CANDOR.

One framing note for the eventual write-up: be careful to claim **unimodal/dimensional**, not "normal." Normality is neither necessary (a skewed continuum is still a continuum) nor sufficient (two heavily overlapping latent types can produce a unimodal, even Gaussian-looking, observed distribution — which is exactly why the taxometrics and reliability analyses in §A6 and §B10 are what make the claim airtight, not the histograms). Framed that way — *dimensional structure, stable individual differences, dyadic accommodation, and mismatch costs* — the unimodal result isn't the boring outcome; it's the corrected foundation Tannen's continuum language always implied.

---

## Pipeline Plan 7/2/2026

> **Goal:** every marker in §3 — items 1–5 **and** the minor line — reads ✅ per the legend
> ("implemented *and in use*"). Written 2026-07-02 after verifying the tree cell-by-cell and
> file-by-file; where a fact below contradicts a §3 status line above, **this section is current**.
>
> **Executor contract.** This plan is self-contained for a fresh model with no other context.
> Paths are repo-relative to `/Users/dainbrownlow/switchboard`. `analysis/07_final.ipynb`
> (**NB07**, 27 cells) is the **only living notebook** — every analysis-side change lands there
> as appended Steps. NB00–06, `utterances_v2/merge_test.csv`, `utterances_v2/paper_aligned_*`,
> and the three `scripts/*.py` builders are **frozen replication artifacts**: never edit their
> behavior, never re-point them (docstring banners are the one allowed edit). Follow
> `docs/PIPELINE.md` for the change loop, and the project `CLAUDE.md` for audit bookkeeping:
> updating the §3 status lines, the dashboard, and `docs/FEATURES.md` **is part of each task**,
> not a wrap-up chore.

### P0. Ground truth, verified against the tree 2026-07-02

Facts the executor should *not* re-derive (and stale claims they supersede):

- **The canonical table is built, fresh, and complete.** `utterances_v2/derived/features_table.csv`
  (2026-06-29, 214,204 rows × 56 cols) already contains every §3 fix's columns: `FTO Sec` + 4
  helpers (`Onset Gap Sec`, `Turn Initial/Backchannel/Interjection Flag`), the 5 laughter columns,
  `Personal Hits`/`Impersonal Hits`/`Analyzed Tokens` (+ the degenerate per-utterance
  `Personal Focus Score`), `Machine Gun Question Score/Flag` (per-side pitch baseline),
  `Rising Terminal Flag`/`Terminal F0 Slope` (missingness preserved), `Topic Label`, and the
  legacy `Turn Gap` (Deprecated bucket). **Nothing in §3 needs "merging" anymore — every
  remaining gap is trust adjudication or analysis use.**
- **Registry & loader are live.** `docs/FEATURES.md` = 17 Trusted / 33 WIP / 4 Deprecated
  (initial buckets, unconfirmed); `src/swb_extract/registry.py` parses it strictly;
  `src/swb_extract/analysis.py::load_features_table` enforces the stale-data + registry guards
  on every load. Promotion mechanics: **move a row between sections in `docs/FEATURES.md`** —
  that is the whole workflow.
- **NB07 state.** Loads via `load_features_table(include="provisional")` (cell 3); masks FTO to
  [−2, +2] s; rolls sides→callers via `../tables/call_con_tab.csv` (so NB07 must execute with
  cwd = `analysis/`); 11-feature paper-aligned volume set = `token_count`, `loudness mean/std/range`,
  `FTO Sec`, `word_rate`, `syllable_rate`, `Pronouns per Second`, `Repetitions In Current/Previous
  Utterance`, `Filler Words per Second`. Steps 1–10 + a final Conclusion cell (reconciled
  2026-06-30: K=2, PC1 42.5%, dip p .994/.993, ΔBIC(2−1)=+11, gender+education robust,
  region/generation null). **Three of its eleven live features are WIP** (the two repetition
  counters and the filler rate) — the trust walkthrough is not optional hygiene; NB07's headline
  numbers ride on it.
- **Repetition "SUSPECT max 109 > max token_count 81" is probably by design, not a bug.**
  `repetitions_in_current.py` counts **token pairs**: Σ over words of C(n_w, 2) (docstring:
  legacy `FERepeats` semantics with corrected tokenization). 15 repeats of one word = 105 pairs
  from 15 tokens. The walkthrough item is *confirm and document the quadratic tail*, not "fix."
- **Flags already serialize as ints in every `features/*.csv`** (verified: latching, question,
  rising_terminal, machine_gun, mutual_revelation, fto, laughter). The "merged flags serialize
  as floats" minor item is a `merge_test.csv` artifact (pandas merge) plus pandas' NaN→float64
  promotion at load time — closed by retirement (C2) + loader dtype coercion (C3), no
  re-extraction needed.
- **MGQ's 49.7% null is documented semantics, not obviously a broken gate**: the composite is
  *empty by design* for backchannel rows (~25% of utterances) and when the intent gate is
  indeterminate (syntactic no + prosody unmeasurable; `Rising Terminal Flag` is 30.3% null).
  The walkthrough item is to *decompose and reconcile* the 49.7%, and only fix if it doesn't
  reconcile.
- **CLI + env.** `swb-extract` is installed (editable, framework Python 3.14, on PATH).
  Subcommands used here: `swb-extract features <name> [--transcript-root …]` and
  `swb-extract table`. Re-run costs: transcript-based and CSV-composite extractors are cheap
  (minutes; `machine_gun_question` is a pure composite over sibling CSVs); **audio extractors
  (`pitch`, `loudness`, `rising_terminal`) are hours — do not re-run unless a verdict demands it.**
  Table rebuild ≈ 3 s. `syllable_rate` self-downloads NLTK `cmudict` (SSL workaround built in).
- **NXT gold is on disk, unparsed.** `corpus/nxt_switchboard_ann/xml/dialAct/` = 1,284 side
  files (642 conversations), named `sw####.{A,B}.dialAct.xml`; sibling layers `terminals/`,
  `phonwords/`, etc. No parser exists anywhere in `src/`/`scripts/`/`tests/` (only a docstring
  mention in `backchannels.py`).
- **Per-utterance times for alignment exist**: `swb_extract.transcripts.parse_transcript` yields
  ms98 utterance start/end (the same source `fto.py` builds word-tight turns from); transcripts
  live at `swb_ms98_transcriptions_cleaned/NN/####/sw####X-ms98-a-trans.text`; manifest keys are
  `{call_id//10:03d}/sw{call:04d}{side}-U{utt:04d}.wav`.
- **The entire Jun-29/30 reorg is uncommitted** on branch `audit-fixes-interactional-extractors`
  (untracked: `analysis/07_final.ipynb`, `docs/FEATURES.md`, `docs/PIPELINE.md`,
  `src/swb_extract/analysis.py`, `src/swb_extract/registry.py`, `CLAUDE.md`; modified:
  `docs/AUDIT.md`, `src/swb_extract/features_table.py`). Propose commit checkpoints to Dain as
  workstreams land — losing this tree loses the cutover.

### P1. The change loop (run for every pipeline edit)

1. Edit extractor + its unit test → `swb-extract features <name>` (regenerates only that CSV).
   **Batch all extractor fixes in a workstream before proceeding** to avoid repeated reruns.
2. `swb-extract table` (zip-merge with per-row key assertions — the corruption guard; never
   hand-edit `features_table.csv` or any `features/*.csv`).
3. Re-run NB07 **top-to-bottom**: `cd analysis && jupyter nbconvert --to notebook --execute
   --inplace 07_final.ipynb` (cwd matters for `../tables/…`). The loader's stale guard makes a
   skipped step 2 a hard error, not silent wrong numbers.
4. Reconcile: if any headline number moved (K, dip p's, ΔBIC, the FDR table), say so explicitly
   in the Conclusion cell's reconciliation block, with the cause.
5. Bookkeeping: move `docs/FEATURES.md` rows + update its header counts/date; rewrite the
   touched §3 `↳ Status:` line(s); adjust the dashboard.

**NB07 editing convention** (fixes the ambiguous "appended below this point" in the current
Conclusion): new work = numbered **Steps 11, 12, …**, each a markdown header cell + code cell(s)
ending in a printed one-line verdict, inserted **before** the final Conclusion cell — Step 10
("Recorded summary") and the Conclusion stay last, and the Conclusion is updated at every
landing. Use NotebookEdit (or equivalent); update the Conclusion's "appended below this point"
sentence to state this convention the first time you touch it.

### P2. Workstream A — §3.3 to ✅ (degenerate features, the *use* side)

The extractor halves are done (gated personal-focus ingredients; missingness-preserving rising
terminal). What's missing is sanctioned **consumption** in NB07 and the kill switch on the one
remaining downstream conflation.

- **A1 — NB07 Step 11: personal focus, pool-then-ratio at caller level.** Over each caller's
  *substantive* utterances (reuse the notebook's `sub`): `PF_ratio = Σ Personal Hits /
  (Σ Personal Hits + Σ Impersonal Hits)` and `PF_density = (Σ Personal Hits + Σ Impersonal Hits)
  / Σ Analyzed Tokens`. Print: callers with denominator ≥ {10, 30, 100} hits (sensitivity line),
  null%, and the distribution (histogram + skew) — acceptance is *non-degenerate*: null% ≪ 71.5%
  and no 0/1 saturation spike. Contrast in one line with `mean(per-utterance Personal Focus
  Score)` to record *why* pooling (that mean is the degenerate estimator). Registry: promote
  `Personal Hits`/`Impersonal Hits`/`Analyzed Tokens` → Trusted (note: "raw ingredients — pool
  at speaker level, NB07 Step 11"); move per-utterance `Personal Focus Score` → **Deprecated**
  ("degenerate per-utterance ratio — 71.5% null, saturates 0/1 on short utterances; pool the raw
  hit columns instead; audit §3.3"). After the move, `include="provisional"` stops loading it —
  confirm NB07 never references that column (it currently doesn't).
- **A2 — NB07 Step 12: rising-terminal missingness audit.** Print: null% of `Rising Terminal
  Flag` (expect ≈30.3%) decomposed by cause (short/unvoiced tails — MNAR **by design**, document
  it); % rising among defined; `Terminal F0 Slope` tail check with a chosen, recorded clip bound
  (FEATURES.md flags ±2000 Hz/s outliers — pick e.g. ±1000 Hz/s, justify in a comment).
  **Assert-in-notebook that no live code path fills NaN→0**: a `grep`-style check (or a plain
  statement backed by the C2 banners) that the only `fillna(0)` on Rising Terminal in the repo
  is inside frozen `scripts/build_merge_test.py`. Any question/MGQ composite ever rebuilt in
  NB07 must treat NaN as *indeterminate*, exactly like `machine_gun_question.py`'s intent gate.
  Promotion of the two rows happens in the C1 walkthrough with these results as evidence.
- **A3 — banner the culprit** (part of C2, listed here because §3.3's status names it): prepend
  to `scripts/build_merge_test.py`'s docstring the FROZEN banner (text in C2) naming its two
  deliberately-preserved bugs — `fillna(0)` on Rising Terminal (line ~102, §3.3) and the
  population-median machine-gun pitch composite (§3.4). **Do not fix the arithmetic** — its
  output must stay reproducible for NB01–06.

**Flip §3.3 → ✅** with a status line stating: extractors fixed (hits emitted + gated;
missingness preserved), sanctioned caller-level pooling live in NB07 Step 11, missingness audit
in Step 12, and the sole remaining NaN→0 conflation confined to the bannered frozen builder.

### P3. Workstream B — §3.5 to ✅ (laughter analyzed)

- **NB07 Step 13: caller-level laughter.** Compute per caller (over *all* utterances, not just
  substantive — laughter often rides backchannel turns; state the choice): `laughs_per_100utt`,
  `Σ Laughter Count / Σ token_count`, and laughed-word share. Print: corpus incidence (% of
  utterances with `Laughter Count ≥ 1` — record the number; SWBD laughter is common, so ~0% means
  something is wrong), caller-level spread, and correlations with PC1/PC2 and `bc_rate` (a first
  look at dim-9 vs the volume axes — laughter is *the* involvement signal Tannen's dim 9 was
  waiting on). **Spot-check ~20 nonzero rows**: the count must equal bracket occurrences in that
  row's `Transcript` (brackets are preserved in the manifest text; fall back to the raw ms98 line
  if not). Registry: promote all 5 laughter columns → Trusted if clean (unit tests already exist:
  `tests/test_laughter.py`).
- Note in the step's markdown that this unblocks §4E-f (speaker-level aggregates: FTO + laughter
  now both exist) — do not build §4E here.

**Flip §3.5 → ✅**: "counted (`laughter.py`), merged (Jun-29 table), registered, and analyzed at
caller level in NB07 Step 13."

### P4. Workstream C — §3.2 to ✅ (trust walkthrough + retirement + loader hygiene)

§3.2's own residue list: the per-feature trust walkthrough, `merge_test` retirement, and the
tannen-track fold. The §4A-battery-on-the-loader concern is structural now: the battery will be
*born* in NB07, which can only read through the loader.

- **C1 — the trust walkthrough: adjudicate all 33 WIP rows.** Each row ends **Trusted** (with an
  evidence note) or **Deprecated** (with its replacement named). Allowed rump: a row may stay WIP
  **only** if its blocker is a named §4 item and it is not consumed by NB07's analysis — only
  `Question Flag`/`Echo Question Flag`/`MGQ ×2` plausibly qualify (gate = §4C12). Work in tiers:

  | Tier / columns | Check | Decision rule |
  |---|---|---|
  | **1 — NB07's live features.** `Repetitions In Current Utterance`, `Repetitions In Previous Utterance`, `Repetitions per Second`, `Filler Words per Second` | Pull the argmax utterance(s) for both counters from the table and confirm massed literal repeats in the transcript (pair-count C(n,2) semantics, P0); add a boundary unit test pinning C(n,2); confirm `Repetitions per Second` shares `tokenize` + numerator; confirm the 0.9% null on "…Previous" = first-utterance-of-side; verify the filler allowlist against its docstring | Promote all four unless transcript inspection contradicts. Notes must record: quadratic tail ("consider log1p at analysis time" — the winsorize+log1p gap is §2.8, not §3) and "filled-pause vs discourse-marker unsplit, opposite signs cancel (§4E-c)" — a construct limitation, not a correctness bug |
  | **2 — FTO companions.** `Onset Gap Sec`, `Turn Initial Flag`, `Backchannel Flag`, `Interjection Flag` | Ride `tests/test_fto.py` (12 tests); decompose `Onset Gap`'s 25.4% null per `fto.py` docstring (conversation-initial etc.); crosstab: `Backchannel Flag` rate ≈ NB07's allowlist bc rate (~25%); `FTO Sec` defined ⟺ `Turn Initial Flag`=1 | Promote with "lexical-allowlist heuristic — NXT gold P/R pending (§4C12)" on the flags |
  | **3 — columns Steps 11–13 consume.** 5 laughter, `Personal/Impersonal Hits`, `Analyzed Tokens`, `Personal Focus Score`, `Mutual Revelation Flag`, `Topic Label` | Laughter → Step 13 evidence; personal-focus → Step 11 evidence (+ spot-check ~10 utterances' hits against the Empath category); `Mutual Revelation`: spot-check ~10 positives for face validity + per-side rate distribution; `Topic Label`: nonnull%, exactly one label per conversation, values ⊆ `tables/topic_tab.csv` | Promote; `Personal Focus Score` → Deprecated (A1). Topic's analytic *use* (§4D16/§2.6) is not a §3 gate — promotion here is join-validity only |
  | **4 — remaining interactional.** `Latching Flag`; `Overlap Duration Sec/Count/Onset Flag`; `Within Pause ×4`; `Rising Terminal Flag`, `Terminal F0 Slope`; `MGQ Score/Flag`; `Question Flag`, `Echo Question Flag` | Latching: read `latching_flag.py`, confirm Latching=1 ⇒ FTO defined & small; 17.8% latched is literature-plausible. Overlap: incidence coheres with Step 1b's 37.7% negative-FTO; spot-check 5 overlapping pairs' word timings. Within-pause: Count=0 ⇒ Total=0, Max consistent; zero-inflation documented. Rising terminal: Step 12 evidence + recorded clip bound. MGQ: **decompose the 49.7% null** into backchannel / indeterminate-intent / defined and reconcile the arithmetic (P0); only if it does *not* reconcile, revisit gates (`MAX_TOKENS`, `MAX_ONSET_SEC`, `HIGH_PITCH_PCTL=75`, `MIN_SIDE_PITCH_N=10`) — cheap CSV-composite rerun. Question/Echo: **gated on D3** | Promote with evidence notes; MGQ promotes only after its ingredients (Question, Rising Terminal, pitch) are adjudicated, since intent = their OR; Question/Echo promote if D3 shows high precision and an *understood* recall deficit ("syntactic questions only; declarative-question recall → §4C12"), else stay WIP with a dated adjudication note |

  **Capstone self-check** (add to Step 10's recorded-summary cell): print the provisional
  columns the analysis actually consumes — `feature_cols` plus every Step-11/12/13 ingredient —
  and require the list be **empty** (validation-only reads like Step 14's are exempt and printed
  separately). Keep `include="provisional"` in cell 3 — it is the documented living-notebook
  default and the loader's warning shrinking to nothing is the visible progress signal.
- **C2 — retire the merge_test track.** (a) FROZEN banners on `scripts/build_merge_test.py`,
  `scripts/merge_features.py`, `scripts/build_tannen_features.py`: *"FROZEN — replication-only
  (AUDIT.md §3.2 cutover). Superseded by `swb-extract table` → `utterances_v2/derived/
  features_table.csv` + the `docs/FEATURES.md` registry. Known, deliberately-preserved bugs:
  fillna(0) on Rising Terminal (§3.3) and population-median machine-gun pitch (§3.4)
  [build_merge_test.py]. Do not run for new analysis; do not fix — outputs must stay reproducible
  for NB01–06."* (b) `docs/PIPELINE.md`: "trustworthy line (notebook 06 onward)" → "notebook 07,
  the living notebook"; the Frozen layer's readers "replication notebooks (01/02/03)" → "frozen
  experiment notebooks (01–06)"; extend the closing "Replication tier" section likewise. Fix
  NB07's Conclusion where it quotes "notebook 06 onward". (c) Verify by grep that nothing outside
  NB01–06 *reads* `merge_test`/`paper_aligned` (NB07's prose mentions are fine). (d) The frozen
  CSVs **stay on disk untouched** — NB01–06 read them by relative path. This also completes the
  tannen-track fold: its three features are canonical-table columns (family=tannen) and its
  builder is bannered.
- **C3 — loader flag-dtype hygiene** (closes the float-flags minor item at the analysis surface):
  in `load_features_table`, after `read_csv`, coerce every loaded column ending in `" Flag"` to
  pandas nullable `"Int64"`; extract it as a pure helper (e.g. `_coerce_flag_dtypes(df)`) and add
  `tests/test_analysis.py` covering the helper plus `registry._parse_doc`,
  `registry.validate_against_table`, and `assert_table_fresh` (tmp-path fixtures; the guards have
  no tests today). Document in the module docstring. NB07's `feature_cols` contain no flags —
  a rerun confirms nothing shifts.

**Flip §3.2 → ✅** when: walkthrough complete under the C1 rule (target end state ≈ 49 Trusted /
5 Deprecated / 0 WIP of 54 registered, or the explicitly-noted Question/MGQ rump); Step-10
self-check prints empty; retirement docs/banners landed; table fresh. Rewrite the (very long)
§3.2 status block as a short final-state statement + one-line history pointer.

### P5. Workstream D — the §3 minor line to ✅

- **D1 — declare the real dependencies** in `pyproject.toml`: `dependencies = ["empath>=0.89",
  "spacy>=3.8", "librosa", "numpy", "nltk", "textstat", "pandas"]` (verified import sites:
  librosa+numpy in `loudness`/`pitch`/`rising_terminal`, nltk+textstat in `syllable_rate`,
  pandas in `analysis.py`); add `[project.optional-dependencies] analysis = ["scikit-learn",
  "scipy", "seaborn", "matplotlib", "diptest"]` (NB07 cell-1 imports). Keep `dev = ["pytest>=7"]`.
  Acceptance: `pip install -e '.[dev,analysis]'` resolves in the working env (packages are
  already present; this is declarative correctness for fresh envs). Note in `docs/PIPELINE.md`
  that `cmudict` self-downloads.
- **D2 — float flags**: closed by P0's verification (ints at source) + C2 (the floats live in
  frozen `merge_test.csv`) + C3 (nullable `Int64` at load). The audit line should say exactly that.
- **D3 — validate `Question Flag` against NXT gold** (the minor line's own ask; the *classifier
  improvement* stays §4C12). Two deliverables:
  1. `src/swb_extract/nxt.py` + `tests/test_nxt.py` (fixture = a truncated real dialAct file):
     parse `corpus/nxt_switchboard_ann/xml/dialAct/sw####.{A,B}.dialAct.xml` into records
     `(conv, side, swbd_tag, start_s, end_s)`. **Open one file first** and confirm whether `da`
     elements carry `nite:start`/`nite:end` directly; if not, resolve spans via their child
     links into `terminals/` (terminals carry times). Write it as shared infrastructure — the
     same layer holds the gold backchannel tags §4C12 needs next.
  2. **NB07 Step 14**: intersect the 642 NXT conversations with the manifest; per side, get our
     utterance spans from `swb_extract.transcripts.parse_transcript` (P0 gives path layout and
     key reconstruction); assign each gold DA to the same-side utterance with maximal time
     overlap (require ≥50% of the shorter span; print the match rate — expect >90%, investigate
     below that). **Print the full `swbd_tag` inventory with counts before hardcoding anything**,
     then define the gold question set as the q-family tags (`qy`, `qw`, `qo`, `qh`, `qrr`, `qr`)
     *including* `^d`/`^g` variants as they actually appear in the inventory. Report: gold
     question rate vs our 3.26% `Question Flag` rate; precision/recall/F1 on matched utterances;
     miss decomposition by gold tag — explicitly test the audit's hypothesis that **declarative
     questions dominate the misses**, and print the recall of (`Question Flag` OR
     `Rising Terminal Flag`) to show how much the fixed prosody feature recovers (§3.3 ↔ §4C12
     bridge). Update the FEATURES.md Question/Echo notes with the measured P/R and adjudicate
     per C1 Tier 4.

  The minor item flips ✅ as "validated against gold, rate gap measured and explained"; if the
  verdict is "recall deficit real, classifier needed," that work item stays where it already
  lives (§4C12) — §3 does not absorb it.

### P6. Sequencing

1. **D1** (minutes, independent).
2. **C1 Tiers 1–2** (read-only diagnosis on the current table; expected outcome: *zero extractor
   changes* — pair counts are by design, FTO helpers are tested). Add the Step-10 self-check
   line while in the notebook. If a genuine extractor bug surfaces, batch its fix per P1.
3. **Steps 11–13** (A1, A2, B) + their registry moves; one NB07 rerun + reconciliation per
   landing (or one combined rerun if landed together).
4. **`nxt.py` + tests → Step 14 (D3)** → Question/Echo adjudication. The heaviest single item.
5. **C1 Tiers 3–4 remainder** incl. the MGQ decomposition; any gate fix is a cheap composite
   rerun (`swb-extract features machine_gun_question && swb-extract table`).
6. **C2 banners + PIPELINE.md edits; C3 loader patch + `tests/test_analysis.py`.** Run `pytest`.
7. **Final pass:** full NB07 execution; Conclusion re-reconciled; `docs/FEATURES.md` counts +
   date bumped; every §3 status line rewritten to present-tense truth; dashboard recounted;
   **propose a commit checkpoint to Dain** (P0: the reorg is still uncommitted).

Estimated effort: D1 minutes · Tiers 1–2 an hour or two · Steps 11–13 an afternoon ·
nxt.py + Step 14 about a day · Tiers 3–4 + C2/C3 another half-day. No audio re-extraction
anywhere on the expected path.

### P7. Acceptance checklist ("green" means all of this is true)

- **§3.1 ✅** (already): every NB07 rerun re-prints Step 1b's FTO validation (median ≈ +0.140 s,
  ~37.7% overlap) — treat as a standing regression guard, not a one-time check.
- **§3.2 ✅**: 33 WIP rows adjudicated (0 WIP, or only the Question/Echo/MGQ rump with dated
  §4C12-gated notes); Step-10 provisional-consumption line prints empty; `merge_test`/
  `paper_aligned` readers = NB01–06 only; three builders bannered; PIPELINE.md names NB07 as the
  trustworthy line; table fresh under the stale guard.
- **§3.3 ✅**: Step 11 (pooled personal focus) and Step 12 (rising-terminal missingness audit)
  live; `Personal Focus Score` Deprecated; no NaN→0 on Rising Terminal outside the bannered
  frozen builder.
- **§3.4 ✅** (already): MGQ null decomposition recorded in its FEATURES.md note.
- **§3.5 ✅**: Step 13 live; 5 laughter rows Trusted.
- **Minor line ✅s**: deps declared (+ analysis extra); float-flags item closed with the
  ints-at-source note; `Question Flag` P/R vs NXT gold measured, recorded, and the
  3.26%-vs-~6–7% gap explained in NB07 Step 14.
- **Bookkeeping**: every touched §3 `↳ Status:` line rewritten; dashboard recounted; FEATURES.md
  header counts/date current; NB07 Conclusion reconciles every new step's numbers.

### P8. Do-not list

- Do not edit NB00–06, `merge_test.csv`, `paper_aligned_*`, or the *behavior* of the three
  frozen builder scripts (banners only). Do not delete anything, including `utterances_v2/_archive/`.
- Do not hand-edit `features_table.csv` or any `features/*.csv` — always regenerate
  (`swb-extract features <name>`, `swb-extract table`); the zip-merge key assertions are the
  §1 row-scramble defense.
- Do not re-run audio extractors (`pitch`, `loudness`, `rising_terminal`) without a walkthrough
  verdict that demands it.
- Do not change the FTO response window ([−2, +2] s) or the 38-token backchannel allowlist in
  this plan — their sensitivity analyses are §4A8/§4C12, and silent definition drift would
  unmoor the NB06↔NB07 reconciliation that currently anchors the cutover's faithfulness.
- Do not add or rename a table column without its FEATURES.md row (the registry guard will
  refuse the load — that is intended).

---

## Continuum Validation Plan 7/2/2026

> **Goal:** every §4A marker — items 1–8 — reads ✅. This is the "unimodality battery" of §5.2,
> the defensible core of the continuum paper. Written 2026-07-02 after verifying the tree and
> the Python environment; where a fact below contradicts a §4A status line above, this section
> is current.
>
> **Executor contract.** Self-contained for a fresh model, with one pointer: read **Pipeline
> Plan 7/2/2026 → P0 (ground truth) and P1 (change loop + NB07 editing conventions)** first;
> they are not repeated here. Everything lands in `analysis/07_final.ipynb` (NB07) + two new
> tested modules under `src/swb_extract/`. NB00–06 and the frozen CSVs stay untouched.
>
> **Integrity clause — read twice.** "Green" means each analysis is *implemented, validated on
> synthetic known-answer data, run on the real data, and recorded* — it does **not** mean
> "unimodality confirmed." The decision rules below are fixed in advance. If Silverman rejects
> at some level, if CCFI lands taxonic, if a multiverse cell goes bimodal — that gets recorded
> and investigated in place, never tuned away. A discordant result is a finding, not a bug.
> Likewise the write-up vocabulary: claim **unimodal / dimensional**, never "normal" (see the
> framing note at the end of §5).

### Q0. Ground truth + preconditions (verified 2026-07-02)

- **Environment is sufficient — zero new dependencies.** Framework Python 3.14 with:
  sklearn 1.8.0 (`FactorAnalysis(..., rotation="varimax")` available), scipy 1.17.1
  (`skewnorm`, **`jf_skew_t`** (Jones–Faddy skew-t), `lognorm`, `ndimage.gaussian_filter1d` all
  present), diptest 0.11.0, numpy 2.4.4, pandas 3.0.2. `statsmodels` is **not installed and not
  needed** here (mixed models are §4B9, out of scope).
- **No new feature columns** → no `docs/FEATURES.md` / registry changes anywhere in this plan.
  These are analyses over existing columns; the loader path is unchanged.
- **Current §4A state:** A1 partial — dip at **caller** level on the retained volume PCs
  (NB07: p = 0.994/0.993), missing the utterance/side levels and the interactional factor.
  A2–A8 ⬜ (NB06/NB07 keep a single GMM-BIC line, which is model selection, not A3's bootstrap
  LR null; PC1 skew is measured but no skewed alternative is fitted — A4).
- **Where the k=2 ghost actually lives.** At caller level BIC already prefers k=1 (ΔBIC +11).
  The **side level** is where BIC picked k=2 (ΔBIC −110, §2.2) — A3 and A4 must run *there*,
  not just at caller level, or the battery misses the one result a hostile reviewer will cite.
- **Units:** 214,204 utterances (≈75% substantive after the 25.2% backchannel strip) →
  4,876 sides → 493 mapped callers, **487 retained** (≥20 substantive utterances).
- **The paper's claimed structure** (A7's simulation target, from §4A7): PC1 mixture
  `0.55·N(−0.31, 0.13²) + 0.45·N(+0.36, 0.31²)`.
- **NB05's F3 signature** (Step 15's identification rule): overlap onset +0.87, overlap
  duration +0.79, turn gap −0.56 — with FTO replacing Turn Gap, expect the timing loading to
  flip context: FTO should load *negative* on an engagement factor (faster floor-taking).
- **FTO Sec is 64.5% null at utterance level by design** (floor transfers only). Utterance-level
  PCA therefore **cannot include it** without imputation (forbidden — Q5): use the 10-feature
  variant (vol11 minus FTO, documented) plus a turn-initial-subset sensitivity. Side/caller
  levels aggregate with NaN-skipping means and are unaffected.
- **Precondition:** Pipeline Plan 7/2/2026 executed at least through its sequencing step 2
  (Tier-1/2 trust adjudication — NB07's own eleven features plus the FTO helpers), ideally in
  full. This is §5's explicit ordering (fixes → battery) and §3.2's stated rationale: the
  airtight stats never run on unadjudicated columns. The Q1 modules + tests have **no data
  dependency** and can be built in parallel at any time; the NB07 steps land after.
- **Step numbering** below assumes Pipeline Plan Steps 11–14 exist. If they don't yet, take the
  next free step numbers and keep this order — the labels are nominal, the order is not.

### Q1. New modules — every algorithm tested before any battery number ships

Pure statistics goes in `src/swb_extract/` with pytest known-answer tests (repo pattern);
notebook cells stay analysis narrative. Non-negotiable: each test statistic must demonstrably
(a) detect planted structure and (b) pass a planted null, seeded, before touching real data.
A7's recovery arms then re-validate everything end-to-end at our exact n.

**`src/swb_extract/stats_modality.py`** (+ `tests/test_stats_modality.py`):

- `count_modes(x, h, grid_n=4096)` — Gaussian-KDE mode count via binned density: standardize,
  bin to a grid padded 3h beyond the range, density = `scipy.ndimage.gaussian_filter1d(counts,
  sigma=h/binwidth)`, count strict local maxima with density > 1e-6·max (float-plateau guard).
  O(n + grid) per bandwidth — this is what makes utterance-level bootstraps feasible. Unit
  tests: plateau and edge-mode cases.
- `h_crit(x, k=1)` — the smallest bandwidth at which the KDE has ≤ k modes. Valid because the
  Gaussian-kernel mode count is nonincreasing in h (Silverman 1981 — the theorem the test rides
  on; do not swap kernels). Binary search on [1e-3·σ̂, 2·range], ~40 iterations.
- `silverman_test(x, k=1, B=999, seed=0)` — critical-bandwidth bootstrap (§4A2): draw B
  smoothed-bootstrap samples from the KDE at h_crit with the variance-preserving rescale
  `y = x̄ + (x_J − x̄ + h_crit·ε) / sqrt(1 + h_crit²/σ̂²)`, ε~N(0,1); p = #{count_modes(y_b,
  h_crit) > k}/B. Returns (h_crit, p). Report the known conservatism (Hall & York 2001) when
  citing; a `calibrate=` flag is optional polish, not a blocker. Tests: N(0,1) n=500 → p
  comfortably > 0.1; `0.5·N(−2,0.5²)+0.5·N(+2,0.5²)` → p < 0.01 (seeded).
- `gmm_blrt(x, k0=1, k1=2, B=999, n_init=10, seed=0)` — McLachlan (1987) parametric bootstrap
  LRT (§4A3), the direct, correct answer to "but BIC picked k=2": fit both k with
  `GaussianMixture` (n_init=10 against local optima), `LR_obs = 2·n·(score_k1 − score_k0)`
  (sklearn's `score` is the *mean* per-sample log-lik — multiply by n); simulate B datasets from
  the fitted k0, refit both on each, `p = (1 + #{LR_b ≥ LR_obs}) / (B + 1)`. Tests: one normal →
  p spread over (0,1]; well-separated mixture → p = 1/(B+1).
- `fit_family_bic(x)` — §4A4, the single highest-leverage analysis (it formally resolves §2.2):
  MLE via scipy `.fit()` for **norm** (2 params), **skewnorm** (3), **lognorm with free loc**
  (3), **jf_skew_t** (4); plus `GaussianMixture(2)` (5 params in 1-D; use `gmm.bic`) and
  GMM(1) ≡ norm as a cross-check. ll = Σ logpdf with −inf guards (lognorm/jf support edges);
  BIC = p·ln n − 2·ll; return the sorted table. Tests: n=2000 draws of skewnorm(a=4) → skewnorm
  beats GMM(2) on BIC; a well-separated 2-Gaussian sample → GMM(2) wins.

**`src/swb_extract/taxometrics.py`** (+ `tests/test_taxometrics.py`) — §4A6, the centerpiece;
nobody in the conversational-style literature has applied taxometrics:

- `gen_data(target_corr, marginals, n, taxonic=False, base_rate=0.45, seed=0)` — Ruscio &
  Kaczetow (2008) comparison-data generator (their GenData pseudocode is explicit — follow it):
  iterate an intermediate correlation matrix, mapping ranks onto bootstrap-resampled empirical
  marginals until the reproduced correlations converge; the **taxonic** variant draws two
  classes at `base_rate`, within-class correlations ≈ 0, per-indicator class separations chosen
  to reproduce `target_corr`. Tests: reproduced correlations within ±0.05 elementwise of
  target; marginals KS-close to the empirical ones.
- `mambac(X, cut_buffer=25)` — Meehl & Yonce: for every **ordered** indicator pair (input,
  output): sort by input; for each cut at case ranks `cut_buffer..n−cut_buffer` compute
  `mean(output | above cut) − mean(output | below cut)`; return per-pair curves + grand mean.
  Taxonic → peaked curve; dimensional → concave dish rising at the ends.
- `maxeig(X, windows=50, overlap=0.9)` — per input indicator: overlapping windows along the
  sorted input; in each window, covariance matrix of the *other* indicators with the diagonal
  zeroed → largest eigenvalue (`np.linalg.eigvalsh`); taxonic → interior peak (hitmax);
  dimensional → flat/monotone.
- `lmode(X)` — `FactorAnalysis(1)` scores → factor-score density + dip (ties back to A1).
- `ccfi(observed_curve, dim_curves, tax_curves)` — **CCFI = RMSR_dim / (RMSR_dim + RMSR_tax)**
  against the comparison populations' mean curves (plot the 10–90% envelopes): **< 0.45 ⇒
  dimensional, > 0.55 ⇒ taxonic, between ⇒ ambiguous** (Ruscio). Compute per procedure
  (MAMBAC / MAXEIG / L-Mode) and report each plus the mean.
- **End-to-end known-answer tests** (these double as the A7 machinery validation):
  dimensional-generated data → CCFI < 0.45; taxonic-generated data at the paper's parameters →
  CCFI > 0.55; both seeded.

### Q2. NB07 steps

Conventions per Pipeline Plan P1 (numbered step = markdown header + code cell(s) ending in a
printed one-line verdict, inserted before the Conclusion; Conclusion reconciled every landing;
fixed seeds; print every B and a runtime line).

**Phase 0 — refactor Steps 2–3 in place** (prerequisite for the level variants and A8, not a
new step): extract two functions defined where Steps 2–3 live —
`build_unit_table(df, unit ∈ {"caller","side","utterance"}, bc_def, transform, min_utt,
feature_set)` and `run_pca(tbl, features)` — and have Steps 2–3 call them with the primary
specification (caller / allowlist38 / no transform / min 20 / vol11). **Regression gate:** the
printed headline numbers must be identical before anything else lands (487 callers, PC1 42.5%,
K=2, dip p 0.994/0.993, ΔBIC(2−1)=+11).

**Step 15 — interactional caller profile + F_int** (feeds A1's third axis and A5's full-space
matrix). Caller-level NaN-aware means over substantive utterances of the walkthrough-adjudicated
interactional columns — minimally: `FTO Sec` (in-window), `Overlap Duration Sec`, `Overlap
Count`, `Overlap Onset Flag`, `Latching Flag`, the four within-pause columns, `Question Flag`,
`Echo Question Flag`, `Rising Terminal Flag`, laughter rate (from Step 13), `Repetitions In
Previous Utterance`. Standardize; choose the factor count by Horn's parallel analysis (reuse
Step 4's machinery on this matrix); `FactorAnalysis(k, rotation="varimax")`. **Identify F_int**
as the factor matching NB05's F3 signature (strong + on overlap onset/duration, opposite-sign
timing/FTO); print the full loading table and variance shares. If **no** factor matches, that
is a real finding about the construct (record it; cross-ref §4C11) — the battery then runs on
the closest engagement-flavored factor, stated plainly. Scope fence: this step exists to give
the battery its interactional axis; the NB05 claim-reframes (gender–F3 §2.4, NYC §2.5/§4D17)
stay out of scope.

**Step 16 — the formal modality battery (closes A1 + A2 + A3 + A4).** Every cell of this
matrix appends a row to a shared `BATTERY` list `(axis, level, test, statistic, p-or-criterion,
verdict)`:

| axis | level | tests |
|---|---|---|
| PC1, PC2 (volume) | caller (n=487) | dip · Silverman(k=1, B=999) · BLRT(1v2, B=999) · fit-family BIC |
| PC1 (volume) | side (n=4,876) | dip · Silverman · BLRT · **fit-family BIC ← the ΔBIC −110 site** |
| PC1 (volume, 10-feature — FTO excluded, documented; + turn-initial sensitivity) | utterance (n≈160k) | dip · Silverman(B=199) · BLRT(B=199; print runtime, drop to B=99 with a printed note if >20 min) |
| F_int | caller (side optional) | dip · Silverman · BLRT · fit-family BIC |

Side- and utterance-level axes come from `build_unit_table(unit=…)` + `run_pca`. **The
side-level fit-family row is the vulnerability-closer:** if one skewed component (skewnorm /
jf_skew_t / lognorm) beats the 2-Gaussian mixture on BIC there, §2.2's ΔBIC −110 is formally
explained as skew-fitting rather than types — record the verdict in exactly those terms,
whichever way it goes. Report every p; **no FDR across the battery** (confirmatory per-test
reporting — state this in the step's markdown).

**Step 17 — multivariate clusterability (closes A5).** "No clusters in the full feature space"
is a much stronger claim than "no clusters on PC1." On two standardized caller×feature
matrices — vol11 and vol+interactional (Step 15's columns): (a) **dip on pairwise Euclidean
distances** (487²/2 ≈ 118k values; diptest handles it); (b) **Hopkins statistic** (m=50,
synthetic points uniform over the PCA-aligned bounding box) — do **not** interpret against the
0.5 folklore: correlated features inflate H, so calibrate with B=500 draws of N(0, Σ̂) at the
same n → empirical p (this *is* the audit's "proper null"); (c) **gap statistic** k=1..6
(KMeans n_init=10, B=100 reference draws from N(0, Σ̂)), decision rule: smallest k with
gap(k) ≥ gap(k+1) − s_{k+1} — expect it to choose k=1 if the continuum story is right. One
verdict line per matrix; BATTERY rows appended.

**Step 18 — taxometrics (closes A6).** Indicators: 4–6 standardized caller-level features
chosen by |PC1 loading|, sign-aligned positive, dropping one of any pair with |r| > 0.85
(near-duplicates degrade MAXEIG); print the indicator correlation matrix and a validity note.
Run `mambac` / `maxeig` / `lmode`; CCFI per procedure against B=100 dimensional + B=100 taxonic
`gen_data` comparison sets, with **base-rate sensitivity π ∈ {0.30, 0.45, 0.55}** (the paper
claimed 0.55/0.45; also report the GMM(2)-weight estimate). Produce the observed-curve-over-
comparison-envelopes figure — that is the publishable exhibit. Optional 3-line CSV export of
the indicator matrix for `RTaxometrics` as the citable cross-check (do not block on R).
Verdict per the Ruscio bands, per procedure + mean.

**Step 19 — recovery simulation / power (closes A7).** Converts the null into "we had the
power to see two styles; they are not there."
- **Arm T (the paper's taxon):** B=200 synthetic datasets at n=487 (and a side-level variant at
  n=4,876) drawn from the Q0 mixture; run dip, Silverman(B=199), BLRT(B=199), fit-family on
  each → per-test **detection rate**. Multivariate arm: taxonic `gen_data` 5-indicator sets at
  the paper's base rate → CCFI on a B=25 subset (runtime).
- **Arm D (matched skewed continuum):** skew-normal fitted to the observed caller PC1 → same
  battery → per-test **false-alarm rate**.
- Verdict sentence for the Conclusion: "at n=487 the battery detects the paper's claimed
  structure with power ≈X per test; on a matched skewed continuum it false-alarms at ≈Y."
  Budget: tens of minutes total; print per-arm timings; reduce inner B before reducing arms.

**Step 20 — multiverse / specification curve (closes A8; what "exhaustive" looks like to a
reviewer).** Full grid over `build_unit_table`/`run_pca`:
`bc_def ∈ {allowlist38, token≤2, allowlist∪token≤2, none}` (the DAMSL-trained-classifier
definition joins later — pending §4C12; say so in the cell) × `transform ∈ {none,
winsorize[1,99]+log1p}` (per-feature policy: log1p only for nonnegative skewed counts/rates,
never FTO — document; this also discharges §2.8's "skipped winsorize+log1p" inside a
sensitivity frame) × `unit ∈ {caller, side}` × `min_utt ∈ {10, 20, 40}` × `feature_set ∈
{vol11, vol11+pitch3, vol+interactional}` = **144 specifications**. Per spec record: dip p on
PC1, max KDE modes over bw {0.25, 0.40, 0.60}, sign of ΔBIC(2−1), skew. Deliverables: the
specification-curve figure (specs ranked by dip p, parameter panel beneath) and the headline
"X of 144 specifications yield unimodal PC1." **Any non-unimodal specs get enumerated and
diagnosed in place** — what they share is the interesting result, not an embarrassment.

**Step 21 — battery summary table + Conclusion block.** Render `BATTERY` as one tidy table —
this is simultaneously A1's required "single table" and the paper's methods table. Update the
Conclusion with a "Formal battery" block stating, per axis×level: dip / Silverman / BLRT /
fit-family verdicts, the multivariate and taxometric verdicts, power/false-alarm rates, and
the multiverse fraction — in the audit's closing-note vocabulary (**dimensional, not normal**;
what remains for "airtight" is the §4B10 reliability line, out of scope here).

### Q3. Sequencing

1. **Q1 modules + tests** — no data dependency; can start immediately, in parallel with the
   Pipeline Plan. Gate: `pytest` green including the known-answer tests.
2. **Phase-0 refactor** + its regression gate (identical headline numbers).
3. **Step 15** (needs the walkthrough-adjudicated interactional columns — the precondition).
4. **Steps 16 → 17** (pure consumption of Q1 + Phase 0; fast).
5. **Step 18**, then **19** (19 re-validates the whole stack at our n), then **20**.
6. **Step 21 + bookkeeping:** flip §4A1–A8 with rewritten present-tense status lines; update the
   cross-referencing soft spots — §2.1 (dip-only caveat → closed by Silverman+BLRT), §2.2
   (skew-fit now actually fitted, side level included) — and the §5.2 line; recount the
   dashboard; add a dated one-sentence addendum to the **Verdict** paragraph (its "single most
   important thing you have not yet done" is precisely this battery); propose a commit
   checkpoint to Dain (the repo's reorg is still uncommitted per Pipeline Plan P0).

Effort: Q1 ≈ 1–2 days (gen_data + taxometrics dominate) · Phase 0 + Steps 15–17 ≈ 1 day ·
Steps 18–20 ≈ 1.5 days · Step 21 + bookkeeping ≈ half a day. Fits §5.2's "1–2 weeks" with slack.

### Q4. Acceptance checklist ("green" means all of this is true)

- **A1 ✅**: dip rows for PC1/PC2/F_int at caller level **and** PC1 at side + utterance level,
  all in the single battery table (its literal ask).
- **A2 ✅**: Silverman implemented per Q1 (unit-tested, Gaussian kernel, variance-preserving
  smoothed bootstrap), run everywhere dip ran; conservatism noted where cited.
- **A3 ✅**: BLRT p-values at caller **and side** level with B printed — model selection (BIC)
  no longer stands in for a test anywhere in the doc.
- **A4 ✅**: fit-family BIC tables at caller and side level; the side-level
  skewed-vs-2-Gaussian verdict recorded in the Conclusion in "formally explained (or not) as
  skew-fitting" terms.
- **A5 ✅**: dip-on-distances + null-calibrated Hopkins + gap-vs-N(0,Σ̂), on both matrices.
- **A6 ✅**: MAMBAC + MAXEIG + L-Mode curves, per-procedure + mean CCFI with base-rate
  sensitivity, comparison-envelope figure; verdict per Ruscio bands.
- **A7 ✅**: per-test power at n=487 (and n=4,876) against the paper's parameters + false-alarm
  rates on the matched skewed continuum; the "we had the power" sentence in the Conclusion.
- **A8 ✅**: all 144 specifications run; the specification-curve figure; unimodal fraction
  headline; non-unimodal specs enumerated and diagnosed.
- **Modules**: pytest green (known-answer tests included); seeds fixed; **no pyproject changes**
  (environment verified sufficient 2026-07-02); **no FEATURES.md changes** (no new columns).
- **Bookkeeping**: §4A status lines rewritten; §2.1 / §2.2 / §5.2 cross-refs updated; Verdict
  addendum added; dashboard recounted; NB07 Conclusion carries the battery block and reconciles
  every new step.

### Q5. Do-not list

- Do not run any battery analysis on `merge_test.csv` / `paper_aligned_*` or inside NB00–06 —
  NB07-on-the-loader only (this is §3.2's whole point).
- Do not tune, re-seed, or re-specify a test because its answer is inconvenient — decision
  rules are fixed above; discordant results are recorded and investigated in place.
- Do not FDR-correct across the battery's modality tests, and do not write "normal" where the
  claim is unimodal/dimensional.
- Do not impute utterance-level FTO to force it into a PCA — its missingness is by design
  (§3.1); use the documented 10-feature variant + turn-initial sensitivity.
- Do not let Step 15 grow into the NB05 reframe — the gender–F3 test (§2.4) and the
  overlap-specific NYC rerun (§2.5/§4D17) are separate work items.
- Do not swap the Gaussian kernel in `h_crit`/Silverman (the monotone-mode-count theorem is
  kernel-specific), and do not add dependencies — `statsmodels` stays uninstalled; nothing in
  §4A needs it.
