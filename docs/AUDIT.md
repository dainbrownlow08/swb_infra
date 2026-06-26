# Audit: Switchboard conversational-styles repo

## Verdict

Your correction arc is sound, and the evidence behind it is stronger than your notebooks currently claim. The paper's "two styles" result fails at four independent levels — any one of which would invalidate it — and your backchannel diagnosis is the correct explanation for the one surviving bimodality. However, your continuum conclusion currently rests on **KDE peak-counting with hand-picked bandwidths**, and at speaker level **BIC actually selected k=2** (ΔBIC = −110) — you dismissed this by magnitude comparison, but a hostile reviewer will seize on it. The single most important thing you have not yet done is convert "I didn't find bimodality" into formal, positive statistical evidence for dimensionality. The good news: the tools for that exist, your corpus has untapped gold annotations sitting on disk that enable them, and there are several theory-driven analyses (trait stability, accommodation, style-mismatch → call quality) that turn the unimodal result from a disappointment into a foundation.

---

## 1. The paper and the legacy code: what the audit established

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

1. **No formal multimodality test anywhere.** Unimodality claims rest on KDE mode counts with ad-hoc prominence thresholds that drift between notebooks (1% in NB3/NB4, 5% in NB5).
2. **The speaker-level BIC picked k=2** (15,756 vs 15,866 for k=1). You argued ΔBIC −110 ≪ the backchannel split's −53,000, which is fair rhetoric but not a test. A 2-Gaussian fit beating 1 Gaussian is *expected* for any skewed unimodal distribution — but you never fit the one-component skewed alternative that would prove it.
3. **Pseudoreplication everywhere.** "Speaker" = conversation-side (`sw####A/B`), but `call_con_tab.csv` shows 4,876 sides come from only **543 callers (~9 calls each)**. Every demographic test (gender d = −0.330, NYC +0.83, education) treats ~9 dependent observations per person as independent. None of this is acknowledged in the notebooks.
4. **The gender–F3 claim** (one of NB5's three headline conclusions) is supported by group means only — no test, no effect size, unlike the NYC comparisons.
5. **The NYC-isn't-overlap finding is premature**, because it depends on Turn Gap and Latching, both of which are currently broken (see §3). The overlap features themselves are sound (word-alignment based), but F3 mixes them with the broken timing features.
6. **No topic control** — `topic_label.py` exists but is never joined; Switchboard topics are assigned, and topic is a plausible confound for both region and gender effects.
7. **No reliability accounting** — median 36 substantive utterances per side means speaker profiles are noisy; PC variance and all effect sizes are attenuated by an unquantified amount.
8. Smaller items: the 38-token backchannel allowlist is copy-pasted across three notebooks with no sensitivity check; NB4/NB5 skip the winsorize+log1p step NB3 Part C established; no parallel analysis for component retention; 17,053 pitch-null rows silently dropped; `tannen_features.csv` — the file named for the initiative — is never read by any notebook.

## 3. Pipeline: fix these before the next analysis round

The Jun 9 extractors are well-built (clean word-alignment infrastructure, every extractor unit-tested, no join fan-out, zero duplicate keys). Five things need fixing, in order:

1. **Turn Gap semantics.** It's computed against the chronological predecessor *including backchannels*, so 60.3% of gaps are negative and the median "gap" is −0.49 s. This poisons Latching (1.15% positives — near-constant) and the machine-gun composite (its "fast follow ≤0.5 s" criterion is satisfied by every negative gap). Redefine as **floor-transfer offset (FTO)**: merge consecutive same-speaker utterances into turns, exclude backchannel-only predecessors, and measure gap at genuine floor transfers (Heldner & Edlund 2010 is the standard reference — SWBD FTOs should center ~+200 ms, which would also situate your data in the turn-taking literature).
2. **One canonical table.** `tannen_features.csv` (May 9) has only topic/personal-focus/mutual-revelation; `merge_test.csv` (Jun 9, 39 cols) has everything else; and no script in the repo builds merge_test's original base columns. Write one version-controlled builder producing a single table.
3. **Degenerate features.** `personal_focus_score`: 54.7% NaN and 64.6% of defined values exactly 0.0 or 1.0 — Empath saturates on short utterances; aggregate to speaker level over content words, or replace with a pronoun+lexicon score over a minimum-token window. `rising_terminal`: 68.5% missing **not at random** (short/unvoiced tails), and `build_merge_test.py` fills NaN→0, conflating "unmeasurable" with "not rising."
4. **The machine-gun composite's pitch term** uses the whole-population median — i.e., it's substantially a female-speaker indicator. Use speaker-relative pitch (z within speaker).
5. **Laughter** (Tannen dim 9) is still stripped, not counted, by every extractor — your own feature map flagged the one-pass fix (count brackets, then strip). It remains the only dimension with raw signal on disk and zero columns.

(Minor: `librosa`, `textstat`, `nltk`, `pandas` still undeclared in `pyproject.toml`; merged flags serialize as floats; the question-flag rate of 3.26% is roughly half the ~6–7% question rate reported for SwDA — validate against gold, see below.)

## 4. Avenues you have not pursued

This is the core of your ask. Organized from "makes the current claim rigorous" to "builds the future program." Tools are open-source and Python-first per your constraints; R noted only where the canonical implementation is R-only.

### A. Make "unimodal continuum" a formal statistical claim

1. **Hartigan's dip test** (`pip install diptest`) on PC0/PC1/F3 at utterance, side, and caller level — the standard test of unimodality, replacing KDE eyeballing. Report dip statistic + bootstrap p for every axis in a single table.
2. **Silverman's bandwidth test** (critical-bandwidth bootstrap) as the complementary mode test — dip and Silverman together are the accepted pair.
3. **Parametric bootstrap likelihood-ratio test for k=1 vs k=2** (simulate from the fitted 1-component model, refit both, build the LR null — ~30 lines with sklearn; `mclustBootstrapLRT` in R if you want the citable canonical version). This is the direct, correct answer to "but BIC picked k=2."
4. **Fit a single skew-normal / skew-t / lognormal against the 2-Gaussian mixture** (scipy `skewnorm`/`skewt` MLE; `mixsmsn` in R for skewed mixtures). If one skewed component beats two Gaussians on BIC — very likely given PC0 skew +0.49 — the k=2 BIC result is formally explained as skew-fitting, not types. *This single analysis closes your biggest vulnerability.*
5. **Multivariate clusterability, not just 1-D projections.** Unimodal PCs don't preclude structure elsewhere in the 19-dim space. Run the dip test on pairwise distances, the Hopkins statistic, and the gap statistic against a proper null (PCA-shaped Gaussian reference). "No clusters in the full feature space" is a much stronger claim than "no clusters on PC0."
6. **Taxometric analysis — the centerpiece I'd recommend.** Types-vs-continuum is a solved methodological problem in psychopathology research: Meehl's MAXEIG/MAMBAC/L-Mode with the **Comparison Curve Fit Index** (CCFI < 0.45 ⇒ dimensional; `RTaxometrics` in R; MAMBAC is simple enough to hand-roll in Python). Nobody in the conversational-style literature has applied taxometrics. With 543 callers (or 3,595 sides) and your indicator set, this is feasible and would be a publishable methods contribution on its own: "Conversational style is dimensional, not taxonic."
7. **Recovery simulation (power analysis for types).** Simulate data with the paper's own claimed structure (means −0.31/+0.36, SDs 0.13/0.31, weights 0.55/0.45), push it through your corrected pipeline, show dip/BLRT/taxometrics all detect it. This converts your null into "we had the power to see two styles; they are not there."
8. **Multiverse / specification-curve analysis.** Your conclusion currently depends on one backchannel definition (38-token allowlist), one transform (winsorize+log1p), one aggregation (mean over ≥20 utterances). Run the grid — backchannel definition (allowlist / token-count threshold / DAMSL-trained classifier), scaling, feature subsets, with/without pitch, level — and report the fraction of specifications yielding unimodality. This is what "exhaustive" looks like to a reviewer.

### B. Fix the unit of analysis; test style as a *trait*

9. **Caller-level deduplication** via `call_con_tab.csv` → caller_no. Rerun the distributional battery at true speaker level (543 callers). All demographic claims need re-estimation as **mixed-effects models**: `feature ~ sex + region + generation + education + topic + (1|caller)`, with FDR correction (statsmodels `MixedLM` / `pymer4`). This will also properly adjudicate NYC-vs-rest (currently confoundable with gender/generation composition) and give the gender–F3 claim its missing test.
10. **ICC / split-half trait stability — the highest-value unpursued analysis in the repo.** Each caller has ~9 conversations: compute ICC of style scores across a caller's calls, and split-half (odd/even utterances) reliability within calls. If style position is stable across conversations with different strangers on different topics, you have *positive* evidence for style as an individual-difference trait distributed along a continuum — exactly the foundation Tannen's continuum reading needs. Low ICC is equally informative (style is situational/dyadic). Reliability estimates also let you disattenuate every effect size you report.

### C. Construct validity — is there an "involvement" axis at all?

11. **Confirmatory factor analysis** (`semopy`). Tannen predicts the HI markers covary: short FTOs, latching, overlap, speed, loudness, questions, personal focus. Your NB5 varimax already shows volume, loudness, rate, and interaction splitting into *separate* factors — i.e., a single HI–HC axis may not exist as a unidimensional construct. Test a hierarchical model (general involvement factor over the four group factors) vs orthogonal factors and report fit. Either result reshapes the paper: a continuum needs an axis, and right now your best candidate is F3, not PC0.
12. **Validate features against the gold annotations already on disk.** `corpus/nxt_switchboard_ann/xml/` contains NXT Switchboard: **dialog acts for 642 conversations (1,284 sides) including gold backchannel tags**, gold disfluency annotation, and accent/break (prosody) layers for a subset (150 accent files; 451 accents/breaks files in `corpus/annotated_files/`). Use it to (a) measure precision/recall of your 38-token backchannel allowlist, then train a small classifier and apply corpus-wide; (b) validate Question Flag (your 3.26% vs ~6–7% expected — likely missing declarative questions, which rising-terminal should catch once its missingness is fixed); (c) split repetition into **disfluent repair vs rhetorical repetition** — Tannen means the latter, and your current counters conflate them with disfluency, which is a real confound for the "repetition = involvement" reading.
13. **Perceptual anchoring.** Tannen's construct is ultimately a hearer judgment. Have 2–3 raters (lab members) rate a stratified sample of ~100 conversation excerpts on involvement (and naturalness); correlate with your composite scores. Without this, the continuum is a continuum of *something measured*; with it, it's a continuum of *perceived style*.

### D. Interactional and dyadic analyses — where Tannen's theory actually lives

14. **Accommodation / entrainment.** Tannen's framework is about what happens *between* speakers. Levitan & Hirschberg's proximity/convergence/synchrony measures were developed on precisely this kind of data — compute whether partners converge on rate, loudness, FTO, overlap over the call, and whether B's style position depends on A's. "Style is a continuum along which speakers *move toward their partners*" is a positive, theory-rich result.
15. **Style mismatch → conversation quality.** `tables/rating_tab.csv` has per-call **DIFFICULTY, TOPICALITY, NATURALNESS** ratings for all 2,438 calls (column semantics confirmed against the DAMSL `.utt` headers). Tannen's central claim is that style *clash* — not style itself — causes interactional trouble. Test: |style_A − style_B| (and dyad mean) predicting rated difficulty/naturalness, mixed model with caller random effects. This requires no types whatsoever and is the most direct test of Tannen's actual thesis anyone could run at scale. I'd call this the most exciting unpursued avenue in the repo.
16. **Topic as within-speaker manipulation.** Join `topic_label` (built, never used). Topics differ in personal focus (the prompts in `topic_tab.csv` are classifiable); each caller appears across multiple topics. Test whether speakers shift toward involvement on personal topics *within caller* — Tannen predicts yes; it also doubles as the topic control for all demographic claims.
17. **Ground turn-gap distributions in the turn-taking literature** (Levinson & Torreira 2015; Heldner & Edlund 2010): after the FTO fix, your gap distribution should reproduce the canonical unimodal ~+200 ms shape with an overlap tail — a free external-validity check that also retroactively explains the paper's −4 s nonsense.

### E. Feature gaps still worth closing (from your map, re-prioritized)

Highest theory-per-effort first: (a) **cooperative vs obstructive overlap** — the split your `overlap.py` explicitly deferred; operationalize as overlap after which the original speaker retains vs loses the floor — this is *the* HI-diagnostic distinction; (b) **laughter counter** (map #17, trivial, dim 9 currently empty); (c) **filled-pause vs discourse-marker split** (map #2 — the two halves of your current filler rate have *opposite* theoretical signs and are cancelling); (d) **voice quality** — jitter/shimmer/HNR/H1–H2 via `praat-parselmouth` or openSMILE eGeMAPS (both open-source; dim 2d has zero columns); (e) **marked-shift dynamics** — pitch slope per syllable, reset counts, contour entropy instead of static moments (your map's own closing argument); validate against the NXT accent/break gold subset; (f) **speaker-level aggregates** (map #29–33: silence ratio, overlap-initiation rate, persistence, narrative share, humor rate) — cheap groupbys once FTO and laughter exist; (g) the Ch.5 narrative block (#25–28) last — heaviest lift, save for after the distributional paper.

### F. External validity

18. **Fisher** (same genre, ~10× larger) to replicate unimodality; **CallHome/CallFriend** for the critical contrast — *familiar* dyads. The paper itself conceded strangers may suppress involvement; if intimates shift toward HI but the distribution stays unimodal, the continuum is robust *and* you've explained the suppression; if bimodality appears with intimates, you've found a real boundary condition. Either outcome strengthens the work. **CANDOR** (free for research) adds modern data with post-conversation outcome surveys — a replication target for the mismatch→quality analysis.

## 5. Suggested order of work

1. **Fixes** (days): FTO turn gap; one canonical table; backchannel classifier validated on NXT dialAct; caller dedup; laughter counter; filled-pause split.
2. **The unimodality battery** (1–2 weeks): dip + Silverman + bootstrap-LRT + skew-fit comparison + multivariate clusterability + recovery simulation + multiverse grid. This is the defensible core of the "continuum" paper.
3. **The positive story** (2–4 weeks): ICC trait stability; CFA of the involvement construct; mixed-model demographics; accommodation; mismatch → rating_tab quality.
4. **Feature expansion** (ongoing, dimension-at-a-time per your preference): overlap split, voice quality, marked shifts, speaker aggregates.
5. **Replication** (later): Fisher / CallHome / CANDOR.

One framing note for the eventual write-up: be careful to claim **unimodal/dimensional**, not "normal." Normality is neither necessary (a skewed continuum is still a continuum) nor sufficient (two heavily overlapping latent types can produce a unimodal, even Gaussian-looking, observed distribution — which is exactly why the taxometrics and reliability analyses in §A6 and §B10 are what make the claim airtight, not the histograms). Framed that way — *dimensional structure, stable individual differences, dyadic accommodation, and mismatch costs* — the unimodal result isn't the boring outcome; it's the corrected foundation Tannen's continuum language always implied.
