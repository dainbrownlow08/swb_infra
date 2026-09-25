# c_alpha — Thomas et al. (2018) reliability + PCA method, copied and applied to Switchboard

`thomas_method.py` re-runs §3.2–§4.1 of Thomas, Czerwinski, McDuff, Craswell & Mark
(CHIIR '18) on our extraction of their eleven variables (`src/swb_extract/features/thomas2018/`)
at their unit (participant × task ≈ one conversation side), step for step — cleaning,
z-scoring, Cronbach's α (raw and keyed), PCA of the correlation matrix, PC1 as the unit-norm
eigenvector (their Table 2 convention), and involvement = Σ loading × z. It then compares
our PC1 with their Table 2: sign agreement, Tucker's congruence φ, bootstrap CIs, and a
sampling-error yardstick at their n (168 / 98).

Stated deviations from the paper: their outlier removal was manual (ours is a |z| > 5 rule
plus the project's ≥ 20-utterance floor, with a sensitivity block without them); Revelle's
GLB is an R routine (McDonald's ω_t from a one-factor fit is reported beside α instead).

Inputs: `utterances_v2/derived/thomas2018_side.csv` (all eleven extracted, then
`swb-extract thomas2018-side`). Outputs (written here, regenerable):

| File | Contents |
|---|---|
| `thomas_method_sides.csv` | per side: n_utts, the eleven raw and z-scored, involvement under our PC1 weights and under their Table 2 weights, percentile |
| `thomas_method_loadings.csv` | per variable: Table 2 weight, our PC1 weight, bootstrap 95% CI, sign match, α-if-deleted, mean/sd/skew/kurtosis |
| `thomas_method_results.csv` | key–value: n at each cleaning step, r̄, α (both keyings), ω_t, PC variance shares, Horn K, corr(ours, their weights), φ, sign agreement, subsample-φ yardstick, sensitivity re-runs |

Run: `PYTHONPATH=src python3 analysis/c_alpha/thomas_method.py` (repo root).

## Results — MISC, their data through our code (2026-08-22)

`misc_variables.py` computes the eleven on the MISC release along the paper's routes (STT
lines as utterances; OpenSMILE `prosodyShs` F0/loudness tracks for pauses, pv, lv; pair clocks
verified shared) with the paper's exclusions: 220 → 176 (tasks 2–5) → **101** after the
ten-minute and named-outlier removals (paper: 98). Magnitudes match the paper's stated ones
(between-own pauses median 0.105 s vs "order of 0.1 s"; words per pair-task 853 vs 857).
`thomas_method.py` on that table (`misc_method_*.csv`):

| Statistic | Paper (MISC, their code) | MISC, our code (n = 100) |
|---|---|---|
| PC1 congruence with Table 2 (Tucker φ) | — | **.988**; signs 10/11 (olap flips: −.01 vs +.09) |
| PC1 variance share | 29% | 25% |
| Cronbach's α | .67 | .60 keyed by Table 2 signs (.44 unkeyed — so the paper keyed) |
| GLB / ω_t | GLB .85 | ω_t .66 |
| involvement, our PC1 vs their published weights | — | r = .995 |

Reading: the replication of the *instrument* holds — on their data our routes reproduce their
component to within a few hundredths of congruence. Residual differences sit in what the
paper could not specify (pronoun and stopword lists, overlap handling in poplen, three
cleaning cases). One route note: on their own F0 signal, literal between-own pauses
outnumber words (wpp ≈ 0.84), so their "approximately the length of each spoken phrase"
gloss was loose or carried an unstated minimum-pause threshold.

## Results — Switchboard through the same method (2026-08-22)

*(In-house routes: pyin + RMS, word-tight spans. Superseded for cross-corpus comparison by the like-for-like section at the end — same detector on both corpora.)*

`thomas_method.py` on `utterances_v2/derived/thomas2018_side.csv` (sides; 3,988 → 3,836 after the
≥20-utterance floor and |z| > 5) and on `thomas_caller_table.csv` (the eleven pooled per caller
across ~7 calls; 493 → 482). Outputs `thomas_method_*.csv`, `thomas_method_caller_*.csv`.

| Statistic | Paper (MISC) | MISC, our code (n=100) | Switchboard sides (n=3,836) | Switchboard callers (n=482) |
|---|---|---|---|---|
| PC1 congruence with Table 2 (φ) | — | .988, signs 10/11 | **.839**, 9/11 (boplen, pv flip; both ≈ 0 everywhere) | .887, 8/11 |
| φ, MISC-ours vs Switchboard | | | .841 | |
| PC1 variance share (Horn K) | 29% | 25% (2) | 24% (5) | 26% (3) |
| Cronbach's α, Table 2 keying | .67 | .60 | .36 | .39 |
| ω_t | GLB .85 | .66 | .57 | .62 |
| involvement: our PC1 vs their fixed weights | — | .995 | .947 | .976 |
| subsample yardstick: φ q05 at n = 168 | | | .90 | .92 |

Reading. The method reproduces the paper on its own data (.988), so the .84 on Switchboard is
the corpus, and it lies outside what their sample size alone would produce (q05 .90). What
moves: the repetition pair and utterance length take over PC1 (rept .58, repu .55, wpu .51 vs
.39/.39/.45), words-per-pause collapses (.44 → .07), speech rate and post-other pause
halve (.39 → .22, −.27 → −.10). The eleven cohere far less here (keyed r̄ .05 vs their .16);
three arithmetic facts explain most of it: `wpp` and `boplen` share a denominator (the
pause count) and correlate +.83 raw, which the Table 2 keying turns into −.83; `rept` and
`repu` are near-duplicates (+.91) and both track utterance length (+.73 / +.65 with wpu) even
after stopword removal; and keying olap negative follows a loading the paper itself called
zero — keying it as Tannen predicts lifts α to .47, leaving the within-turn pause variables
unsigned lifts it to .56. The *score* is more robust than the *loadings*: involvement under
their fixed Table 2 weights correlates .95 (sides) / .98 (callers) with our own PC1 score.

Level differences that are route, not style: wpu 12.3 vs MISC 7.0 words per line (human vs
STT segmentation), boplen .20 vs .105 s and wpp 2.0 vs .84 (pause runs from pyin at 16 ms vs
OpenSMILE at 10 ms on longer lines), pv 1,571 vs 3,712 Hz² (SHS octave errors inflate MISC's),
olap .36 vs .16, poplen .40 vs 1.6 s (intermediaries searching the web).

## Method identification on MISC — how close can 1:1 get? (2026-08-23)

`misc_identify.py` (outputs `misc_identify_variants.csv`, `misc_identify_samples.csv`,
`misc_variables_n168.csv`). Two things were identified from the release itself:

- **Their reliability/PCA sample (168) is exact:** tasks 2–5 (176) minus the five named
  outlier participant-tasks (171) minus the three participant-tasks with no questionnaire
  responses — participant 22, tasks 2, 3, 5 — = 168. **Their modelling sample (98):** the same
  minus tasks whose *audio* runs past 600 s gives 97 (one boundary case; the transcript-time
  version gives 101).
- **Their pronoun list is LIWC-like:** our 16-form list correlates .97 with LIWC's i+we+you
  in the bundled LIWC file and counts 39% more — the paper's "captured more terms than did
  LIWC".

Unstated routes varied one at a time on the 168 (pause definition: F0 = 0 vs voicing
probability < .5 vs ≥ 2-frame runs; pv/lv frames: F0 > 0 vs voicing probability vs within
utterances only; poplen: overlaps blanked vs naive negative gaps; olap: transcript lines vs
partner voicing; stemmer: Porter vs Snowball; pronoun list: ours vs LIWC-extended), then a
greedy search (17 combinations): φ moves at most .02, α at most .03.

| Sample / routes | n | α keyed | α (psych check.keys) | GLB-fa (1/2/3 factors) | PC1 share | φ vs Table 2 | signs |
|---|---|---|---|---|---|---|---|
| Paper | 168 (α, GLB) / 98? | .67 | | .85 | .29 | 1 | 11 |
| their 168, baseline routes | 168 | .535 | .474 | .57 / .66 / .71 | .225 | .923 | 10 |
| their 168, greedy-best routes | 168 | .522 | .566 | .56 / .65 / .70 | .226 | .941 | 10 |
| their 98, baseline routes | 97 | .576 | .499 | .61 / .69 / .74 | .237 | .974 | 10 |
| their 98, greedy-best routes | 97 | .562 | .603 | .60 / .68 / .73 | .241 | .983 | 10 |

| Variable | Table 2 | ours, their 168 | ours, their 98 |
|---|---|---|---|
| ppron | +0.13 | +0.28 | +0.19 |
| wps | +0.39 | +0.49 | +0.44 |
| wpu | +0.45 | +0.47 | +0.48 |
| wpp | +0.44 | +0.50 | +0.45 |
| boplen | -0.10 | -0.18 | -0.13 |
| poplen | -0.27 | -0.06 | -0.11 |
| pv | +0.09 | +0.20 | +0.16 |
| lv | +0.21 | +0.19 | +0.22 |
| olap | -0.01 | +0.07 | +0.07 |
| rept | +0.39 | +0.24 | +0.36 |
| repu | +0.39 | +0.21 | +0.32 |

Reading. The published loadings are reproduced on the **98-type** sample (φ .97–.98; every
weight within range) far better than on the 168 (φ .92–.94, where poplen shrinks from −.27
to −.06 and rept/repu from .39 to .24/.21) — so Table 2 was most likely computed on the
cleaned 98, with the 168 used for the GLB/α sentence. Direction replicates under every
identifiable choice (10/11 signs always; the flip is olap at ≈ 0). Coherence does not fully:
α lands .10–.15 below .67 and the FA-based GLB .11–.14 below .85 on both samples, and no
route switch closes that. What remains unrecoverable from the release: their exact pronoun
and stopword lists, how their OpenSMILE frames were matched to utterances, and their GLB
estimator (the algebraic GLB exceeds the factor-based one). The one remaining move is the
paper's own offer — the code "available on request".

**Is the coherence gap chance? No.** Resampling our own variables on the identified 168
(2,000 bootstrap draws) and on random 98-subsets: keyed α reaches .67 in 0.1% / 0% of draws
(5–95% bands .46–.61 / .46–.60) and PC1 reaches 29% in 0.05% / 0% (.21–.26 / .21–.25). The
paper's coherence is therefore not a sample-composition effect of *our* variables; something
in the construction differs systematically. Where: on our build the pause, prosody and overlap
variables barely correlate with the rest (mean keyed r with the other ten — poplen .02,
boplen .05, pv .07, lv .08, olap −.06) while the length/rate/repetition block carries the
component; in the paper poplen (−.27), lv (.21) and boplen (−.10) sat inside it. The likeliest
locus is how the OpenSMILE-derived and timing variables were computed (frame-to-utterance
matching, pause definition, overlap handling), not the word lists — the text-side variables
already match Table 2 on the 98-type sample.

## Like-for-like: one detector, one route, both corpora (2026-08-23)

Question: how much of the Switchboard coherence gap was *route* (different pitch tracker, different
line spans) rather than *corpus*? Answer by removing the route difference entirely: run openSMILE
3.0 `prosodyShs` (the paper's config) over every Switchboard side recording **and** over MISC
(`run_opensmile.py`; the MISC release's own tracks came from an older openSMILE build — loudness
matches ours at r .9998, F0/voicing do not), then build the eleven on Switchboard exactly along the
MISC route (`swb_misc_route.py`: lines = ms98 utterances with speech-tight spans from the word
alignments, the analogue of an STT segment; pauses = F0 = 0 runs inside lines; pv/lv over the whole
side channel; olap/poplen from line onsets vs the partner's lines), and score everything with
`thomas_method.py` (`like_for_like.py` drives it; `compare_routes.py` prints the table).

| Data, tracks, unit | n | α keyed | r̄ | ω_t | PC1 % (Horn K) | φ vs Table 2 | signs | r(PC1 score, their weights) |
|---|---|---|---|---|---|---|---|---|
| Paper | 168 / 98 | .67 | .16 | GLB .85 | 29 | 1 | 11 | |
| MISC, shipped tracks, their 168 | 166 | .568 | .107 | .633 | 23.8 (4) | .957 | 10 | .984 |
| MISC, shipped tracks, their 98 | 96 | .616 | .127 | .668 | 25.4 (3) | **.990** | 10 | .997 |
| MISC, openSMILE-3 tracks, their 168 | 166 | .535 | .095 | .628 | 24.8 (3) | .822 | 10 | .891 |
| MISC, openSMILE-3 tracks, their 98 | 96 | .583 | .113 | .660 | 25.6 (2) | .942 | 9 | .978 |
| Switchboard, in-house routes (pyin, word-tight), sides | 3,836 | .359 | .048 | .573 | 24.4 (5) | .839 | 9 | .947 |
| **Switchboard, openSMILE-3, MISC route, sides** | 3,921 | **.407** | .059 | .559 | 24.3 (5) | **.813** | 9 | .918 |
| same, pv/lv inside own lines only | 3,915 | .422 | .062 | .570 | 24.4 (5) | .823 | 9 | .926 |
| same, padded ms98 spans for wps/olap/poplen | 2,192 | .479 | .077 | .626 | 27.6 (3) | .810 | 8 | .932 |
| Switchboard, openSMILE-3, MISC route, callers (pooled) | 484 | .430 | .064 | .609 | 25.5 (4) | .816 | 8 | .913 |

(The |z| > 5 outlier rule is applied everywhere as in the paper; min-utts 20 on Switchboard only.
The padded-span row is a failed sensitivity, not a route: padded ms98 lines overlap so heavily —
olap .72 — that an onset after a partner pause never occurs for 1,750 of 3,988 sides and poplen is
undefined.) Re-tracking MISC with openSMILE 3 halves its boplen (.114 → .059 s), cuts pv
(3,791 → 2,401 Hz²) and flips lv's loading sign (+.21 → −.07/−.19), so the detector build is itself a
measurable route axis on MISC — the shipped tracks stay the reference for Table 2.

PC1 weights (same ordering as Table 2):

| | ppron | wps | wpu | wpp | boplen | poplen | pv | lv | olap | rept | repu |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Table 2 | .13 | .39 | .45 | .44 | −.10 | −.27 | .09 | .21 | −.01 | .39 | .39 |
| MISC shipped, their 98 | .20 | .39 | .47 | .42 | −.08 | −.27 | .15 | .21 | .08 | .39 | .33 |
| MISC openSMILE-3, their 98 | .17 | .43 | .45 | .46 | −.12 | −.22 | .24 | −.07 | .03 | .38 | .31 |
| Switchboard openSMILE-3, sides | .11 | .21 | .51 | −.04 | −.04 | −.11 | .01 | .15 | .02 | .58 | .56 |
| Switchboard openSMILE-3, callers | .10 | .27 | .48 | −.08 | −.12 | −.18 | −.01 | .14 | .06 | .56 | .54 |
| Switchboard pyin (in-house), sides | .10 | .22 | .51 | .07 | .09 | −.10 | −.04 | .11 | −.01 | .58 | .55 |

**What the route change did on Switchboard.** Keyed α .36 → .41 (sides) / .43 (callers); φ .84 → .81;
PC1 share unchanged (24–25 %). Per-side involvement under Table 2 weights correlates .962 between the
two Switchboard routes (own-PC1 scores .991); per variable, wpp (r .48) and pv (.41) are the
detector-dependent ones, boplen .75, lv .93, the seven text/timing variables 1.00. Route was never
the main term.

**Where the remaining gap lives — one variable.** Keyed mean inter-item correlation per variable,
MISC openSMILE-3 (98) vs Switchboard openSMILE-3 (sides): ppron .06/.05 · wps .18/.09 · wpu .23/.18 ·
**wpp .22/−.06** · boplen .02/−.08 · poplen .02/.01 · pv .13/.01 · lv −.06/.06 · olap −.02/−.07 ·
rept .15/.21 · repu .11/.20; mean .095/.055. The largest pairwise differences are all wpp's:
wps–wpp .84 on MISC vs .04 on Switchboard, wpu–wpp .51 vs −.01, wpp–boplen (keyed) +.32 vs −.72.
On MISC words-per-pause behaves as a fluency measure moving with speech rate; on Switchboard it
detaches from rate and locks onto boplen through the shared pause-count denominator.

**Why: the pause object differs between the two audio types, under the identical detector.**
Inside lines, on an 80-conversation Switchboard sample vs the 176 MISC task recordings (tasks 2–5):

| F0 = 0 runs inside lines | Switchboard (8 kHz telephone) | MISC (48 kHz headset) |
|---|---|---|
| voiced share of line time | .53 | .79 |
| runs per word | .69 | 1.24 |
| run length median / mean / p90 (ms) | 100 / 163 / 370 | 40 / 60 / 130 |
| share of runs < 50 ms · ≥ 200 ms | .23 · .23 | .53 · .03 |
| words per voiced second | 6.6 | 3.4 |
| wps (speech-tight / STT segments) | 3.5 | 2.7 |

MISC's "pauses" are mostly sub-50-ms unvoiced stretches between voiced ones (the tracker follows F0
through 79 % of speech); Switchboard's are long. Against the ms98 word alignments (98,865 words) the
long ones are not silences: real inter-word gaps ≥ 200 ms occur .062 per word, openSMILE's F0 = 0
runs ≥ 200 ms .159 per word (2.6×), 65 % of those runs lie mostly *inside* an aligned word, and 75 % of
all F0 = 0 frames inside lines fall within word spans. On narrowband telephone audio the tracker loses
voicing during words, so wpp and boplen measure dropout density rather than pausing — and the same
decoupling appears under pyin (wpp mean keyed r −.057 vs −.059 here), so it is the audio, not the
detector. This is a corpus-level bound on faithful replication: Thomas et al.'s pause definition
("periods where OpenSMILE reports no F0") is not portable to 8-kHz recordings. A minimum-run
threshold would not rescue it symmetrically (only 3 % of MISC's runs reach 200 ms — the threshold
erases their variable); defining pauses from the word alignments (the in-house route) restores a
pause measure on Switchboard at the cost of leaving the paper's definition.

Files: `run_opensmile.py` (+ `opensmile_conf/`), `swb_misc_route.py` → `swb_misc_route_{sides,callers}.csv`,
`like_for_like.py` → `misc_{shipped,v3}_{168,98}_*`, `swb_os_{sides,sides_trans,sides_inlines,callers}_*`,
`compare_routes.py`.
