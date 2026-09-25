# Related work — the nearest research, tiered by engagement depth

Compiled 2026-08-30 from three targeted literature sweeps (2026-08-23/25: psychometrics lineage,
projection-methods lineage, conversation-style quantification) plus session-verified anchors.
Citations marked **[V]** were verified against publisher/index pages or the PDF itself during the
sweeps; **[std]** are canonical textbook citations added from general knowledge — verify page
numbers at cite time. Free PDFs listed where confirmed.

## Tier 1 — the three contrast papers (engage in detail, not just cite)

1. **[V] Thomas, Czerwinski, McDuff, Craswell & Mark (2018).** Style and alignment in
   information-seeking conversation. *CHIIR '18*. doi:10.1145/3176349.3176388.
   11 automatic features on MISC, PCA, PC1 (29%) *named* "involvement" post hoc (their §4.1
   parenthetical drops PCs 2+ as hard to interpret / less person-varying / small). The data-first
   contrast we replicate exactly (`analysis/c_alpha/`: φ .99 on their identified 98; α gap located
   in pause/prosody routes; the "no-F0" pause definition not portable to telephone audio).
   Free PDF: microsoft.com/en-us/research/wp-content/uploads/2018/01/chiirfp025-thomasA.pdf
2. **[V] Ward & Avila (2023).** A dimensional model of interaction style variation in spoken
   dialog. *Speech Communication* 149, 47–62. ~84 prosodic-construction features per 30-s
   Switchboard fragment, z-scored, PCA, 8 dimensions retained, dimension 1 mapped to Tannen
   post hoc (verify exact feature count against the PDF when quoting). Same corpus, inverse
   inference direction. Engage also their instability finding (individual tendencies beat a
   speaker-independent model by only 3.6%) — our S7 ICC design answers it.
   Free PDF: cs.utep.edu/nigel/istyles/ward-avila-submitted.pdf
   Companions: **[V] Ward (2021)** SIGDIAL, aclanthology.org/2021.sigdial-1.4 (individual styles
   "far from stable"); **[V] Ward (2022)** SIGDIAL, cs.utep.edu/nigel/papers/istyles-sigdial2022.pdf.
3. **[V] Qiu, Gadiraju & Bozzon (2020).** Estimating conversational styles in conversational
   microtask crowdsourcing. *PACM HCI* 4(CSCW1), Article 32. doi:10.1145/3392837.
   The only prior theory-signed ±1 Tannen scoring: five hand-coded dimensions summed, sign →
   binary HI/HC class, text chat; a learned classifier predicts the label from chat features.
   Theory-first like us; manual, binary, text — the three axes we differ on.
   Free PDF: sihangqiu.com/convsty/publication/cscw2020estimating.pdf · code: github.com/qiusihang/convsty
   (WIP precursor: HCOMP 2019, humancomputation.com/2019/assets/papers/130.pdf.)

## Tier 2 — theory and data primary sources (must cite)

4. **[std] Tannen, D. (1984/2005).** *Conversational Style: Analyzing Talk Among Friends.*
   New ed., Oxford UP 2005 — the theory; every map direction cites its pages.
5. **[V] Tannen, D. (1980).** The parameters of conversational style. *ACL 18*, 39–40.
   aclanthology.org/P80-1011 — the theory offered to NLP in 1980; the scorer went unbuilt.
6. **[std] Thomas, McDuff, Czerwinski & Craswell (2017).** MISC: A data set of information-seeking
   conversations. *CAIR '17 (SIGIR workshop)* — required citation under the MISC data terms.
7. **[std] Godfrey, Holliman & McDaniel (1992).** SWITCHBOARD: telephone speech corpus for
   research and development. *ICASSP '92* — the corpus.
8. **[std] Calhoun, Carletta, Brenier, Mayo, Jurafsky, Steedman & Beaver (2010).** The NXT-format
   Switchboard Corpus. *Language Resources and Evaluation* 44(4) — the gold annotation layer.
9. **[std] Jurafsky, Shriberg & Biasca (1997).** SWBD-DAMSL labeling manual; and
   **Stolcke et al. (2000).** Dialogue act modeling… *Computational Linguistics* 26(3) — SwDA tags.

## Tier 3 — estimator lineage (the named parts of the score)

10. **[V] Dawes (1979).** The robust beauty of improper linear models. *Am. Psychologist* 34, 571–582 — the signed-z mean's license.
11. **[V] Wainer (1976).** It don't make no nevermind. *Psych. Bulletin* 83, 213–217 — equal weights.
12. **[V] Bobko, Roth & Buster (2007).** Usefulness of unit weights. *ORM* 10, 689–709 — modern review.
13. **[std] Burgess (1928).** Factors determining success or failure on parole. In *The Workings of
    the Indeterminate-Sentence Law…* (Illinois) — the original binary marker count.
14. **[V] Cohen (1983).** The cost of dichotomization. *Applied Psych. Measurement* 7, 249–253 — why the continuous score is primary.
15. **[V] MacCallum, Zhang, Preacher & Rucker (2002).** On the practice of dichotomization. *Psych. Methods* 7, 19–40.
16. **[V] Davison & Davenport (2002).** Criterion profile analysis. *Psych. Methods* 7, 468–484 — identical person-scoring algebra, empirical vector.
17. **[V] Prediger (1982).** Dimensions underlying Holland's hexagon. *J. Vocational Behavior* 21, 259–287 — closest ancestor: theory-signed axes, persons scored.
18. **[V] Grice (2001).** Computing and evaluating factor scores. *Psych. Methods* 6, 430–450 — coarse ±1 weights, mirror image.
19. **[std] Little, Cunningham, Shahar & Widaman (2002).** To parcel or not to parcel. *SEM* 9(2) — the parceling decision.
20. **[V] Furr (2008).** A framework for profile similarity. *J. Personality* 76, 1267–1316 — z-scored score = distinctive similarity.
21. **[std] Campbell & Fiske (1959).** Convergent and discriminant validation… *Psych. Bulletin* 56, 81–105 — gate-vs-outcome logic.
22. **[std] Loevinger (1954).** The attenuation paradox. *Psych. Bulletin* 51, 493–504 — why coherence is never optimized.
23. **[std] Cronbach (1951).** Coefficient alpha… *Psychometrika* 16, 297–334.
24. **[std] Spearman (1904).** The proof and measurement of association. *Am. J. Psychology* 15 — disattenuation (S7).
25. **[std] Horn (1965).** A rationale and test for the number of factors. *Psychometrika* 30, 179–185 — parallel analysis.
26. **[V] Lorenzo-Seva & ten Berge (2006).** Tucker's congruence coefficient. *Methodology* 2, 57–64 — φ bands.
27. **[std] Shrout & Fleiss (1979).** Intraclass correlations. *Psych. Bulletin* 86, 420–428 — S7 ICCs.
28. **[std] Hurlbert (1984).** Pseudoreplication… *Ecological Monographs* 54, 187–211 — the S3 rationale's name (optional).
29. **[V] McNeish & Wolf (2020)** *BRM* 52, 2287–2305 and **[V] Widaman & Revelle (2023)** *BRM* 55, 788–806 — the sum-score debate (defensive).

## Tier 4 — projection/space lineage (appendix C + the CPCA footnote)

30. **[V] Mante, Sussillo, Shenoy & Newsome (2013).** *Nature* 503, 78–84 — targeted dimensionality
    reduction; the algebraic precedent for the cut u = zLLᵀt variant. (Formalized: Aoi & Pillow, NeurIPS 2018.)
31. **[V] Gurtman (1992).** *JPSP* 63, 105–118 (+ Wright et al. 2009 *JPA* 91, 311–322; Zimmermann &
    Wright 2017 *Assessment* 24, 3–23) — circumplex structural summary: persons projected onto
    theory-located directions; rotation-invariance parallel.
32. **[V] Grand, Blank, Pereira & Fedorenko (2022).** Semantic projection… *Nat. Hum. Behav.* 6, 975–987 — modern named-axis projection.
33. **[V] Kozlowski, Taddy & Evans (2019).** The geometry of culture. *ASR* 84, 905–949.
34. **[V] Carroll (1972)** external analysis / **Chang & Carroll (1969)** PROFIT (Bell Labs report) — the historical root; alive as vegan's envfit.
35. **[V] Takane & Shibayama (1991).** *Psychometrika* 56, 97–120 — "CPCA" = constrained PCA; the
    name-collision footnote (our combined trusted-space PCA must be spelled out).
36. **[V] Dwyer (1937)** *Psychometrika* 2, 173–178; **Gorsuch (1997)** *EPM* 57, 725–740 — extension analysis (appendix only).

## Tier 5 — domain-adjacent (delimit the field, one clause each)

37. **[V] Biber (1988).** *Variation across Speech and Writing* — Dimension 1 "Involved vs
    Informational": the historical weighted-composite involvement score; data-first, texts not speakers.
38. **[V] Shamekhi, Czerwinski, Mark, Novotny & Bennett (2016).** *IVA '16*, LNCS, 40–50 — HI/HC
    agent + the self-report Tannen questionnaire later reused.
39. **[V] Hoegen, Aneja, McDuff & Czerwinski (2019).** An end-to-end conversational style matching
    agent. *IVA '19*, 111–118 (arXiv:1904.02760) — Tannen-motivated automatic features for dyadic
    matching; person's axis position is self-report.
40. **[V] Kostric, Balog & Gadiraju (2025).** Should we tailor the talk? *UMAP '25* (arXiv:2504.13095) — HI/HC on the system side.
41. **[V] Coker & Burgoon (1987).** *Human Communication Research* 13, 463–494 — coded-behavior involvement, regression weights.
42. **[V] Cegala (1981)** *Comm. Education* 30, 109–121; **Norton (1978)** *HCR* 4, 99–112 — self-report style scales.
43. **[V] Niederhoffer & Pennebaker (2002).** *JLSP* 21, 337–360 — LSM: dyadic alignment, not a trait axis.
44. **[V] Grothendieck, Borges & Gorin (2011).** Social correlates of turn-taking style. *CSL* 25, 789–801 — Switchboard style clusters.
45. **[V] Mairesse, Walker, Mehl & Moore (2007).** *JAIR* 30, 457–500 — trait scores from conversation cues (supervised).
46. **[V] Oertel, Scherer & Campbell (2011)** Interspeech; **Wrede & Shriberg (2003)** Eurospeech — momentary involvement (state, not trait).
47. **[V] Ranganath, Jurafsky & McFarland (2009/2013)** EMNLP / *CSL* — interactional style detection, folk labels, learned weights.

Statistical-test citations used inside NB08 (dip, Silverman, BLRT, gap statistic, bootstrap) are
cited where used, per the NB07 battery conventions.

## The gap boundary — support for the novelty claim

The claim (scoped, "to our knowledge"): *the first continuous, automatic, theory-first
operationalization of Tannen's involvement–considerateness on natural conversational speech at
corpus scale, with a falsifiable, reliability-separated coherence test.* Each boundary paper is the
nearest neighbor that misses on the named axes (✓ has it · ~ partial · ✗ lacks it):

| Paper | Tannen, theory-first | Automatic features | Continuous score | Speech | Per-speaker at scale | Falsifiable coherence test |
|---|---|---|---|---|---|---|
| **Ours** | ✓ | ✓ | ✓ | ✓ | ✓ (487 callers × ~7–9 calls) | ✓ (P1–P3 + ICC separation) |
| Qiu et al. 2020 | ✓ | ✗ manual ±1 coding | ✗ binary class | ✗ text chat | ~ | ✗ |
| Ward & Avila 2023 | ✗ data-first, named post hoc | ✓ | ✓ | ✓ | ~ fragments; individuals unstable | ✗ |
| Thomas et al. 2018 | ✗ PC1 named post hoc | ✓ | ✓ | ✓ | ✗ 98–168 tasks, 1 occasion | **~ §4.1 GLB/scree** — no reliability separation, no person-level test, no pre-stated predictions |
| Biber 1988 | ✗ (Chafe-adjacent, data-first) | ✓ | ✓ | ✗ texts/registers | ✓ texts | ✗ |
| Hoegen et al. 2019 | ✓ Tannen-motivated | ✓ (for dyadic matching) | ✗ axis = self-report | ✓ | ✗ | ✗ |
| Shamekhi et al. 2016 | ✓ | ✗ questionnaire | ✗ | ~ | ✗ | ✗ |
| Coker & Burgoon 1987 | ✗ (Burgoon's construct) | ✗ hand-coded | ~ rater-weighted | ✓ | ✗ 52 dyads | ✗ |
| Mairesse et al. 2007 | ✗ (Big Five, supervised) | ✓ | ✓ | ✓ | ~ | ✗ |
| Ward 2021 | ✗ | ✓ | ✓ | ✓ | ~ stability measured, noise/repertoire not separable | ✗ |
| Tannen 1980; 1984/2005 | the theory itself — no measurement | | | | | |

Honesty note (cite it this way): Thomas et al.'s §4.1 explicitly frames GLB/α + scree as a test of
"a coherent consideration-involvement dimension," so the coherence *question* is theirs as partial
precedent; what no prior work has is the combination in the last three columns of our row —
theory-fixed direction, repeated-conversations reliability separating repertoire from noise, and
person-level typology predictions stated before looking.

### Public PDFs for the boundary set (checked 2026-08-30)

Verification: **[opened]** = PDF fetched and title page confirmed this session; **[repo]** = open
repository by policy (ACL Anthology / arXiv / JAIR), link from a verified sweep; **[none]** = no
legitimate public PDF exists.

| Boundary paper | Public PDF |
|---|---|
| Thomas et al. 2018, CHIIR | [opened] microsoft.com/en-us/research/wp-content/uploads/2018/01/chiirfp025-thomasA.pdf |
| Ward & Avila 2023, Speech Comm. | [opened] cs.utep.edu/nigel/istyles/ward-avila-submitted.pdf (author accepted manuscript) |
| Ward 2021, SIGDIAL | [repo] aclanthology.org/2021.sigdial-1.4.pdf |
| Qiu et al. 2020, CSCW | [opened] sihangqiu.com/convsty/publication/cscw2020estimating.pdf (author copy; precursor: humancomputation.com/2019/assets/papers/130.pdf) |
| Hoegen et al. 2019, IVA | [repo] arxiv.org/abs/1904.02760 |
| Shamekhi et al. 2016, IVA (LNAI 10011, 40–50) | [opened] microsoft.com/en-us/research/wp-content/uploads/2017/10/IVA2016.pdf |
| Mairesse et al. 2007, JAIR 30, 457–500 | [opened] jair.org/index.php/jair/article/download/10520/25197 (JAIR is open access) |
| Coker & Burgoon 1987, HCR 13, 463–494 | [none] — OpenAlex: is_oa false, no OA location. Paywalled at OUP/Wiley; free abstract: eric.ed.gov/?id=EJ385205 |
| Biber 1988 (book) | [none] — no public PDF; borrowable via library / archive.org lending |
| Tannen 1980, ACL | [repo] aclanthology.org/P80-1011.pdf |
| Tannen 1984/2005 (book) | [none] — no public PDF |
