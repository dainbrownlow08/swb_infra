"""Side-by-side of the paper's statistics across data sets and detector routes (reads *_results.csv)."""
import sys
from pathlib import Path
import pandas as pd

H = Path(__file__).resolve().parent
V = ["ppron", "wps", "wpu", "wpp", "boplen", "poplen", "pv", "lv", "olap", "rept", "repu"]
ROWS = [  # (label, results csv, loadings csv)
    ("MISC shipped tracks, their 168", "misc_shipped_168_results.csv", "misc_shipped_168_loadings.csv"),
    ("MISC shipped tracks, their 98", "misc_shipped_98_results.csv", "misc_shipped_98_loadings.csv"),
    ("MISC openSMILE-3 tracks, their 168", "misc_v3_168_results.csv", "misc_v3_168_loadings.csv"),
    ("MISC openSMILE-3 tracks, their 98", "misc_v3_98_results.csv", "misc_v3_98_loadings.csv"),
    ("Switchboard, in-house routes (pyin, word-tight), sides", "thomas_method_results.csv", "thomas_method_loadings.csv"),
    ("Switchboard, MISC route (openSMILE-3, speech-tight lines), sides", "swb_os_sides_results.csv", "swb_os_sides_loadings.csv"),
    ("  same, padded transcript spans for wps/olap/poplen", "swb_os_sides_trans_results.csv", "swb_os_sides_trans_loadings.csv"),
    ("  same, pv/lv inside own lines only", "swb_os_sides_inlines_results.csv", "swb_os_sides_inlines_loadings.csv"),
    ("Switchboard, MISC route, callers (pooled over calls)", "swb_os_callers_results.csv", "swb_os_callers_loadings.csv"),
]
keys = [("n_sides_analysed", "n"), ("alpha_keyed_table2_signs", "α keyed"), ("alpha_raw_signs", "α unkeyed"), ("mean_r_keyed", "r̄"),
        ("omega_t_1factor_keyed", "ω"), ("pc1_variance_share", "PC1"), ("horn_K", "K"), ("congruence_phi_ours_vs_thomas_pc1", "φ"),
        ("sign_agreement_k_of_11", "signs"), ("corr_involvement_ours_vs_thomas_weights", "r(their w)")]
out = []
for label, rf, lf in ROWS:
    if not (H / rf).is_file():
        continue
    r = pd.read_csv(H / rf).set_index("key").value
    out.append({"data / route": label, **{k2: r.get(k1, "") for k1, k2 in keys}})
print("paper (MISC, their code): n 168/98 · α .67 · GLB .85 · PC1 .29 · φ 1 · signs 11")
print(pd.DataFrame(out).to_string(index=False))
print("\nPC1 weights:")
tab = {"Table 2": [.13, .39, .45, .44, -.10, -.27, .09, .21, -.01, .39, .39]}
for label, rf, lf in ROWS:
    if (H / lf).is_file():
        L = pd.read_csv(H / lf).set_index("variable"); tab[label[:38]] = [round(L.loc[v, "our_pc1_weight"], 2) for v in V]
print(pd.DataFrame(tab, index=V).T.to_string())
