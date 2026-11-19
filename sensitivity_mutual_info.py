"""Sensitivity check for the Hop 2 feature selector.

pipeline.py calls mutual_info_classif on a sparse TF-IDF matrix with the default
discrete_features="auto". scikit-learn sets this to True for sparse input, so every
distinct TF-IDF value is treated as a category. This script re-runs the unchanged
pipeline with two other ways of scoring the same candidate features and prints how
the Hop 2 selection and the main results change.

Run from the repository root:  python sensitivity_mutual_info.py
Output: results_sensitivity.json (pipeline.py and results.json are not modified)
"""
import contextlib
import io
import json
import warnings

import numpy as np

warnings.filterwarnings("ignore")

SOURCE = open("pipeline.py").read()
MI_LINE = "mi_scores = mutual_info_classif(X_train_retrieved, y_train_enc, random_state=RNG_SEED)"
WRITE_BLOCK = 'with open("results.json", "w") as f:\n    json.dump(results, f, indent=2)'
assert MI_LINE in SOURCE and WRITE_BLOCK in SOURCE

VARIANTS = {
    "as published (sparse input, discrete MI)": MI_LINE,
    "continuous MI (dense input, discrete_features=False)":
        "mi_scores = mutual_info_classif(X_train_retrieved.toarray(), y_train_enc, "
        "discrete_features=False, random_state=RNG_SEED)",
    "chi-squared":
        "from sklearn.feature_selection import chi2\n"
        "mi_scores = chi2(X_train_retrieved, y_train_enc)[0]",
}

signal = {w for ws in json.load(open("dataset.json"))["signal_words"].values() for w in ws}
summary = {}
for name, line in VARIANTS.items():
    code = SOURCE.replace(MI_LINE, line).replace(WRITE_BLOCK, "pass")
    ns = {"__name__": "__sensitivity__"}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(code, "pipeline_variant", "exec"), ns)
    res = ns["results"]
    kept = ns["selected_terms"]
    summary[name] = {
        "signal_words_kept_of_60": sum(t in signal for t in kept),
        "noise_tokens_kept_of_60": sum(t not in signal for t in kept),
        "fidelity_hop1": res["fidelity"]["hop1_mean"],
        "fidelity_hop2": res["fidelity"]["hop2_mean"],
        "fidelity_hop3": res["fidelity"]["hop3_mean"],
        "fidelity_baseline": res["fidelity"]["baseline_mean"],
        "faithfulness_chained": res["faithfulness"]["chained_mean"],
        "faithfulness_baseline": res["faithfulness"]["baseline_mean"],
        "accuracy_chained": res["performance"]["chained_accuracy"],
        "accuracy_baseline": res["performance"]["baseline_accuracy"],
    }

with open("results_sensitivity.json", "w") as f:
    json.dump(summary, f, indent=2)

header = f"{'Hop 2 selector':55s}{'signal':>7s}{'noise':>7s}{'F1':>7s}{'F2':>7s}{'F3':>7s}{'base':>7s}{'acc':>7s}"
print(header)
for name, s in summary.items():
    print(f"{name:55s}{s['signal_words_kept_of_60']:7d}{s['noise_tokens_kept_of_60']:7d}"
          f"{s['fidelity_hop1']:7.3f}{s['fidelity_hop2']:7.3f}{s['fidelity_hop3']:7.3f}"
          f"{s['fidelity_baseline']:7.3f}{s['accuracy_chained']:7.3f}")
print("\nsignal/noise = features among the 60 kept by Hop 2; F1-F3 = mean fidelity at each hop;")
print("base = baseline fidelity; acc = accuracy of the chained pipeline.")
