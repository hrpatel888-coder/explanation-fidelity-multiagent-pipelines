# Explanation fidelity in sequential multi-agent pipelines

Code, data and results for the paper "Explanation Fidelity Degradation in Sequential Multi-Agent AI Pipelines: A Synthetic Benchmark and Quantitative Framework for Cross-Hop Information Loss" by Harsha V. Patil, Kanchan G. Rathi and Geeta Chhabra Gandhi.

The study measures how well the explanation at each stage of a three-agent pipeline (retriever, feature selector, classifier) matches known ground truth, and compares it with a single non-chained classifier on the same task.

## Files

| File | Purpose |
|---|---|
| `generate_data.py` | Generates the synthetic benchmark and writes `dataset.json` |
| `dataset.json` | The benchmark: 1,500 documents in five classes, with the signal words of each class |
| `pipeline.py` | Runs the three-hop pipeline and the baseline, computes all metrics and tests, writes `results.json` |
| `results.json` | Numbers behind the tables and figures in the manuscript |
| `make_figures.py` | Draws the four figures (2D and 3D, 300 dpi) from `results.json` into `figures/` |
| `sensitivity_mutual_info.py` | Re-runs the pipeline with two other Hop 2 selectors (see Known issue) |
| `Explanation_Fidelity_Degradation_MultiAgent.ipynb` | Google Colab notebook with the whole analysis |
| `checksums.sha256` | SHA-256 of `dataset.json` and `results.json` |

## Reproducing the results

Run every command from the repository root.

```bash
pip install -r requirements.txt
python generate_data.py                  # writes dataset.json
python pipeline.py                       # writes results.json
python make_figures.py                   # writes figures/2d and figures/3d
sha256sum -c checksums.sha256            # both files should report OK
```

Python 3.10 or later is needed. The random seed is 42 throughout. Both generated files were reproduced byte for byte on Python 3.12.3 with NumPy 2.5.4, SciPy 1.18.1, scikit-learn 1.9.1, pandas 3.0.6 and matplotlib 3.11.2, and the same numbers were obtained on Google Colab. `pipeline.py` prints many scikit-learn `UserWarning` lines from `mutual_info_classif`; see the next section for what they mean.

To use Colab, upload the notebook (File, Upload notebook) and choose Runtime, Run all. It takes under a minute on the free CPU tier.

## Dataset

The benchmark is fully synthetic and no text was collected from any source. It has 1,500 documents, 300 in each of five classes (technology, sports, health, finance, education). Each class has 12 signal words. A signal word enters a document of its class with probability 0.55 and is then repeated once or twice. Each document also receives 15 to 39 noise tokens drawn from 150 tokens named `word0` to `word149`, and each signal word of another class leaks in with probability 0.04.

`dataset.json` has four fields: `docs` (token strings), `labels`, `signal_words` and `classes`. Documents average 39.0 tokens (SD 7.9, range 19 to 60). The 70/30 stratified split into 1,050 training and 450 test documents is made inside `pipeline.py`.

## Pipeline

1. Hop 1 keeps the 8 highest TF-IDF terms of each document.
2. Hop 2 pools the terms retrieved from the training documents, scores them with mutual information and keeps 60.
3. Hop 3 trains a logistic regression on those 60 features and explains each prediction with coefficient times feature value.
4. The baseline is one logistic regression on the full TF-IDF space with the same explanation method.

Fidelity at each hop is the share of explanation terms that are signal words of the document's class, with an empty explanation scored 0. Faithfulness is the drop in predicted probability when the explanation terms are removed.

## Results in the manuscript

| Stage | Mean fidelity | SD |
|---|---|---|
| Hop 1, retriever | 0.595 | 0.163 |
| Hop 2, selector | 0.202 | 0.362 |
| Hop 3, classifier | 0.090 | 0.111 |
| Baseline | 0.797 | 0.173 |

The chained pipeline reaches 0.620 accuracy and the baseline 1.000. Mean faithfulness is 0.240 for the chained pipeline and 0.684 for the baseline.

## Known issue: how Hop 2 scores features

`pipeline.py` calls `mutual_info_classif` on a sparse TF-IDF matrix with the default `discrete_features="auto"`. scikit-learn sets this to `True` for sparse input, so each distinct TF-IDF value is handled as a category rather than as a continuous number. Noise tokens have more distinct values than signal words, which inflates their scores. The warning lines mentioned above come from this.

The values in `results.json` were produced this way, and they depend on it. With a continuous estimator or a chi-squared score, Hop 2 keeps all 60 signal words and the fall in fidelity does not appear. `sensitivity_mutual_info.py` reproduces the comparison:

| Hop 2 selector | Signal words kept | Noise tokens kept | Fidelity at Hop 2 | Fidelity at Hop 3 | Chained accuracy |
|---|---|---|---|---|---|
| As published (sparse input, discrete MI) | 6 | 54 | 0.202 | 0.090 | 0.620 |
| Continuous MI (dense input) | 60 | 0 | 0.873 | 0.864 | 1.000 |
| Chi-squared | 60 | 0 | 0.873 | 0.864 | 1.000 |

Hop 1 (0.595) and the baseline (0.797) are the same in all three rows. The full output is in `results_sensitivity.json`.

## Other limits

The agents are classical machine-learning components, not language-model agents. The explanation method is the same at every hop, and the pipeline has one topology. The fidelity of a misclassified document is scored against its true class, so part of the Hop 3 fidelity reflects the 38% of test documents the chained pipeline misclassifies.

## Citation and licence

See `CITATION.cff`. Code and data are released under the MIT licence (`LICENSE`).

## Contact

Corresponding author: Geeta Chhabra Gandhi, geeta.gandhi@jaipur.manipal.edu
