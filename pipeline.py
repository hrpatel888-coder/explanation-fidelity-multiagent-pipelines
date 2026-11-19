"""
Multi-agent pipeline (3 hops) with explanation-fidelity measurement at each stage,
compared against a single-agent (non-chained) baseline.

HOP 1 - Retriever Agent:
    TF-IDF-based per-document term retrieval. For each document, retrieves the
    top-K terms by TF-IDF weight. Explanation E1(doc) = set of retrieved terms.

HOP 2 - Feature-Selector Agent:
    Operates on the GLOBAL vocabulary of retrieved terms (union across HOP1
    outputs for the training set) and applies mutual-information-based
    filtering to select the top-M globally informative features.
    Explanation E2 = global feature importance ranking (retained features).

HOP 3 - Classifier Agent:
    Logistic Regression trained ONLY on the HOP2-selected feature subset.
    Explanation E3(doc) = per-document feature attribution obtained from
    |coefficient * feature_value| (standard linear-model attribution),
    restricted to features present in the document.

BASELINE (single agent, no chaining):
    Logistic Regression trained directly on the full TF-IDF feature space
    (no retrieval / no filtering step), explained the same way (E_base).

METRICS (computed per document, averaged):
  - Fidelity@k: precision of explanation against KNOWN ground-truth signal
    words for that document's class (i.e., what fraction of the top-k
    explanation terms are true signal words).
  - Hop-to-hop consistency: Jaccard(E_i, E_{i+1}) - how much of the earlier
    explanation survives into the next stage.
  - Faithfulness (deletion AOPC): drop in predicted-class probability when
    top explanation terms are removed from the document (higher = more
    faithful explanation).
  - Predictive performance: accuracy / macro-F1 at final classification.
"""
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from scipy import stats

RNG_SEED = 42
TOP_K = 8          # explanation size (top-k terms) at each hop
TOP_M = 60         # global features retained after HOP2 filtering

with open("dataset.json") as f:
    data = json.load(f)

docs = data["docs"]
labels = data["labels"]
signal_words = data["signal_words"]
classes = data["classes"]

label_to_signal = {c: set(signal_words[c]) for c in classes}

X_train_docs, X_test_docs, y_train, y_test = train_test_split(
    docs, labels, test_size=0.30, random_state=RNG_SEED, stratify=labels
)

# ---------- Full TF-IDF space (used by baseline AND as input to retriever) ----------
vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
X_train_full = vectorizer.fit_transform(X_train_docs)
X_test_full = vectorizer.transform(X_test_docs)
vocab = np.array(vectorizer.get_feature_names_out())
vocab_index = {w: i for i, w in enumerate(vocab)}

def topk_terms_for_row(csr_row, k):
    row = csr_row.toarray().ravel()
    if np.all(row == 0):
        return set()
    top_idx = np.argsort(row)[::-1][:k]
    top_idx = [i for i in top_idx if row[i] > 0]
    return set(vocab[top_idx])

# ================= HOP 1: Retriever Agent =================
E1_train = [topk_terms_for_row(X_train_full[i], TOP_K) for i in range(X_train_full.shape[0])]
E1_test = [topk_terms_for_row(X_test_full[i], TOP_K) for i in range(X_test_full.shape[0])]

def fidelity_at_k(explanation_sets, y_true_labels):
    scores = []
    for expl, lab in zip(explanation_sets, y_true_labels):
        if len(expl) == 0:
            scores.append(0.0)
            continue
        true_signal = label_to_signal[lab]
        hits = len(expl & true_signal)
        scores.append(hits / len(expl))
    return np.array(scores)

fid_hop1 = fidelity_at_k(E1_test, y_test)

# ================= HOP 2: Feature-Selector Agent =================
# Global vocabulary retrieved by HOP1 (union across training explanations)
retrieved_vocab = sorted(set().union(*E1_train)) if E1_train else []
retrieved_idx = [vocab_index[w] for w in retrieved_vocab if w in vocab_index]
retrieved_terms = vocab[retrieved_idx]

X_train_retrieved = X_train_full[:, retrieved_idx]
X_test_retrieved = X_test_full[:, retrieved_idx]

# mutual information on the RETRIEVED subspace only (this is the info loss point:
# HOP2 never sees anything outside what HOP1 retrieved)
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
mi_scores = mutual_info_classif(X_train_retrieved, y_train_enc, random_state=RNG_SEED)
top_m_local_idx = np.argsort(mi_scores)[::-1][:TOP_M]
selected_terms = set(retrieved_terms[top_m_local_idx])
global_select_idx = [retrieved_idx[i] for i in top_m_local_idx]

# HOP2 explanation per document = intersection of doc's HOP1 explanation with the
# globally selected feature set (this IS what survives to hop 3)
E2_test = [expl & selected_terms for expl in E1_test]
fid_hop2 = fidelity_at_k(E2_test, y_test)
consistency_1_2 = np.array([
    len(e1 & e2) / len(e1) if len(e1) > 0 else 0.0
    for e1, e2 in zip(E1_test, E2_test)
])

# ================= HOP 3: Classifier Agent (chained) =================
X_train_final = X_train_full[:, global_select_idx]
X_test_final = X_test_full[:, global_select_idx]
final_terms = vocab[global_select_idx]

clf_chained = LogisticRegression(max_iter=2000, random_state=RNG_SEED)
clf_chained.fit(X_train_final, y_train)
y_pred_chained = clf_chained.predict(X_test_final)
acc_chained = accuracy_score(y_test, y_pred_chained)
f1_chained = f1_score(y_test, y_pred_chained, average="macro")

class_order = clf_chained.classes_

def linear_explanation(clf, X_row_dense, terms, k, predicted_class_idx):
    coefs = clf.coef_[predicted_class_idx]
    attributions = coefs * X_row_dense
    if np.all(attributions == 0):
        return set(), attributions
    top_idx = np.argsort(np.abs(attributions))[::-1][:k]
    top_idx = [i for i in top_idx if attributions[i] != 0]
    return set(terms[top_idx]), attributions

E3_test = []
X_test_final_dense = X_test_final.toarray()
pred_class_indices = [list(class_order).index(p) for p in y_pred_chained]
for i in range(X_test_final_dense.shape[0]):
    expl, _ = linear_explanation(clf_chained, X_test_final_dense[i], final_terms, TOP_K, pred_class_indices[i])
    E3_test.append(expl)

fid_hop3 = fidelity_at_k(E3_test, y_test)
consistency_2_3 = np.array([
    len(e2 & e3) / len(e2) if len(e2) > 0 else 0.0
    for e2, e3 in zip(E2_test, E3_test)
])
consistency_1_3 = np.array([
    len(e1 & e3) / len(e1) if len(e1) > 0 else 0.0
    for e1, e3 in zip(E1_test, E3_test)
])

# ================= Faithfulness (deletion AOPC) for HOP3 (chained) =================
def faithfulness_deletion(clf, X_row_dense, terms, explanation, predicted_class_idx):
    """Probability drop when explanation terms are zeroed out."""
    if len(explanation) == 0:
        return 0.0
    base_prob = clf.predict_proba(X_row_dense.reshape(1, -1))[0][predicted_class_idx]
    modified = X_row_dense.copy()
    term_set_idx = [i for i, t in enumerate(terms) if t in explanation]
    modified[term_set_idx] = 0.0
    new_prob = clf.predict_proba(modified.reshape(1, -1))[0][predicted_class_idx]
    return base_prob - new_prob

faith_hop3 = np.array([
    faithfulness_deletion(clf_chained, X_test_final_dense[i], final_terms, E3_test[i], pred_class_indices[i])
    for i in range(X_test_final_dense.shape[0])
])

# ================= BASELINE: single-agent, non-chained =================
clf_baseline = LogisticRegression(max_iter=2000, random_state=RNG_SEED)
clf_baseline.fit(X_train_full, y_train)
y_pred_baseline = clf_baseline.predict(X_test_full)
acc_baseline = accuracy_score(y_test, y_pred_baseline)
f1_baseline = f1_score(y_test, y_pred_baseline, average="macro")

class_order_b = clf_baseline.classes_
pred_class_indices_b = [list(class_order_b).index(p) for p in y_pred_baseline]
X_test_full_dense = X_test_full.toarray()

E_base_test = []
for i in range(X_test_full_dense.shape[0]):
    expl, _ = linear_explanation(clf_baseline, X_test_full_dense[i], vocab, TOP_K, pred_class_indices_b[i])
    E_base_test.append(expl)

fid_baseline = fidelity_at_k(E_base_test, y_test)

faith_baseline = np.array([
    faithfulness_deletion(clf_baseline, X_test_full_dense[i], vocab, E_base_test[i], pred_class_indices_b[i])
    for i in range(X_test_full_dense.shape[0])
])

# ================= Statistical tests =================
t_12, p_12 = stats.ttest_rel(fid_hop1, fid_hop2)
t_23, p_23 = stats.ttest_rel(fid_hop2, fid_hop3)
t_13, p_13 = stats.ttest_rel(fid_hop1, fid_hop3)
t_base_chain, p_base_chain = stats.ttest_ind(fid_baseline, fid_hop3)
t_faith, p_faith = stats.ttest_ind(faith_baseline, faith_hop3)

results = {
    "n_test": len(y_test),
    "fidelity": {
        "hop1_mean": float(np.mean(fid_hop1)), "hop1_std": float(np.std(fid_hop1)),
        "hop2_mean": float(np.mean(fid_hop2)), "hop2_std": float(np.std(fid_hop2)),
        "hop3_mean": float(np.mean(fid_hop3)), "hop3_std": float(np.std(fid_hop3)),
        "baseline_mean": float(np.mean(fid_baseline)), "baseline_std": float(np.std(fid_baseline)),
    },
    "consistency": {
        "hop1_to_hop2_mean": float(np.mean(consistency_1_2)),
        "hop2_to_hop3_mean": float(np.mean(consistency_2_3)),
        "hop1_to_hop3_mean": float(np.mean(consistency_1_3)),
    },
    "faithfulness": {
        "chained_mean": float(np.mean(faith_hop3)), "chained_std": float(np.std(faith_hop3)),
        "baseline_mean": float(np.mean(faith_baseline)), "baseline_std": float(np.std(faith_baseline)),
    },
    "performance": {
        "chained_accuracy": float(acc_chained), "chained_macro_f1": float(f1_chained),
        "baseline_accuracy": float(acc_baseline), "baseline_macro_f1": float(f1_baseline),
    },
    "significance": {
        "hop1_vs_hop2": {"t": float(t_12), "p": float(p_12)},
        "hop2_vs_hop3": {"t": float(t_23), "p": float(p_23)},
        "hop1_vs_hop3": {"t": float(t_13), "p": float(p_13)},
        "baseline_vs_chained_fidelity": {"t": float(t_base_chain), "p": float(p_base_chain)},
        "baseline_vs_chained_faithfulness": {"t": float(t_faith), "p": float(p_faith)},
    },
    "config": {"top_k": TOP_K, "top_m": TOP_M, "n_classes": len(classes), "n_docs": len(docs)},
}

with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
