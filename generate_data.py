"""
Synthetic multi-class text classification benchmark with KNOWN ground-truth
informative ("signal") words per class. This allows rigorous, objective
measurement of explanation fidelity at each stage of a multi-agent pipeline,
since we know exactly which features SHOULD be considered important.

Design:
- 5 classes, each with a disjoint set of 12 "signal" words strongly
  associated with that class (embedded via a base occurrence probability).
- A shared pool of 150 "noise" words drawn uniformly across all classes
  (semantically irrelevant to the label).
- Each document = signal words (with class-specific probability) + noise
  words (background rate), shuffled, length 25-60 tokens.
- This mirrors a real bag-of-words classification setting while giving us
  ground truth for fidelity measurement.
"""
import numpy as np
import json

rng = np.random.default_rng(42)

CLASSES = ["technology", "sports", "health", "finance", "education"]

SIGNAL_WORDS = {
    "technology": ["algorithm","software","processor","network","database","encryption",
                   "server","bandwidth","firmware","compiler","semiconductor","interface"],
    "sports": ["tournament","striker","referee","stadium","athlete","scoreboard",
               "midfielder","championship","sprint","dribble","goalpost","relay"],
    "health": ["diagnosis","antibody","symptom","vaccine","clinician","dosage",
               "pathogen","therapy","biopsy","cardiac","immunity","prescription"],
    "finance": ["dividend","portfolio","interest","equity","liquidity","inflation",
                "hedge","valuation","amortize","treasury","leverage","yield"],
    "education": ["curriculum","pedagogy","assessment","syllabus","tutor","semester",
                  "literacy","enrollment","scholarship","classroom","rubric","thesis"],
}

NOISE_POOL = [f"word{i}" for i in range(150)]  # generic filler tokens

N_PER_CLASS = 300          # 5 classes * 300 = 1500 documents
SIGNAL_PROB = 0.55         # prob each signal word for the class appears
NOISE_MIN, NOISE_MAX = 15, 40

def generate_document(cls):
    tokens = []
    for w in SIGNAL_WORDS[cls]:
        if rng.random() < SIGNAL_PROB:
            reps = rng.integers(1, 3)
            tokens.extend([w] * reps)
    # small chance of "leaked" signal words from other classes (realistic noise)
    for other_cls in CLASSES:
        if other_cls == cls:
            continue
        for w in SIGNAL_WORDS[other_cls]:
            if rng.random() < 0.04:
                tokens.append(w)
    n_noise = rng.integers(NOISE_MIN, NOISE_MAX)
    tokens.extend(rng.choice(NOISE_POOL, size=n_noise).tolist())
    rng.shuffle(tokens)
    return " ".join(tokens)

def main():
    docs, labels = [], []
    for cls in CLASSES:
        for _ in range(N_PER_CLASS):
            docs.append(generate_document(cls))
            labels.append(cls)

    idx = rng.permutation(len(docs))
    docs = [docs[i] for i in idx]
    labels = [labels[i] for i in idx]

    with open("dataset.json", "w") as f:
        json.dump({"docs": docs, "labels": labels,
                   "signal_words": SIGNAL_WORDS, "classes": CLASSES}, f)

    print(f"Generated {len(docs)} documents across {len(CLASSES)} classes.")
    print(f"Example doc: {docs[0][:120]}...")
    print(f"Example label: {labels[0]}")

if __name__ == "__main__":
    main()
