"""
Multi-label classifier mapping free-text vulnerability descriptions to
MITRE ATT&CK techniques.

Why this isn't just a CWE lookup: many real-world text sources (vendor
advisories, blog writeups, exploit-db descriptions) don't carry a
structured CWE field at all — only NVD does. This classifier trains on
NVD records (which *do* have CWE -> technique via the static seed table
in config.py) so it can generalize to text sources that lack structured
CWE metadata, by learning the linguistic patterns that correlate with
each technique.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
from sentence_transformers import SentenceTransformer 
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from config import CWE_TO_ATTACK_TECHNIQUE, MODELS_DIR, RAW_DIR

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OUTPUT_PATH = Path(MODELS_DIR) / "attack_classifier.joblib"
MLB_PATH = Path(MODELS_DIR) / "attack_label_binarizer.joblib"


def build_weak_labels() -> tuple[list[str], list[list[str]]]:
    """
    Build (text, technique_ids[]) training pairs from NVD records using
    the static CWE -> technique seed table as the label source.
    """
    cve_files = list(Path(RAW_DIR).glob("clean_cwe*.json"))

    if not cve_files:
        raise FileNotFoundError(
            f"No clean_cwe*.json files found in {RAW_DIR}. Run nvd_client.py first."
        )

    texts, label_sets = [], []

    for path in cve_files:
        with open(path, encoding="utf-8") as f:
            records = json.load(f)

        for record in records:
            description = record.get("description", "")
            cwe_ids = record.get("cwe_ids", [])

            techniques = set()

            for cwe in cwe_ids:
                for entry in CWE_TO_ATTACK_TECHNIQUE.get(cwe, []):
                    techniques.add(entry["technique_id"])

            if description and techniques:
                texts.append(description)
                label_sets.append(sorted(techniques))

    return texts, label_sets


def train_classifier():
    texts, label_sets = build_weak_labels()
    if not texts:
        raise RuntimeError("No labeled examples found — run nvd_client.py first.")

    logger.info("Training ATT&CK classifier on %d examples", len(texts))

    mlb = MultiLabelBinarizer() 
    y = mlb.fit_transform(label_sets)

    embedder = SentenceTransformer(EMBEDDING_MODEL)
    X = embedder.encode(texts, show_progress_bar=True)

    clf = OneVsRestClassifier(LogisticRegression(max_iter=1000))
    clf.fit(X, y)

    joblib.dump(clf, OUTPUT_PATH)
    joblib.dump(mlb, MLB_PATH)
    logger.info("Saved classifier to %s", OUTPUT_PATH)
    return clf, mlb, embedder


def predict_techniques(texts: list[str], clf=None, mlb=None, embedder=None) -> list[list[str]]:
    """Run inference: free text -> predicted ATT&CK technique IDs."""
    if clf is None or mlb is None:
        clf = joblib.load(OUTPUT_PATH)
        mlb = joblib.load(MLB_PATH)
    if embedder is None:
        embedder = SentenceTransformer(EMBEDDING_MODEL)

    X = embedder.encode(texts, show_progress_bar=False)
    y_pred = clf.predict(X)
    return [list(mlb.classes_[row.astype(bool)]) for row in y_pred]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_classifier()
