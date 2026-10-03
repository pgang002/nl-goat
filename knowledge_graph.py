"""
Builds the Stage 1 output artifact: a structured Vulnerability Knowledge
Base combining NER-extracted entities, ATT&CK technique predictions, and
exploit-db cross-references, plus a FAISS vector index over the raw text
for retrieval (this is what Stage 2's agent will query via RAG).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from sentence_transformers import SentenceTransformer
import faiss

from attack_classifier import (
    predict_techniques,
    OUTPUT_PATH as ATTACK_MODEL_PATH,
    MLB_PATH as ATTACK_LABEL_PATH,
)
from config import MODELS_DIR, PROCESSED_DIR, RAW_DIR

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
KG_OUTPUT_PATH = Path(PROCESSED_DIR) / "vulnerability_knowledge_base.json"
FAISS_INDEX_PATH = Path(MODELS_DIR) / "vuln_kb.faiss"
FAISS_META_PATH = Path(MODELS_DIR) / "vuln_kb_meta.json"


def load_all_cve_records() -> list[dict]:
    cve_files = list(Path(RAW_DIR).glob("clean_cwe*.json"))

    if not cve_files:
        raise FileNotFoundError(
            f"No clean_cwe*.json files found in {RAW_DIR}. Run nvd_client.py first."
        )

    records = []
    for path in cve_files:
        with open(path, encoding="utf-8") as f:
            records.extend(json.load(f))

    return records


def load_exploitdb_index() -> dict[str, list[dict]]:
    """Map cve_id -> list of matched exploit-db entries."""
    path = Path(RAW_DIR) / "exploitdb_matches.json"
    if not path.exists():
        logger.warning("%s not found — exploit-db cross-references will be empty.", path)
        return {}
    with open(path, encoding="utf-8") as f:
        matches = json.load(f)
    index: dict[str, list[dict]] = {}
    for entry in matches:
        for cve_id in entry.get("cve_ids", []):
            index.setdefault(cve_id, []).append(entry)
    return index


def build_knowledge_base() -> list[dict]:
    cve_records = load_all_cve_records()
    exploit_index = load_exploitdb_index()

    descriptions = [r.get("description", "") for r in cve_records]
    if not ATTACK_MODEL_PATH.exists() or not ATTACK_LABEL_PATH.exists():
        raise FileNotFoundError(
            "ATT&CK classifier files not found. Run attack_classifier.py first."
        )

    technique_predictions = predict_techniques(descriptions) if descriptions else []

    kb_entries = []
    for record, techniques in zip(cve_records, technique_predictions):
        cve_id = record.get("cve_id")
        kb_entries.append({
            "cve_id": cve_id,
            "description": record.get("description"),
            "cwe_ids": record.get("cwe_ids", []),
            "affected_cpe": record.get("affected_cpe", []),
            "cvss_severity": record.get("cvss_severity"),
            "predicted_attack_techniques": techniques,
            "known_exploits": exploit_index.get(cve_id, []),
        })

    with open(KG_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(kb_entries, f, indent=2)
    logger.info("Wrote %d knowledge base entries to %s", len(kb_entries), KG_OUTPUT_PATH)
    return kb_entries


def build_vector_index(kb_entries: list[dict]):
    """Build a FAISS index over CVE descriptions for Stage 2's RAG retrieval."""
    if not kb_entries:
        logger.warning("No knowledge base entries — skipping vector index build.")
        return

    embedder = SentenceTransformer(EMBEDDING_MODEL)
    texts = [entry["description"] or "" for entry in kb_entries]
    embeddings = embedder.encode(texts, show_progress_bar=True).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, str(FAISS_INDEX_PATH))

    # Store row -> cve_id mapping so retrieval results can be mapped back to KB entries
    meta = [{"row": i, "cve_id": entry["cve_id"]} for i, entry in enumerate(kb_entries)]
    with open(FAISS_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info("Saved FAISS index (%d vectors, dim=%d) to %s", index.ntotal, dim, FAISS_INDEX_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    kb = build_knowledge_base()
    build_vector_index(kb)
