"""
LLM-assisted NER annotation using the OpenAI GPT API.

This script REPLACES both ner_bootstrap.py and the manual annotation
step. GPT annotates CVE descriptions with security-specific NER labels,
producing a higher-quality training set than regex heuristics alone.

Run order for Stage 1:
  1. nvd_client.py          ← collect CVE data  (run this first)
  2. exploitdb_client.py    ← cross-reference exploits
  3. THIS SCRIPT            ← replaces ner_bootstrap.py + hand-correction
  4. train_ner.py           ← fine-tune NER model on GPT-annotated labels
  5. attack_classifier.py   ← train ATT&CK classifier
  6. knowledge_graph.py     ← build final KB + vector index

Cost estimate (approximate, mid-2025 pricing):
  gpt-4o-mini → ~$0.002 per CVE;  200 CVEs ≈ $0.40  (recommended)
  gpt-4o      → ~$0.010 per CVE;  200 CVEs ≈ $2.00  (higher quality)

Setup:
  Add to your .env file (same file as NVD_API_KEY):
    OPENAI_API_KEY=your-key-here

Usage:
  python gpt_annotator.py                        # annotate all collected CVEs
  python gpt_annotator.py --max-records 50       # quick test run
  python gpt_annotator.py --spot-check 10        # print 10 random labeled examples to verify quality
  python gpt_annotator.py --model gpt-4o         # use a stronger model
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import time
from pathlib import Path

from openai import OpenAI

from config import PROCESSED_DIR, RAW_DIR
from preprocessing import clean_text

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4o-mini"
OUTPUT_PATH = Path(PROCESSED_DIR) / "ner_silver_dataset.jsonl"
RATE_LIMIT_DELAY = 0.5  # seconds between API calls — adjust if you hit rate limits

# ---------------------------------------------------------------------------
# Prompt design
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a cybersecurity NER (Named Entity Recognition) annotator.
Your job is to extract named entities from CVE vulnerability descriptions.

Entity types and definitions:
  PRODUCT       — software or hardware product name
                  e.g. "Apache Struts", "OpenSSL", "WordPress", "Cisco IOS"
  VERSION       — version number or version range string
                  e.g. "2.3.5", "before 2.3.32", "1.x through 1.4.2", "prior to 9.8"
  VULN_TYPE     — the category or class of vulnerability
                  e.g. "SQL injection", "remote code execution", "buffer overflow", "XSS"
  ATTACK_VECTOR — the method or mechanism an attacker uses to trigger the vulnerability
                  e.g. "via a crafted HTTP request", "through the username parameter"
                  IMPORTANT: stop before any "in the <component>" phrase — that is COMPONENT
  COMPONENT     — a specific internal module, function, or subcomponent that is vulnerable
                  e.g. "Jakarta Multipart parser", "login module", "XML parser", "mod_cgi"

Rules:
  - Only extract entities explicitly present in the text — do not infer
  - Each entity text must be an exact substring of the input description
  - ATTACK_VECTOR and COMPONENT are separate: "via a crafted header in the Foo parser"
    → ATTACK_VECTOR = "via a crafted header", COMPONENT = "Foo parser"
  - Return ONLY valid JSON, no explanation, no markdown fences
  - Format: {"entities": [{"text": "<exact substring>", "label": "<TYPE>"}, ...]}
  - If a type has no match in the text, omit it from the list entirely
"""

# Two few-shot examples shown to GPT before every annotation request.
# These are the clearest signal for what we want — keep them accurate.
FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": (
            "Annotate this CVE description:\n"
            "Apache Struts versions 2.3.5 contain a remote code execution vulnerability "
            "exploitable via a crafted Content-Type header in the Jakarta Multipart parser."
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "entities": [
                {"text": "Struts",                          "label": "PRODUCT"},
                {"text": "2.3.5",                           "label": "VERSION"},
                {"text": "remote code execution",           "label": "VULN_TYPE"},
                {"text": "via a crafted Content-Type header", "label": "ATTACK_VECTOR"},
                {"text": "Jakarta Multipart parser",        "label": "COMPONENT"},
            ]
        }),
    },
    {
        "role": "user",
        "content": (
            "Annotate this CVE description:\n"
            "SQL injection vulnerability in ExampleCMS 3.2.1 and earlier allows remote "
            "attackers to execute arbitrary SQL commands via the username parameter in "
            "the login module."
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "entities": [
                {"text": "ExampleCMS",                      "label": "PRODUCT"},
                {"text": "3.2.1 and earlier",               "label": "VERSION"},
                {"text": "SQL injection",                   "label": "VULN_TYPE"},
                {"text": "via the username parameter",      "label": "ATTACK_VECTOR"},
                {"text": "login module",                    "label": "COMPONENT"},
            ]
        }),
    },
]


# ---------------------------------------------------------------------------
# Core annotation logic
# ---------------------------------------------------------------------------

def _annotate_single(client: OpenAI, text: str, model: str) -> list[dict]:
    """
    Call GPT to annotate one CVE description. Returns a list of
    {"text": ..., "label": ...} entity dicts, or [] on failure.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *FEW_SHOT_EXAMPLES,
        {"role": "user", "content": f"Annotate this CVE description:\n{text}"},
    ]
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,   # deterministic — annotation is not a creative task
            max_tokens=512,
        )
        raw = response.choices[0].message.content.strip()
        parsed = json.loads(raw)
        return parsed.get("entities", [])
    except json.JSONDecodeError:
        logger.warning("GPT returned non-JSON response — skipping record.")
        return []
    except Exception as e:
        logger.warning("GPT API error: %s — skipping record.", e)
        return []


def _entities_to_spans(
    text: str, entities: list[dict]
) -> list[tuple[int, int, str]]:
    """
    Convert entity dicts ({"text": ..., "label": ...}) to character-offset
    spans by locating each entity's exact text in the description.
    Multiple matches: take the first occurrence (fine for short CVE text).
    """
    spans = []
    for entity in entities:
        needle = entity.get("text", "").strip()
        label = entity.get("label", "")
        if not needle or not label:
            continue
        match = re.search(re.escape(needle), text, flags=re.IGNORECASE)
        if match:
            spans.append((match.start(), match.end(), label))
        else:
            logger.debug("Entity text %r not found in: %r", needle, text[:80])
    return spans




def _spans_to_bio(text: str, spans: list[tuple[int, int, str]]) -> tuple[list[str], list[str]]:
    """
    Convert character-offset entity spans into whitespace-tokenized BIO tags.

    Each span is a tuple: (start_character, end_character, label).
    Example span: (15, 29, "VULN_TYPE").

    Output:
      tokens   -> words from the text
      ner_tags -> BIO labels aligned with those words
    """
    spans = sorted(spans, key=lambda s: s[0])
    tokens, tags = [], []
    offset = 0

    for token in text.split():
        start = text.find(token, offset)
        if start == -1:
            start = offset
        end = start + len(token)
        offset = end

        label = "O"
        for s_start, s_end, s_label in spans:
            if start >= s_start and end <= s_end:
                label = f"I-{s_label}" if start > s_start else f"B-{s_label}"
                break

        tokens.append(token)
        tags.append(label)

    return tokens, tags

def annotate_record(client: OpenAI, record: dict, model: str) -> dict:
    """Annotate one CVE record and return a training example in JSONL format."""
    text = clean_text(record.get("description", ""))
    entities = _annotate_single(client, text, model)
    spans = _entities_to_spans(text, entities)
    tokens, tags = _spans_to_bio(text, spans)
    return {
        "cve_id": record.get("cve_id"),
        "text": text,
        "tokens": tokens,
        "ner_tags": tags,
    }


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def load_cve_records() -> list[dict]:
    records = []
    for path in Path(RAW_DIR).glob("clean_cwe*.json"):
        with open(path, encoding="utf-8") as f:
            records.extend(json.load(f))
    return records


def build_annotated_dataset(
    max_records: int | None = None,
    model: str = DEFAULT_MODEL,
) -> list[dict]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not set. Add it to your .env file: OPENAI_API_KEY=your-key-here"
        )
    client = OpenAI(api_key=api_key)

    records = load_cve_records()
    if not records:
        raise RuntimeError(
            f"No CVE records found in {RAW_DIR} — run nvd_client.py first."
        )
    if max_records:
        records = records[:max_records]

    logger.info("Annotating %d CVE records with %s ...", len(records), model)
    examples = []
    for i, record in enumerate(records, 1):
        if not record.get("description"):
            continue
        example = annotate_record(client, record, model)
        examples.append(example)
        if i % 10 == 0:
            logger.info("  Annotated %d / %d", i, len(records))
        time.sleep(RATE_LIMIT_DELAY)

    Path(PROCESSED_DIR).mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    logger.info("Saved %d annotated examples to %s", len(examples), OUTPUT_PATH)
    return examples


# ---------------------------------------------------------------------------
# Spot-check: print N random examples so you can verify quality visually
# ---------------------------------------------------------------------------

def spot_check(n: int = 10):
    """
    Print N random examples from the annotated dataset in a readable format.
    Run this after build_annotated_dataset() to sanity-check GPT's labels
    before fine-tuning.
    """
    if not OUTPUT_PATH.exists():
        print(f"No annotated dataset found at {OUTPUT_PATH} — run annotation first.")
        return

    with open(OUTPUT_PATH, encoding="utf-8") as f:
        examples = [json.loads(line) for line in f]

    sample = random.sample(examples, min(n, len(examples)))
    for ex in sample:
        print(f"\n{'='*60}")
        print(f"CVE: {ex['cve_id']}")
        print(f"Text: {ex['text'][:120]}...")
        print(f"{'Token':<25} {'Tag'}")
        print("-" * 40)
        for tok, tag in zip(ex["tokens"], ex["ner_tags"]):
            if tag != "O":
                print(f"{tok:<25} {tag}")   # only print non-O tokens to keep output short
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="GPT-assisted NER annotation for CVE descriptions")
    parser.add_argument("--max-records", type=int, default=None,
                        help="Cap number of CVEs to annotate (useful for a quick test run)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"OpenAI model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--spot-check", type=int, default=0, metavar="N",
                        help="After annotating, print N random examples to verify quality")
    args = parser.parse_args()

    build_annotated_dataset(max_records=args.max_records, model=args.model)

    if args.spot_check > 0:
        spot_check(n=args.spot_check)
