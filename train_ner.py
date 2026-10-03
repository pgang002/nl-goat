"""
Fine-tunes a transformer token-classification model
on the CVE description dataset
produced by gpt_annotator.py.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from config import MODELS_DIR, NER_LABELS, PROCESSED_DIR

logger = logging.getLogger(__name__)

BASE_MODEL = "distilbert-base-uncased"  # small + fast; swap for a larger model once the pipeline is validated
TRAIN_FILE = Path(PROCESSED_DIR) / "ner_silver_dataset.jsonl"  # point to hand-corrected file when available
OUTPUT_DIR = Path(MODELS_DIR) / "ner_model"

LABEL2ID = {label: i for i, label in enumerate(NER_LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}


def load_dataset(path: Path) -> Dataset:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return Dataset.from_list(records)


def tokenize_and_align_labels(examples, tokenizer):
    tokenized = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True,
        padding="max_length", max_length=128,
    )
    all_labels = []
    for i, tags in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != prev_word_id:
                label_ids.append(LABEL2ID.get(tags[word_id], LABEL2ID["O"]))
            else:
                label_ids.append(-100)  # only label the first subtoken of each word
            prev_word_id = word_id
        all_labels.append(label_ids)
    tokenized["labels"] = all_labels
    return tokenized


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    true_labels, true_preds = [], []
    for pred_row, label_row in zip(predictions, labels):
        for p, l in zip(pred_row, label_row):
            if l != -100:
                true_labels.append(ID2LABEL[l])
                true_preds.append(ID2LABEL[p])

    correct = sum(p == l for p, l in zip(true_preds, true_labels))
    accuracy = correct / max(len(true_labels), 1)
    return {"token_accuracy": accuracy}


def main():
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"{TRAIN_FILE} not found — run gpt_annotator.py first "
            "(and ideally spot-check a sample of it)."
        )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL, num_labels=len(NER_LABELS), id2label=ID2LABEL, label2id=LABEL2ID
    )

    dataset = load_dataset(TRAIN_FILE)
    dataset = dataset.train_test_split(test_size=0.15, seed=42)

    tokenized = dataset.map(
        lambda ex: tokenize_and_align_labels(ex, tokenizer), batched=True
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=8,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="token_accuracy",
        report_to= "none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    logger.info("Saved fine-tuned NER model to %s", OUTPUT_DIR)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
