from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer

from src.utils import (
    compute_metrics_dict,
    encode_labels,
    load_text_dataset,
    save_confusion_matrix,
    save_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned transformer model.")
    parser.add_argument("--data-path", type=Path, default=Path("data/sample_cyber_text.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/minilm-eval-lab"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=160)
    args = parser.parse_args()

    df = load_text_dataset(args.data_path)
    label_to_id, id_to_label = encode_labels(df["label"].tolist())
    df["label_id"] = df["label"].map(label_to_id)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    dataset = Dataset.from_pandas(df[["text", "label_id"]].rename(columns={"label_id": "labels"}))
    dataset = dataset.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=args.max_length),
        batched=True,
    )

    trainer = Trainer(
        model=model,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    predictions = trainer.predict(dataset)
    y_true = predictions.label_ids.tolist()
    y_pred = np.argmax(predictions.predictions, axis=1).tolist()

    class_names = [id_to_label[idx] for idx in sorted(id_to_label)]
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    print(report)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    save_confusion_matrix(y_true, y_pred, class_names, args.results_dir / "confusion_matrix_full_eval.png")
    save_json(
        {
            "model_dir": str(args.model_dir),
            "metrics": compute_metrics_dict(y_true, y_pred),
            "classification_report": report,
        },
        args.results_dir / "eval_metrics.json",
    )


if __name__ == "__main__":
    main()
