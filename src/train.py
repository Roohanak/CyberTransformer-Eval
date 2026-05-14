from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.utils import (
    compute_metrics_dict,
    encode_labels,
    load_text_dataset,
    save_confusion_matrix,
    save_json,
    split_dataframe,
)


def tokenize_dataset(dataset: Dataset, tokenizer, max_length: int) -> Dataset:
    return dataset.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_length),
        batched=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a transformer for cybersecurity text classification.")
    parser.add_argument("--data-path", type=Path, default=Path("data/sample_cyber_text.csv"))
    parser.add_argument("--model-name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--output-dir", type=Path, default=Path("models/minilm-eval-lab"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    df = load_text_dataset(args.data_path)
    label_to_id, id_to_label = encode_labels(df["label"].tolist())
    df["label_id"] = df["label"].map(label_to_id)

    train_df, test_df = split_dataframe(df, args.test_size, args.seed)

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(train_df[["text"]], train_df["label_id"])
    baseline_pred = baseline.predict(test_df[["text"]]).tolist()
    baseline_metrics = compute_metrics_dict(test_df["label_id"].tolist(), baseline_pred)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(label_to_id),
        id2label={idx: label for idx, label in id_to_label.items()},
        label2id=label_to_id,
    )

    train_dataset = Dataset.from_pandas(train_df[["text", "label_id"]].rename(columns={"label_id": "labels"}))
    test_dataset = Dataset.from_pandas(test_df[["text", "label_id"]].rename(columns={"label_id": "labels"}))

    train_dataset = tokenize_dataset(train_dataset, tokenizer, args.max_length)
    test_dataset = tokenize_dataset(test_dataset, tokenizer, args.max_length)

    def compute_eval_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=1)
        return compute_metrics_dict(labels.tolist(), predictions.tolist())

    training_config = {
        "output_dir": str(args.output_dir),
        "save_strategy": "epoch",
        "learning_rate": 2e-5,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "num_train_epochs": args.epochs,
        "weight_decay": 0.01,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1_macro",
        "logging_steps": 5,
        "report_to": [],
        "seed": args.seed,
    }

    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        training_config["eval_strategy"] = "epoch"
    else:
        training_config["evaluation_strategy"] = "epoch"

    training_args = TrainingArguments(**training_config)

    trainer_config = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": test_dataset,
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": compute_eval_metrics,
    }

    trainer_signature = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_config["processing_class"] = tokenizer
    elif "tokenizer" in trainer_signature.parameters:
        trainer_config["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_config)

    trainer.train()
    predictions = trainer.predict(test_dataset)

    y_true = predictions.label_ids.tolist()
    y_pred = np.argmax(predictions.predictions, axis=1).tolist()
    fine_tuned_metrics = compute_metrics_dict(y_true, y_pred)

    class_names = [id_to_label[idx] for idx in sorted(id_to_label)]
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    print(report)

    errors = test_df.copy()
    errors["predicted_label"] = [id_to_label[pred] for pred in y_pred]
    errors = errors[errors["label"] != errors["predicted_label"]][["text", "label", "predicted_label"]]
    errors.to_csv(args.results_dir / "error_analysis.csv", index=False)

    save_confusion_matrix(y_true, y_pred, class_names, args.results_dir / "confusion_matrix.png")

    save_json(
        {
            "model_name": args.model_name,
            "labels": label_to_id,
            "baseline": baseline_metrics,
            "fine_tuned": fine_tuned_metrics,
            "classification_report": report,
        },
        args.results_dir / "metrics.json",
    )

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Saved model to {args.output_dir}")
    print(f"Saved metrics to {args.results_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
