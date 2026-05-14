from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


def load_text_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_columns = {"text", "label"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Dataset must contain columns: {sorted(required_columns)}")
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str)
    return df


def encode_labels(labels: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    unique_labels = sorted(set(labels))
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    return label_to_id, id_to_label


def split_dataframe(df: pd.DataFrame, test_size: float, seed: int):
    stratify = df["label"] if df["label"].value_counts().min() >= 2 else None
    return train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )


def compute_metrics_dict(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def save_confusion_matrix(y_true: List[int], y_pred: List[int], class_names: List[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    display.plot(cmap="Blues", xticks_rotation=35)
    plt.title("Cybersecurity Text Classification")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
