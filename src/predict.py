from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one prediction with the fine-tuned model.")
    parser.add_argument("--model-dir", type=Path, default=Path("models/minilm-eval-lab"))
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--max-length", type=int, default=160)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.eval()

    inputs = tokenizer(args.text, return_tensors="pt", truncation=True, max_length=args.max_length)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)[0]
        pred_id = int(torch.argmax(probs))

    label = model.config.id2label[pred_id]
    confidence = float(probs[pred_id])
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.4f}")


if __name__ == "__main__":
    main()
