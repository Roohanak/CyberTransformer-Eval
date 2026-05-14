from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from pypdf import PdfReader


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def extract_pdf_text(pdf_path: Path, max_pages: int | None) -> str:
    reader = PdfReader(str(pdf_path))
    pages = reader.pages if max_pages is None else reader.pages[:max_pages]
    text_parts = []

    for page in pages:
        text_parts.append(page.extract_text() or "")

    return clean_text(" ".join(text_parts))


def build_dataset(papers_dir: Path, output_path: Path, max_pages: int | None, min_chars: int) -> None:
    rows = []
    class_dirs = [path for path in sorted(papers_dir.iterdir()) if path.is_dir()]

    if not class_dirs:
        raise ValueError(
            "No label folders found. Create folders like data/papers/phishing/ and put PDFs inside them."
        )

    for class_dir in class_dirs:
        label = class_dir.name
        for pdf_path in sorted(class_dir.glob("*.pdf")):
            text = extract_pdf_text(pdf_path, max_pages=max_pages)
            if len(text) < min_chars:
                print(f"Skipping {pdf_path.name}: only {len(text)} characters extracted")
                continue

            rows.append(
                {
                    "text": text,
                    "label": label,
                    "source_file": pdf_path.name,
                }
            )

    if not rows:
        raise ValueError("No usable PDF text was extracted.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Saved {len(rows)} examples to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a text classification CSV from labeled PDF folders.")
    parser.add_argument("--papers-dir", type=Path, default=Path("data/papers"))
    parser.add_argument("--output-path", type=Path, default=Path("data/papers_dataset.csv"))
    parser.add_argument("--max-pages", type=int, default=3, help="Use first N pages from each paper. Use 0 for all pages.")
    parser.add_argument("--min-chars", type=int, default=500)
    args = parser.parse_args()

    max_pages = None if args.max_pages == 0 else args.max_pages
    build_dataset(args.papers_dir, args.output_path, max_pages=max_pages, min_chars=args.min_chars)


if __name__ == "__main__":
    main()
