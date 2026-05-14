# CyberTransformer-Eval

CyberTransformer-Eval is a small research-style NLP project for cybersecurity text classification using PyTorch and Hugging Face Transformers.

The project fine-tunes transformer models on cybersecurity research paper text and compares their performance against a simple baseline using accuracy, precision, recall, F1-score, confusion matrices, and error analysis.

## Project Goal

Classify cybersecurity research papers into topic categories based on their extracted text.

Current labels:

- cryptography
- intrusion_detection
- machine_learning
- network_security

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- scikit-learn
- Pandas
- Matplotlib
- Jupyter
- pypdf

## Project Structure

```text
CyberTransformer-Eval/
|-- data/
|   |-- papers/
|   |-- sample_cyber_text.csv
|   `-- papers_dataset.csv
|-- notebooks/
|   `-- experiment.ipynb
|-- src/
|   |-- build_dataset_from_pdfs.py
|   |-- train.py
|   |-- evaluate.py
|   |-- predict.py
|   `-- utils.py
|-- models/
|   |-- distilbert/
|   |-- bert/
|   `-- scibert/
|-- results/
|   |-- distilbert/
|   |-- bert/
|   `-- scibert/
|-- requirements.txt
`-- README.md
```

## Dataset

The dataset is built from cybersecurity research papers. PDFs are organized into folders where each folder name is the class label.

```text
data/papers/
|-- cryptography/
|   |-- paper1.pdf
|   `-- paper2.pdf
|-- intrusion_detection/
|   |-- paper3.pdf
|   `-- paper4.pdf
|-- machine_learning/
|   |-- paper5.pdf
|   `-- paper6.pdf
`-- network_security/
    |-- paper7.pdf
    `-- paper8.pdf
```

The dataset builder extracts text from the PDFs and creates a CSV file with:

```text
text,label
```

For this experiment, only the first page of each paper is used because it usually contains the title, abstract, keywords, and topic context.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Build Dataset from PDFs

```powershell
python -m src.build_dataset_from_pdfs --papers-dir data/papers --output-path data/papers_dataset.csv --max-pages 1
```

## Train Models

### DistilBERT

```powershell
python -m src.train --data-path data/papers_dataset.csv --model-name distilbert-base-uncased --output-dir models/distilbert --results-dir results/distilbert --epochs 5
```

### BERT

```powershell
python -m src.train --data-path data/papers_dataset.csv --model-name bert-base-uncased --output-dir models/bert --results-dir results/bert --epochs 5
```

### SciBERT

```powershell
python -m src.train --data-path data/papers_dataset.csv --model-name allenai/scibert_scivocab_uncased --output-dir models/scibert --results-dir results/scibert --epochs 5
```

## Evaluate a Saved Model

```powershell
python -m src.evaluate --model-dir models/scibert --data-path data/papers_dataset.csv
```

## Run a Prediction

```powershell
python -m src.predict --model-dir models/scibert --text "This paper proposes a method for detecting attacks in industrial control systems."
```

## Model Comparison

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| Baseline | 0.1667 | 0.0417 | 0.2500 | 0.0714 |
| BERT | 0.1667 | 0.0833 | 0.1250 | 0.1000 |
| DistilBERT | 0.5000 | 0.2500 | 0.5000 | 0.3333 |
| SciBERT | 0.5000 | 0.3333 | 0.5000 | 0.3750 |

SciBERT achieved the best macro F1-score among the evaluated transformer models. This is expected because SciBERT is pretrained on scientific text, making it a better fit for research paper classification.

DistilBERT matched SciBERT on accuracy but had a lower macro F1-score. BERT performed poorly on this small dataset and only slightly improved over the baseline.

## Results Interpretation

The baseline model uses a majority-class strategy and does not learn from text features. Its macro F1-score was 0.0714.

Fine-tuning improved performance:

- DistilBERT improved macro F1 to 0.3333.
- SciBERT improved macro F1 to 0.3750.
- SciBERT was the strongest model overall.

The evaluation set is small, so the results should be interpreted as a proof-of-concept rather than a large-scale benchmark.

## Error Analysis

The project saves misclassified examples to:

```text
results/<model_name>/error_analysis.csv
```

This file is used to inspect which classes the model confuses.

Observed issues:

- network_security papers were often confused with cryptography or machine_learning.
- machine_learning papers overlapped with security/privacy papers.
- Some papers contain multiple topics, making single-label classification difficult.

## Outputs

Each training run saves:

```text
models/<model_name>/
results/<model_name>/metrics.json
results/<model_name>/confusion_matrix.png
results/<model_name>/error_analysis.csv
```

## What I Learned

This project demonstrates:

- dataset preparation from PDFs
- transformer tokenization
- fine-tuning with PyTorch and Hugging Face
- baseline comparison
- model evaluation using macro F1-score
- confusion matrix analysis
- error analysis on misclassified samples
- comparison of DistilBERT, BERT, and SciBERT

## Limitations

- The dataset is small.
- Some cybersecurity categories overlap.
- Only the first page of each paper is used.
- Results may change with a larger train/test split.
- This is a proof-of-concept experiment, not a production classifier.

## Future Work

- Add more papers per class
- Use abstracts only when available
- Add RoBERTa and DeBERTa models
- Try multi-label classification
- Use a larger dataset with 1,000+ examples
- Add cross-validation
- Add an LLM-based zero-shot and few-shot comparison
- Build a separate larger CyberLLM evaluation project


