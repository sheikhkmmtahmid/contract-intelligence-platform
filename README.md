# Contract Intelligence and Power Imbalance Platform

A legal AI system that ingests commercial contracts, classifies every clause against a unified 100-type taxonomy, detects anomalous language against market-standard patterns, and scores bilateral power imbalance between contracting parties. All outputs are explained at the token level via SHAP.

Built by [SKMMT](https://skmmt.rootexception.com/)

---

## What It Does

| Capability | Technology | Output |
|---|---|---|
| Clause classification | LegalBERT + Legal-RoBERTa-large ensemble, fine-tuned on CUAD + LEDGAR + MAUD (100 types) | Per-clause type labels and probabilities |
| Anomaly detection | Isolation Forest + Shallow Autoencoder on Legal-RoBERTa embeddings | Anomaly risk score 0 to 100, flagged above 70 |
| Power imbalance scoring | RoBERTa sentiment + modal verb analysis + obligation assignment + assertiveness | Party A/B leverage scores, bilateral imbalance index from -100 to +100 |
| Explainability | SHAP KernelExplainer (token-level) | PNG attribution plots per clause |
| Report generation | ReportLab PDF | Downloadable legal-quality PDF report |
| Taxonomy expansion | EDGAR pipeline using UMAP + HDBSCAN + TF-IDF + cosine similarity routing | Auto-discovers and names new clause types from SEC filings |

---

## Architecture

```
contract-intelligence-platform/
+-- config.py                      All paths, hyperparameters, thresholds (single source of truth)
+-- requirements.txt
+-- src/
|   +-- data_pipeline.py           CUAD/LEDGAR/MAUD download, EDGAR ingestion, segmentation, DB
|   +-- clause_classifier.py       LegalBERT + DeBERTa + LegalRoBERTa training, ensemble, inference
|   +-- anomaly_detector.py        Isolation Forest + Autoencoder + score fusion
|   +-- power_scorer.py            Feature-engineered bilateral power imbalance scorer
|   +-- explainability.py          SHAP KernelExplainer for classifier and power scorer
|   +-- report_generator.py        ReportLab PDF generation
+-- api/
|   +-- main.py                    FastAPI application (6 endpoints)
|   +-- schemas.py                 Pydantic v2 request/response models
|   +-- database.py                SQLAlchemy ORM + SQLite
+-- templates/
|   +-- dashboard.html             Single-page dashboard (Chart.js, dark navy/gold theme)
+-- data/
|   +-- raw/                       Raw uploaded contracts
|   +-- processed/                 CSVs, DB, SHAP plots, PDF reports, review queues
|   +-- cuad/                      Cached CUAD dataset
|   +-- ledgar/                    Cached LEDGAR dataset
|   +-- maud/                      Cached MAUD dataset
|   +-- edgar/                     SEC EDGAR raw and processed exhibits
+-- models/
    +-- clause_classifier/
    |   +-- legalbert/best/        Best LegalBERT checkpoint
    |   +-- deberta/best/          Best DeBERTa-v3 checkpoint
    |   +-- legalroberta/best/     Best Legal-RoBERTa-large checkpoint
    |   +-- production_config.json Active ensemble configuration
    +-- anomaly_detector/          isolation_forest.pkl, autoencoder.pt, normaliser.pkl
    +-- power_scorer/              (feature-engineered, no checkpoint needed)
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

GPU is strongly recommended for training. CPU inference works but is slow.

### 2. Download and process training data

```bash
python src/data_pipeline.py --mode all
```

Downloads CUAD (~500MB), LEDGAR (~80k clauses), and MAUD (~39k clauses) from HuggingFace,
processes all clauses, cleans noise (short texts, MAUD redaction tags), and creates
stratified 80/10/10 train/val/test CSV splits.

### 3. Train the clause classifiers

```bash
python src/clause_classifier.py --mode train --backbone legalbert
python src/clause_classifier.py --mode train --backbone deberta
python src/clause_classifier.py --mode train --backbone legalroberta
```

Each backbone trains independently. Legal-RoBERTa-large uses gradient checkpointing
and a smaller batch size (8) to fit in 8GB VRAM.

### 4. Select the best ensemble

```bash
python src/clause_classifier.py --mode select
```

Evaluates all 7 combinations (3 singles, 3 pairs, 1 triple) on the validation set
using a weight grid search. Writes `production_config.json` with the winner.

### 5. Evaluate on the test set

```bash
python src/clause_classifier.py --mode evaluate
```

### 6. Train the anomaly detector

```bash
python src/anomaly_detector.py --mode train
python src/anomaly_detector.py --mode evaluate
```

Generates Legal-RoBERTa embeddings from the fine-tuned classifier checkpoint for all
training clauses (approximately 80 minutes on GPU), then fits Isolation Forest and
trains the Autoencoder.

### 7. Evaluate the power scorer

```bash
python src/power_scorer.py --mode evaluate
```

### 8. Launch the API server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000` for the dashboard, or `http://localhost:8000/docs` for
the interactive API documentation.

---

## Current Evaluation Results

All numbers are on the held-out test set (10% of the full dataset, 100% human-annotated,
drawn exclusively from CUAD, LEDGAR, and MAUD, not EDGAR auto-labels).

### Clause Classifier

| Model | Val F1 Macro | Notes |
|---|---|---|
| LegalBERT (solo) | 0.5706 | `nlpaueb/legal-bert-base-uncased` |
| DeBERTa-v3 (solo) | 0.5803 | `microsoft/deberta-v3-base` |
| Legal-RoBERTa-large (solo) | 0.5756 | `lexlms/legal-roberta-large` |
| **ensemble_legalbert_legalroberta** | **0.6212 val / 0.6058 test** | Production model, w_lb=0.6, w_lr=0.4 |

Test set precision macro: 0.5858. Test set recall macro: 0.6429.

### Why the F1 Is 0.6058 and Not Higher

This is an honest explanation of the current result.

The task is 100-class multi-label classification across three datasets with very
different label distributions, annotation styles, and noise levels. The main factors
limiting F1 are:

**1. Class imbalance at 100 types.**
Frequent types like "Governing Law" and "Indemnification" have thousands of training
examples and score well individually. Rare types like "Source Code Escrow" or
"Benchmark Rights" may have fewer than 100 examples total. Macro F1 weights all
100 classes equally, so poor performance on rare classes pulls the average down
significantly regardless of how well common classes perform.

**2. Label noise in EDGAR-sourced training data.**
EDGAR clauses are auto-labeled by the classifier itself at confidence above 0.70.
This means mislabeled examples enter the training set and teach the model incorrect
associations. EDGAR contributes up to 40% of the training set (the EDGAR_TRAIN_CAP
parameter), and that cap is the main protection against noise dominating.

**3. Hard-to-distinguish adjacent types.**
Several type pairs are semantically very close: License Grant vs Non-Transferable
License, Affiliate License-Licensor vs Affiliate License-Licensee, No-Solicit of
Customers vs No-Solicit of Employees. The model regularly confuses these pairs.
They share vocabulary and sentence structure, differing mainly in which party is
named or a single modifier word.

**4. 512-token truncation.**
Clauses that exceed 512 tokens (roughly 350+ words) are split into overlapping
chunks via sliding window. Some clauses span 1,000 to 1,200 tokens. The chunk
with the highest maximum probability is used for the final label, but this is an
approximation. Long clauses that develop their key language past the 512-token
boundary may be misclassified.

**5. Multi-source dataset heterogeneity.**
CUAD, LEDGAR, and MAUD were each annotated independently with different guidelines,
granularities, and definitions of what constitutes a clause boundary. When merged,
some types from one dataset partially overlap with types from another, creating
ambiguous training signal.

**6. DeBERTa-v3 not contributing to the ensemble.**
The model selection process (grid search over all 7 combinations with per-class
threshold optimisation) determined that adding DeBERTa gives zero improvement
(w_db=0.0 in the winning config). DeBERTa likely overfits differently from
LegalBERT and Legal-RoBERTa, adding noise rather than complementary signal.

### Probable Ways to Improve

In rough order of expected impact:

| Approach | Expected Gain | Effort |
|---|---|---|
| More human-annotated data for rare types | High | High |
| Per-class weighted loss (focal loss or class weights) | Medium | Low |
| Use Legal-RoBERTa-large as sole backbone with longer training | Medium | Low |
| Resolve adjacent-type confusion via type hierarchy | Medium | Medium |
| Reduce EDGAR noise cap or raise auto-label confidence threshold | Low to medium | Low |
| Contrastive learning to separate similar types | Medium | High |
| Hierarchical classification (coarse type first, fine type second) | High | High |
| Better handling of long clauses (Longformer or chunk aggregation) | Medium | Medium |

A realistic ceiling with the current datasets and architecture is approximately
F1 Macro 0.70 to 0.72. Reaching F1 above 0.80 would require substantially more
high-quality annotated data for the 40+ rare types.

### Anomaly Detector

Evaluated using proxy labels (clauses shorter than 15 words or longer than the
95th percentile by word count as approximate anomaly indicators). No ground-truth
anomaly labels exist in the training datasets.

| Metric | Value |
|---|---|
| Precision@50 | 0.24 |
| Recall@50 | 0.033 |

These numbers reflect the weakness of the proxy label, not necessarily the
detector quality. Clauses with extreme word counts are not the same as legally
anomalous clauses. See `model_card.md` for a full discussion.

### Power Scorer

Rule-based, no supervised labels available for calibration.

| Metric | Value |
|---|---|
| Mean imbalance score (test set) | 2.19 |
| Standard deviation | 27.49 |
| Consistency (same clause, same result) | True |

---

## EDGAR Taxonomy Expansion Pipeline

The platform can learn new clause types from SEC EDGAR EX-10 filings continuously.

```
Download EX-10 exhibits from SEC EDGAR
    clause segmentation
    AutoLabeler (trained classifier, above 0.70 confidence goes to DB)
    Unknown clauses go to UMAP + HDBSCAN clustering
        SimilarityRouter (cosine similarity to existing 100 type embeddings)
            Above 0.80 similarity: relabel to matched type and save to DB
            0.50 to 0.80 similarity: taxonomy_review.csv for optional human review
            Below 0.50 similarity: TaxonomyExpander
                Cluster size above 50: auto-name, add to dynamic_taxonomy.json, save to DB
                Cluster size 10 to 49: taxonomy_review.csv
                Cluster size below 10: noise, discard
```

```bash
python src/data_pipeline.py --mode edgar --limit 5000
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | /health | Service health check |
| POST | /analyse | Analyse a contract (PDF upload or raw text) |
| GET | /clauses/{contract_id} | All clauses with scores |
| GET | /anomalies/{contract_id} | Anomalous clauses only |
| GET | /imbalance/{contract_id} | Power imbalance breakdown |
| GET | /report/{contract_id} | Download PDF report |

### Example: Analyse a contract via cURL

```bash
curl -X POST http://localhost:8000/analyse \
  -F "file=@contract.pdf" \
  -F "party_a=Acme Corp" \
  -F "party_b=Client Ltd"
```

---

## Key Design Decisions

**Why 100-type unified taxonomy?**
CUAD covers 41 core commercial provisions. LEDGAR adds 32 boilerplate types. MAUD
adds 14 M&A deal-point types. Merging all three into a single label space gives
broader coverage without splitting training data across incompatible schemas.

**Why three backbones?**
LegalBERT has domain-specific legal pre-training. DeBERTa-v3 has stronger
disentangled attention. Legal-RoBERTa-large is a 125M-parameter model trained
on the Pile of Law corpus, a much larger legal text collection than LegalBERT's
training data. Training all three and selecting the best ensemble by validation
F1 is more reliable than committing to one architecture upfront.

**Why feature-engineered power imbalance scoring?**
No publicly available dataset of human-annotated power imbalance labels exists
for commercial contracts. The feature-engineering approach is academically grounded,
transparent, and reproducible. See `model_card.md` for the full feature architecture
and references.

**Why dual anomaly signals?**
Isolation Forest detects global outliers in the embedding space efficiently.
The Autoencoder captures structural reconstruction anomalies that Isolation Forest
can miss. Combining both with equal weights reduces false positives compared to
either detector alone.

---

## Configuration

All hyperparameters, thresholds, and file paths live in `config.py`. Nothing is
hardcoded elsewhere. Key parameters:

```python
NUM_CLAUSE_TYPES              = 100    # base taxonomy size
ANOMALY_FLAG_THRESHOLD        = 70     # 0 to 100 scale
IMBALANCE_HIGH_THRESHOLD      = 40     # |score| above 40 is HIGH
EDGAR_AUTO_LABEL_CONFIDENCE   = 0.70   # minimum classifier confidence to auto-accept EDGAR label
EDGAR_TRAIN_CAP               = 0.40   # maximum EDGAR fraction of training set
UNDERSAMPLE_MAX_PER_CLASS     = None   # set to integer to cap per class; None disables
```

---

## Legal Disclaimer

This platform is an AI research tool. It does not provide legal advice. All outputs
are probabilistic estimates that must be reviewed and verified by qualified legal
counsel before reliance. Clause classifications, anomaly flags, and power imbalance
scores may contain errors.
