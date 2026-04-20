# Model Card: Contract Intelligence and Power Imbalance Platform

**Version:** 2.0.0 | **Date:** April 2026 | **Framework:** PyTorch + HuggingFace Transformers

---

## 1. Model Overview

This platform comprises four distinct model components working in sequence, plus an automated taxonomy expansion pipeline:

| Component | Model | Task | Training Paradigm |
|---|---|---|---|
| Clause Classifier | LegalBERT + Legal-RoBERTa-large ensemble | Multi-label classification (100 types) | Supervised fine-tuning |
| Anomaly Detector (1) | Isolation Forest | Unsupervised anomaly detection | Unsupervised |
| Anomaly Detector (2) | Shallow Autoencoder | Reconstruction-error anomaly detection | Self-supervised |
| Power Imbalance Scorer | Feature engineering + RoBERTa sentiment | Bilateral leverage scoring | Feature-engineered (no labels) |
| Taxonomy Expander | UMAP + HDBSCAN + TF-IDF + cosine similarity | New clause type discovery | Unsupervised |

---

## 2. Intended Use

**Primary use:** Legal technology professionals, contract analysts, and legal teams
analysing commercial contracts for:
- Clause-type classification and organisation across 100 unified types
- Detection of non-standard or anomalous language versus market standard
- Preliminary assessment of power balance between contracting parties
- Explainability of AI-generated contract assessments
- Continuous discovery of emerging clause types from live SEC filings

**Not intended for:**
- Replacement of qualified legal advice
- Binding legal determinations
- Consumer contracts (trained on commercial contracts only)
- Non-English language contracts

---

## 3. Clause Classifier

### Architecture

Three backbones are trained independently. The best-performing ensemble is selected
by validation F1 macro via grid search across all single-backbone and ensemble
combinations. The production model is determined by `production_config.json`.

| Backbone | HuggingFace ID | Parameters | Domain Pre-training |
|---|---|---|---|
| Legal-BERT | `nlpaueb/legal-bert-base-uncased` | 110M | Legal text corpus |
| DeBERTa-v3 | `microsoft/deberta-v3-base` | 183M | General (disentangled attention) |
| Legal-RoBERTa-large | `lexlms/legal-roberta-large` | 355M | Pile of Law (large-scale legal corpus) |

All three use a linear classification head over the encoder output with BCEWithLogitsLoss.
DeBERTa uses mean-pooling over last hidden states (its AutoModelForSequenceClassification
pooler is unstable at the low learning rates required to avoid NaN).
Legal-BERT and Legal-RoBERTa use the standard CLS pooler.

### Per-Backbone Training Configuration

| Hyperparameter | Legal-BERT | DeBERTa-v3 | Legal-RoBERTa-large |
|---|---|---|---|
| Learning rate | 2e-5 | 3e-6 | 1e-5 |
| Max epochs | 90 | 90 | 90 |
| Head LR multiplier | 10x | 1x | 10x |
| Adam epsilon | 1e-8 | 1e-6 | 1e-8 |
| Gradient clip | 1.0 | 0.5 | 1.0 |
| Early stopping patience | 6 | 8 | 6 |
| Min epochs before early stop | 10 | 20 | 10 |
| Per-device batch size | 16 | 16 | 8 |
| Gradient accumulation steps | 2 | 2 | 4 |
| Effective batch size | 32 | 32 | 32 |
| Pooling strategy | CLS | Mean-pool | CLS |

Notes on DeBERTa hyperparameter choices:
- LR 3e-6: LR above 5e-6 causes NaN gradients via DeBERTa's disentangled attention.
  LR 2e-6 is stable but converges too slowly. 3e-6 is the empirical midpoint.
- Head multiplier 1x: any multiplier above 1 causes NaN for DeBERTa.
- Adam epsilon 1e-6: required by the DeBERTa paper to prevent NaN; 1e-8 is insufficient.
- Min epochs 20: DeBERTa near-zero F1 for the first 15+ epochs as representations stabilise.

Notes on Legal-RoBERTa hyperparameters:
- Batch size 8 + grad accumulation 4: 355M parameters do not fit in 8GB VRAM at batch 16.
  Gradient accumulation compensates to preserve effective batch size of 32.
- Gradient checkpointing enabled: recomputes activations during backward pass instead
  of storing them, reducing activation memory by approximately 60%.

Common settings across all backbones: warmup steps 500, max sequence length 256,
AsymmetricLoss (gamma_neg=4, gamma_pos=1, clip=0.05), AdamW with linear warmup and
linear decay scheduler.

### Training Data

| Dataset | Source | Clauses | Clause Types |
|---|---|---|---|
| CUAD | theatticusproject/cuad-qa (HuggingFace) | ~13,000 | 41 core commercial types |
| LEDGAR | coastalcph/lex_glue config=ledgar (HuggingFace) | ~80,000 | 32 boilerplate provision types |
| MAUD | theatticusproject/maud (HuggingFace) | ~39,000 | 14 M&A deal-point types |
| EDGAR (ongoing) | SEC EDGAR EX-10 exhibits | growing | auto-discovered new types |

**Total base training set:** approximately 132,000 clauses across 100 unified types.
**Split:** 80% train / 10% val / 10% test, stratified by primary clause type.

EDGAR clauses are auto-labeled by the classifier at confidence above 0.80 and
contribute up to 40% of the training set (EDGAR_TRAIN_CAP parameter). They are
excluded from val and test splits, which contain only human-annotated clauses.

### Unified Taxonomy (100 Types)

The taxonomy merges four sources into a single label space:
- 41 CUAD core types: e.g. Indemnification, Governing Law, IP Ownership Assignment
- 32 LEDGAR types: e.g. Force Majeure, Representations And Warranties, Severability
- 14 MAUD types: e.g. Material Adverse Effect, Termination Fee, No-Shop
- 13 additional commercial provisions: e.g. Data Protection And Privacy, Service Level Agreement
- Dynamic types: auto-discovered from EDGAR, stored in `dynamic_taxonomy.json`

### Calibration and Threshold Optimisation

1. Train backbone on training set.
2. Run inference on validation set.
3. Apply per-class threshold search (grid 0.1 to 0.9) to maximise per-class F1.
4. Store optimised thresholds in checkpoint alongside model weights.

### Ensemble Selection

All 7 combinations of the 3 backbones (3 singles, 3 pairs, 1 triple) are evaluated
on the validation set with a weight grid search. The winning combination and weights
are written to `production_config.json` for deterministic deployment.

### Current Performance

| Model | Val F1 Macro | Notes |
|---|---|---|
| LegalBERT (solo) | 0.5706 | `nlpaueb/legal-bert-base-uncased` |
| DeBERTa-v3 (solo) | 0.5803 | `microsoft/deberta-v3-base` |
| Legal-RoBERTa-large (solo) | 0.5756 | `lexlms/legal-roberta-large` |
| **ensemble_legalbert_legalroberta** | **0.6212 val / 0.6058 test** | Production model, w_lb=0.6, w_lr=0.4 |

Test set precision macro: 0.5858. Test set recall macro: 0.6429.

DeBERTa-v3 received weight 0.0 in the winning ensemble. Model selection determined
that adding DeBERTa gives no improvement: DeBERTa likely overfits differently from
the two Legal-domain models, adding noise rather than complementary signal at this
scale and data mix.

---

## 4. Why the F1 Is 0.6058 and Not Higher

This section provides an honest, technical account of the current performance ceiling.
The F1 Macro of 0.6058 on the held-out test set is the honest result achievable with
the current data and architecture. It is not an implementation error.

### Root Cause 1: Class Imbalance at 100 Types

Macro F1 weights all 100 classes equally regardless of frequency. Frequent types
such as Governing Law and Indemnification have thousands of training examples and
score well individually (F1 above 0.80 for some). Rare types such as Source Code
Escrow or Benchmarking Rights may have fewer than 100 examples total across all
three datasets. Poor performance on rare classes pulls the macro average down
significantly, even if common classes perform strongly.

The 100-type taxonomy is deliberately inclusive: types from CUAD, LEDGAR, and MAUD
were all retained rather than collapsing them, which means rare types were introduced
from smaller datasets without proportional training data.

### Root Cause 2: EDGAR Label Noise

EDGAR clauses are auto-labeled by the classifier itself at confidence above 0.80.
This means mislabeled examples enter the training set and teach the model incorrect
associations. EDGAR contributes up to 40% of the training set. The confidence
threshold is the main protection against noise dominating: lowering it below 0.80
was experimentally shown to degrade F1.

### Root Cause 3: Hard-to-Distinguish Adjacent Types

Several type pairs are semantically very close: License Grant vs Non-Transferable
License, Affiliate License-Licensor vs Affiliate License-Licensee, No-Solicit of
Customers vs No-Solicit of Employees. The model regularly confuses these pairs.
They share vocabulary and sentence structure, differing mainly in which party is
named or a single modifier word. Any model operating on token embeddings without
explicit structural reasoning over party names will struggle here.

### Root Cause 4: 512-Token Truncation

Clauses exceeding 512 tokens (roughly 350+ words) are split into overlapping chunks
via sliding window. Some clauses span 1,000 to 1,200 tokens. The chunk with the
highest maximum probability determines the final label. This is an approximation:
clauses whose key discriminating language falls past the 512-token boundary may be
systematically misclassified, with no mechanism to recover the truncated context.

### Root Cause 5: Multi-Source Dataset Heterogeneity

CUAD, LEDGAR, and MAUD were each annotated independently with different guidelines,
granularities, and clause boundary definitions. When merged, some types from one
dataset partially overlap with types from another, creating ambiguous training signal.
The model receives contradictory supervision for structurally similar clauses that
received different labels in different datasets.

### Root Cause 6: DeBERTa-v3 Not Contributing to the Ensemble

The model selection process (grid search over all 7 combinations with per-class
threshold optimisation) determined that adding DeBERTa gives zero improvement
(w_db=0.0 in the winning config). This is a data result, not an architecture
judgement. Three backbones were trained to give the selection process maximum
choice; the selection process chose two.

### Probable Improvement Paths

Listed in approximate descending order of expected impact:

| Approach | Expected Gain | Effort |
|---|---|---|
| More human-annotated data for rare types | High | High |
| Per-class weighted loss (focal loss or explicit class weights) | Medium | Low |
| Use Legal-RoBERTa-large as sole backbone with longer training | Medium | Low |
| Resolve adjacent-type confusion via a type hierarchy (coarse then fine) | Medium | Medium |
| Reduce EDGAR noise cap or raise auto-label confidence threshold further | Low to medium | Low |
| Contrastive learning to separate similar types | Medium | High |
| Hierarchical classification (coarse type first, fine type second) | High | High |
| Long-sequence handling (Longformer or better chunk aggregation) | Medium | Medium |

A realistic ceiling with the current datasets and architecture is approximately
F1 Macro 0.70 to 0.72. Reaching F1 above 0.80 would require substantially more
high-quality annotated data for the 40+ rare types.

### Known Limitations

- Performance on low-frequency clause types (fewer than 100 training examples) is reduced.
- The model regularly confuses adjacent type pairs (see Root Cause 3 above).
- English-only; not trained on non-English legal text.
- Trained on US/UK commercial contracts; may underperform on other jurisdictions.
- Multi-label output: a clause may receive multiple type labels simultaneously.
  This is correct for many real clauses but can inflate error counts on single-label evaluation.

---

## 5. Anomaly Detection Engine

### Input Representation

Both anomaly components use embeddings from the fine-tuned Legal-RoBERTa-large
checkpoint rather than the base pre-trained model. The fine-tuned checkpoint has
learned legal clause semantics through supervised training on 100 clause types:
its [CLS] representations are richer and more discriminative for clause-level
anomaly detection than a base model trained only on masked language modelling.

Embedding dimension: 1024 (Legal-RoBERTa-large hidden size).

### Component 1: Isolation Forest

- **Algorithm:** sklearn IsolationForest
- **Input:** Legal-RoBERTa-large fine-tuned [CLS] embeddings (1024-dim)
- **Contamination:** 5% (estimated proportion of anomalous clauses in training data)
- **Estimators:** 200 trees
- **Interpretation:** More negative decision_function score means more anomalous

### Component 2: Shallow Autoencoder

- **Architecture:** 1024 -> 256 -> 64 -> 256 -> 1024 (ReLU activations, linear output)
- **Loss:** MSELoss (reconstruction error)
- **Training:** 30 epochs, Adam (lr=1e-3), batch size 64
- **Interpretation:** Higher MSE reconstruction error means more anomalous

### Score Fusion

```
combined_score = 0.5 x IF_normalised + 0.5 x AE_normalised
```

Both scores are min-max normalised to [0, 100] on training-set scores.
Clauses with `combined_score > 70` are flagged as anomalous.

### Evaluation Approach and Honest Assessment

CUAD, LEDGAR, and MAUD contain no ground-truth anomaly labels. Evaluation uses
proxy labels: clauses shorter than 15 words or longer than the 95th percentile by
word count are treated as approximate anomaly indicators.

| Metric | Value |
|---|---|
| Precision@50 | 0.24 |
| Recall@50 | 0.033 |

These numbers reflect the weakness of the proxy label more than the detector quality.
Clauses with extreme word counts are not the same as legally anomalous clauses. A
genuinely anomalous clause can be any length: what matters is unusual language, not
unusual length.

The honest position: the anomaly detector reliably flags structural outliers in the
embedding space. Whether those outliers correspond to legally significant anomalies
cannot be quantified without purpose-built ground-truth annotations, which do not
exist in the training datasets.

The dual-signal approach (Isolation Forest + Autoencoder) reduces false positives
relative to either detector used alone. Isolation Forest detects global outliers
efficiently. The Autoencoder captures structural reconstruction anomalies that
Isolation Forest can miss because Isolation Forest partitions the full 1024-dim
space uniformly, while the Autoencoder compresses and reconstructs via a 64-dim
bottleneck that captures only the dominant variation axes.

---

## 6. Power Imbalance Scorer

### Critical Transparency Note

No publicly available dataset of human-annotated power imbalance labels exists for
commercial contracts. This scorer is therefore fully feature-engineered, not
supervised. This is a deliberate design decision grounded in the academic literature
on computational legal studies:

- Lippi et al. (2019): "CLAUDETTE: an automated detector of potentially unfair
  clauses in online terms of service"
- Drawzeski et al. (2021): "A corpus for clause-level analysis of unfairness in
  online terms of service"

The feature-engineering approach is academically valid, transparent, and
reproducible. Scores should be treated as linguistic indicators, not ground truth.

### Feature Architecture

| Feature | Weight | Extractor | Interpretation |
|---|---|---|---|
| Sentiment tone | 30% | RoBERTa (`cardiffnlp/twitter-roberta-base-sentiment-latest`) | Negative tone reflects harsh or one-sided language |
| Modal verb balance | 25% | Rule-based (obligation vs discretion modals) | High obligation modals reflect drafting party advantage |
| Obligation assignment | 25% | Rule-based party indicator matching | Obligations on Party B reflect Party A advantage |
| Assertiveness | 20% | Rule-based (absolute vs conditional language) | Assertive language reflects drafting party advantage |
| Clause length | Amplifier | Z-score normalised | Longer clauses amplify the base imbalance signal |

### Scoring Formula

```python
party_a_leverage = (
    0.30 x (1 - sentiment) +   # low sentiment = harsh
    0.25 x modal_score     +   # obligation modals
    0.25 x obligation_score +  # obligations on Party B
    0.20 x assertiveness
)
party_b_leverage = (
    0.30 x sentiment +
    0.25 x (1 - modal_score) +
    0.25 x (1 - obligation_score) +
    0.20 x (1 - assertiveness)
)
amplifier = 0.8 + 0.4 x length_score  # range [0.8, 1.2]
power_imbalance = (party_a_leverage - party_b_leverage) x amplifier x 100
```

Output range: -100 (maximum Party B advantage) to +100 (maximum Party A advantage).
Scores above +40 or below -40 are flagged as high imbalance.

### Validation

Since no ground-truth labels exist, validation uses:
- Feature contribution analysis: feature mean/std on held-out test set
- Consistency testing: same clause scored twice produces identical results
- Score distribution analysis: checks for degenerate distributions
- Manual spot-checking: qualitative review of high-scoring clauses

Current test set statistics: mean imbalance 2.19, standard deviation 27.49.
Consistency: True (deterministic for identical inputs).

---

## 7. Taxonomy Expansion Pipeline (EDGAR)

### Purpose

Continuously discovers new clause types from SEC EX-10 material contract exhibits,
expanding the classifier vocabulary without manual annotation.

### Pipeline

```
EDGAR EX-10 exhibits (8-K filings via EFTS search)
    SGML parsing + clause segmentation
    AutoLabeler (trained classifier)
        above 0.80 confidence: accepted, written to training DB
        below 0.80 confidence: unknown clauses forwarded to clustering
            UMAP (1024d to 5d, cosine metric)
            sklearn HDBSCAN (auto cluster count)
            TF-IDF keyword extraction per cluster
            SimilarityRouter (cosine sim to 100 type name embeddings)
                above 0.80 similarity: relabel to matched existing type, save to DB
                0.50 to 0.80 similarity: taxonomy_review.csv for optional human review
                below 0.50 similarity: TaxonomyExpander
                    cluster size above 10: auto-name, add to dynamic_taxonomy.json, save to DB
                    cluster size below 10: noise, discarded
```

### Key Thresholds

| Parameter | Value | Meaning |
|---|---|---|
| EDGAR_AUTO_LABEL_CONFIDENCE | 0.80 | Minimum classifier confidence for auto-accept |
| CLUSTER_ROUTE_HIGH_SIM | 0.80 | Cosine similarity threshold to relabel to existing type |
| CLUSTER_ROUTE_LOW_SIM | 0.50 | Below this similarity is a genuinely new type candidate |
| TAXONOMY_AUTO_ADD_MIN_SIZE | 10 | Minimum cluster size for automatic taxonomy addition |
| BERTOPIC_MIN_CLUSTER_SIZE | 10 | HDBSCAN minimum cluster size |

The EDGAR_AUTO_LABEL_CONFIDENCE threshold was raised from 0.70 to 0.80 after
experimental observation that 0.70-threshold auto-labels increased label noise in
the training set without measurable F1 improvement.

### Manual Override

All automated decisions are reviewable via `data/processed/taxonomy_review.csv`.
Set `action=accept` on any row and call `TaxonomyExpander.apply_manual_review()`
to incorporate those types. The pipeline runs fully without human intervention.
Manual review is optional, not required.

---

## 8. SHAP Explainability

- **Method:** SHAP KernelExplainer (model-agnostic)
- **Classifier SHAP:** Word-level masking, classifier probability perturbation
- **Power SHAP:** Feature-level perturbation, imbalance score perturbation
- **Background samples:** 100 random masks / neutral feature values
- **Output:** Per-token and per-feature attribution values with direction (positive = Party A, negative = Party B)

---

## 9. Ethical Considerations

1. **Legal advice disclaimer:** This platform does not provide legal advice.
   All outputs are probabilistic indicators requiring human expert review.

2. **Bias risk:** Models trained on CUAD, LEDGAR, and MAUD (primarily US corporate
   contracts) may exhibit geographic and industry bias. Performance on EU, Asian,
   or consumer contracts is not validated.

3. **Transparency commitment:** All model weights, training configurations, and
   feature weights are documented in this card and centralised in `config.py`.
   Auto-discovered taxonomy additions are stored in `dynamic_taxonomy.json` with
   full audit trail in `edgar_new_clause_types.csv`. No hidden scoring logic exists.

4. **Human oversight:** High-imbalance or anomalous clause flags are designed
   to draw attention for human review, not to automate legal decisions.

5. **Data provenance:** EDGAR filings are public SEC disclosures. Auto-labeled
   training data is tracked with source tags (EDGAR_auto, EDGAR_cluster_relabel,
   EDGAR_new_type) for full traceability.

---

## 10. Training Infrastructure

- **Hardware:** GPU required for classifier training (Colab T4/A100 recommended). CPU inference works but is slow.
- **Training time (GPU):** 3 to 5 hours for Legal-BERT (up to 90 epochs with early stopping). 5 to 10 hours for DeBERTa-v3. 8 to 14 hours for Legal-RoBERTa-large. Approximately 80 minutes for anomaly detector embedding generation plus 10 minutes fitting.
- **EDGAR pipeline (CPU):** Approximately 3 hours per 5,000 contracts (download, inference, clustering).
- **Storage:** Approximately 2GB for Legal-BERT checkpoint. Approximately 2.5GB for DeBERTa checkpoint. Approximately 4GB for Legal-RoBERTa-large checkpoint. Approximately 50MB for anomaly models. Approximately 500MB per 5,000 EDGAR exhibits.
- **Dependencies:** See `requirements.txt`.

---

## 11. Citation

If using this platform in academic work, please cite:

```bibtex
@software{contract_intelligence_platform,
  title  = {Contract Intelligence and Power Imbalance Platform},
  year   = {2026},
  note   = {100-type unified clause classification with LegalBERT and
            Legal-RoBERTa-large ensemble (test F1 macro 0.6058), dual-signal
            anomaly detection using fine-tuned Legal-RoBERTa embeddings,
            feature-engineered bilateral power imbalance scoring, and
            automated taxonomy expansion from SEC EDGAR EX-10 filings},
}
```

CUAD dataset citation:
```bibtex
@article{hendrycks2021cuad,
  title   = {CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review},
  author  = {Hendrycks, Dan and others},
  journal = {arXiv preprint arXiv:2103.06268},
  year    = {2021}
}
```

LEDGAR dataset citation:
```bibtex
@inproceedings{tuggener2020ledgar,
  title     = {LEDGAR: A Large-Scale Multi-label Corpus for Text Classification of Legal Provisions in Contracts},
  author    = {Tuggener, Don and others},
  booktitle = {Proceedings of LREC 2020},
  year      = {2020}
}
```

MAUD dataset citation:
```bibtex
@article{wang2023maud,
  title   = {MAUD: An Expert-Annotated Legal NLP Dataset for Merger Agreement Understanding},
  author  = {Wang, Steven H. and others},
  journal = {arXiv preprint arXiv:2301.00876},
  year    = {2023}
}
```

SEC EDGAR data:
```bibtex
@misc{sec_edgar,
  title        = {EDGAR Full-Text Search and Filing Archive},
  author       = {{U.S. Securities and Exchange Commission}},
  howpublished = {\url{https://www.sec.gov/cgi-bin/browse-edgar}},
  note         = {EX-10.* material contract exhibits accessed via EFTS API.
                  All filings are public disclosures under the Securities
                  Exchange Act of 1934. No redistribution of raw filings;
                  derived clause-level annotations only.},
  year         = {2018--2024}
}
```
