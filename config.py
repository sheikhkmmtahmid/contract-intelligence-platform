"""
config.py — Central configuration for Contract Intelligence Platform.

All paths, hyperparameters, model identifiers, and thresholds are defined here.
No hardcoded values exist anywhere else in the codebase.
"""

from pathlib import Path

# Root paths
ROOT_DIR = Path(__file__).parent.resolve()

DATA_DIR          = ROOT_DIR / "data"
RAW_DIR           = DATA_DIR / "raw"
PROCESSED_DIR     = DATA_DIR / "processed"
CUAD_DIR          = DATA_DIR / "cuad"

LEDGAR_DIR        = DATA_DIR / "ledgar"
MAUD_DIR          = DATA_DIR / "maud"
EDGAR_DIR         = DATA_DIR / "edgar"
EDGAR_RAW_DIR     = EDGAR_DIR / "raw"

MODELS_DIR        = ROOT_DIR / "models"
CLASSIFIER_DIR    = MODELS_DIR / "clause_classifier"
CLASSIFIER_LEGALBERT_DIR      = CLASSIFIER_DIR / "legalbert"
CLASSIFIER_DEBERTA_DIR        = CLASSIFIER_DIR / "deberta"
CLASSIFIER_LEGALROBERTA_DIR   = CLASSIFIER_DIR / "legalroberta"
CLASSIFIER_PRODUCTION_CONFIG  = CLASSIFIER_DIR / "production_config.json"
ANOMALY_DIR       = MODELS_DIR / "anomaly_detector"
POWER_DIR         = MODELS_DIR / "power_scorer"

LOGS_DIR          = ROOT_DIR / "logs"
STATIC_DIR        = ROOT_DIR / "static"
TEMPLATES_DIR     = ROOT_DIR / "templates"

# Database
DB_PATH           = PROCESSED_DIR / "contracts.db"
DB_URL            = f"sqlite:///{DB_PATH}"

# HuggingFace model identifiers
LEGAL_BERT_MODEL        = "nlpaueb/legal-bert-base-uncased"
DEBERTA_MODEL           = "microsoft/deberta-v3-base"
LEGAL_ROBERTA_MODEL     = "lexlms/legal-roberta-large"
SENTIMENT_MODEL         = "cardiffnlp/twitter-roberta-base-sentiment-latest"
CUAD_DATASET_ID         = "theatticusproject/cuad-qa"
LEDGAR_DATASET_ID       = "coastalcph/lex_glue"   # config="ledgar"
MAUD_DATASET_ID         = "theatticusproject/maud"

# EDGAR settings
EDGAR_AUTO_LABEL_CONFIDENCE = 0.80   # probs above this → auto-accepted (raised from 0.70 to reduce label noise)
EDGAR_DOWNLOAD_LIMIT        = 5000   # max contracts per run (SEC rate limit safe)
EDGAR_CLUSTER_N             = 20     # number of clusters for new-type discovery
EDGAR_REVIEW_PATH           = PROCESSED_DIR / "review_queue.csv"
EDGAR_NEW_TYPES_PATH        = PROCESSED_DIR / "edgar_new_clause_types.csv"

# BERTopic cluster discovery
BERTOPIC_MIN_CLUSTER_SIZE  = 10      # HDBSCAN min cluster size — filters noise
BERTOPIC_UMAP_COMPONENTS   = 5       # UMAP target dimensions
BERTOPIC_TOP_N_WORDS       = 10      # keywords extracted per topic

# Cluster routing thresholds (cosine similarity to existing type name embeddings)
CLUSTER_ROUTE_HIGH_SIM     = 0.80    # >= this → relabel to matched existing type
CLUSTER_ROUTE_LOW_SIM      = 0.50    # < this  → genuinely new type candidate

# Taxonomy auto-expansion
TAXONOMY_AUTO_ADD_MIN_SIZE = 10      # clusters >= this size → auto-add to taxonomy
TAXONOMY_REVIEW_MIN_SIZE   = 10      # clusters < this → discard as noise
TAXONOMY_REVIEW_PATH       = PROCESSED_DIR / "taxonomy_review.csv"
DYNAMIC_TAXONOMY_PATH      = PROCESSED_DIR / "dynamic_taxonomy.json"

# Unified clause taxonomy — 100 types
# Sources: CUAD (41 core types) + LEDGAR (32 new provision types) +
#          MAUD (14 M&A deal-point types) + 13 common commercial provisions
CUAD_CLAUSE_TYPES = [
    # ── CUAD core (41) ────────────────────────────────────────────────────
    "Competitive Restriction Exception",
    "Non-Compete",
    "Exclusivity",
    "No-Solicit Of Customers",
    "No-Solicit Of Employees",
    "Non-Disparagement",
    "Termination For Convenience",
    "Rofr/Rofo/Rofn",
    "Change Of Control",
    "Anti-Assignment",
    "Revenue/Profit Sharing",
    "Price Restrictions",
    "Minimum Commitment",
    "Volume Restriction",
    "IP Ownership Assignment",
    "Joint IP Ownership",
    "License Grant",
    "Non-Transferable License",
    "Affiliate License-Licensor",
    "Affiliate License-Licensee",
    "Unlimited/All-You-Can-Eat-License",
    "Irrevocable Or Perpetual License",
    "Source Code Escrow",
    "Post-Termination Services",
    "Audit Rights",
    "Uncapped Liability",
    "Cap On Liability",
    "Liquidated Damages",
    "Warranty Duration",
    "Insurance",
    "Covenant Not To Sue",
    "Third Party Beneficiary",
    "Most Favored Nation",
    "Governing Law",
    "Dispute Resolution",
    "Limitations Of Liability",
    "Indemnification",
    "Agreement Date",
    "Effective Date",
    "Expiration Date",
    "Renewal Term",
    # ── New from LEDGAR (32) ───────────────────────────────────────────────
    "Confidentiality",
    "Amendments",
    "Anti-Corruption Laws",
    "Approvals And Consents",
    "Authority",
    "Compliance With Laws",
    "Consent To Jurisdiction",
    "Definitions",
    "Enforceability",
    "Entire Agreements",
    "Expenses",
    "Force Majeure",
    "Further Assurances",
    "Interests",
    "Liens And Encumbrances",
    "Limitations Of Remedies",
    "Non-Waiver",
    "Notices",
    "Organization And Existence",
    "Payments",
    "Representations And Warranties",
    "Sanctions",
    "Securities Law Compliance",
    "Severability",
    "Specific Performance",
    "Successors And Assigns",
    "Survival",
    "Taxes And Withholding",
    "Trade Controls",
    "Transactions With Affiliates",
    "Waiver Of Jury Trial",
    "Waivers",
    # ── New from MAUD (14) ─────────────────────────────────────────────────
    "No-Shop",
    "Material Adverse Effect",
    "Hell-Or-High-Water",
    "Termination Fee",
    "Reverse Termination Fee",
    "Matching Rights",
    "Superior Offer Definition",
    "Fiduciary Exception",
    "Antitrust Efforts Standard",
    "Operating Covenants",
    "Expense Reimbursement",
    "Intervening Event Definition",
    "Board Recommendation Change",
    "Tail Period",
    # ── Additional commercial provisions (13) ─────────────────────────────
    "Data Protection And Privacy",
    "Employment And Benefits",
    "Capitalization",
    "Publicity And Announcements",
    "Non-Solicitation General",
    "Step-In Rights",
    "Benchmarking Rights",
    "Electronic Signatures And Counterparts",
    "Service Level Agreement",
    "Subcontracting",
    "Assignment Of Receivables",
    "Performance Bonds",
    "Escrow And Holdback",
]

# Dynamic taxonomy — auto-discovered types are persisted in dynamic_taxonomy.json
# and merged at import time without ever editing config.py manually
if DYNAMIC_TAXONOMY_PATH.exists():
    import json as _json
    _dynamic_types = _json.loads(DYNAMIC_TAXONOMY_PATH.read_text())
    CUAD_CLAUSE_TYPES = CUAD_CLAUSE_TYPES + [
        t for t in _dynamic_types if t not in CUAD_CLAUSE_TYPES
    ]

NUM_CLAUSE_TYPES = len(CUAD_CLAUSE_TYPES)  # 100 base + any auto-discovered

# Clause classifier training hyperparameters
CLASSIFIER_LEARNING_RATE  = 2e-5
CLASSIFIER_BATCH_SIZE     = 16
CLASSIFIER_EPOCHS         = 25
CLASSIFIER_WARMUP_STEPS   = 500
CLASSIFIER_MAX_LENGTH     = 256       # 256 covers 95%+ of clauses; 512→256 is ~4× faster attention
CLASSIFIER_THRESHOLD      = 0.5       # Sigmoid threshold for multi-label (fallback)
CLASSIFIER_TARGET_F1      = 0.80

# Open-set / unknown clause detection
UNKNOWN_PROB_THRESHOLD    = 0.3       # max(probs) below this → assign "Other"

# Anomaly detection hyperparameters
ISOLATION_FOREST_CONTAMINATION  = 0.05
AUTOENCODER_HIDDEN_DIMS         = [256, 64]   # encoder bottleneck sizes
AUTOENCODER_EPOCHS              = 30
AUTOENCODER_BATCH_SIZE          = 64
AUTOENCODER_LR                  = 1e-3
EMBEDDING_DIM                   = 1024        # Legal-RoBERTa-large [CLS] dimension

# Anomaly score fusion weights (must sum to 1.0)
ANOMALY_IF_WEIGHT               = 0.5
ANOMALY_AE_WEIGHT               = 0.5
ANOMALY_FLAG_THRESHOLD          = 70          # 0-100 scale

# Power imbalance scoring — feature weights (must sum to 1.0)
POWER_WEIGHT_SENTIMENT      = 0.30
POWER_WEIGHT_MODAL_VERBS    = 0.25
POWER_WEIGHT_OBLIGATIONS    = 0.25
POWER_WEIGHT_ASSERTIVENESS  = 0.20

# Modal verbs: obligation vs discretion
OBLIGATION_MODALS   = {"shall", "must", "will", "required", "obligated"}
DISCRETION_MODALS   = {"may", "might", "could", "should", "permitted", "entitled"}

# Power imbalance flag thresholds
IMBALANCE_HIGH_THRESHOLD    = 40   # |score| > 40 → high imbalance
IMBALANCE_MEDIUM_THRESHOLD  = 20   # |score| > 20 → medium

# SHAP explainability
SHAP_BACKGROUND_SAMPLES = 100   # background dataset size for KernelExplainer
SHAP_MAX_EVALS          = 500   # max model evaluations per explanation
SHAP_OUTPUT_DIR         = PROCESSED_DIR / "shap_plots"

# PDF processing
MAX_PAGES               = 100
CLAUSE_MIN_TOKENS       = 10     # discard very short segments
CLAUSE_MAX_TOKENS       = 512    # truncate to Legal-BERT limit
# Maximum clause character length for train/val/test splits.
# val/test human data tops out at ~2885 tokens (~11,500 chars).
# Anything above 15,000 chars is almost certainly an EDGAR pipeline
# artifact (entire document sections ingested as a single clause).
CLAUSE_MAX_CHARS        = 15000

# Data splits
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10
RANDOM_SEED = 42

# Undersampling — cap dominant classes in the training set only.
# Val/test sets are never touched. Set to None to disable.
UNDERSAMPLE_MAX_PER_CLASS = None

# API / server
API_HOST        = "0.0.0.0"
API_PORT        = 8000
API_TITLE       = "Contract Intelligence & Power Imbalance Platform"
API_VERSION     = "1.0.0"
CORS_ORIGINS    = ["*"]   # tighten in production

# Evaluation output
EVAL_RESULTS_PATH   = ROOT_DIR / "evaluation_results.json"
MODEL_CARD_PATH     = ROOT_DIR / "model_card.md"

# Ensure all required directories exist at import time
for _dir in [
    RAW_DIR, PROCESSED_DIR, CUAD_DIR, LEDGAR_DIR, MAUD_DIR,
    EDGAR_DIR, EDGAR_RAW_DIR,
    CLASSIFIER_DIR, CLASSIFIER_LEGALBERT_DIR, CLASSIFIER_DEBERTA_DIR,
    CLASSIFIER_LEGALROBERTA_DIR,
    ANOMALY_DIR, POWER_DIR,
    LOGS_DIR, SHAP_OUTPUT_DIR,
]:
    _dir.mkdir(parents=True, exist_ok=True)
