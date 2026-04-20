"""
config_colab.py — Central configuration for Contract Intelligence Platform.
Colab-compatible version: auto-detects Google Drive mount point.

Place this file at:
  My Drive/contract-intelligence-platform/config.py

All paths, hyperparameters, model identifiers, and thresholds are defined here.
No hardcoded values exist anywhere else in the codebase.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Root path — Colab-aware
# ---------------------------------------------------------------------------

try:
    import google.colab  # noqa: F401
    _IN_COLAB = True
except ImportError:
    _IN_COLAB = False

if _IN_COLAB:
    # Standard Google Drive mount point in Colab
    ROOT_DIR = Path("/content/drive/MyDrive/contract-intelligence-platform")
else:
    # Local development: resolve relative to this file's location
    ROOT_DIR = Path(__file__).parent.resolve()

# ---------------------------------------------------------------------------
# Directory structure
# ---------------------------------------------------------------------------

DATA_DIR      = ROOT_DIR / "data"
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CUAD_DIR      = DATA_DIR / "cuad"

MODELS_DIR                   = ROOT_DIR / "models"
CLASSIFIER_DIR               = MODELS_DIR / "clause_classifier"
CLASSIFIER_LEGALBERT_DIR     = CLASSIFIER_DIR / "legalbert"
CLASSIFIER_DEBERTA_DIR       = CLASSIFIER_DIR / "deberta"
CLASSIFIER_PRODUCTION_CONFIG = CLASSIFIER_DIR / "production_config.json"
ANOMALY_DIR                  = MODELS_DIR / "anomaly_detector"
POWER_DIR                    = MODELS_DIR / "power_scorer"

LOGS_DIR      = ROOT_DIR / "logs"
STATIC_DIR    = ROOT_DIR / "static"
TEMPLATES_DIR = ROOT_DIR / "templates"

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

DB_PATH = PROCESSED_DIR / "contracts.db"
DB_URL  = f"sqlite:///{DB_PATH}"

# ---------------------------------------------------------------------------
# HuggingFace model identifiers
# ---------------------------------------------------------------------------

LEGAL_BERT_MODEL = "nlpaueb/legal-bert-base-uncased"
DEBERTA_MODEL    = "microsoft/deberta-v3-base"
SENTIMENT_MODEL  = "cardiffnlp/twitter-roberta-base-sentiment-latest"
#CUAD_DATASET_ID  = "theatticusproject/cuad"
CUAD_DATASET_ID  = "theatticusproject/cuad-qa"

# ---------------------------------------------------------------------------
# CUAD clause types (41 categories)
# ---------------------------------------------------------------------------

CUAD_CLAUSE_TYPES = [
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
]

NUM_CLAUSE_TYPES = len(CUAD_CLAUSE_TYPES)  # 41

# ---------------------------------------------------------------------------
# Clause classifier training hyperparameters
# ---------------------------------------------------------------------------

CLASSIFIER_LEARNING_RATE  = 2e-5
CLASSIFIER_BATCH_SIZE     = 16
CLASSIFIER_EPOCHS         = 10
CLASSIFIER_WARMUP_STEPS   = 500
CLASSIFIER_MAX_LENGTH     = 512      # Legal-BERT / DeBERTa max tokens
CLASSIFIER_THRESHOLD      = 0.5     # Sigmoid threshold fallback
CLASSIFIER_TARGET_F1      = 0.80

# Open-set / unknown clause detection
UNKNOWN_PROB_THRESHOLD    = 0.3     # max(probs) below this → assign "Other"

# ---------------------------------------------------------------------------
# Anomaly detection hyperparameters
# ---------------------------------------------------------------------------

ISOLATION_FOREST_CONTAMINATION = 0.05
AUTOENCODER_HIDDEN_DIMS        = [256, 64]
AUTOENCODER_EPOCHS             = 30
AUTOENCODER_BATCH_SIZE         = 64
AUTOENCODER_LR                 = 1e-3
EMBEDDING_DIM                  = 768   # Legal-BERT [CLS] dimension

ANOMALY_IF_WEIGHT      = 0.5
ANOMALY_AE_WEIGHT      = 0.5
ANOMALY_FLAG_THRESHOLD = 70           # 0-100 scale

# ---------------------------------------------------------------------------
# Power imbalance scoring — feature weights (must sum to 1.0)
# ---------------------------------------------------------------------------

POWER_WEIGHT_SENTIMENT     = 0.30
POWER_WEIGHT_MODAL_VERBS   = 0.25
POWER_WEIGHT_OBLIGATIONS   = 0.25
POWER_WEIGHT_ASSERTIVENESS = 0.20

OBLIGATION_MODALS  = {"shall", "must", "will", "required", "obligated"}
DISCRETION_MODALS  = {"may", "might", "could", "should", "permitted", "entitled"}

IMBALANCE_HIGH_THRESHOLD   = 40    # |score| > 40 → high imbalance
IMBALANCE_MEDIUM_THRESHOLD = 20    # |score| > 20 → medium

# ---------------------------------------------------------------------------
# SHAP explainability
# ---------------------------------------------------------------------------

SHAP_BACKGROUND_SAMPLES = 100
SHAP_MAX_EVALS          = 500
SHAP_OUTPUT_DIR         = PROCESSED_DIR / "shap_plots"

# ---------------------------------------------------------------------------
# PDF processing
# ---------------------------------------------------------------------------

MAX_PAGES         = 100
CLAUSE_MIN_TOKENS = 10    # discard very short segments
CLAUSE_MAX_TOKENS = 512   # truncate to model max

# ---------------------------------------------------------------------------
# Data splits
# ---------------------------------------------------------------------------

TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# API / server
# ---------------------------------------------------------------------------

API_HOST     = "0.0.0.0"
API_PORT     = 8000
API_TITLE    = "Contract Intelligence & Power Imbalance Platform"
API_VERSION  = "1.0.0"
CORS_ORIGINS = ["*"]

# ---------------------------------------------------------------------------
# Evaluation output
# ---------------------------------------------------------------------------

EVAL_RESULTS_PATH = ROOT_DIR / "evaluation_results.json"
MODEL_CARD_PATH   = ROOT_DIR / "model_card.md"

# ---------------------------------------------------------------------------
# Ensure all required directories exist at import time
# ---------------------------------------------------------------------------

for _dir in [
    RAW_DIR, PROCESSED_DIR, CUAD_DIR,
    CLASSIFIER_DIR, CLASSIFIER_LEGALBERT_DIR, CLASSIFIER_DEBERTA_DIR,
    ANOMALY_DIR, POWER_DIR,
    LOGS_DIR, SHAP_OUTPUT_DIR,
]:
    _dir.mkdir(parents=True, exist_ok=True)
