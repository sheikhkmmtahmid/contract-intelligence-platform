"""
data_pipeline_colab.py — Colab-compatible contract ingestion pipeline.

Self-contained: database ORM is defined inline so only config.py and this
file are needed in Google Drive. No api/ folder required.

Place files at:
  My Drive/contract-intelligence-platform/config.py         (config_colab.py renamed)
  My Drive/contract-intelligence-platform/src/data_pipeline.py  (this file renamed)

Run in Colab (after mounting Drive and installing deps):
    !python "src/data_pipeline.py" --mode cuad
    !python "src/data_pipeline.py" --mode split

Responsibilities:
  1. Download and parse the CUAD dataset from HuggingFace.
  2. Ingest raw contract PDFs using pdfplumber.
  3. Segment contract text into individual clauses.
  4. Persist processed clauses to SQLite via SQLAlchemy.
  5. Produce stratified 80/10/10 train/val/test splits.
"""

# ---------------------------------------------------------------------------
# Colab bootstrap — run once before anything else
# ---------------------------------------------------------------------------

import subprocess
import sys

def _install_if_missing(packages: list) -> None:
    """Silently install any missing packages via pip."""
    import importlib
    missing = []
    _pkg_map = {
        "pdfplumber": "pdfplumber",
        "loguru":     "loguru",
        "tqdm":       "tqdm",
        "datasets":   "datasets",
        "transformers": "transformers",
        "sklearn":    "scikit-learn",
        "sqlalchemy": "sqlalchemy",
    }
    for import_name, pip_name in _pkg_map.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing.append(pip_name)

    if missing:
        print(f"Installing missing packages: {missing}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + missing)

_install_if_missing([])

# ---------------------------------------------------------------------------
# Standard imports
# ---------------------------------------------------------------------------

import argparse
import hashlib
import re
import textwrap
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd
import pdfplumber
import torch
from datasets import load_dataset
from loguru import logger
from sklearn.model_selection import train_test_split
from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey,
    Integer, String, Text, create_engine, event, text,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Project root — works whether running from Drive or locally
# ---------------------------------------------------------------------------

# When this file lives at .../src/data_pipeline.py its parent.parent is the
# project root, which is also where config.py lives.
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger.remove()
logger.add(
    config.LOGS_DIR / "data_pipeline.log",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG",
)
logger.add(sys.stderr, level="INFO")


# ===========================================================================
# INLINE DATABASE — replaces api/database.py so no api/ folder is needed
# ===========================================================================

_engine = create_engine(
    config.DB_URL,
    connect_args={"check_same_thread": False},
    echo=False,
)


@event.listens_for(_engine, "connect")
def _set_sqlite_pragma(dbapi_connection, connection_record):
    """Enable WAL mode and foreign key enforcement on every new SQLite connection."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)


class Base(DeclarativeBase):
    pass


class Contract(Base):
    """One row per ingested contract document."""
    __tablename__ = "contracts"

    contract_id = Column(String(64), primary_key=True, index=True)
    filename    = Column(String(512), nullable=False)
    source      = Column(String(64),  nullable=False, default="upload")
    page_count  = Column(Integer, nullable=True)
    created_at  = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    clauses = relationship("Clause", back_populates="contract", cascade="all, delete-orphan")


class Clause(Base):
    """One row per extracted clause."""
    __tablename__ = "clauses"

    clause_id             = Column(String(64), primary_key=True, index=True)
    contract_id           = Column(String(64), ForeignKey("contracts.contract_id"), nullable=False, index=True)
    clause_text           = Column(Text,        nullable=False)
    clause_type           = Column(String(512), nullable=False, default="")
    party_a               = Column(String(256), nullable=True,  default="")
    party_b               = Column(String(256), nullable=True,  default="")
    source                = Column(String(64),  nullable=True,  default="")
    anomaly_score         = Column(Float,   nullable=True)
    is_anomalous          = Column(Boolean, nullable=True, default=False)
    power_imbalance_score = Column(Float,   nullable=True)
    party_a_leverage      = Column(Float,   nullable=True)
    party_b_leverage      = Column(Float,   nullable=True)
    sentiment_score       = Column(Float,   nullable=True)
    modal_score           = Column(Float,   nullable=True)
    obligation_score      = Column(Float,   nullable=True)
    assertiveness_score   = Column(Float,   nullable=True)
    shap_plot_path        = Column(String(512), nullable=True)
    created_at            = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    contract = relationship("Contract", back_populates="clauses")


def create_tables() -> None:
    """Create all database tables if they do not already exist."""
    Base.metadata.create_all(bind=_engine)


@contextmanager
def _managed_session() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ===========================================================================
# 1. PDF INGESTION
# ===========================================================================

class PDFIngester:
    """Extracts raw text from PDF contracts using pdfplumber.

    Handles multi-column layouts by joining lines per page and normalising
    whitespace. Respects the MAX_PAGES limit defined in config to avoid
    memory issues on very large documents.
    """

    def __init__(self, max_pages: int = config.MAX_PAGES):
        self.max_pages = max_pages

    def extract_text(self, pdf_path: Path) -> str:
        """Extract and return the full text of a PDF contract.

        Args:
            pdf_path: Absolute path to the PDF file.

        Returns:
            Cleaned, concatenated text from all pages up to max_pages.

        Raises:
            FileNotFoundError: If pdf_path does not exist.
            ValueError: If the PDF yields no extractable text.
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        pages_text: List[str] = []
        with pdfplumber.open(pdf_path) as pdf:
            total = min(len(pdf.pages), self.max_pages)
            for page in pdf.pages[:total]:
                page_text = page.extract_text(x_tolerance=2, y_tolerance=2)
                if page_text:
                    pages_text.append(page_text)
            logger.info(f"Extracted text from {total} pages in {pdf_path.name}")

        full_text = self._clean_text("\n".join(pages_text))
        if not full_text.strip():
            raise ValueError(f"No extractable text found in {pdf_path.name}")
        return full_text

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalise whitespace and remove PDF artefacts."""
        text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]", " ", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


# ===========================================================================
# 2. CLAUSE SEGMENTATION
# ===========================================================================

class ClauseSegmenter:
    """Segments contract text into individual clauses.

    Strategy:
      - Primary split: numbered section headers (e.g., "1.", "1.1", "Article 2")
      - Secondary split: paragraph boundaries (double newline)
      - Filter: discard segments shorter than CLAUSE_MIN_TOKENS words
      - Truncate to max token length if tokenizer is provided.
    """

    SECTION_PATTERNS = [
        r"^(?:ARTICLE|SECTION|CLAUSE)\s+\d+",
        r"^\d+\.\d*\s+[A-Z]",
        r"^\d+\.\s+[A-Z]",
        r"^[A-Z][A-Z\s]{4,}$",
    ]
    COMPILED_PATTERNS = [re.compile(p, re.MULTILINE) for p in SECTION_PATTERNS]

    def __init__(
        self,
        tokenizer: Optional[AutoTokenizer] = None,
        min_tokens: int = config.CLAUSE_MIN_TOKENS,
        max_tokens: int = config.CLAUSE_MAX_TOKENS,
    ):
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.tokenizer  = tokenizer

    def segment(self, text: str) -> List[str]:
        """Split a full contract text into a list of clause strings.

        Args:
            text: Full contract text (already cleaned).

        Returns:
            List of clause strings.
        """
        clauses = self._split_by_headers(text)
        if len(clauses) < 5:
            clauses = self._split_by_paragraphs(text)

        clauses = [self._clean_clause(c) for c in clauses]
        clauses = [c for c in clauses if self._is_valid(c)]

        if self.tokenizer is not None:
            clauses = [self._truncate(c) for c in clauses]

        logger.debug(f"Segmented into {len(clauses)} clauses")
        return clauses

    def _split_by_headers(self, text: str) -> List[str]:
        lines = text.split("\n")
        segments: List[str] = []
        current: List[str] = []
        for line in lines:
            if self._is_header(line):
                if current:
                    segments.append("\n".join(current).strip())
                current = [line]
            else:
                current.append(line)
        if current:
            segments.append("\n".join(current).strip())
        return segments

    def _split_by_paragraphs(self, text: str) -> List[str]:
        return [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]

    def _is_header(self, line: str) -> bool:
        line = line.strip()
        if not line:
            return False
        return any(p.match(line) for p in self.COMPILED_PATTERNS)

    def _is_valid(self, clause: str) -> bool:
        return len(clause.split()) >= self.min_tokens

    @staticmethod
    def _clean_clause(clause: str) -> str:
        return re.sub(r"[ \t]{2,}", " ", clause).strip()

    def _truncate(self, clause: str) -> str:
        tokens = self.tokenizer(
            clause,
            max_length=self.max_tokens,
            truncation=True,
            return_tensors="pt",
        )
        return self.tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)


# ===========================================================================
# 3. LEGAL-BERT EMBEDDING GENERATOR
# ===========================================================================

class EmbeddingGenerator:
    """Generates [CLS] token embeddings from Legal-BERT.

    Used for anomaly detection feature generation.
    """

    def __init__(self, model_name: str = config.LEGAL_BERT_MODEL, device: str = "auto"):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading Legal-BERT: {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def embed(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate [CLS] embeddings for a list of clause texts.

        Args:
            texts: List of clause strings.
            batch_size: Clauses per forward pass.

        Returns:
            NumPy array of shape (N, 768).
        """
        all_embeddings: List[np.ndarray] = []
        for start in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch = texts[start : start + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=config.CLAUSE_MAX_TOKENS,
                return_tensors="pt",
            ).to(self.device)
            outputs = self.model(**encoded)
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_emb)
        return np.vstack(all_embeddings)


# ===========================================================================
# 4. CUAD DATASET PROCESSOR
# ===========================================================================

class CUADProcessor:
    """Downloads and processes the CUAD dataset from HuggingFace.

    Converts QA-format CUAD annotations into (clause_text, clause_types) pairs.
    Merges both train and test splits. Groups by contract title first, then
    falls back to ID prefix.
    """

    def __init__(self):
        self.clause_types = config.CUAD_CLAUSE_TYPES
        self.type_to_idx  = {ct: i for i, ct in enumerate(self.clause_types)}

    #def load_and_process(self, cache_dir: Path = config.CUAD_DIR) -> pd.DataFrame:
        """Download CUAD from HuggingFace and convert to a flat clause DataFrame.

        Args:
            cache_dir: Local directory to cache the raw dataset.

        Returns:
            DataFrame with columns:
                contract_id, clause_id, clause_text, clause_types (list),
                party_a, party_b, source
        """
    #    logger.info("Loading CUAD dataset from HuggingFace...")
        #dataset = load_dataset(
        #    config.CUAD_DATASET_ID,
        #    cache_dir=str(cache_dir),
        #    trust_remote_code=True,
        #)
    #    dataset = load_dataset(
    #        "theatticusproject/cuad-qa",
    #        cache_dir=str(cache_dir),
    #    )

        #records: List[Dict] = []
        # CUAD is structured as a single 'train' split with QA pairs
        #split = dataset["train"]

        # Group by contract (identified by 'id' prefix before '__')
        #contract_map: Dict[str, List] = {}
        #for item in tqdm(split, desc="Parsing CUAD"):
        #    # item['id'] format: "ContractName__ClauseType__0"
        #    parts = item["id"].split("__")
        #    contract_name = parts[0] if len(parts) > 0 else "unknown"

        #    if contract_name not in contract_map:
        #        contract_map[contract_name] = []
        #    contract_map[contract_name].append(item)
        #new
    #    records: List[Dict] = []

    #    # Merge train and test because CUAD-QA ships with both
    #    all_items = list(dataset["train"]) + list(dataset["test"])

    #    # Group by contract title first, then fall back to id prefix
    #    contract_map: Dict[str, List] = {}
    #    for item in tqdm(all_items, desc="Parsing CUAD"):
    #        contract_name = item.get("title", "").strip()

    #        if not contract_name:
    #            raw_id = item.get("id", "")
    #            parts = raw_id.split("__")
    #            contract_name = parts[0].strip() if parts else "unknown"

    #        if not contract_name:
    #            contract_name = "unknown"

    #        contract_map.setdefault(contract_name, []).append(item)
        #new
    #    logger.info(f"Found {len(contract_map)} unique contracts in CUAD")

    #    for contract_name, items in tqdm(
    #        contract_map.items(), desc="Processing contracts"
    #    ):
    #        contract_id = self._make_contract_id(contract_name)

            # Collect all answer spans as clauses
    #        clause_texts_seen: Dict[str, List[str]] = {}

    #        for item in items:
    #            clause_type = self._infer_clause_type(item)
    #            if clause_type is None:
    #                continue

                #answers = item.get("answers", {})
                #answer_texts = answers.get("text", [])

                #for ans_text in answer_texts:
                #    ans_text = ans_text.strip()
                #    if not ans_text or len(ans_text.split()) < config.CLAUSE_MIN_TOKENS:
                #        continue
                #new
            #    answers = item.get("answers", {}) or {}
            #    answer_texts = answers.get("text", []) or []

            #    if isinstance(answer_texts, str):
            #        answer_texts = [answer_texts]

            #    for ans_text in answer_texts:
            #        if not isinstance(ans_text, str):
            #            continue
            #        ans_text = ans_text.strip()
            #        if not ans_text:
            #            continue
            #        if len(ans_text.split()) < config.CLAUSE_MIN_TOKENS:
            #            continue
            #        #new
            #        if ans_text not in clause_texts_seen:
            #            clause_texts_seen[ans_text] = []
            #        clause_texts_seen[ans_text].append(clause_type)

            #for clause_text, clause_type_list in clause_texts_seen.items():
            #    clause_id = self._make_clause_id(contract_id, clause_text)
            #    records.append(
            #        {
            #            "contract_id": contract_id,
            #            "clause_id": clause_id,
            #            "clause_text": clause_text,
            #            "clause_types": list(set(clause_type_list)),
            #            "party_a": self._extract_party(clause_text, "a"),
            #            "party_b": self._extract_party(clause_text, "b"),
            #            "source": "CUAD",
            #        }
            #    )


        #df = pd.DataFrame(records)
        #logger.info(f"Processed {len(df)} clauses from {len(contract_map)} contracts")
        #return df
    #new
    def load_and_process(self, cache_dir: Path = config.CUAD_DIR) -> pd.DataFrame:
        """Download CUAD from HuggingFace and convert to a flat clause DataFrame.

        Args:
            cache_dir: Local directory to cache the raw dataset.

        Returns:
            DataFrame with columns:
                contract_id, clause_id, clause_text, clause_types,
                party_a, party_b, source
        """
        logger.info("Loading CUAD dataset from HuggingFace...")
        dataset = load_dataset(
            config.CUAD_DATASET_ID,
            cache_dir=str(cache_dir),
        )

        records: List[Dict] = []
        all_items = list(dataset["train"]) + list(dataset["test"])

        contract_map: Dict[str, List] = {}
        for item in tqdm(all_items, desc="Parsing CUAD"):
            contract_name = item.get("title", "").strip()

            if not contract_name:
                raw_id = item.get("id", "")
                parts  = raw_id.split("__")
                contract_name = parts[0].strip() if parts else "unknown"

            if not contract_name:
                contract_name = "unknown"

            contract_map.setdefault(contract_name, []).append(item)

        logger.info(f"Found {len(contract_map)} unique contracts in CUAD")

        for contract_name, items in tqdm(contract_map.items(), desc="Processing contracts"):
            contract_id        = self._make_contract_id(contract_name)
            clause_texts_seen: Dict[str, List[str]] = {}

            for item in items:
                clause_type = self._infer_clause_type(item)
                if clause_type is None:
                    continue

                answers      = item.get("answers", {}) or {}
                answer_texts = answers.get("text", []) or []

                if isinstance(answer_texts, str):
                    answer_texts = [answer_texts]

                for ans_text in answer_texts:
                    if not isinstance(ans_text, str):
                        continue
                    ans_text = ans_text.strip()
                    if not ans_text:
                        continue
                    if len(ans_text.split()) < config.CLAUSE_MIN_TOKENS:
                        continue
                    clause_texts_seen.setdefault(ans_text, []).append(clause_type)

            for clause_text, clause_type_list in clause_texts_seen.items():
                clause_id = self._make_clause_id(contract_id, clause_text)
                records.append(
                    {
                        "contract_id":  contract_id,
                        "clause_id":    clause_id,
                        "clause_text":  clause_text,
                        "clause_types": sorted(set(clause_type_list)),
                        "party_a":      self._extract_party(clause_text, "a"),
                        "party_b":      self._extract_party(clause_text, "b"),
                        "source":       "CUAD",
                    }
                )

        df = pd.DataFrame(records)
        logger.info(f"Processed {len(df)} clauses from {len(contract_map)} contracts")
        return df
    #new

    #def _infer_clause_type(self, item: Dict) -> Optional[str]:
        """Map a CUAD QA item back to a clause type name.

        CUAD question IDs encode the clause type. We match against the
        canonical clause type list.

        Args:
            item: A single CUAD dataset row.

        Returns:
            Matched clause type string, or None if no match found.
        """
    #    question = item.get("question", "").lower()
    #    for ct in self.clause_types:
    #        if ct.lower() in question:
    #            return ct
    #    return None
    #new
    def _infer_clause_type(self, item: Dict) -> Optional[str]:
        """Infer clause type from item ID first, fall back to question text.

        Args:
            item: A single CUAD dataset row.

        Returns:
            Matched clause type string, or None if no match found.
        """
        raw_id   = item.get("id", "")
        question = item.get("question", "").strip().lower()

        # First try extracting from id, which is usually more stable
        if "__" in raw_id:
            parts = raw_id.split("__")
            if len(parts) >= 2:
                candidate = parts[-1].replace("_", " ").strip().lower()
                for ct in self.clause_types:
                    if ct.lower() == candidate or ct.lower() in candidate or candidate in ct.lower():
                        return ct

        # Fallback to matching on question text
        for ct in self.clause_types:
            if ct.lower() in question or question in ct.lower():
                return ct

        return None
    # new

    @staticmethod
    def _make_contract_id(name: str) -> str:
        return hashlib.md5(name.encode()).hexdigest()[:12]

    @staticmethod
    def _make_clause_id(contract_id: str, text: str) -> str:
        return hashlib.md5(f"{contract_id}{text}".encode()).hexdigest()[:16]

    @staticmethod
    def _extract_party(text: str, party: str) -> str:
        """Heuristically extract party name from clause text."""
        patterns_a = [
            r'"(Company|Licensor|Seller|Vendor|Provider|Employer|Franchisor)"',
            r"(Company|Licensor|Seller|Vendor|Provider)",
        ]
        patterns_b = [
            r'"(Counterparty|Licensee|Buyer|Customer|Employee|Franchisee)"',
            r"(Counterparty|Licensee|Buyer|Customer)",
        ]
        patterns = patterns_a if party == "a" else patterns_b
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return ""


# ===========================================================================
# 5. DATABASE STORAGE
# ===========================================================================

class DataStore:
    """Persists processed contracts and clauses to SQLite via SQLAlchemy."""

    def __init__(self):
        create_tables()

    def save_contract(self, contract_id: str, source: str, filename: str) -> None:
        """Insert a contract record if it does not already exist.

        Args:
            contract_id: Unique contract identifier.
            source: Origin of the contract ('CUAD', 'upload', etc.).
            filename: Original filename or identifier.
        """
        with SessionLocal() as session:
            if session.get(Contract, contract_id) is None:
                session.add(
                    Contract(
                        contract_id=contract_id,
                        filename=filename,
                        source=source,
                    )
                )
                session.commit()

    def save_clauses(self, df: pd.DataFrame) -> int:
        """Bulk-insert clauses from a DataFrame into the database.

        Args:
            df: DataFrame with columns matching the Clause ORM model.

        Returns:
            Number of new clauses inserted.
        """
        inserted = 0
        with SessionLocal() as session:
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Saving clauses"):
                if session.get(Clause, row["clause_id"]) is not None:
                    continue

                clause_types = row.get("clause_types", [])
                clause_types_str = (
                    "|".join(clause_types) if isinstance(clause_types, list)
                    else str(clause_types)
                )

                session.add(
                    Clause(
                        clause_id=row["clause_id"],
                        contract_id=row["contract_id"],
                        clause_text=row["clause_text"],
                        clause_type=clause_types_str,
                        party_a=row.get("party_a", ""),
                        party_b=row.get("party_b", ""),
                        source=row.get("source", ""),
                    )
                )
                inserted += 1
            session.commit()

        logger.info(f"Saved {inserted} new clauses to database")
        return inserted

    def load_all_clauses(self) -> pd.DataFrame:
        """Load all clauses from the database as a DataFrame.

        Returns:
            DataFrame with all clause fields.
        """
        with SessionLocal() as session:
            rows = session.execute(text("SELECT * FROM clauses")).fetchall()
            if not rows:
                return pd.DataFrame()
            return pd.DataFrame(rows, columns=rows[0]._fields)


# ===========================================================================
# 6. TRAIN / VAL / TEST SPLITTER
# ===========================================================================

class DataSplitter:
    """Produces stratified 80/10/10 splits by primary clause type.

    Raises the minimum class size to 10 so that after the 80/20 first
    split the temp pool reliably has >=2 samples per class for the
    second stratified split. Rare classes are routed to train.
    """

    #def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    #    """Generate stratified train, validation, and test DataFrames.

    #    Args:
    #        df: Full clause DataFrame with a 'clause_type' column.

    #    Returns:
    #        Tuple of (train_df, val_df, test_df).
    #    """
    #    # Use the first (primary) clause type for stratification
    #    df = df.copy()
    #    df["primary_type"] = df["clause_type"].apply(
    #        lambda x: x.split("|")[0] if isinstance(x, str) and x else "Unknown"
    #    )

        # Remove classes with fewer than 3 samples (can't stratify)
    #    counts = df["primary_type"].value_counts()
    #    valid_types = counts[counts >= 3].index
    #    df_filtered = df[df["primary_type"].isin(valid_types)]
    #    df_rare = df[~df["primary_type"].isin(valid_types)]

    #    train_df, temp_df = train_test_split(
    #        df_filtered,
    #        test_size=(1 - config.TRAIN_RATIO),
    #        stratify=df_filtered["primary_type"],
    #        random_state=config.RANDOM_SEED,
    #    )

    #    relative_val = config.VAL_RATIO / (config.VAL_RATIO + config.TEST_RATIO)
    #    val_df, test_df = train_test_split(
    #        temp_df,
    #        test_size=(1 - relative_val),
    #        stratify=temp_df["primary_type"],
    #        random_state=config.RANDOM_SEED,
    #    )

    #    # Append rare-class samples to train to avoid losing them entirely
    #    train_df = pd.concat([train_df, df_rare], ignore_index=True)

    #    logger.info(
    #        f"Split sizes — Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
    #    )
    #    return train_df, val_df, test_df
    #new
    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate stratified train, validation, and test DataFrames."""

        df = df.copy()
        df["primary_type"] = df["clause_type"].apply(
            lambda x: x.split("|")[0] if isinstance(x, str) and x else "Unknown"
        )

        # Need enough samples so that after the first split, temp still has
        # at least 2 examples per class for the val/test stratified split.
        # With an 80/10/10 split, temp is 20%, so classes need roughly 10+
        # examples to safely preserve >=2 in temp.
        counts      = df["primary_type"].value_counts()
        valid_types = counts[counts >= 10].index

        df_filtered = df[df["primary_type"].isin(valid_types)].copy()
        df_rare     = df[~df["primary_type"].isin(valid_types)].copy()

        if df_filtered.empty:
            raise RuntimeError("No classes have enough samples for stratified splitting.")

        train_df, temp_df = train_test_split(
            df_filtered,
            test_size=(1 - config.TRAIN_RATIO),
            stratify=df_filtered["primary_type"],
            random_state=config.RANDOM_SEED,
        )

        temp_counts      = temp_df["primary_type"].value_counts()
        temp_valid_types = temp_counts[temp_counts >= 2].index

        temp_stratified = temp_df[temp_df["primary_type"].isin(temp_valid_types)].copy()
        temp_rare       = temp_df[~temp_df["primary_type"].isin(temp_valid_types)].copy()

        relative_val = config.VAL_RATIO / (config.VAL_RATIO + config.TEST_RATIO)

        if temp_stratified.empty:
            val_df   = pd.DataFrame(columns=df.columns)
            test_df  = pd.DataFrame(columns=df.columns)
            train_df = pd.concat([train_df, df_rare, temp_rare], ignore_index=True)
        else:
            val_df, test_df = train_test_split(
                temp_stratified,
                test_size=(1 - relative_val),
                stratify=temp_stratified["primary_type"],
                random_state=config.RANDOM_SEED,
            )
            # Put unsplittable rare leftovers back into train
            train_df = pd.concat([train_df, df_rare, temp_rare], ignore_index=True)

        logger.info(
            f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
        )
        return train_df, val_df, test_df
    #new

    def save_splits(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> None:
        """Persist train/val/test splits to CSV files.

        Args:
            train_df: Training set DataFrame.
            val_df: Validation set DataFrame.
            test_df: Test set DataFrame.
        """
        train_df.to_csv(config.PROCESSED_DIR / "train.csv", index=False)
        val_df.to_csv(config.PROCESSED_DIR / "val.csv",   index=False)
        test_df.to_csv(config.PROCESSED_DIR / "test.csv", index=False)
        logger.info(f"Splits saved to {config.PROCESSED_DIR}")


# ===========================================================================
# 7. UNIFIED PIPELINE RUNNER
# ===========================================================================

class ContractIntelligencePipeline:
    """End-to-end pipeline orchestrator for data preparation."""

    def __init__(self):
        self.pdf_ingester   = PDFIngester()
        self.segmenter      = ClauseSegmenter()
        self.cuad_processor = CUADProcessor()
        self.datastore      = DataStore()
        self.splitter       = DataSplitter()

    def run_cuad(self) -> pd.DataFrame:
        """Download and process the full CUAD dataset end-to-end.

        Returns:
            Processed DataFrame saved to database and returned.
        """
        logger.info("=== Starting CUAD pipeline ===")
        df = self.cuad_processor.load_and_process()

        for contract_id in df["contract_id"].unique():
            self.datastore.save_contract(
                contract_id=contract_id,
                source="CUAD",
                filename=contract_id,
            )

        self.datastore.save_clauses(df)
        logger.info("=== CUAD pipeline complete ===")
        return df

    def run_pdf(self, pdf_path: Path, contract_id: Optional[str] = None) -> pd.DataFrame:
        """Ingest a single PDF contract and store its clauses.

        Args:
            pdf_path: Path to the PDF file.
            contract_id: Optional override; auto-generated from filename if None.

        Returns:
            DataFrame of extracted clauses.
        """
        logger.info(f"=== Processing PDF: {pdf_path.name} ===")
        if contract_id is None:
            contract_id = hashlib.md5(pdf_path.name.encode()).hexdigest()[:12]

        raw_text = self.pdf_ingester.extract_text(pdf_path)
        clauses  = self.segmenter.segment(raw_text)

        records = []
        for i, clause_text in enumerate(clauses):
            clause_id = hashlib.md5(f"{contract_id}{i}{clause_text}".encode()).hexdigest()[:16]
            records.append(
                {
                    "contract_id":  contract_id,
                    "clause_id":    clause_id,
                    "clause_text":  clause_text,
                    "clause_types": [],
                    "party_a":      "",
                    "party_b":      "",
                    "source":       "upload",
                }
            )

        df = pd.DataFrame(records)
        self.datastore.save_contract(contract_id=contract_id, source="upload", filename=pdf_path.name)
        self.datastore.save_clauses(df)
        logger.info(f"Extracted {len(df)} clauses from {pdf_path.name}")
        return df

    def run_text(self, text: str, contract_id: str) -> pd.DataFrame:
        """Ingest a raw text contract (e.g., from API upload).

        Args:
            text: Full contract text string.
            contract_id: Unique identifier for this contract.

        Returns:
            DataFrame of extracted clauses.
        """
        clauses = self.segmenter.segment(text)
        records = []
        for i, clause_text in enumerate(clauses):
            clause_id = hashlib.md5(f"{contract_id}{i}{clause_text}".encode()).hexdigest()[:16]
            records.append(
                {
                    "contract_id":  contract_id,
                    "clause_id":    clause_id,
                    "clause_text":  clause_text,
                    "clause_types": [],
                    "party_a":      "",
                    "party_b":      "",
                    "source":       "text_input",
                }
            )

        df = pd.DataFrame(records)
        self.datastore.save_contract(contract_id=contract_id, source="text_input", filename=contract_id)
        self.datastore.save_clauses(df)
        return df

    def run_split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load all stored clauses and produce stratified train/val/test splits.

        Returns:
            Tuple of (train_df, val_df, test_df) saved as CSVs.
        """
        logger.info("=== Generating data splits ===")
        df = self.datastore.load_all_clauses()
        if df.empty:
            raise RuntimeError("No clauses in database. Run --mode cuad first.")

        train_df, val_df, test_df = self.splitter.split(df)
        self.splitter.save_splits(train_df, val_df, test_df)
        return train_df, val_df, test_df


# ===========================================================================
# CLI ENTRY POINT
# ===========================================================================

def main():
    """Command-line interface for the data pipeline."""
    parser = argparse.ArgumentParser(
        description="Contract Intelligence Data Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples:
              python src/data_pipeline.py --mode cuad
              python src/data_pipeline.py --mode pdf --file contracts/my_contract.pdf
              python src/data_pipeline.py --mode split
            """
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["cuad", "pdf", "split"],
        required=True,
        help="Pipeline mode to run",
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="Path to PDF file (required for --mode pdf)",
    )
    args = parser.parse_args()

    pipeline = ContractIntelligencePipeline()

    if args.mode == "cuad":
        pipeline.run_cuad()

    elif args.mode == "pdf":
        if args.file is None:
            parser.error("--file is required when --mode is pdf")
        pipeline.run_pdf(args.file)

    elif args.mode == "split":
        train_df, val_df, test_df = pipeline.run_split()
        print(f"\nSplit complete:")
        print(f"  Train: {len(train_df)} clauses")
        print(f"  Val:   {len(val_df)} clauses")
        print(f"  Test:  {len(test_df)} clauses")


if __name__ == "__main__":
    main()
