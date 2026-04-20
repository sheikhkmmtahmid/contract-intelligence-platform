"""
data_pipeline.py — Contract ingestion, segmentation, and storage pipeline.

Responsibilities:
  1. Download and parse the CUAD dataset from HuggingFace.
  2. Download and parse the LEDGAR dataset (100 provision types, ~80k clauses).
  3. Download and parse the MAUD dataset (14 M&A deal-point types, ~39k clauses).
  4. Ingest raw contract PDFs using pdfplumber.
  5. Segment contract text into individual clauses via rule-based sentence
     boundary detection combined with Legal-BERT embedding similarity.
  6. Label each clause with its unified clause type (multi-label, 100 types).
  7. Persist processed clauses to SQLite via SQLAlchemy.
  8. Produce stratified 80/10/10 train/val/test splits.

Usage:
    python src/data_pipeline.py --mode cuad        # download + process CUAD
    python src/data_pipeline.py --mode ledgar      # download + process LEDGAR
    python src/data_pipeline.py --mode maud        # download + process MAUD
    python src/data_pipeline.py --mode all         # all three datasets + split
    python src/data_pipeline.py --mode pdf --file path/to/contract.pdf
    python src/data_pipeline.py --mode split       # generate train/val/test CSVs
"""

import argparse
import hashlib
import re
import sys
import textwrap
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pdfplumber
import requests
import torch
from datasets import load_dataset
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sqlalchemy import text
from sqlalchemy.orm import Session
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# Resolve project root so the script is importable from any working directory
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from api.database import Clause, Contract, SessionLocal, create_tables

# Logging setup
logger.remove()
logger.add(
    config.LOGS_DIR / "data_pipeline.log",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG",
)
logger.add(sys.stderr, level="INFO")


# 1. PDF INGESTION

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
            for i, page in enumerate(pdf.pages[:total]):
                page_text = page.extract_text(x_tolerance=2, y_tolerance=2)
                if page_text:
                    pages_text.append(page_text)
            logger.info(f"Extracted text from {total} pages in {pdf_path.name}")

        full_text = "\n".join(pages_text)
        full_text = self._clean_text(full_text)

        if not full_text.strip():
            raise ValueError(f"No extractable text found in {pdf_path.name}")

        return full_text

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalise whitespace and remove PDF artefacts.

        Args:
            text: Raw extracted text.

        Returns:
            Cleaned text with normalised whitespace.
        """
        # Remove null bytes and non-printable characters except newlines/tabs
        text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]", " ", text)
        # Collapse multiple spaces to single
        text = re.sub(r"[ \t]{2,}", " ", text)
        # Collapse more than two consecutive newlines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


# 2. CLAUSE SEGMENTATION

class ClauseSegmenter:
    """Segments contract text into individual clauses.

    Strategy:
      - Primary split: numbered section headers (e.g., "1.", "1.1", "Article 2")
      - Secondary split: paragraph boundaries (double newline)
      - Filter: discard segments shorter than CLAUSE_MIN_TOKENS words
      - Legal-BERT embeddings used for semantic deduplication of near-duplicate
        segments that the rule-based splitter may produce.
    """

    # Regex patterns for common legal section headers
    SECTION_PATTERNS = [
        r"^(?:ARTICLE|SECTION|CLAUSE)\s+\d+",          # ARTICLE 1, SECTION 2
        r"^\d+\.\d*\s+[A-Z]",                           # 1.1 LIABILITY
        r"^\d+\.\s+[A-Z]",                              # 1. DEFINITIONS
        r"^[A-Z][A-Z\s]{4,}$",                          # ALL-CAPS HEADERS
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
        self.tokenizer = tokenizer

    def segment(self, text: str) -> List[str]:
        """Split a full contract text into a list of clause strings.

        Args:
            text: Full contract text (already cleaned).

        Returns:
            List of clause strings, each representing a distinct clause.
        """
        # Try header-based segmentation first
        clauses = self._split_by_headers(text)

        # Fallback to paragraph-based if too few clauses found
        if len(clauses) < 5:
            clauses = self._split_by_paragraphs(text)

        # Filter out noise
        clauses = [self._clean_clause(c) for c in clauses]
        clauses = [c for c in clauses if self._is_valid(c)]

        # Truncate to max token length if tokenizer is available
        if self.tokenizer is not None:
            clauses = [self._truncate(c) for c in clauses]

        logger.debug(f"Segmented into {len(clauses)} clauses")
        return clauses

    def _split_by_headers(self, text: str) -> List[str]:
        """Split text on detected section headers.

        Args:
            text: Full contract text.

        Returns:
            List of text segments between headers.
        """
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
        """Split text on paragraph boundaries (double newlines).

        Args:
            text: Full contract text.

        Returns:
            List of paragraphs.
        """
        return [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]

    def _is_header(self, line: str) -> bool:
        """Return True if the line looks like a legal section header."""
        line = line.strip()
        if not line:
            return False
        return any(p.match(line) for p in self.COMPILED_PATTERNS)

    def _is_valid(self, clause: str) -> bool:
        """Return True if clause meets minimum length requirements."""
        word_count = len(clause.split())
        return word_count >= self.min_tokens

    @staticmethod
    def _clean_clause(clause: str) -> str:
        """Strip leading/trailing whitespace and normalise internal spacing."""
        clause = re.sub(r"[ \t]{2,}", " ", clause)
        return clause.strip()

    def _truncate(self, clause: str) -> str:
        """Truncate clause text to max_tokens using the tokenizer.

        Args:
            clause: Raw clause text.

        Returns:
            Truncated clause text decoded back to string.
        """
        tokens = self.tokenizer(
            clause,
            max_length=self.max_tokens,
            truncation=True,
            return_tensors="pt",
        )
        return self.tokenizer.decode(
            tokens["input_ids"][0],
            skip_special_tokens=True,
        )


# 3. LEGAL-BERT EMBEDDING GENERATOR

class EmbeddingGenerator:
    """Generates [CLS] token embeddings from Legal-BERT.

    Embeddings are used both for semantic clause deduplication during
    segmentation and as input features for the anomaly detection engine.
    """

    def __init__(self, model_name: str = config.LEGAL_BERT_MODEL, device: str = "auto"):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading Legal-BERT model: {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def embed(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate [CLS] embeddings for a list of clause texts.

        Args:
            texts: List of clause strings.
            batch_size: Number of clauses to encode per forward pass.

        Returns:
            NumPy array of shape (N, 768) where N = len(texts).
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
            # [CLS] token is at position 0 of last_hidden_state
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeddings)

        return np.vstack(all_embeddings)


# 4. CUAD DATASET PROCESSOR

class CUADProcessor:
    """Downloads and processes the CUAD dataset from HuggingFace.

    CUAD contains 510 commercial contracts with 41 clause-type annotations
    provided as question-answer pairs per clause type.

    This processor:
      - Converts QA-format CUAD annotations into (clause_text, clause_types) pairs.
      - Extracts the answer spans as clause-level text.
      - Assigns multi-hot labels over the 41 clause types.
    """

    def __init__(self):
        self.clause_types = config.CUAD_CLAUSE_TYPES
        self.type_to_idx = {ct: i for i, ct in enumerate(self.clause_types)}

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
                parts = raw_id.split("__")
                contract_name = parts[0].strip() if parts else "unknown"

            if not contract_name:
                contract_name = "unknown"

            contract_map.setdefault(contract_name, []).append(item)

        logger.info(f"Found {len(contract_map)} unique contracts in CUAD")

        for contract_name, items in tqdm(contract_map.items(), desc="Processing contracts"):
            contract_id = self._make_contract_id(contract_name)
            clause_texts_seen: Dict[str, List[str]] = {}

            for item in items:
                clause_type = self._infer_clause_type(item)
                if clause_type is None:
                    continue

                answers = item.get("answers", {}) or {}
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
                        "contract_id": contract_id,
                        "clause_id": clause_id,
                        "clause_text": clause_text,
                        "clause_types": sorted(set(clause_type_list)),
                        "party_a": self._extract_party(clause_text, "a"),
                        "party_b": self._extract_party(clause_text, "b"),
                        "source": "CUAD",
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
        raw_id = item.get("id", "")
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
        """Produce a stable, short contract identifier from a contract name."""
        return hashlib.md5(name.encode()).hexdigest()[:12]

    @staticmethod
    def _make_clause_id(contract_id: str, text: str) -> str:
        """Produce a stable clause identifier from contract ID + clause text."""
        return hashlib.md5(f"{contract_id}{text}".encode()).hexdigest()[:16]

    @staticmethod
    def _extract_party(text: str, party: str) -> str:
        """Heuristically extract party name from clause text.

        Args:
            text: Clause text.
            party: 'a' or 'b' — which party to look for.

        Returns:
            Party name string if found, else empty string.
        """
        # Common legal party indicators
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


# 5. LEDGAR DATASET PROCESSOR

# Maps LEDGAR provision-type strings → unified taxonomy names.
# Keys are LEDGAR's exact ClassLabel strings; None = skip (too generic / OOV).
_LEDGAR_LABEL_MAP: Dict[str, Optional[str]] = {
    "Adjustments":              None,
    "Agreements":               None,
    "Amendments":               "Amendments",
    "Anti-Corruption Laws":     "Anti-Corruption Laws",
    "Approvals":                "Approvals And Consents",
    "Arbitration":              "Dispute Resolution",
    "Assignments":              "Anti-Assignment",
    "Authority":                "Authority",
    "Base Salary":              "Employment And Benefits",
    "Benefits":                 "Employment And Benefits",
    "Capitalization":           "Capitalization",
    "Closings":                 None,
    "Compliance With Laws":     "Compliance With Laws",
    "Confidentiality":          "Confidentiality",
    "Consent To Jurisdiction":  "Consent To Jurisdiction",
    "Construction":             "Definitions",
    "Definitions":              "Definitions",
    "Disability":               "Employment And Benefits",
    "Effective Dates":          "Effective Date",
    "Employment":               "Employment And Benefits",
    "Enforceability":           "Enforceability",
    "Entire Agreements":        "Entire Agreements",
    "Erisa":                    "Employment And Benefits",
    "Expenses":                 "Expenses",
    "Further Assurances":       "Further Assurances",
    "General":                  None,
    "Governing Laws":           "Governing Law",
    "Headings":                 None,
    "Indemnifications":         "Indemnification",
    "Insurances":               "Insurance",
    "Integration":              "Entire Agreements",
    "Intellectual Property":    "IP Ownership Assignment",
    "Interests":                "Interests",
    "Jurisdictions":            "Consent To Jurisdiction",
    "Liens":                    "Liens And Encumbrances",
    "Limitations Of Remedies":  "Limitations Of Remedies",
    "Non-Waivers":              "Non-Waiver",
    "Notices":                  "Notices",
    "No Third-Party Beneficiaries": "Third Party Beneficiary",
    "Obligations":              None,
    "Organization":             "Organization And Existence",
    "Payments":                 "Payments",
    "Remedies":                 "Limitations Of Remedies",
    "Representations":          "Representations And Warranties",
    "Sanctions":                "Sanctions",
    "Securities Laws":          "Securities Law Compliance",
    "Severability":             "Severability",
    "Specific Performance":     "Specific Performance",
    "Successors":               "Successors And Assigns",
    "Survival":                 "Survival",
    "Taxes":                    "Taxes And Withholding",
    "Terminations":             "Termination For Convenience",
    "Titles":                   None,
    "Trade Controls":           "Trade Controls",
    "Transactions With Affiliates": "Transactions With Affiliates",
    "Transfers":                "Anti-Assignment",
    "Waiver Of Jury Trial":     "Waiver Of Jury Trial",
    "Waivers":                  "Waivers",
    "Warranties":               "Representations And Warranties",
    "Withholding":              "Taxes And Withholding",
}


class LEDGARProcessor:
    """Downloads and processes the LEDGAR dataset from HuggingFace (via lex_glue).

    LEDGAR contains ~80,000 contract provisions labelled with 100 provision
    types. This processor maps those labels to the unified 100-type taxonomy
    and returns a DataFrame with the same schema as CUADProcessor.
    """

    def load_and_process(self, cache_dir: Path = config.LEDGAR_DIR) -> pd.DataFrame:
        logger.info("Loading LEDGAR dataset from HuggingFace (lex_glue/ledgar)...")
        dataset = load_dataset(
            config.LEDGAR_DATASET_ID,
            "ledgar",
            cache_dir=str(cache_dir),
        )

        records: List[Dict] = []
        skipped = 0

        for split_name in ("train", "validation", "test"):
            if split_name not in dataset:
                continue
            split = dataset[split_name]
            label_feature = split.features["label"]

            for item in tqdm(split, desc=f"Processing LEDGAR {split_name}"):
                text = (item.get("text") or "").strip()
                if not text or len(text.split()) < config.CLAUSE_MIN_TOKENS:
                    skipped += 1
                    continue

                # Decode integer label → LEDGAR string → unified type
                label_int = item["label"]
                ledgar_label = label_feature.int2str(label_int)
                unified_type = _LEDGAR_LABEL_MAP.get(ledgar_label)
                if unified_type is None:
                    skipped += 1
                    continue

                clause_id = hashlib.md5(f"LEDGAR{text}".encode()).hexdigest()[:16]
                contract_id = hashlib.md5(f"LEDGAR_{label_int}".encode()).hexdigest()[:12]

                records.append({
                    "contract_id":  contract_id,
                    "clause_id":    clause_id,
                    "clause_text":  text,
                    "clause_types": [unified_type],
                    "party_a":      "",
                    "party_b":      "",
                    "source":       "LEDGAR",
                })

        df = pd.DataFrame(records)
        # De-duplicate on clause_id (identical texts may appear across splits)
        df = df.drop_duplicates(subset="clause_id").reset_index(drop=True)
        logger.info(
            f"LEDGAR: {len(df)} clauses retained, {skipped} skipped "
            f"(too short or unmapped label)"
        )
        return df


# 6. MAUD DATASET PROCESSOR

# Maps substrings found in MAUD 'question' field → unified taxonomy names.
# Matching is done with .lower() contains checks in priority order.
_MAUD_QUESTION_MAP: List[Tuple[str, str]] = [
    ("reverse termination fee",     "Reverse Termination Fee"),
    ("termination fee",             "Termination Fee"),
    ("hell-or-high-water",          "Hell-Or-High-Water"),
    ("no-shop",                     "No-Shop"),
    ("non-solicitation",            "No-Shop"),
    ("fiduciary",                   "Fiduciary Exception"),
    ("operating covenant",          "Operating Covenants"),
    ("mae",                         "Material Adverse Effect"),
    ("material adverse effect",     "Material Adverse Effect"),
    ("antitrust",                   "Antitrust Efforts Standard"),
    ("specific performance",        "Specific Performance"),
    ("superior offer",              "Superior Offer Definition"),
    ("intervening event",           "Intervening Event Definition"),
    ("matching right",              "Matching Rights"),
    ("tail period",                 "Tail Period"),
    ("expense reimbursement",       "Expense Reimbursement"),
    ("board recommendation",        "Board Recommendation Change"),
    ("change of recommendation",    "Board Recommendation Change"),
    ("change in recommendation",    "Board Recommendation Change"),
]


class MAUDProcessor:
    """Downloads and processes the MAUD dataset from HuggingFace.

    MAUD contains ~39,200 M&A contract excerpts annotated with 22 deal-point
    types. This processor maps those types to the unified taxonomy and returns
    a DataFrame with the same schema as CUADProcessor.
    """

    def load_and_process(self, cache_dir: Path = config.MAUD_DIR) -> pd.DataFrame:
        logger.info("Loading MAUD dataset from HuggingFace...")
        dataset = load_dataset(
            config.MAUD_DATASET_ID,
            cache_dir=str(cache_dir),
        )

        records: List[Dict] = []
        skipped = 0

        for split_name in ("train", "validation", "test"):
            if split_name not in dataset:
                continue
            split = dataset[split_name]

            for item in tqdm(split, desc=f"Processing MAUD {split_name}"):
                text = (item.get("text") or "").strip()
                if not text or len(text.split()) < config.CLAUSE_MIN_TOKENS:
                    skipped += 1
                    continue

                question = (item.get("question") or "").strip().lower()
                unified_type = self._map_question(question)
                if unified_type is None:
                    skipped += 1
                    continue

                # Use category + question hash as synthetic contract grouping
                category = (item.get("category") or "MAUD").strip()
                contract_id = hashlib.md5(
                    f"MAUD_{category}_{question}".encode()
                ).hexdigest()[:12]
                clause_id = hashlib.md5(f"MAUD{text}".encode()).hexdigest()[:16]

                records.append({
                    "contract_id":  contract_id,
                    "clause_id":    clause_id,
                    "clause_text":  text,
                    "clause_types": [unified_type],
                    "party_a":      "",
                    "party_b":      "",
                    "source":       "MAUD",
                })

        df = pd.DataFrame(records)
        df = df.drop_duplicates(subset="clause_id").reset_index(drop=True)
        logger.info(
            f"MAUD: {len(df)} clauses retained, {skipped} skipped "
            f"(too short or unmapped question)"
        )
        return df

    @staticmethod
    def _map_question(question: str) -> Optional[str]:
        """Return unified type for a MAUD question string, or None if unmapped."""
        for keyword, unified_type in _MAUD_QUESTION_MAP:
            if keyword in question:
                return unified_type
        return None


# 7. EDGAR DOWNLOADER

class EDGARDownloader:
    """Downloads EX-10.* material contract exhibits from SEC EDGAR.

    Strategy
    --------
    1. Query EDGAR EFTS (full-text search) for 8-K filings that contain
       contract language.  Each EFTS hit includes ``file_type`` (the exhibit
       type within the filing) and ``adsh`` (accession number).
    2. Keep only hits where ``file_type`` starts with ``EX-10``.
    3. For each hit, download the full SGML submission ``.txt`` file
       (one request per filing).
    4. Parse the SGML ``<DOCUMENT>`` blocks to extract the text of every
       EX-10.* exhibit and save it to raw_dir.

    One HTTP request per filing (the SGML file contains all embedded docs).
    Rate-limited to ≤10 requests/second per SEC fair-use policy.
    """

    EFTS_URL      = "https://efts.sec.gov/LATEST/search-index"
    ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"
    HEADERS       = {"User-Agent": "ContractIntelligencePlatform research@example.com"}
    REQUEST_DELAY = 0.12   # stay safely under 10 req/sec

    def __init__(self, raw_dir: Path = config.EDGAR_RAW_DIR):
        self.raw_dir = raw_dir
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    # EFTS errors above ~1000 results per query window, so we paginate
    # within narrow quarterly date windows instead of scrolling one big range.
    _DATE_WINDOWS = [
        ("2018-01-01", "2018-06-30"), ("2018-07-01", "2018-12-31"),
        ("2019-01-01", "2019-06-30"), ("2019-07-01", "2019-12-31"),
        ("2020-01-01", "2020-06-30"), ("2020-07-01", "2020-12-31"),
        ("2021-01-01", "2021-06-30"), ("2021-07-01", "2021-12-31"),
        ("2022-01-01", "2022-06-30"), ("2022-07-01", "2022-12-31"),
        ("2023-01-01", "2023-06-30"), ("2023-07-01", "2023-12-31"),
        ("2024-01-01", "2024-06-30"), ("2024-07-01", "2024-12-31"),
    ]

    def download(self, limit: int = config.EDGAR_DOWNLOAD_LIMIT) -> List[Path]:
        """Download up to *limit* EX-10.* exhibit files.

        Iterates over semi-annual date windows to avoid the EFTS deep-pagination
        500 error (server rejects ``from`` > ~1000 for any single query window).
        Within each window, pages with ``from`` / ``size`` until exhausted.

        Args:
            limit: Maximum number of exhibit files to save.

        Returns:
            List of paths to downloaded files.
        """
        logger.info(f"Starting EDGAR download — target: {limit} contracts")
        downloaded: List[Path] = []
        seen_accessions: set   = set()
        batch = 100

        for startdt, enddt in self._DATE_WINDOWS:
            if len(downloaded) >= limit:
                break
            logger.info(f"  Window {startdt} → {enddt}")
            start = 0

            while len(downloaded) < limit:
                try:
                    resp = requests.get(
                        self.EFTS_URL,
                        params={
                            "q":         "agreement",
                            "forms":     "8-K",
                            "dateRange": "custom",
                            "startdt":   startdt,
                            "enddt":     enddt,
                            "from":      start,
                            "size":      batch,
                        },
                        headers=self.HEADERS,
                        timeout=30,
                    )
                    resp.raise_for_status()
                    hits = resp.json().get("hits", {}).get("hits", [])
                except Exception as e:
                    logger.error(f"EFTS search error ({startdt}→{enddt} from={start}): {e}")
                    break  # move to next window on error

                if not hits:
                    break  # window exhausted

                for hit in hits:
                    if len(downloaded) >= limit:
                        break
                    src       = hit.get("_source", {})
                    file_type = src.get("file_type", "")
                    adsh      = src.get("adsh", "")
                    ciks      = src.get("ciks", [])

                    if not file_type.upper().startswith("EX-10"):
                        continue
                    if not adsh or not ciks:
                        continue
                    if adsh in seen_accessions:
                        continue
                    seen_accessions.add(adsh)

                    cik   = str(int(ciks[0]))
                    paths = self._extract_exhibits_from_sgml(cik, adsh, src.get("file_date", "unknown"))
                    downloaded.extend(paths)
                    if paths:
                        logger.debug(f"  +{len(paths)} from {adsh} ({file_type})")
                    time.sleep(self.REQUEST_DELAY)

                start += len(hits)
                logger.info(f"Downloaded {len(downloaded)}/{limit} contracts so far")

        logger.info(f"EDGAR download complete: {len(downloaded)} contracts")
        return downloaded

    def _extract_exhibits_from_sgml(
        self, cik: str, adsh: str, file_date: str
    ) -> List[Path]:
        """Download the full SGML submission and extract all EX-10.* texts.

        The EDGAR submission ``.txt`` file is a multi-document SGML container.
        Each embedded document is wrapped in ``<DOCUMENT>...</DOCUMENT>`` with
        ``<TYPE>`` and ``<TEXT>`` tags.  We extract the text of every EX-10.*
        document and save each as a separate file in raw_dir.

        Args:
            cik:       Filer CIK (numeric, no leading zeros).
            adsh:      Accession number with dashes e.g. 0001234567-24-000001.
            file_date: Filing date string for output filename.

        Returns:
            List of paths to saved exhibit files.
        """
        paths: List[Path] = []
        accession_nd = adsh.replace("-", "")
        sgml_url     = f"{self.ARCHIVES_BASE}/{cik}/{accession_nd}/{adsh}.txt"

        try:
            resp = requests.get(sgml_url, headers=self.HEADERS, timeout=45)
            resp.raise_for_status()
            sgml = resp.text
        except Exception as e:
            logger.debug(f"Could not fetch SGML for {adsh}: {e}")
            return paths

        # Split into <DOCUMENT> blocks and process each
        doc_blocks = re.split(r"<DOCUMENT>", sgml, flags=re.IGNORECASE)[1:]
        for block in doc_blocks:
            type_match = re.search(r"<TYPE>([^\n<]+)", block, re.IGNORECASE)
            if not type_match:
                continue
            doc_type = type_match.group(1).strip()
            if not doc_type.upper().startswith("EX-10"):
                continue

            text_match = re.search(
                r"<TEXT>(.*?)(?:</TEXT>|</DOCUMENT>)", block,
                re.IGNORECASE | re.DOTALL,
            )
            if not text_match:
                continue

            exhibit_text = text_match.group(1).strip()
            if not exhibit_text or len(exhibit_text) < 200:
                continue

            safe_type   = re.sub(r"[^\w]", "_", doc_type)
            exhibit_id  = hashlib.md5(f"{adsh}{doc_type}".encode()).hexdigest()[:8]
            output_path = self.raw_dir / f"{file_date}_{cik}_{safe_type}_{exhibit_id}.txt"

            if not output_path.exists():
                output_path.write_text(exhibit_text, encoding="utf-8", errors="replace")
            paths.append(output_path)

        return paths


# 8. EDGAR PROCESSOR

class EDGARProcessor:
    """Cleans and segments raw EDGAR contract text files.

    Strips HTML tags, SEC boilerplate headers (page numbers, form headers,
    EDGAR filing metadata), then feeds clean text through ClauseSegmenter.
    Returns same DataFrame schema as CUADProcessor.
    """

    BOILERPLATE_PATTERNS = [
        re.compile(r"<[^>]+>"),
        re.compile(r"&\w+;"),
        re.compile(r"={3,}"),
        re.compile(r"-{3,}"),
        re.compile(r"Page \d+ of \d+", re.IGNORECASE),
        re.compile(r"EXHIBIT \d+[\.\d]*", re.IGNORECASE),
        re.compile(r"<DOCUMENT>.*?</DOCUMENT>", re.DOTALL | re.IGNORECASE),
        re.compile(r"<TEXT>|</TEXT>", re.IGNORECASE),
        re.compile(r"\[EDGAR[^\]]*\]", re.IGNORECASE),
    ]

    def __init__(self):
        self.segmenter = ClauseSegmenter()

    def process_file(self, file_path: Path) -> pd.DataFrame:
        """Process a single EDGAR raw file into a clause DataFrame.

        Args:
            file_path: Path to the raw downloaded EDGAR file.

        Returns:
            DataFrame of clauses, or empty DataFrame if file yields nothing.
        """
        try:
            raw = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.debug(f"Could not read {file_path.name}: {e}")
            return pd.DataFrame()

        clean = self._clean(raw)
        if len(clean.split()) < 100:
            return pd.DataFrame()

        clauses = self.segmenter.segment(clean)
        if not clauses:
            return pd.DataFrame()

        contract_id = hashlib.md5(file_path.name.encode()).hexdigest()[:12]
        records = []
        for clause_text in clauses:
            clause_id = hashlib.md5(f"EDGAR{contract_id}{clause_text}".encode()).hexdigest()[:16]
            records.append({
                "contract_id":  contract_id,
                "clause_id":    clause_id,
                "clause_text":  clause_text,
                "clause_types": [],
                "party_a":      "",
                "party_b":      "",
                "source":       "EDGAR",
            })
        return pd.DataFrame(records)

    def process_directory(self, raw_dir: Path = config.EDGAR_RAW_DIR) -> pd.DataFrame:
        """Process only NEW raw .txt files (not already in the database).

        Checks the SQLite database for contract_ids already ingested from EDGAR
        and skips those files, so repeated runs only process newly downloaded files.

        Args:
            raw_dir: Directory containing downloaded EDGAR .txt files.

        Returns:
            Combined DataFrame of new clauses only.
        """
        files = list(raw_dir.glob("*.txt"))
        if not files:
            logger.warning(f"No .txt files found in {raw_dir}")
            return pd.DataFrame()

        # Compute contract_id for each file (same hash as process_file uses)
        already_processed = self._get_processed_contract_ids()
        new_files = [
            f for f in files
            if hashlib.md5(f.name.encode()).hexdigest()[:12] not in already_processed
        ]

        if not new_files:
            logger.info("All EDGAR files already processed — nothing new to label.")
            return pd.DataFrame()

        logger.info(
            f"Processing {len(new_files)} new files "
            f"({len(files) - len(new_files)} already in DB, skipped)"
        )

        dfs = []
        for f in tqdm(new_files, desc="Processing EDGAR files"):
            df = self.process_file(f)
            if not df.empty:
                dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.drop_duplicates(subset="clause_id").reset_index(drop=True)
        logger.info(f"EDGAR processor: {len(combined)} clauses from {len(new_files)} new files")
        return combined

    @staticmethod
    def _get_processed_contract_ids() -> set:
        """Return set of contract_ids already stored in the database from EDGAR."""
        try:
            with SessionLocal() as session:
                rows = session.execute(
                    text("SELECT contract_id FROM contracts WHERE source LIKE 'EDGAR%'")
                ).fetchall()
            return {r[0] for r in rows}
        except Exception:
            return set()

    def _clean(self, text: str) -> str:
        """Strip HTML, SEC boilerplate and normalise whitespace."""
        for pattern in self.BOILERPLATE_PATTERNS:
            text = pattern.sub(" ", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


# 9. AUTO LABELER

class AutoLabeler:
    """Labels EDGAR clauses using the trained production classifier.

    Splits clauses into two buckets:
      - confidence >= EDGAR_AUTO_LABEL_CONFIDENCE → auto-accepted, saved to DB
      - confidence <  EDGAR_AUTO_LABEL_CONFIDENCE → written to review_queue.csv
        for manual correction before adding to training data
    """

    def __init__(self):
        from src.clause_classifier import ClauseClassifierInference
        self.classifier = ClauseClassifierInference()
        self.threshold  = config.EDGAR_AUTO_LABEL_CONFIDENCE

    def label(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run inference and split into accepted / review queues.

        Args:
            df: DataFrame from EDGARProcessor (clause_types column is empty).

        Returns:
            Tuple of (accepted_df, review_df).
        """
        texts       = df["clause_text"].tolist()
        predictions = self.classifier.predict(texts)

        accepted_rows: List[Dict] = []
        review_rows:   List[Dict] = []

        for pred, row in zip(predictions, df.itertuples()):
            clause_types = pred["clause_types"]
            probs        = pred["probabilities"]
            max_conf     = max(probs.values()) if probs else 0.0

            row_dict = {
                "contract_id":  row.contract_id,
                "clause_id":    row.clause_id,
                "clause_text":  row.clause_text,
                "clause_types": clause_types,
                "party_a":      row.party_a,
                "party_b":      row.party_b,
                "source":       "EDGAR_auto",
                "confidence":   round(max_conf, 4),
            }

            if clause_types == ["Other"] or max_conf < self.threshold:
                review_rows.append(row_dict)
            else:
                accepted_rows.append(row_dict)

        accepted_df = pd.DataFrame(accepted_rows) if accepted_rows else pd.DataFrame()
        review_df   = pd.DataFrame(review_rows)   if review_rows   else pd.DataFrame()

        logger.info(
            f"AutoLabeler: {len(accepted_df)} auto-accepted, "
            f"{len(review_df)} sent to review queue"
        )
        return accepted_df, review_df

    def save_review_queue(self, review_df: pd.DataFrame) -> None:
        """Append low-confidence clauses to review_queue.csv.

        Args:
            review_df: DataFrame of clauses needing human review.
        """
        if review_df.empty:
            return
        path = config.EDGAR_REVIEW_PATH
        if path.exists():
            existing  = pd.read_csv(path)
            review_df = pd.concat([existing, review_df], ignore_index=True)
            review_df = review_df.drop_duplicates(subset="clause_id")
        review_df.to_csv(path, index=False)
        logger.info(f"Review queue saved → {path}  ({len(review_df)} total rows)")


# 10. CLUSTER DISCOVERY  →  SIMILARITY ROUTING  →  TAXONOMY EXPANSION


class ClusterDiscovery:
    """Clusters unknown clauses using UMAP + sklearn HDBSCAN + TF-IDF keywords.

    No BERTopic dependency — uses only umap-learn (pure Python) and
    scikit-learn >= 1.3 (which ships sklearn.cluster.HDBSCAN).

    Pipeline:
      Legal-BERT [CLS] embeddings
        → UMAP (768d → 5d, cosine metric)
        → sklearn HDBSCAN (auto cluster count, noise = -1)
        → TF-IDF per cluster for keyword extraction

    Returns a cluster_data dict consumed by SimilarityRouter.
    Also writes edgar_new_clause_types.csv as a manual-review escape hatch.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading Legal-BERT for clustering on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.LEGAL_BERT_MODEL, use_fast=False
        )
        self.model = AutoModel.from_pretrained(config.LEGAL_BERT_MODEL).to(self.device)
        self.model.eval()

    def discover(self, review_df: pd.DataFrame) -> Dict:
        """Cluster unknown clauses and return rich cluster data.

        Args:
            review_df: Low-confidence clauses from AutoLabeler.

        Returns:
            Dict with keys:
              df        - review_df enriched with cluster_id, cluster_keywords,
                          cluster_size columns
              centroids - {cluster_id: np.ndarray(768)} centroid per cluster
              words     - {cluster_id: List[str]} top keywords per cluster
        """
        if review_df.empty:
            logger.info("No unknown clauses to cluster.")
            return {"df": review_df, "centroids": {}, "words": {}}

        texts = review_df["clause_text"].tolist()
        logger.info(f"Generating embeddings for {len(texts)} unknown clauses...")
        embeddings = self._embed(texts)

        # Step 1 — UMAP dimensionality reduction (pure Python, no C++ needed)
        from umap import UMAP
        from sklearn.cluster import HDBSCAN
        from sklearn.feature_extraction.text import TfidfVectorizer

        logger.info("Running UMAP dimensionality reduction...")
        umap_embs = UMAP(
            n_components=config.BERTOPIC_UMAP_COMPONENTS,
            min_dist=0.0,
            metric="cosine",
            random_state=config.RANDOM_SEED,
        ).fit_transform(embeddings)

        # Step 2 — HDBSCAN density clustering (sklearn >= 1.3, no compiler needed)
        logger.info("Running HDBSCAN clustering...")
        labels = HDBSCAN(
            min_cluster_size=config.BERTOPIC_MIN_CLUSTER_SIZE,
            metric="euclidean",
        ).fit_predict(umap_embs)
        labels = np.array(labels)

        noise_count     = int((labels == -1).sum())
        unique_clusters = [int(l) for l in np.unique(labels) if l != -1]
        logger.info(
            f"Clustering: {len(unique_clusters)} clusters, "
            f"{noise_count} noise clauses discarded"
        )

        # Step 3 — TF-IDF keyword extraction per cluster
        tfidf = TfidfVectorizer(
            max_features=10000,
            stop_words="english",
            ngram_range=(1, 2),
        )
        tfidf.fit(texts)
        vocab = np.array(tfidf.get_feature_names_out())

        df = review_df.copy()
        df["cluster_id"]       = labels
        df["cluster_keywords"] = ""
        df["cluster_size"]     = 0

        centroids: Dict[int, np.ndarray] = {}
        words: Dict[int, List[str]]      = {}
        csv_rows: List[Dict]             = []

        for cid in unique_clusters:
            mask         = labels == cid
            member_texts = [texts[i] for i, m in enumerate(mask) if m]
            member_embs  = embeddings[mask]
            centroid     = member_embs.mean(axis=0)
            centroids[cid] = centroid

            # Mean TF-IDF scores across cluster members → top keywords
            cluster_tfidf = tfidf.transform(member_texts).toarray().mean(axis=0)
            top_idx       = cluster_tfidf.argsort()[::-1][:config.BERTOPIC_TOP_N_WORDS]
            topic_words   = vocab[top_idx].tolist()
            words[cid]    = topic_words
            kw_str        = ", ".join(topic_words[:6])

            df.loc[mask, "cluster_keywords"] = kw_str
            df.loc[mask, "cluster_size"]     = int(mask.sum())

            samples = sorted(member_texts, key=len)[:3]
            csv_rows.append({
                "cluster_id":     cid,
                "size":           int(mask.sum()),
                "keywords":       kw_str,
                "sample_1":       samples[0] if len(samples) > 0 else "",
                "sample_2":       samples[1] if len(samples) > 1 else "",
                "sample_3":       samples[2] if len(samples) > 2 else "",
                "suggested_name": "",   # auto-filled by TaxonomyExpander; editable
                "auto_route":     "",   # filled by SimilarityRouter
            })

        # Write full cluster report — manual override always available
        if csv_rows:
            csv_df = pd.DataFrame(csv_rows).sort_values("size", ascending=False)
            csv_df.to_csv(config.EDGAR_NEW_TYPES_PATH, index=False)
            logger.info(
                f"Cluster report → {config.EDGAR_NEW_TYPES_PATH} "
                f"({len(csv_rows)} clusters)"
            )

        return {"df": df, "centroids": centroids, "words": words}

    @torch.no_grad()
    def _embed(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate Legal-BERT [CLS] embeddings for a list of texts."""
        all_emb: List[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch   = texts[i : i + batch_size]
            encoded = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=config.CLAUSE_MAX_TOKENS, return_tensors="pt",
            ).to(self.device)
            out = self.model(**encoded)
            all_emb.append(out.last_hidden_state[:, 0, :].cpu().numpy())
        return np.vstack(all_emb)


class SimilarityRouter:
    """Routes discovered clusters to existing types, review queue, or new types.

    For each cluster centroid, computes cosine similarity against embeddings of
    all existing clause type name strings and applies two thresholds:

        sim >= CLUSTER_ROUTE_HIGH_SIM  →  relabel clauses to matched existing type
        CLUSTER_ROUTE_LOW_SIM <= sim < HIGH  →  ambiguous → taxonomy_review.csv
        sim <  CLUSTER_ROUTE_LOW_SIM   →  new type candidate → TaxonomyExpander
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.LEGAL_BERT_MODEL, use_fast=False
        )
        self.model = AutoModel.from_pretrained(config.LEGAL_BERT_MODEL).to(self.device)
        self.model.eval()

        logger.info("Embedding existing clause type names for similarity routing...")
        self.type_names      = list(config.CUAD_CLAUSE_TYPES)
        self.type_embeddings = self._embed_types()   # (N_types, 768)

    def route(
        self, cluster_data: Dict
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """Route every cluster to the correct downstream destination.

        Args:
            cluster_data: Dict returned by ClusterDiscovery.discover().

        Returns:
            Tuple of:
              relabeled_df   - clauses mapped to a matched existing type
              ambiguous_df   - clauses in ambiguous clusters (for human review)
              new_candidates - {cluster_id: {words, size, df, best_existing,
                               best_sim}} for genuinely new types
        """
        df        = cluster_data["df"]
        centroids = cluster_data["centroids"]
        words     = cluster_data["words"]

        relabeled_rows: List[pd.DataFrame] = []
        ambiguous_rows: List[pd.DataFrame] = []
        new_candidates: Dict[int, Dict]    = {}

        for cid, centroid in centroids.items():
            cluster_df = df[df["cluster_id"] == cid].copy()
            size       = len(cluster_df)
            kw         = ", ".join(words.get(cid, []))

            sims      = cosine_similarity(centroid.reshape(1, -1), self.type_embeddings)[0]
            best_idx  = int(sims.argmax())
            best_sim  = float(sims[best_idx])
            best_type = self.type_names[best_idx]

            if best_sim >= config.CLUSTER_ROUTE_HIGH_SIM:
                # Clearly covered by an existing type — relabel
                cluster_df["clause_types"] = [[best_type]] * size
                cluster_df["confidence"]   = round(best_sim, 4)
                cluster_df["source"]       = "EDGAR_cluster_relabel"
                relabeled_rows.append(cluster_df)
                logger.info(
                    f"  Cluster {cid:3d} ({size:4d} clauses) [{kw}] "
                    f"→ RELABEL '{best_type}' (sim={best_sim:.3f})"
                )

            elif best_sim >= config.CLUSTER_ROUTE_LOW_SIM:
                # Ambiguous — queue for human review
                cluster_df["best_match"] = best_type
                cluster_df["best_sim"]   = round(best_sim, 4)
                ambiguous_rows.append(cluster_df)
                logger.info(
                    f"  Cluster {cid:3d} ({size:4d} clauses) [{kw}] "
                    f"→ REVIEW  best='{best_type}' (sim={best_sim:.3f})"
                )

            else:
                # Genuinely new type not covered by existing taxonomy
                new_candidates[cid] = {
                    "words":        words.get(cid, []),
                    "size":         size,
                    "df":           cluster_df,
                    "best_existing": best_type,
                    "best_sim":     round(best_sim, 4),
                }
                logger.info(
                    f"  Cluster {cid:3d} ({size:4d} clauses) [{kw}] "
                    f"→ NEW TYPE (closest='{best_type}', sim={best_sim:.3f})"
                )

        relabeled_df = (
            pd.concat(relabeled_rows, ignore_index=True)
            if relabeled_rows else pd.DataFrame()
        )
        ambiguous_df = (
            pd.concat(ambiguous_rows, ignore_index=True)
            if ambiguous_rows else pd.DataFrame()
        )

        logger.info(
            f"SimilarityRouter: {len(relabeled_rows)} relabeled, "
            f"{len(ambiguous_rows)} ambiguous, "
            f"{len(new_candidates)} new type candidates"
        )
        return relabeled_df, ambiguous_df, new_candidates

    @torch.no_grad()
    def _embed_types(self) -> np.ndarray:
        """Embed all existing clause type name strings (short, batch of 32)."""
        all_emb: List[np.ndarray] = []
        for i in range(0, len(self.type_names), 32):
            batch   = self.type_names[i : i + 32]
            encoded = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=64, return_tensors="pt",
            ).to(self.device)
            out = self.model(**encoded)
            all_emb.append(out.last_hidden_state[:, 0, :].cpu().numpy())
        return np.vstack(all_emb)


class TaxonomyExpander:
    """Auto-names and persists genuinely new clause types found by SimilarityRouter.

    Decision logic per cluster:
        size >= TAXONOMY_AUTO_ADD_MIN_SIZE  → auto-add name to dynamic_taxonomy.json
                                              label clauses → training data
        TAXONOMY_REVIEW_MIN_SIZE <= size < AUTO_ADD  → write to taxonomy_review.csv
        size < TAXONOMY_REVIEW_MIN_SIZE     → discard (noise)

    dynamic_taxonomy.json is merged into CUAD_CLAUSE_TYPES at config import time,
    so new types are available on the next run without any manual config edits.

    Manual override: edit taxonomy_review.csv, change 'action' column to 'accept',
    then call TaxonomyExpander.apply_manual_review() to process accepted rows.
    """

    def __init__(self):
        self.n_auto_added = 0
        self.n_review     = 0
        self.n_noise      = 0

    def expand(self, new_candidates: Dict[int, Dict]) -> pd.DataFrame:
        """Process new type candidates and return labeled clauses for training.

        Args:
            new_candidates: Dict from SimilarityRouter.route().

        Returns:
            DataFrame of newly labeled clauses ready for DataStore.save_clauses().
        """
        if not new_candidates:
            return pd.DataFrame()

        auto_rows:   List[pd.DataFrame] = []
        review_rows: List[Dict]         = []

        dynamic_types = self._load_dynamic_taxonomy()
        all_known     = set(config.CUAD_CLAUSE_TYPES) | set(dynamic_types)
        newly_added:  List[str] = []

        for cid, candidate in new_candidates.items():
            size  = candidate["size"]
            words = candidate["words"]
            cdf   = candidate["df"].copy()

            if size < config.TAXONOMY_REVIEW_MIN_SIZE:
                self.n_noise += 1
                logger.debug(f"Cluster {cid} ({size} clauses) → noise, discarded")
                continue

            type_name = self._auto_name(words)
            if type_name in all_known:
                type_name = f"{type_name} Variant"

            if size >= config.TAXONOMY_AUTO_ADD_MIN_SIZE:
                cdf["clause_types"] = [[type_name]] * len(cdf)
                cdf["confidence"]   = 0.0
                cdf["source"]       = "EDGAR_new_type"
                auto_rows.append(cdf)
                newly_added.append(type_name)
                all_known.add(type_name)
                self.n_auto_added += 1
                logger.info(
                    f"  Cluster {cid} ({size} clauses) → AUTO-ADD '{type_name}'"
                )

            else:
                review_rows.append({
                    "cluster_id":    cid,
                    "suggested_name": type_name,
                    "size":          size,
                    "keywords":      ", ".join(words),
                    "best_existing": candidate.get("best_existing", ""),
                    "best_sim":      candidate.get("best_sim", 0.0),
                    "sample_1":      cdf["clause_text"].iloc[0] if len(cdf) > 0 else "",
                    "sample_2":      cdf["clause_text"].iloc[1] if len(cdf) > 1 else "",
                    "sample_3":      cdf["clause_text"].iloc[2] if len(cdf) > 2 else "",
                    "action":        "review",   # change to 'accept' to add to taxonomy
                })
                self.n_review += 1
                logger.info(
                    f"  Cluster {cid} ({size} clauses) → REVIEW '{type_name}'"
                )

        # Persist auto-added types to dynamic_taxonomy.json
        if newly_added:
            merged = list(dict.fromkeys(dynamic_types + newly_added))
            self._save_dynamic_taxonomy(merged)
            logger.info(
                f"dynamic_taxonomy.json updated "
                f"(+{len(newly_added)} types, {len(merged)} total)"
            )

        # Persist review candidates to taxonomy_review.csv
        if review_rows:
            rdf  = pd.DataFrame(review_rows)
            path = config.TAXONOMY_REVIEW_PATH
            if path.exists():
                existing = pd.read_csv(path)
                rdf = pd.concat([existing, rdf], ignore_index=True).drop_duplicates(
                    subset="cluster_id"
                )
            rdf.to_csv(path, index=False)
            logger.info(
                f"Taxonomy review queue → {path} "
                f"({len(review_rows)} new clusters)"
            )

        return (
            pd.concat(auto_rows, ignore_index=True) if auto_rows else pd.DataFrame()
        )

    @staticmethod
    def apply_manual_review() -> List[str]:
        """Process taxonomy_review.csv: accept rows where action='accept'.

        Call this after a human has reviewed taxonomy_review.csv and marked
        rows with action='accept'. Those types are appended to dynamic_taxonomy.json
        and become part of the classifier taxonomy on the next retrain.

        Returns:
            List of newly accepted type name strings.
        """
        path = config.TAXONOMY_REVIEW_PATH
        if not path.exists():
            logger.info("No taxonomy_review.csv found.")
            return []

        df       = pd.read_csv(path)
        accepted = df[df["action"].str.lower() == "accept"]
        if accepted.empty:
            logger.info("No rows marked 'accept' in taxonomy_review.csv.")
            return []

        new_names     = accepted["suggested_name"].dropna().tolist()
        dynamic_types = TaxonomyExpander._load_dynamic_taxonomy()
        merged        = list(dict.fromkeys(dynamic_types + new_names))
        TaxonomyExpander._save_dynamic_taxonomy(merged)

        # Mark accepted rows as 'done' so they aren't re-processed
        df.loc[df["action"].str.lower() == "accept", "action"] = "done"
        df.to_csv(path, index=False)

        logger.info(
            f"Manual review applied: {len(new_names)} types accepted → "
            f"dynamic_taxonomy.json"
        )
        return new_names

    @staticmethod
    def _auto_name(words: List[str]) -> str:
        """Title-case top keywords into a clause type name, skipping stopwords."""
        stops = {
            "the", "a", "an", "of", "in", "to", "and", "or", "for",
            "is", "are", "be", "such", "as", "by", "any", "all", "with",
        }
        clean = [w for w in words[:6] if w.lower() not in stops and len(w) > 2]
        return " ".join(w.title() for w in clean[:4]) if clean else "Unknown Clause Type"

    @staticmethod
    def _load_dynamic_taxonomy() -> List[str]:
        import json
        path = config.DYNAMIC_TAXONOMY_PATH
        return json.loads(path.read_text()) if path.exists() else []

    @staticmethod
    def _save_dynamic_taxonomy(types: List[str]) -> None:
        import json
        config.DYNAMIC_TAXONOMY_PATH.write_text(
            json.dumps(types, indent=2), encoding="utf-8"
        )


# 11. DATABASE STORAGE

class DataStore:
    """Persists processed contracts and clauses to SQLite via SQLAlchemy."""

    def __init__(self):
        create_tables()

    def save_contract(self, contract_id: str, source: str, filename: str) -> None:
        """Insert or update a contract record.

        Args:
            contract_id: Unique contract identifier.
            source: Origin of the contract ('CUAD', 'upload', etc.).
            filename: Original filename or identifier.
        """
        with SessionLocal() as session:
            existing = session.get(Contract, contract_id)
            if existing is None:
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
            Number of clauses inserted.
        """
        inserted = 0
        with SessionLocal() as session:
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Saving clauses"):
                existing = session.get(Clause, row["clause_id"])
                if existing is not None:
                    continue

                clause_types = row.get("clause_types", [])
                if isinstance(clause_types, list):
                    clause_types_str = "|".join(clause_types)
                else:
                    clause_types_str = str(clause_types)

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


# 12. TRAIN / VAL / TEST SPLITTER

class DataSplitter:
    """Produces stratified 80/10/10 splits by primary clause type.

    Stratification ensures each split contains proportional representation
    of all 41 clause types, preventing train/test distribution mismatch.
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
    # EDGAR auto-labels may account for at most this fraction of training rows.
    # Capping prevents noisy machine-generated labels from dominating clean data.
    EDGAR_TRAIN_CAP = 0.40

    # Maximum number of pipe-separated labels kept per clause.
    # EDGAR auto-labeling sometimes assigns 20-40 labels to one clause —
    # these are spurious combinations that corrupt the multi-hot label space.
    MAX_LABELS_PER_CLAUSE = 3

    def _clean_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cap multi-label combinations and strip unknown/empty labels.

        EDGAR auto-labeling can assign up to 40 labels per clause, which
        creates thousands of unique pipe-joined strings and corrupts the
        multi-hot label space. We keep only the first MAX_LABELS_PER_CLAUSE
        labels (sorted so the most common appears first after pipe-split).

        Args:
            df: DataFrame with a 'clause_type' column.

        Returns:
            DataFrame with cleaned 'clause_type' column.
        """
        valid = set(config.CUAD_CLAUSE_TYPES)

        def _cap(val):
            if not isinstance(val, str) or not val.strip():
                return ""
            parts = [p.strip() for p in val.split("|") if p.strip() in valid]
            return "|".join(parts[: self.MAX_LABELS_PER_CLAUSE])

        df = df.copy()
        df["clause_type"] = df["clause_type"].apply(_cap)
        # Drop rows where all labels were stripped (unknown types only)
        df = df[df["clause_type"] != ""].reset_index(drop=True)

        # Strip inline <omitted> redaction tags (MAUD dataset artifact).
        # Replace with a single space so surrounding words don't merge.
        # Rows that have too little real content left are caught by the
        # min-word-count filter below.
        omit_mask = df["clause_text"].str.contains(r"<omitted>", case=False, na=False)
        if omit_mask.any():
            df.loc[omit_mask, "clause_text"] = (
                df.loc[omit_mask, "clause_text"]
                .str.replace(r"<omitted>", " ", flags=re.IGNORECASE, regex=True)
                .str.replace(r"\s{2,}", " ", regex=True)
                .str.strip()
            )
            logger.info(f"Stripped <omitted> tags from {omit_mask.sum():,} MAUD clauses")

        # Drop clauses below minimum word count (single words, numbers, section headers)
        before_min = len(df)
        df = df[df["clause_text"].str.split().str.len() >= config.CLAUSE_MIN_TOKENS].reset_index(drop=True)
        removed_min = before_min - len(df)
        if removed_min:
            logger.info(
                f"Removed {removed_min:,} clauses below {config.CLAUSE_MIN_TOKENS} words "
                f"(noise: single words, numbers, section headers)"
            )

        # Drop clauses that exceed the character length ceiling.
        # The val/test human data tops out at ~11,500 chars (~2885 tokens).
        # Rows beyond CLAUSE_MAX_CHARS are EDGAR pipeline artifacts where
        # entire document sections were ingested as a single "clause".
        before = len(df)
        df["_char_len"] = df["clause_text"].str.len()
        df = df[df["_char_len"] <= config.CLAUSE_MAX_CHARS].drop(columns=["_char_len"])
        df = df.reset_index(drop=True)
        removed = before - len(df)
        if removed:
            logger.info(
                f"Removed {removed:,} clauses exceeding {config.CLAUSE_MAX_CHARS:,} chars "
                f"(EDGAR document-level artifacts)"
            )
        return df

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate stratified train/val/test splits with clean eval sets.

        Key design decisions:
          - Val and test sets are built exclusively from human-annotated data
            (CUAD, LEDGAR, MAUD). This ensures F1 metrics measure real
            generalisation, not accuracy against noisy auto-generated labels.
          - EDGAR auto-labels are restricted to training only and capped at
            EDGAR_TRAIN_CAP (40%) of total training rows.
          - Multi-label combinations are capped at MAX_LABELS_PER_CLAUSE (3)
            to remove pathological EDGAR assignments of 20-40 labels per clause.

        Args:
            df: Full clause DataFrame loaded from SQLite.

        Returns:
            Tuple of (train_df, val_df, test_df).
        """
        df = self._clean_labels(df)
        df["primary_type"] = df["clause_type"].apply(
            lambda x: x.split("|")[0] if x else "Unknown"
        )

        # ── Split clean vs EDGAR ───────────────────────────────────────────
        is_edgar = df["source"].str.startswith("EDGAR")
        clean_df = df[~is_edgar].copy()
        edgar_df = df[is_edgar].copy()

        logger.info(
            f"Split input — clean: {len(clean_df):,}  EDGAR: {len(edgar_df):,}"
        )

        # ── Val / test from clean data only ────────────────────────────────
        counts = clean_df["primary_type"].value_counts()
        valid_types = counts[counts >= 10].index
        clean_ok   = clean_df[clean_df["primary_type"].isin(valid_types)].copy()
        clean_rare = clean_df[~clean_df["primary_type"].isin(valid_types)].copy()

        if clean_ok.empty:
            raise RuntimeError("Not enough clean-source rows to build val/test sets.")

        relative_val = config.VAL_RATIO / (config.VAL_RATIO + config.TEST_RATIO)

        # First split: carve out 20% for val+test
        train_clean, temp_df = train_test_split(
            clean_ok,
            test_size=(1 - config.TRAIN_RATIO),
            stratify=clean_ok["primary_type"],
            random_state=config.RANDOM_SEED,
        )

        # Second split: split temp into val and test
        temp_counts      = temp_df["primary_type"].value_counts()
        temp_valid_types = temp_counts[temp_counts >= 2].index
        temp_ok          = temp_df[temp_df["primary_type"].isin(temp_valid_types)].copy()
        temp_rare        = temp_df[~temp_df["primary_type"].isin(temp_valid_types)].copy()

        if temp_ok.empty:
            val_df  = pd.DataFrame(columns=df.columns)
            test_df = pd.DataFrame(columns=df.columns)
        else:
            val_df, test_df = train_test_split(
                temp_ok,
                test_size=(1 - relative_val),
                stratify=temp_ok["primary_type"],
                random_state=config.RANDOM_SEED,
            )

        # ── Training set: clean train portion + capped EDGAR ──────────────
        clean_train = pd.concat(
            [train_clean, clean_rare, temp_rare], ignore_index=True
        )

        # Cap EDGAR at EDGAR_TRAIN_CAP fraction of total training rows
        max_edgar = int(len(clean_train) * self.EDGAR_TRAIN_CAP / (1 - self.EDGAR_TRAIN_CAP))
        if len(edgar_df) > max_edgar:
            edgar_sampled = edgar_df.sample(n=max_edgar, random_state=config.RANDOM_SEED)
            logger.info(
                f"EDGAR capped: {len(edgar_df):,} → {max_edgar:,} rows "
                f"({self.EDGAR_TRAIN_CAP:.0%} of training set)"
            )
        else:
            edgar_sampled = edgar_df

        train_df = pd.concat([clean_train, edgar_sampled], ignore_index=True).sample(
            frac=1, random_state=config.RANDOM_SEED
        ).reset_index(drop=True)

        # Undersample dominant classes in the training set only.
        # Val/test are never touched — evaluation stays unbiased.
        cap = config.UNDERSAMPLE_MAX_PER_CLASS
        if cap is not None:
            before_us = len(train_df)
            train_df = (
                train_df
                .groupby("primary_type", group_keys=False)
                .apply(
                    lambda g: g.sample(
                        n=min(len(g), cap),
                        random_state=config.RANDOM_SEED,
                    )
                )
                .sample(frac=1, random_state=config.RANDOM_SEED)
                .reset_index(drop=True)
            )
            logger.info(
                f"Undersampling (cap={cap}/class): {before_us:,} → {len(train_df):,} training rows"
            )

        logger.info(
            f"Split sizes — Train: {len(train_df):,}  "
            f"(clean: {len(clean_train):,}, EDGAR: {len(edgar_sampled):,})  |  "
            f"Val: {len(val_df):,}  |  Test: {len(test_df):,}  "
            f"[val/test are 100% human-annotated]"
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
        val_df.to_csv(config.PROCESSED_DIR / "val.csv", index=False)
        test_df.to_csv(config.PROCESSED_DIR / "test.csv", index=False)
        logger.info(f"Splits saved to {config.PROCESSED_DIR}")


# 13. UNIFIED PIPELINE RUNNER

class ContractIntelligencePipeline:
    """End-to-end pipeline orchestrator for data preparation.

    Combines PDF ingestion, clause segmentation, CUAD / LEDGAR / MAUD
    processing, database storage, and data splitting into a single callable
    interface.
    """

    def __init__(self):
        self.pdf_ingester     = PDFIngester()
        self.segmenter        = ClauseSegmenter()
        self.cuad_processor   = CUADProcessor()
        self.ledgar_processor = LEDGARProcessor()
        self.maud_processor   = MAUDProcessor()
        self.edgar_downloader = EDGARDownloader()
        self.edgar_processor  = EDGARProcessor()
        self.datastore        = DataStore()
        self.splitter         = DataSplitter()

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

    def run_ledgar(self) -> pd.DataFrame:
        """Download and process the LEDGAR dataset end-to-end.

        Returns:
            Processed DataFrame saved to database and returned.
        """
        logger.info("=== Starting LEDGAR pipeline ===")
        df = self.ledgar_processor.load_and_process()

        for contract_id in df["contract_id"].unique():
            self.datastore.save_contract(
                contract_id=contract_id,
                source="LEDGAR",
                filename=contract_id,
            )

        self.datastore.save_clauses(df)
        logger.info("=== LEDGAR pipeline complete ===")
        return df

    def run_maud(self) -> pd.DataFrame:
        """Download and process the MAUD dataset end-to-end.

        Returns:
            Processed DataFrame saved to database and returned.
        """
        logger.info("=== Starting MAUD pipeline ===")
        df = self.maud_processor.load_and_process()

        for contract_id in df["contract_id"].unique():
            self.datastore.save_contract(
                contract_id=contract_id,
                source="MAUD",
                filename=contract_id,
            )

        self.datastore.save_clauses(df)
        logger.info("=== MAUD pipeline complete ===")
        return df

    def run_all(self) -> Dict[str, pd.DataFrame]:
        """Run CUAD + LEDGAR + MAUD pipelines then generate splits.

        Returns:
            Dict with keys 'cuad', 'ledgar', 'maud', 'train', 'val', 'test'.
        """
        result: Dict[str, pd.DataFrame] = {}
        result["cuad"]   = self.run_cuad()
        result["ledgar"] = self.run_ledgar()
        result["maud"]   = self.run_maud()
        train_df, val_df, test_df = self.run_split()
        result["train"], result["val"], result["test"] = train_df, val_df, test_df
        return result

    def run_edgar(self, limit: int = config.EDGAR_DOWNLOAD_LIMIT) -> Dict[str, pd.DataFrame]:
        """Full automated EDGAR pipeline with manual review escape hatches.

        Flow
        ----
        1. Download EX-10 exhibits from SEC EDGAR
        2. Segment into clauses (EDGARProcessor)
        3. Auto-label known types (AutoLabeler, confidence >= threshold)
           → accepted clauses → DataStore + training data
           → unknown clauses → ClusterDiscovery
        4. Cluster unknowns with BERTopic (UMAP + HDBSCAN)
        5. Route clusters by similarity to existing type name embeddings
           (SimilarityRouter):
           → HIGH sim → relabel to matched type → DataStore
           → MID  sim → taxonomy_review.csv (human reviews 'action' column)
           → LOW  sim → new type candidates → TaxonomyExpander
        6. Expand taxonomy (TaxonomyExpander):
           → large clusters  → auto-name + dynamic_taxonomy.json + DataStore
           → medium clusters → taxonomy_review.csv
           → small clusters  → discarded (noise)

        Manual review files (all optional — pipeline runs fully without them):
          data/processed/review_queue.csv          low-confidence auto-labels
          data/processed/edgar_new_clause_types.csv all BERTopic clusters
          data/processed/taxonomy_review.csv        ambiguous + medium new types
          data/processed/dynamic_taxonomy.json      auto-added new type names

        After running:
          1. Optionally review taxonomy_review.csv, set action='accept' on good rows
          2. Run TaxonomyExpander.apply_manual_review() to process accepted rows
          3. Run --mode split to rebuild train/val/test
          4. Retrain classifier in Colab

        Args:
            limit: Max EX-10 contracts to download.

        Returns:
            Dict with pipeline stage DataFrames.
        """
        logger.info("=== Starting EDGAR pipeline ===")

        # Step 1 — Download
        self.edgar_downloader.download(limit=limit)

        # Step 2 — Segment
        df_raw = self.edgar_processor.process_directory()
        if df_raw.empty:
            logger.warning("No EDGAR clauses extracted. Check data/edgar/raw/")
            return {}

        # Step 3 — Auto-label known types
        labeler = AutoLabeler()
        accepted_df, review_df = labeler.label(df_raw)
        labeler.save_review_queue(review_df)

        if not accepted_df.empty:
            for cid in accepted_df["contract_id"].unique():
                self.datastore.save_contract(cid, "EDGAR_auto", cid)
            self.datastore.save_clauses(accepted_df)

        # Steps 4-6 — Cluster → route → expand (only if there are unknown clauses)
        relabeled_df  = pd.DataFrame()
        ambiguous_df  = pd.DataFrame()
        labeled_new_df = pd.DataFrame()

        if not review_df.empty:
            # Step 4 — BERTopic clustering
            discoverer   = ClusterDiscovery()
            cluster_data = discoverer.discover(review_df)
            del discoverer   # free GPU memory before loading SimilarityRouter

            if cluster_data["centroids"]:
                # Step 5 — Similarity routing
                router = SimilarityRouter()
                relabeled_df, ambiguous_df, new_candidates = router.route(cluster_data)
                del router   # free GPU memory before TaxonomyExpander

                # Save relabeled clauses
                if not relabeled_df.empty:
                    for cid in relabeled_df["contract_id"].unique():
                        self.datastore.save_contract(cid, "EDGAR_cluster_relabel", cid)
                    self.datastore.save_clauses(relabeled_df)

                # Save ambiguous clusters to review CSV
                if not ambiguous_df.empty:
                    self._save_ambiguous_clusters(ambiguous_df)

                # Step 6 — Taxonomy expansion
                expander       = TaxonomyExpander()
                labeled_new_df = expander.expand(new_candidates)

                if not labeled_new_df.empty:
                    for cid in labeled_new_df["contract_id"].unique():
                        self.datastore.save_contract(cid, "EDGAR_new_type", cid)
                    self.datastore.save_clauses(labeled_new_df)

                logger.info("=== EDGAR pipeline complete ===")
                logger.info(f"  Raw clauses        : {len(df_raw)}")
                logger.info(f"  Auto-labeled       : {len(accepted_df)}")
                logger.info(f"  Cluster-relabeled  : {len(relabeled_df)}")
                logger.info(f"  New types auto-added : {expander.n_auto_added}")
                logger.info(f"  For manual review  : {expander.n_review} clusters → {config.TAXONOMY_REVIEW_PATH}")
                logger.info(f"  Noise discarded    : {expander.n_noise} clusters")
            else:
                logger.info("=== EDGAR pipeline complete (no clusters formed) ===")
        else:
            logger.info("=== EDGAR pipeline complete (all clauses auto-labeled) ===")

        return {
            "raw_clauses":  df_raw,
            "accepted":     accepted_df,
            "relabeled":    relabeled_df,
            "labeled_new":  labeled_new_df,
            "ambiguous":    ambiguous_df,
            "review":       review_df,
        }

    def _save_ambiguous_clusters(self, ambiguous_df: pd.DataFrame) -> None:
        """Summarise ambiguous clusters to CSV for human review."""
        summary = (
            ambiguous_df.groupby(
                ["cluster_id", "cluster_keywords", "best_match", "best_sim"]
            )
            .agg(
                size=("clause_id", "count"),
                sample_1=("clause_text", "first"),
            )
            .reset_index()
        )
        summary["suggested_name"] = ""
        summary["action"] = "review"   # change to 'accept' to use best_match label

        path = config.TAXONOMY_REVIEW_PATH
        if path.exists():
            existing = pd.read_csv(path)
            summary  = pd.concat([existing, summary], ignore_index=True).drop_duplicates(
                subset="cluster_id"
            )
        summary.to_csv(path, index=False)
        logger.info(
            f"Ambiguous cluster review → {path} ({len(summary)} clusters)"
        )

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
        clauses = self.segmenter.segment(raw_text)

        records = []
        for i, clause_text in enumerate(clauses):
            clause_id = hashlib.md5(f"{contract_id}{i}{clause_text}".encode()).hexdigest()[:16]
            records.append(
                {
                    "contract_id": contract_id,
                    "clause_id": clause_id,
                    "clause_text": clause_text,
                    "clause_types": [],
                    "party_a": "",
                    "party_b": "",
                    "source": "upload",
                }
            )

        df = pd.DataFrame(records)
        self.datastore.save_contract(
            contract_id=contract_id,
            source="upload",
            filename=pdf_path.name,
        )
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
                    "contract_id": contract_id,
                    "clause_id": clause_id,
                    "clause_text": clause_text,
                    "clause_types": [],
                    "party_a": "",
                    "party_b": "",
                    "source": "text_input",
                }
            )

        df = pd.DataFrame(records)
        self.datastore.save_contract(
            contract_id=contract_id,
            source="text_input",
            filename=contract_id,
        )
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


# CLI ENTRY POINT

def main():
    """Command-line interface for the data pipeline."""
    parser = argparse.ArgumentParser(
        description="Contract Intelligence Data Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples:
              python src/data_pipeline.py --mode all                             # recommended first run
              python src/data_pipeline.py --mode cuad                            # CUAD only
              python src/data_pipeline.py --mode ledgar                          # LEDGAR only
              python src/data_pipeline.py --mode maud                            # MAUD only
              python src/data_pipeline.py --mode pdf --file contracts/my.pdf
              python src/data_pipeline.py --mode split
            """
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["cuad", "ledgar", "maud", "all", "edgar", "pdf", "split"],
        required=True,
        help="Pipeline mode to run",
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="Path to PDF file (required for --mode pdf)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=config.EDGAR_DOWNLOAD_LIMIT,
        help="Max contracts to download (--mode edgar only)",
    )
    args = parser.parse_args()

    pipeline = ContractIntelligencePipeline()

    if args.mode == "cuad":
        pipeline.run_cuad()

    elif args.mode == "ledgar":
        df = pipeline.run_ledgar()
        print(f"\nLEDGAR pipeline complete: {len(df)} clauses")

    elif args.mode == "maud":
        df = pipeline.run_maud()
        print(f"\nMAUD pipeline complete: {len(df)} clauses")

    elif args.mode == "all":
        result = pipeline.run_all()
        print(f"\nAll datasets processed:")
        print(f"  CUAD   : {len(result['cuad'])} clauses")
        print(f"  LEDGAR : {len(result['ledgar'])} clauses")
        print(f"  MAUD   : {len(result['maud'])} clauses")
        print(f"  Train  : {len(result['train'])} clauses")
        print(f"  Val    : {len(result['val'])} clauses")
        print(f"  Test   : {len(result['test'])} clauses")

    elif args.mode == "edgar":
        result = pipeline.run_edgar(limit=args.limit)
        if result:
            print(f"\nEDGAR pipeline complete:")
            print(f"  Raw clauses      : {len(result['raw_clauses'])}")
            print(f"  Auto-accepted    : {len(result['accepted'])}")
            print(f"  Cluster-relabeled: {len(result['relabeled'])}")
            print(f"  New type labeled : {len(result['labeled_new'])}")
            print(f"  For review       : {len(result['review'])}  → {config.TAXONOMY_REVIEW_PATH}")
            print(f"  New types path   : {config.EDGAR_NEW_TYPES_PATH}")

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
