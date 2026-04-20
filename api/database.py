"""
database.py — SQLAlchemy ORM models and session management.

Tables:
  contracts — one row per ingested contract
  clauses   — one row per extracted clause (FK → contracts)
  analysis_results — one row per analysis run linking all scores

All timestamps are stored as ISO-8601 UTC strings for SQLite compatibility.
"""

import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    event,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# Engine & Session

engine = create_engine(
    config.DB_URL,
    connect_args={"check_same_thread": False},  # SQLite requires this for FastAPI
    echo=False,
)

# Enable WAL mode for better concurrent read performance with SQLite
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Enable WAL mode and foreign key enforcement on every new SQLite connection."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# ORM Base

class Base(DeclarativeBase):
    """Declarative base for all ORM models."""
    pass


# ORM Models

class Contract(Base):
    """Represents a single ingested contract document.

    Attributes:
        contract_id: MD5-based unique identifier derived from filename.
        filename: Original filename or identifier string.
        source: Origin of contract ('CUAD', 'upload', 'text_input').
        page_count: Number of pages (for PDFs).
        created_at: UTC timestamp of ingestion.
        clauses: Relationship to associated Clause rows.
    """

    __tablename__ = "contracts"

    contract_id = Column(String(64), primary_key=True, index=True)
    filename    = Column(String(512), nullable=False)
    source      = Column(String(64), nullable=False, default="upload")
    page_count  = Column(Integer, nullable=True)
    created_at  = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    clauses     = relationship("Clause", back_populates="contract", cascade="all, delete-orphan")
    results     = relationship("AnalysisResult", back_populates="contract", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Contract id={self.contract_id} file={self.filename}>"


class Clause(Base):
    """Represents a single clause extracted from a contract.

    Attributes:
        clause_id: MD5-based unique identifier.
        contract_id: FK to contracts table.
        clause_text: Raw text of the clause.
        clause_type: Pipe-separated list of CUAD clause type labels.
        party_a: Name of Party A as detected in clause.
        party_b: Name of Party B as detected in clause.
        source: Origin ('CUAD', 'upload', 'text_input').
        anomaly_score: Combined anomaly risk score (0–100).
        is_anomalous: True if anomaly_score > ANOMALY_FLAG_THRESHOLD.
        power_imbalance_score: Bilateral imbalance score (-100 to +100).
        party_a_leverage: Party A leverage score (0–100).
        party_b_leverage: Party B leverage score (0–100).
        sentiment_score: Sentiment feature value.
        modal_score: Modal verb feature value.
        obligation_score: Obligation assignment feature value.
        assertiveness_score: Assertiveness feature value.
        shap_plot_path: Filesystem path to the SHAP PNG for this clause.
        created_at: UTC timestamp of processing.
    """

    __tablename__ = "clauses"

    clause_id             = Column(String(64), primary_key=True, index=True)
    contract_id           = Column(String(64), ForeignKey("contracts.contract_id"), nullable=False, index=True)
    clause_text           = Column(Text, nullable=False)
    clause_type           = Column(String(512), nullable=False, default="")
    party_a               = Column(String(256), nullable=True, default="")
    party_b               = Column(String(256), nullable=True, default="")
    source                = Column(String(64), nullable=True, default="")

    # Anomaly detection fields
    anomaly_score         = Column(Float, nullable=True)
    is_anomalous          = Column(Boolean, nullable=True, default=False)

    # Power imbalance fields
    power_imbalance_score = Column(Float, nullable=True)
    party_a_leverage      = Column(Float, nullable=True)
    party_b_leverage      = Column(Float, nullable=True)

    # Feature-level scores
    sentiment_score       = Column(Float, nullable=True)
    modal_score           = Column(Float, nullable=True)
    obligation_score      = Column(Float, nullable=True)
    assertiveness_score   = Column(Float, nullable=True)

    # Explainability
    shap_plot_path        = Column(String(512), nullable=True)

    created_at            = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    contract = relationship("Contract", back_populates="clauses")

    def __repr__(self) -> str:
        return f"<Clause id={self.clause_id} type={self.clause_type[:30]}>"


class AnalysisResult(Base):
    """Stores aggregate analysis results at the contract level.

    Attributes:
        result_id: Auto-incremented primary key.
        contract_id: FK to contracts table.
        overall_imbalance_index: Contract-level power imbalance (-100 to +100).
        total_clauses: Total clause count.
        anomalous_clauses: Count of flagged anomalous clauses.
        dominant_clause_type: Most frequent clause type in the contract.
        analysis_metadata: JSON string with additional metadata.
        created_at: UTC timestamp.
    """

    __tablename__ = "analysis_results"

    result_id               = Column(Integer, primary_key=True, autoincrement=True)
    contract_id             = Column(String(64), ForeignKey("contracts.contract_id"), nullable=False, index=True)
    overall_imbalance_index = Column(Float, nullable=True)
    total_clauses           = Column(Integer, nullable=True)
    anomalous_clauses       = Column(Integer, nullable=True)
    dominant_clause_type    = Column(String(256), nullable=True)
    analysis_metadata       = Column(Text, nullable=True)  # JSON string
    created_at              = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    contract = relationship("Contract", back_populates="results")

    def __repr__(self) -> str:
        return f"<AnalysisResult contract={self.contract_id} imbalance={self.overall_imbalance_index}>"


# Utility functions

def create_tables() -> None:
    """Create all database tables if they do not already exist.

    Safe to call multiple times — uses CREATE TABLE IF NOT EXISTS semantics.
    """
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that provides a database session per request.

    Yields:
        SQLAlchemy Session instance. Automatically closed after use.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def managed_session() -> Generator[Session, None, None]:
    """Context manager for database sessions outside of FastAPI request scope.

    Usage:
        with managed_session() as session:
            session.add(some_object)
            session.commit()
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
