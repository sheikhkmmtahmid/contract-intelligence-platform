"""
schemas.py — Pydantic v2 request and response models for the Contract Intelligence API.

Every endpoint request and response is typed via these models. This enforces
schema validation, enables automatic OpenAPI docs, and ensures structured
error responses throughout the codebase.
"""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# SHARED / PRIMITIVE SCHEMAS

class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str = Field(..., description="Always 'ok' when the service is healthy")
    version: str = Field(..., description="API version string")
    timestamp: datetime = Field(..., description="UTC timestamp of the health check")

    model_config = {"json_schema_extra": {"example": {
        "status": "ok",
        "version": "1.0.0",
        "timestamp": "2024-01-15T12:00:00Z",
    }}}


class ErrorResponse(BaseModel):
    """Structured error envelope returned by all error paths."""

    error: str = Field(..., description="Short error identifier")
    detail: str = Field(..., description="Human-readable error description")
    contract_id: Optional[str] = Field(None, description="Contract ID if applicable")

    model_config = {"json_schema_extra": {"example": {
        "error": "contract_not_found",
        "detail": "No contract found with ID abc123",
        "contract_id": "abc123",
    }}}


# CLAUSE SCHEMAS

class ClauseSHAPDetail(BaseModel):
    """SHAP attribution detail for a single clause."""

    classifier_shap_path: Optional[str] = Field(None, description="Path to classifier SHAP PNG")
    power_shap_path:      Optional[str] = Field(None, description="Path to power SHAP PNG")
    classifier_shap_values: List[Dict[str, float]] = Field(
        default_factory=list,
        description="List of {word, shap_value} pairs for top tokens",
    )
    power_shap_values: Dict[str, float] = Field(
        default_factory=dict,
        description="Feature name → SHAP value for power imbalance features",
    )


class ClauseScore(BaseModel):
    """Full score and metadata for a single clause."""

    clause_id:             str = Field(..., description="Unique clause identifier")
    contract_id:           str = Field(..., description="Parent contract identifier")
    clause_text:           str = Field(..., description="Raw clause text")
    clause_types:          List[str] = Field(default_factory=list, description="Predicted CUAD clause types")
    party_a:               Optional[str] = Field(None, description="Party A name if detected")
    party_b:               Optional[str] = Field(None, description="Party B name if detected")
    anomaly_score:         Optional[float] = Field(None, ge=0, le=100, description="Combined anomaly risk score (0–100)")
    is_anomalous:          Optional[bool]  = Field(None, description="True if anomaly score > threshold")
    power_imbalance_score: Optional[float] = Field(None, ge=-100, le=100, description="Bilateral imbalance (-100 to +100)")
    party_a_leverage:      Optional[float] = Field(None, ge=0, le=100, description="Party A leverage score (0–100)")
    party_b_leverage:      Optional[float] = Field(None, ge=0, le=100, description="Party B leverage score (0–100)")
    sentiment_score:       Optional[float] = Field(None, ge=0, le=1)
    modal_score:           Optional[float] = Field(None, ge=0, le=1)
    obligation_score:      Optional[float] = Field(None, ge=0, le=1)
    assertiveness_score:   Optional[float] = Field(None, ge=0, le=1)
    imbalance_label:       Optional[str]   = Field(None, description="'HIGH', 'MEDIUM', or 'BALANCED'")
    shap_plot_path:        Optional[str]   = Field(None, description="Path to SHAP PNG file")

    @field_validator("clause_types", mode="before")
    @classmethod
    def parse_pipe_separated_types(cls, v):
        """Convert pipe-separated string to list if needed."""
        if isinstance(v, str):
            return [t.strip() for t in v.split("|") if t.strip()] if v else []
        return v or []


class ClauseListResponse(BaseModel):
    """Response for GET /clauses/{contract_id}."""

    contract_id:   str = Field(..., description="Contract identifier")
    total_clauses: int = Field(..., description="Total number of clauses")
    clauses:       List[ClauseScore] = Field(..., description="List of clause scores")


class AnomaliesResponse(BaseModel):
    """Response for GET /anomalies/{contract_id}."""

    contract_id:      str = Field(..., description="Contract identifier")
    total_clauses:    int = Field(..., description="Total clauses in contract")
    anomalous_count:  int = Field(..., description="Number of flagged anomalous clauses")
    anomaly_threshold: float = Field(..., description="Anomaly flag threshold used")
    anomalous_clauses: List[ClauseScore] = Field(..., description="Flagged anomalous clauses only")


# POWER IMBALANCE SCHEMAS

class ImbalanceByType(BaseModel):
    """Power imbalance score for a single clause type."""

    clause_type:     str   = Field(..., description="Clause type name")
    imbalance_score: float = Field(..., ge=-100, le=100, description="Mean imbalance for this clause type")
    clause_count:    int   = Field(..., description="Number of clauses of this type")


class PowerImbalanceResponse(BaseModel):
    """Response for GET /imbalance/{contract_id}."""

    contract_id:              str   = Field(..., description="Contract identifier")
    overall_imbalance_index:  float = Field(..., ge=-100, le=100, description="Contract-level imbalance")
    dominant_party:           str   = Field(..., description="Party holding more leverage")
    total_clauses:            int   = Field(..., description="Total clauses analysed")
    imbalance_by_type:        List[ImbalanceByType] = Field(..., description="Per clause-type imbalance")
    party_a_name:             str   = Field(default="Party A")
    party_b_name:             str   = Field(default="Party B")


# ANALYSIS SCHEMAS

class AnalyseRequest(BaseModel):
    """Request body for POST /analyse when submitting raw text."""

    text:      Optional[str] = Field(None, description="Raw contract text (alternative to PDF upload)")
    party_a:   str           = Field(default="Party A", description="Display name for Party A")
    party_b:   str           = Field(default="Party B", description="Display name for Party B")
    run_shap:  bool          = Field(default=False, description="Whether to generate SHAP explanations (slow)")

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v):
        if v is not None and not v.strip():
            raise ValueError("text field cannot be blank")
        return v


class AnalysisSummary(BaseModel):
    """Summary section of the full analysis response."""

    contract_id:              str   = Field(..., description="Unique identifier for this analysis")
    total_clauses:            int   = Field(..., description="Total clauses extracted")
    anomalous_clauses:        int   = Field(..., description="Clauses flagged as anomalous")
    overall_imbalance_index:  float = Field(..., description="Contract-level power imbalance (-100 to +100)")
    dominant_party:           str   = Field(..., description="Party holding more leverage")
    dominant_clause_type:     Optional[str] = Field(None, description="Most frequent clause type")


class AnalyseResponse(BaseModel):
    """Full response body for POST /analyse."""

    contract_id:  str             = Field(..., description="Contract identifier")
    summary:      AnalysisSummary = Field(..., description="High-level analysis summary")
    clauses:      List[ClauseScore]     = Field(..., description="Per-clause scores")
    imbalance:    PowerImbalanceResponse = Field(..., description="Power imbalance breakdown")
    report_url:   str             = Field(..., description="URL to download full PDF report")


# REPORT SCHEMAS

class ReportResponse(BaseModel):
    """Response for GET /report/{contract_id} (metadata only; PDF served separately)."""

    contract_id:   str      = Field(..., description="Contract identifier")
    report_path:   str      = Field(..., description="Path to generated PDF report")
    generated_at:  datetime = Field(..., description="UTC timestamp of report generation")
    clauses_count: int      = Field(..., description="Number of clauses in report")
