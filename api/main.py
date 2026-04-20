"""
main.py — FastAPI application entry point for the Contract Intelligence Platform.

Endpoints:
    GET  /health                    — Service health check
    POST /analyse                   — Analyse a contract (PDF upload or raw text)
    GET  /report/{contract_id}      — Download full PDF report
    GET  /clauses/{contract_id}     — All clauses with scores
    GET  /anomalies/{contract_id}   — Anomalous clauses only
    GET  /imbalance/{contract_id}   — Power imbalance breakdown

All errors are returned as structured JSON (ErrorResponse schema).
All model checkpoints are loaded at startup and reused across requests.
"""

import hashlib
import json
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from loguru import logger
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from api.database import (
    AnalysisResult,
    Clause,
    Contract,
    SessionLocal,
    create_tables,
    get_db,
)
from api.schemas import (
    AnalyseResponse,
    AnalysisSummary,
    AnomaliesResponse,
    ClauseListResponse,
    ClauseScore,
    ErrorResponse,
    HealthResponse,
    ImbalanceByType,
    PowerImbalanceResponse,
    ReportResponse,
)

logger.remove()
logger.add(config.LOGS_DIR / "api.log", rotation="20 MB", level="DEBUG")
logger.add(sys.stderr, level="INFO")

# Application factory

def create_app() -> FastAPI:
    """Construct and configure the FastAPI application instance.

    Returns:
        Configured FastAPI app with all routes and middleware registered.
    """
    app = FastAPI(
        title=config.API_TITLE,
        version=config.API_VERSION,
        description=(
            "Production-grade Contract Intelligence & Power Imbalance Platform. "
            "Analyses commercial contracts for clause classification, anomaly detection, "
            "bilateral power imbalance scoring, and SHAP explainability."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Static files and templates
    app.mount("/static", StaticFiles(directory=str(config.STATIC_DIR)), name="static")
    app.mount(
        "/shap",
        StaticFiles(directory=str(config.SHAP_OUTPUT_DIR)),
        name="shap",
    )
    app.state.templates = Jinja2Templates(directory=str(config.TEMPLATES_DIR))

    return app


app = create_app()

# Model registry — loaded once at startup

class ModelRegistry:
    """Holds singleton references to all loaded models.

    Models are loaded lazily on first request to avoid startup failure
    if checkpoints haven't been trained yet.
    """

    def __init__(self):
        self._classifier   = None
        self._anomaly      = None
        self._power_scorer = None
        self._explainability = None
        self._pipeline     = None
        self._report_gen   = None

    @property
    def classifier(self):
        if self._classifier is None:
            from src.clause_classifier import ClauseClassifierInference
            self._classifier = ClauseClassifierInference()
        return self._classifier

    @property
    def anomaly(self):
        if self._anomaly is None:
            from src.anomaly_detector import AnomalyDetectorInference
            self._anomaly = AnomalyDetectorInference()
        return self._anomaly

    @property
    def power_scorer(self):
        if self._power_scorer is None:
            from src.power_scorer import PowerImbalanceScorer
            self._power_scorer = PowerImbalanceScorer()
        return self._power_scorer

    @property
    def explainability(self):
        if self._explainability is None:
            from src.explainability import ExplainabilityEngine
            self._explainability = ExplainabilityEngine()
        return self._explainability

    @property
    def pipeline(self):
        if self._pipeline is None:
            from src.data_pipeline import ContractIntelligencePipeline
            self._pipeline = ContractIntelligencePipeline()
        return self._pipeline

    @property
    def report_gen(self):
        if self._report_gen is None:
            from src.report_generator import ReportGenerator
            self._report_gen = ReportGenerator()
        return self._report_gen


registry = ModelRegistry()


# Startup / shutdown events

@app.on_event("startup")
async def startup_event():
    """Initialise database tables on application startup."""
    logger.info("Starting Contract Intelligence Platform API...")
    create_tables()
    logger.info("Database tables ready.")


# Global exception handler

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch all unhandled exceptions and return structured JSON error.

    Args:
        request: Incoming FastAPI request.
        exc: Unhandled exception.

    Returns:
        JSONResponse with ErrorResponse schema and 500 status.
    """
    logger.error(f"Unhandled exception: {exc}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="internal_server_error",
            detail=str(exc),
        ).model_dump(),
    )


# Utility helpers

def _get_contract_or_404(contract_id: str, db: Session) -> Contract:
    """Fetch a contract from the database or raise HTTP 404.

    Args:
        contract_id: Contract identifier.
        db: Database session.

    Returns:
        Contract ORM instance.

    Raises:
        HTTPException: 404 if contract not found.
    """
    contract = db.get(Contract, contract_id)
    if contract is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ErrorResponse(
                error="contract_not_found",
                detail=f"No contract found with ID: {contract_id}",
                contract_id=contract_id,
            ).model_dump(),
        )
    return contract


def _clause_orm_to_schema(clause: Clause) -> ClauseScore:
    """Convert a Clause ORM object to a ClauseScore Pydantic schema.

    Args:
        clause: SQLAlchemy Clause instance.

    Returns:
        ClauseScore schema instance.
    """
    return ClauseScore(
        clause_id=clause.clause_id,
        contract_id=clause.contract_id,
        clause_text=clause.clause_text,
        clause_types=clause.clause_type or "",
        party_a=clause.party_a or "",
        party_b=clause.party_b or "",
        anomaly_score=clause.anomaly_score,
        is_anomalous=clause.is_anomalous,
        power_imbalance_score=clause.power_imbalance_score,
        party_a_leverage=clause.party_a_leverage,
        party_b_leverage=clause.party_b_leverage,
        sentiment_score=clause.sentiment_score,
        modal_score=clause.modal_score,
        obligation_score=clause.obligation_score,
        assertiveness_score=clause.assertiveness_score,
        imbalance_label=_derive_imbalance_label(clause.power_imbalance_score),
        shap_plot_path=clause.shap_plot_path,
    )


def _derive_imbalance_label(score: Optional[float]) -> Optional[str]:
    """Derive a human-readable imbalance label from a numeric score.

    Args:
        score: Power imbalance score (-100 to +100) or None.

    Returns:
        'HIGH', 'MEDIUM', 'BALANCED', or None if score is missing.
    """
    if score is None:
        return None
    if abs(score) >= config.IMBALANCE_HIGH_THRESHOLD:
        return "HIGH" if score > 0 else "HIGH (Party B)"
    if abs(score) >= config.IMBALANCE_MEDIUM_THRESHOLD:
        return "MEDIUM" if score > 0 else "MEDIUM (Party B)"
    return "BALANCED"


def _run_full_analysis(
    contract_id: str,
    party_a: str,
    party_b: str,
    run_shap: bool,
    db: Session,
) -> AnalyseResponse:
    """Execute the full analysis pipeline on an already-ingested contract.

    Pipeline:
      1. Classify all clauses using production classifier (Legal-BERT / DeBERTa / Legal-RoBERTa / ensemble)
      2. Score anomalies using IF + Autoencoder
      3. Score power imbalance using feature-engineered scorer
      4. Optionally generate SHAP explanations
      5. Persist all results to database
      6. Compute contract-level aggregate metrics

    Args:
        contract_id: Contract identifier (must already exist in DB).
        party_a: Display name for Party A.
        party_b: Display name for Party B.
        run_shap: Whether to generate SHAP visualisations.
        db: Database session.

    Returns:
        Full AnalyseResponse schema.
    """
    clauses = db.query(Clause).filter(Clause.contract_id == contract_id).all()
    if not clauses:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=ErrorResponse(
                error="no_clauses",
                detail=f"No clauses found for contract {contract_id}. Ingestion may have failed.",
                contract_id=contract_id,
            ).model_dump(),
        )

    texts = [c.clause_text for c in clauses]

    # Step 1: Clause Classification
    logger.info(f"Classifying {len(clauses)} clauses for contract {contract_id}")
    try:
        classifications = registry.classifier.predict(texts)
        for clause, pred in zip(clauses, classifications):
            clause.clause_type = "|".join(pred["clause_types"]) if pred["clause_types"] else ""
        db.commit()
    except FileNotFoundError:
        logger.warning("Classifier checkpoint not found — skipping classification.")

    # Step 2: Anomaly Detection
    logger.info(f"Scoring anomalies for contract {contract_id}")
    try:
        anomaly_scores = registry.anomaly.score(texts)
        for clause, a_score in zip(clauses, anomaly_scores):
            clause.anomaly_score = a_score["combined_score"]
            clause.is_anomalous  = a_score["is_anomalous"]
        db.commit()
    except FileNotFoundError:
        logger.warning("Anomaly detector not found — skipping anomaly scoring.")

    # Step 3: Power Imbalance Scoring
    logger.info(f"Scoring power imbalance for contract {contract_id}")
    registry.power_scorer.length_extractor.fit(texts)
    registry.power_scorer._length_fitted = True
    power_scores = registry.power_scorer.score(texts, party_a, party_b)

    for clause, p_score in zip(clauses, power_scores):
        clause.power_imbalance_score = p_score["power_imbalance_score"]
        clause.party_a_leverage      = p_score["party_a_leverage"]
        clause.party_b_leverage      = p_score["party_b_leverage"]
        clause.sentiment_score       = p_score["sentiment_score"]
        clause.modal_score           = p_score["modal_score"]
        clause.obligation_score      = p_score["obligation_score"]
        clause.assertiveness_score   = p_score["assertiveness_score"]
    db.commit()

    # Step 4: Optional SHAP
    if run_shap:
        logger.info("Generating SHAP explanations (this may take several minutes)...")
        try:
            registry.explainability.explain_contract(contract_id, max_clauses=5)
        except Exception as e:
            logger.warning(f"SHAP generation failed: {e}")

    # Step 5: Aggregate Metrics
    db.refresh_all = True
    clauses = db.query(Clause).filter(Clause.contract_id == contract_id).all()

    imbalance_values = [
        c.power_imbalance_score for c in clauses
        if c.power_imbalance_score is not None
    ]
    overall_index = float(sum(imbalance_values) / len(imbalance_values)) if imbalance_values else 0.0
    anomalous_count = sum(1 for c in clauses if c.is_anomalous)

    # Dominant clause type
    type_counts: dict = {}
    for c in clauses:
        if c.clause_type:
            primary = c.clause_type.split("|")[0]
            type_counts[primary] = type_counts.get(primary, 0) + 1
    dominant_type = max(type_counts, key=type_counts.get) if type_counts else None

    dominant_party = (
        party_a if overall_index > 0
        else (party_b if overall_index < 0 else "Balanced")
    )

    # Persist analysis result
    result_row = AnalysisResult(
        contract_id=contract_id,
        overall_imbalance_index=overall_index,
        total_clauses=len(clauses),
        anomalous_clauses=anomalous_count,
        dominant_clause_type=dominant_type,
        analysis_metadata=json.dumps({
            "party_a": party_a,
            "party_b": party_b,
            "run_shap": run_shap,
        }),
    )
    db.add(result_row)
    db.commit()

    # Build response
    clause_schemas = [_clause_orm_to_schema(c) for c in clauses]

    imbalance_by_type: dict = {}
    for c in clauses:
        if c.clause_type and c.power_imbalance_score is not None:
            primary = c.clause_type.split("|")[0]
            imbalance_by_type.setdefault(primary, []).append(c.power_imbalance_score)

    imbalance_list = [
        ImbalanceByType(
            clause_type=ct,
            imbalance_score=float(sum(vals) / len(vals)),
            clause_count=len(vals),
        )
        for ct, vals in imbalance_by_type.items()
    ]

    power_response = PowerImbalanceResponse(
        contract_id=contract_id,
        overall_imbalance_index=overall_index,
        dominant_party=dominant_party,
        total_clauses=len(clauses),
        imbalance_by_type=imbalance_list,
        party_a_name=party_a,
        party_b_name=party_b,
    )

    return AnalyseResponse(
        contract_id=contract_id,
        summary=AnalysisSummary(
            contract_id=contract_id,
            total_clauses=len(clauses),
            anomalous_clauses=anomalous_count,
            overall_imbalance_index=overall_index,
            dominant_party=dominant_party,
            dominant_clause_type=dominant_type,
        ),
        clauses=clause_schemas,
        imbalance=power_response,
        report_url=f"/report/{contract_id}",
    )


# ROUTES

@app.get("/", include_in_schema=False)
async def dashboard(request: Request):
    """Serve the main HTML dashboard.

    Args:
        request: Incoming HTTP request.

    Returns:
        Jinja2 HTML response with the dashboard template.
    """
    return app.state.templates.TemplateResponse(
        "dashboard.html",
        {"request": request, "api_version": config.API_VERSION},
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Returns service health status. Use this for load balancer probes.",
)
async def health_check() -> HealthResponse:
    """Return service health status.

    Returns:
        HealthResponse with status 'ok' and current UTC timestamp.
    """
    return HealthResponse(
        status="ok",
        version=config.API_VERSION,
        timestamp=datetime.now(timezone.utc),
    )


@app.post(
    "/analyse",
    response_model=AnalyseResponse,
    summary="Analyse Contract",
    description=(
        "Accept a contract as a PDF upload or raw text, run the full analysis pipeline "
        "(clause classification, anomaly detection, power imbalance scoring), and return "
        "a structured analysis result."
    ),
    status_code=status.HTTP_200_OK,
)
async def analyse_contract(
    file:    Optional[UploadFile] = File(None,  description="PDF file upload"),
    text:    Optional[str]        = Form(None,  description="Raw contract text"),
    party_a: str                  = Form("Party A", description="Party A display name"),
    party_b: str                  = Form("Party B", description="Party B display name"),
    run_shap: bool                = Form(False, description="Generate SHAP explanations"),
    db: Session = Depends(get_db),
) -> AnalyseResponse:
    """Analyse a commercial contract from PDF or raw text.

    The endpoint accepts either:
      - A multipart PDF file upload (file parameter)
      - Raw contract text (text parameter)

    One of the two must be provided. If both are provided, the file takes precedence.

    Args:
        file: Optional PDF file upload.
        text: Optional raw contract text string.
        party_a: Display name for Party A (default: 'Party A').
        party_b: Display name for Party B (default: 'Party B').
        run_shap: If True, generate SHAP visualisations (significantly increases latency).
        db: Database session (injected by FastAPI).

    Returns:
        Full AnalyseResponse with clause scores, anomaly flags, and power imbalance breakdown.

    Raises:
        HTTPException 422: If neither file nor text is provided.
        HTTPException 500: On any processing failure.
    """
    if file is None and (text is None or not text.strip()):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=ErrorResponse(
                error="missing_input",
                detail="Provide either a PDF file upload or raw contract text.",
            ).model_dump(),
        )

    # Generate contract_id
    if file is not None:
        raw_bytes = await file.read()
        contract_id = hashlib.md5(raw_bytes).hexdigest()[:12]

        # Save PDF to temp file for ingestion
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(raw_bytes)
            tmp_path = Path(tmp.name)

        try:
            registry.pipeline.run_pdf(tmp_path, contract_id=contract_id)
        finally:
            tmp_path.unlink(missing_ok=True)

    else:
        contract_id = hashlib.md5(text.encode()).hexdigest()[:12]
        registry.pipeline.run_text(text, contract_id=contract_id)

    return _run_full_analysis(contract_id, party_a, party_b, run_shap, db)


@app.get(
    "/clauses/{contract_id}",
    response_model=ClauseListResponse,
    summary="Get All Clauses",
    description="Return all extracted clauses for a contract with their classification, anomaly, and imbalance scores.",
)
async def get_clauses(
    contract_id: str,
    db: Session = Depends(get_db),
) -> ClauseListResponse:
    """Return all clauses for a given contract.

    Args:
        contract_id: Contract identifier.
        db: Database session.

    Returns:
        ClauseListResponse with all clause scores.

    Raises:
        HTTPException 404: If contract not found.
    """
    _get_contract_or_404(contract_id, db)
    clauses = db.query(Clause).filter(Clause.contract_id == contract_id).all()
    return ClauseListResponse(
        contract_id=contract_id,
        total_clauses=len(clauses),
        clauses=[_clause_orm_to_schema(c) for c in clauses],
    )


@app.get(
    "/anomalies/{contract_id}",
    response_model=AnomaliesResponse,
    summary="Get Anomalous Clauses",
    description=(
        "Return only clauses flagged as anomalous (combined anomaly score > threshold). "
        f"Default threshold: {config.ANOMALY_FLAG_THRESHOLD}."
    ),
)
async def get_anomalies(
    contract_id: str,
    db: Session = Depends(get_db),
) -> AnomaliesResponse:
    """Return anomalous clauses for a contract.

    Args:
        contract_id: Contract identifier.
        db: Database session.

    Returns:
        AnomaliesResponse with only flagged anomalous clauses.

    Raises:
        HTTPException 404: If contract not found.
    """
    _get_contract_or_404(contract_id, db)
    all_clauses = db.query(Clause).filter(Clause.contract_id == contract_id).all()
    anomalous   = [c for c in all_clauses if c.is_anomalous]

    return AnomaliesResponse(
        contract_id=contract_id,
        total_clauses=len(all_clauses),
        anomalous_count=len(anomalous),
        anomaly_threshold=config.ANOMALY_FLAG_THRESHOLD,
        anomalous_clauses=[_clause_orm_to_schema(c) for c in anomalous],
    )


@app.get(
    "/imbalance/{contract_id}",
    response_model=PowerImbalanceResponse,
    summary="Get Power Imbalance",
    description="Return the full power imbalance breakdown for a contract, per clause type and overall.",
)
async def get_imbalance(
    contract_id: str,
    db: Session = Depends(get_db),
) -> PowerImbalanceResponse:
    """Return power imbalance analysis for a contract.

    Args:
        contract_id: Contract identifier.
        db: Database session.

    Returns:
        PowerImbalanceResponse with overall and per-type imbalance scores.

    Raises:
        HTTPException 404: If contract not found.
    """
    _get_contract_or_404(contract_id, db)
    clauses = db.query(Clause).filter(Clause.contract_id == contract_id).all()

    # Get party names from latest analysis result
    latest = (
        db.query(AnalysisResult)
        .filter(AnalysisResult.contract_id == contract_id)
        .order_by(AnalysisResult.created_at.desc())
        .first()
    )
    party_a, party_b = "Party A", "Party B"
    if latest and latest.analysis_metadata:
        try:
            meta = json.loads(latest.analysis_metadata)
            party_a = meta.get("party_a", "Party A")
            party_b = meta.get("party_b", "Party B")
        except json.JSONDecodeError:
            pass

    imbalance_values = [
        c.power_imbalance_score for c in clauses if c.power_imbalance_score is not None
    ]
    overall = float(sum(imbalance_values) / len(imbalance_values)) if imbalance_values else 0.0

    imbalance_by_type: dict = {}
    for c in clauses:
        if c.clause_type and c.power_imbalance_score is not None:
            primary = c.clause_type.split("|")[0]
            imbalance_by_type.setdefault(primary, []).append(c.power_imbalance_score)

    return PowerImbalanceResponse(
        contract_id=contract_id,
        overall_imbalance_index=overall,
        dominant_party=party_a if overall > 0 else (party_b if overall < 0 else "Balanced"),
        total_clauses=len(clauses),
        imbalance_by_type=[
            ImbalanceByType(
                clause_type=ct,
                imbalance_score=float(sum(vals) / len(vals)),
                clause_count=len(vals),
            )
            for ct, vals in imbalance_by_type.items()
        ],
        party_a_name=party_a,
        party_b_name=party_b,
    )


@app.get(
    "/report/{contract_id}",
    summary="Download PDF Report",
    description="Generate and download a full PDF analysis report for a contract.",
)
async def get_report(
    contract_id: str,
    db: Session = Depends(get_db),
):
    """Generate and serve the PDF report for a contract.

    Args:
        contract_id: Contract identifier.
        db: Database session.

    Returns:
        FileResponse streaming the generated PDF.

    Raises:
        HTTPException 404: If contract not found.
        HTTPException 500: If PDF generation fails.
    """
    contract = _get_contract_or_404(contract_id, db)
    clauses  = db.query(Clause).filter(Clause.contract_id == contract_id).all()

    latest = (
        db.query(AnalysisResult)
        .filter(AnalysisResult.contract_id == contract_id)
        .order_by(AnalysisResult.created_at.desc())
        .first()
    )

    try:
        pdf_path = registry.report_gen.generate(
            contract_id=contract_id,
            filename=contract.filename,
            clauses=[_clause_orm_to_schema(c) for c in clauses],
            overall_imbalance=latest.overall_imbalance_index if latest else 0.0,
            anomalous_count=latest.anomalous_clauses if latest else 0,
        )
    except Exception as exc:
        logger.error(f"PDF generation failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                error="report_generation_failed",
                detail=str(exc),
                contract_id=contract_id,
            ).model_dump(),
        )

    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=f"contract_analysis_{contract_id}.pdf",
    )


# Development server entry point

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True,
        log_level="info",
    )
