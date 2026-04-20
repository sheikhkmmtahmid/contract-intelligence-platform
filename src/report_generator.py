"""
report_generator.py — Production-grade PDF report generation using ReportLab.

Generates a comprehensive legal analysis report including:
  - Executive summary with overall power imbalance index
  - Clause-by-clause breakdown table with anomaly and imbalance scores
  - SHAP visualisation embeds (PNG images)
  - Per-clause-type power imbalance summary
  - Methodology and disclaimer section

Design: dark navy background with gold accents matching the dashboard aesthetic.

Usage:
    from src.report_generator import ReportGenerator
    gen = ReportGenerator()
    pdf_path = gen.generate(contract_id, filename, clauses, overall_imbalance, anomalous_count)
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from loguru import logger
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    HRFlowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from api.schemas import ClauseScore

logger.remove()
logger.add(config.LOGS_DIR / "report_generator.log", rotation="10 MB", level="DEBUG")
logger.add(sys.stderr, level="INFO")

# ---------------------------------------------------------------------------
# Brand colours
# ---------------------------------------------------------------------------
NAVY        = colors.HexColor("#0D1B2A")
GOLD        = colors.HexColor("#D4AF37")
LIGHT_GOLD  = colors.HexColor("#F0E68C")
WHITE       = colors.white
RED         = colors.HexColor("#C0392B")
AMBER       = colors.HexColor("#E67E22")
GREEN       = colors.HexColor("#27AE60")
BLUE        = colors.HexColor("#2980B9")
GREY        = colors.HexColor("#BDC3C7")
DARK_GREY   = colors.HexColor("#2C3E50")


class ReportGenerator:
    """Generates a multi-page PDF analysis report for a single contract.

    The report is styled with the platform's dark navy and gold brand colours
    and includes all analysis results available at the time of generation.
    """

    REPORT_DIR = config.PROCESSED_DIR / "reports"

    def __init__(self):
        self.REPORT_DIR.mkdir(parents=True, exist_ok=True)
        self.styles = self._build_styles()

    def _build_styles(self) -> dict:
        """Build a dict of named ReportLab ParagraphStyles.

        Returns:
            Dict mapping style name → ParagraphStyle instance.
        """
        return {
            "title": ParagraphStyle(
                "title",
                fontName="Helvetica-Bold",
                fontSize=22,
                textColor=GOLD,
                alignment=TA_CENTER,
                spaceAfter=6,
            ),
            "subtitle": ParagraphStyle(
                "subtitle",
                fontName="Helvetica",
                fontSize=12,
                textColor=LIGHT_GOLD,
                alignment=TA_CENTER,
                spaceAfter=12,
            ),
            "section_heading": ParagraphStyle(
                "section_heading",
                fontName="Helvetica-Bold",
                fontSize=14,
                textColor=GOLD,
                spaceBefore=14,
                spaceAfter=6,
            ),
            "body": ParagraphStyle(
                "body",
                fontName="Helvetica",
                fontSize=9,
                textColor=WHITE,
                leading=13,
                spaceAfter=4,
            ),
            "clause_text": ParagraphStyle(
                "clause_text",
                fontName="Helvetica",
                fontSize=8,
                textColor=GREY,
                leading=11,
                spaceAfter=2,
            ),
            "disclaimer": ParagraphStyle(
                "disclaimer",
                fontName="Helvetica-Oblique",
                fontSize=8,
                textColor=GREY,
                leading=11,
                spaceAfter=4,
            ),
            "metric_label": ParagraphStyle(
                "metric_label",
                fontName="Helvetica-Bold",
                fontSize=10,
                textColor=LIGHT_GOLD,
            ),
            "metric_value": ParagraphStyle(
                "metric_value",
                fontName="Helvetica-Bold",
                fontSize=20,
                textColor=GOLD,
                alignment=TA_CENTER,
            ),
        }

    def generate(
        self,
        contract_id: str,
        filename: str,
        clauses: List[ClauseScore],
        overall_imbalance: float,
        anomalous_count: int,
    ) -> Path:
        """Generate the full PDF report and save to disk.

        Args:
            contract_id: Unique contract identifier.
            filename: Original contract filename for display.
            clauses: List of ClauseScore schema instances with all scores.
            overall_imbalance: Contract-level power imbalance index (-100 to +100).
            anomalous_count: Number of anomalous clauses detected.

        Returns:
            Path to the generated PDF file.
        """
        output_path = self.REPORT_DIR / f"report_{contract_id}.pdf"

        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            leftMargin=2*cm,
            rightMargin=2*cm,
            topMargin=2*cm,
            bottomMargin=2*cm,
        )

        story = []
        story += self._build_cover_page(contract_id, filename, overall_imbalance, anomalous_count, clauses)
        story.append(PageBreak())
        story += self._build_executive_summary(overall_imbalance, anomalous_count, clauses)
        story.append(PageBreak())
        story += self._build_clause_table(clauses)
        story += self._build_shap_section(clauses)
        story += self._build_methodology_section()

        doc.build(story, onFirstPage=self._add_background, onLaterPages=self._add_background)

        logger.info(f"PDF report generated: {output_path}")
        return output_path

    def _add_background(self, canvas, doc):
        """Draw dark navy background on every page.

        Args:
            canvas: ReportLab canvas.
            doc: Document instance.
        """
        canvas.saveState()
        canvas.setFillColor(NAVY)
        canvas.rect(0, 0, A4[0], A4[1], fill=True, stroke=False)

        # Footer
        canvas.setFillColor(GOLD)
        canvas.setFont("Helvetica", 7)
        canvas.drawString(
            2*cm, 1.2*cm,
            f"Contract Intelligence Platform | {config.API_TITLE} | Confidential"
        )
        canvas.drawRightString(
            A4[0] - 2*cm, 1.2*cm,
            f"Page {doc.page}"
        )
        canvas.restoreState()

    def _build_cover_page(
        self,
        contract_id: str,
        filename: str,
        overall_imbalance: float,
        anomalous_count: int,
        clauses: List[ClauseScore],
    ) -> list:
        """Build the cover page elements.

        Args:
            contract_id: Contract identifier.
            filename: Original filename.
            overall_imbalance: Overall power imbalance index.
            anomalous_count: Number of anomalous clauses.
            clauses: All clause scores.

        Returns:
            List of ReportLab flowables for the cover page.
        """
        elements = [Spacer(1, 3*cm)]

        elements.append(Paragraph(config.API_TITLE, self.styles["title"]))
        elements.append(Paragraph("Legal Intelligence & Power Imbalance Analysis", self.styles["subtitle"]))
        elements.append(HRFlowable(width="100%", thickness=1, color=GOLD, spaceAfter=20))

        elements.append(Spacer(1, 1*cm))

        # Contract metadata table
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        meta_data = [
            ["Contract ID",    contract_id],
            ["Filename",       filename],
            ["Analysis Date",  now_str],
            ["Total Clauses",  str(len(clauses))],
            ["Anomalous",      f"{anomalous_count} clauses"],
            ["Platform",       f"v{config.API_VERSION}"],
        ]
        meta_table = Table(meta_data, colWidths=[5*cm, 10*cm])
        meta_table.setStyle(TableStyle([
            ("TEXTCOLOR",    (0, 0), (-1, -1), WHITE),
            ("FONTNAME",     (0, 0), (0, -1),  "Helvetica-Bold"),
            ("FONTNAME",     (1, 0), (1, -1),  "Helvetica"),
            ("FONTSIZE",     (0, 0), (-1, -1), 10),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [DARK_GREY, NAVY]),
            ("TOPPADDING",   (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING",  (0, 0), (-1, -1), 8),
            ("GRID",         (0, 0), (-1, -1), 0.5, GOLD),
        ]))
        elements.append(meta_table)

        elements.append(Spacer(1, 2*cm))

        # Big imbalance metric
        imbalance_color = RED if overall_imbalance > 20 else (BLUE if overall_imbalance < -20 else GREEN)
        elements.append(Paragraph("Overall Power Imbalance Index", self.styles["metric_label"]))
        elements.append(Paragraph(
            f'<font color="#{imbalance_color.hexval()[1:]}">{overall_imbalance:+.1f}</font>',
            self.styles["metric_value"],
        ))
        elements.append(Paragraph(
            "(+100 = Maximum Party A advantage | -100 = Maximum Party B advantage | 0 = Balanced)",
            self.styles["clause_text"],
        ))

        return elements

    def _build_executive_summary(
        self,
        overall_imbalance: float,
        anomalous_count: int,
        clauses: List[ClauseScore],
    ) -> list:
        """Build the executive summary section.

        Args:
            overall_imbalance: Contract-level imbalance.
            anomalous_count: Anomalous clause count.
            clauses: All clause scores.

        Returns:
            List of ReportLab flowables.
        """
        elements = [Paragraph("1. Executive Summary", self.styles["section_heading"])]
        elements.append(HRFlowable(width="100%", thickness=0.5, color=GOLD))

        # Imbalance interpretation
        if overall_imbalance > config.IMBALANCE_HIGH_THRESHOLD:
            interp = (
                f"This contract exhibits HIGH power imbalance favouring Party A "
                f"(index: {overall_imbalance:+.1f}). Legal review is strongly recommended "
                f"for the identified high-leverage clauses before signing."
            )
        elif overall_imbalance < -config.IMBALANCE_HIGH_THRESHOLD:
            interp = (
                f"This contract exhibits HIGH power imbalance favouring Party B "
                f"(index: {overall_imbalance:+.1f}). Review liability, indemnity, "
                f"and termination clauses with counsel."
            )
        elif abs(overall_imbalance) > config.IMBALANCE_MEDIUM_THRESHOLD:
            interp = (
                f"This contract shows MODERATE power imbalance "
                f"(index: {overall_imbalance:+.1f}). Key clause types show divergence — "
                f"see the per-clause breakdown for detail."
            )
        else:
            interp = (
                f"This contract appears broadly BALANCED between parties "
                f"(index: {overall_imbalance:+.1f}). No significant structural "
                f"power asymmetry was detected."
            )

        elements.append(Spacer(1, 6))
        elements.append(Paragraph(interp, self.styles["body"]))

        # Key stats
        elements.append(Spacer(1, 8))
        high_clauses = [c for c in clauses if c.imbalance_label and "HIGH" in c.imbalance_label]
        elements.append(Paragraph(
            f"• <b>Total clauses analysed:</b> {len(clauses)}<br/>"
            f"• <b>Anomalous clauses flagged:</b> {anomalous_count}<br/>"
            f"• <b>High-imbalance clauses:</b> {len(high_clauses)}<br/>"
            f"• <b>Anomaly threshold:</b> {config.ANOMALY_FLAG_THRESHOLD}/100<br/>"
            f"• <b>Models used:</b> Legal-BERT (classification), Isolation Forest + "
            f"Autoencoder (anomaly), RoBERTa (sentiment), SHAP (explainability)",
            self.styles["body"],
        ))

        return elements

    def _build_clause_table(self, clauses: List[ClauseScore]) -> list:
        """Build the clause-by-clause analysis table.

        Args:
            clauses: All clause scores.

        Returns:
            List of ReportLab flowables.
        """
        elements = [
            Paragraph("2. Clause-by-Clause Analysis", self.styles["section_heading"]),
            HRFlowable(width="100%", thickness=0.5, color=GOLD),
            Spacer(1, 6),
        ]

        headers = ["#", "Clause Type", "Anomaly\nScore", "Anomaly\nFlag", "Imbalance\nScore", "Label"]
        table_data = [headers]

        for i, clause in enumerate(clauses[:100], 1):  # cap at 100 rows for PDF size
            clause_type = (clause.clause_type.split("|")[0] if clause.clause_type else "Unclassified")[:25]
            anomaly_score = f"{clause.anomaly_score:.1f}" if clause.anomaly_score is not None else "N/A"
            anomaly_flag  = "YES" if clause.is_anomalous else "NO"
            imbalance     = f"{clause.power_imbalance_score:+.1f}" if clause.power_imbalance_score is not None else "N/A"
            label         = clause.imbalance_label or "N/A"

            table_data.append([str(i), clause_type, anomaly_score, anomaly_flag, imbalance, label])

        col_widths = [1*cm, 5.5*cm, 2*cm, 2*cm, 2.5*cm, 3*cm]
        t = Table(table_data, colWidths=col_widths, repeatRows=1)

        style = [
            # Header row
            ("BACKGROUND",    (0, 0), (-1, 0),  DARK_GREY),
            ("TEXTCOLOR",     (0, 0), (-1, 0),  GOLD),
            ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, 0),  8),
            ("ALIGN",         (0, 0), (-1, 0),  "CENTER"),
            # Data rows
            ("TEXTCOLOR",     (0, 1), (-1, -1), WHITE),
            ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE",      (0, 1), (-1, -1), 7),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [NAVY, DARK_GREY]),
            ("ALIGN",         (2, 1), (-1, -1), "CENTER"),
            ("GRID",          (0, 0), (-1, -1), 0.3, GOLD),
            ("TOPPADDING",    (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]

        # Colour-code anomaly flag column
        for row_idx, clause in enumerate(clauses[:100], 1):
            if clause.is_anomalous:
                style.append(("TEXTCOLOR", (3, row_idx), (3, row_idx), RED))
                style.append(("FONTNAME",  (3, row_idx), (3, row_idx), "Helvetica-Bold"))

            # Colour-code imbalance label
            if clause.imbalance_label and "HIGH" in clause.imbalance_label:
                style.append(("TEXTCOLOR", (5, row_idx), (5, row_idx), RED))
            elif clause.imbalance_label == "MEDIUM":
                style.append(("TEXTCOLOR", (5, row_idx), (5, row_idx), AMBER))
            elif clause.imbalance_label == "BALANCED":
                style.append(("TEXTCOLOR", (5, row_idx), (5, row_idx), GREEN))

        t.setStyle(TableStyle(style))
        elements.append(t)

        if len(clauses) > 100:
            elements.append(Spacer(1, 4))
            elements.append(Paragraph(
                f"Note: Table shows first 100 of {len(clauses)} clauses. "
                "Full data available via API GET /clauses/{contract_id}.",
                self.styles["clause_text"],
            ))

        return elements

    def _build_shap_section(self, clauses: List[ClauseScore]) -> list:
        """Embed SHAP visualisation PNGs for clauses that have them.

        Args:
            clauses: Clause scores with optional shap_plot_path.

        Returns:
            List of ReportLab flowables.
        """
        shap_clauses = [c for c in clauses if c.shap_plot_path and Path(c.shap_plot_path).exists()]
        if not shap_clauses:
            return []

        elements = [
            PageBreak(),
            Paragraph("3. SHAP Token-Level Explanations", self.styles["section_heading"]),
            HRFlowable(width="100%", thickness=0.5, color=GOLD),
            Paragraph(
                "The following visualisations show which tokens in each clause drove the "
                "classification and power imbalance predictions. Red bars push toward "
                "Party A; blue bars push toward Party B.",
                self.styles["body"],
            ),
            Spacer(1, 8),
        ]

        for clause in shap_clauses[:5]:  # cap at 5 SHAP plots per report
            elements.append(Paragraph(
                f"Clause ID: {clause.clause_id[:12]}... | "
                f"Type: {clause.clause_type.split('|')[0] if clause.clause_type else 'Unknown'}",
                self.styles["metric_label"],
            ))
            try:
                img = Image(clause.shap_plot_path, width=15*cm, height=7*cm)
                elements.append(img)
            except Exception as e:
                logger.warning(f"Failed to embed SHAP image: {e}")
                elements.append(Paragraph(f"[SHAP image unavailable: {e}]", self.styles["clause_text"]))
            elements.append(Spacer(1, 12))

        return elements

    def _build_methodology_section(self) -> list:
        """Build the methodology and disclaimer section.

        Returns:
            List of ReportLab flowables.
        """
        elements = [
            PageBreak(),
            Paragraph("4. Methodology & Limitations", self.styles["section_heading"]),
            HRFlowable(width="100%", thickness=0.5, color=GOLD),
            Spacer(1, 6),
        ]

        methodology_text = (
            "<b>Clause Classification:</b> Fine-tuned Legal-BERT (nlpaueb/legal-bert-base-uncased) "
            "on the CUAD dataset (510 commercial contracts, 41 clause types). "
            "Multi-label BCEWithLogitsLoss, AdamW optimiser, 5 epochs.<br/><br/>"

            "<b>Anomaly Detection:</b> Isolation Forest (contamination=0.05) and Shallow Autoencoder "
            "(768→256→64→256→768, MSELoss) trained on Legal-BERT [CLS] embeddings of "
            "market-standard clauses. Scores fused via 50/50 weighted average.<br/><br/>"

            "<b>Power Imbalance Scoring:</b> Feature-engineered scorer combining sentiment "
            "(RoBERTa), modal verb balance, obligation assignment, assertiveness analysis, "
            "and normalised clause length. Weights: sentiment 30%, modals 25%, "
            "obligations 25%, assertiveness 20%.<br/><br/>"

            "<b>Important Limitation:</b> No publicly available ground-truth dataset exists "
            "for power imbalance labels in commercial contracts. The power imbalance scorer "
            "is feature-engineered and validated through feature contribution analysis "
            "and consistency testing, not supervised learning. Scores reflect linguistic "
            "patterns and should be interpreted as indicators, not definitive legal judgements.<br/><br/>"

            "<b>SHAP Explainability:</b> SHAP KernelExplainer applied to both classifiers. "
            "Token attribution is model-agnostic and computed via perturbation analysis."
        )
        elements.append(Paragraph(methodology_text, self.styles["body"]))

        elements.append(Spacer(1, 12))
        elements.append(HRFlowable(width="100%", thickness=0.3, color=GREY))
        elements.append(Spacer(1, 6))
        elements.append(Paragraph(
            "DISCLAIMER: This report is generated by an automated AI system and does not "
            "constitute legal advice. Always seek qualified legal counsel before signing or "
            "relying on any contract analysis. The platform authors accept no liability for "
            "decisions made based on this report.",
            self.styles["disclaimer"],
        ))

        return elements
