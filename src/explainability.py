"""
explainability.py — SHAP-based token-level and feature-level explainability.

Two explainability layers are provided:

  1. Classifier SHAP — KernelExplainer on the clause classifier.
     Shows which tokens drove each clause type prediction.
     Output: token-level SHAP bar chart (PNG).

  2. Power Imbalance SHAP — KernelExplainer on the power scorer features.
     Shows which features (sentiment, modal verbs, obligations, assertiveness)
     drove the bilateral power imbalance score.
     Output: feature-importance SHAP bar chart (PNG).

SHAP KernelExplainer is model-agnostic and works across both models without
modification. Token attribution maps directly to legal clause words, making
this approach legally interpretable.

Usage:
    from src.explainability import ExplainabilityEngine
    engine = ExplainabilityEngine()
    png_path = engine.explain_clause(clause_text, clause_id)
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless backend — must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.clause_classifier import ClauseClassifierInference
from src.power_scorer import PowerImbalanceScorer

logger.remove()
logger.add(config.LOGS_DIR / "explainability.log", rotation="10 MB", level="DEBUG")
logger.add(sys.stderr, level="INFO")


# 1. CLASSIFIER SHAP EXPLAINER

class ClassifierSHAPExplainer:
    """Generates token-level SHAP explanations for clause type predictions.

    Uses SHAP KernelExplainer with a bag-of-words input representation.
    The model wrapper converts word-masked inputs back to text before
    calling the classifier, enabling true token-level attribution.

    Note: KernelExplainer is slow (~30s per clause). For production, consider
    caching explanations after first computation.
    """

    def __init__(self, classifier: Optional[ClauseClassifierInference] = None):
        self.classifier = classifier or ClauseClassifierInference()

    def _build_word_mask_fn(self, text: str, target_label_idx: int):
        """Build a SHAP-compatible prediction function using word masking.

        SHAP KernelExplainer requires a function f(X) → R^n where X is a
        binary mask matrix over input features (words here). We mask words
        out of the original text and run the classifier on the masked version.

        Args:
            text: Original clause text.
            target_label_idx: Index of the target clause type in CUAD_CLAUSE_TYPES.

        Returns:
            Tuple of (predict_fn, words) where predict_fn maps masks → probabilities.
        """
        words = text.split()

        def predict_fn(mask_matrix: np.ndarray) -> np.ndarray:
            """Classify masked clause texts and return target class probability.

            Args:
                mask_matrix: Binary matrix of shape (n_samples, n_words).
                             1 = keep word, 0 = mask to [MASK].

            Returns:
                1D array of target class probabilities, shape (n_samples,).
            """
            texts_batch = []
            for mask_row in mask_matrix:
                masked_words = [
                    w if mask_row[i] == 1 else "[MASK]"
                    for i, w in enumerate(words)
                ]
                texts_batch.append(" ".join(masked_words))

            preds = self.classifier.predict(texts_batch, threshold=0.0)
            probs = np.array([
                p["probabilities"].get(config.CUAD_CLAUSE_TYPES[target_label_idx], 0.0)
                for p in preds
            ])
            return probs

        return predict_fn, words

    def explain(
        self,
        clause_text: str,
        target_clause_type: str,
        n_background: int = config.SHAP_BACKGROUND_SAMPLES,
        max_evals: int = config.SHAP_MAX_EVALS,
    ) -> Tuple[np.ndarray, List[str]]:
        """Compute token-level SHAP values for a clause.

        Args:
            clause_text: Raw clause text to explain.
            target_clause_type: Name of the CUAD clause type to explain.
            n_background: Number of background samples for KernelExplainer.
            max_evals: Maximum model evaluations (controls accuracy vs speed).

        Returns:
            Tuple of (shap_values, words) where shap_values is a 1D array
            of per-word attribution scores aligned with the words list.

        Raises:
            ValueError: If target_clause_type is not in CUAD_CLAUSE_TYPES.
        """
        if target_clause_type not in config.CUAD_CLAUSE_TYPES:
            raise ValueError(
                f"Unknown clause type: {target_clause_type}. "
                f"Must be one of {config.CUAD_CLAUSE_TYPES}"
            )

        target_idx = config.CUAD_CLAUSE_TYPES.index(target_clause_type)
        predict_fn, words = self._build_word_mask_fn(clause_text, target_idx)

        n_words = len(words)
        if n_words == 0:
            return np.array([]), []

        # Background data: random binary masks (represent "average" input)
        rng = np.random.RandomState(config.RANDOM_SEED)
        background = rng.randint(0, 2, size=(min(n_background, 50), n_words)).astype(float)

        # Single instance to explain: all words present
        instance = np.ones((1, n_words))

        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(
            instance,
            nsamples=min(max_evals, 200),
            silent=True,
        )

        return shap_values[0], words

    def plot_and_save(
        self,
        shap_values: np.ndarray,
        words: List[str],
        clause_id: str,
        clause_type: str,
        top_n: int = 20,
    ) -> Path:
        """Generate and save a token-level SHAP bar chart as PNG.

        Args:
            shap_values: SHAP attribution values (1D array, len = n_words).
            words: Word list aligned with shap_values.
            clause_id: Unique clause identifier (used in filename).
            clause_type: Clause type label for the plot title.
            top_n: Number of top words to display.

        Returns:
            Path to the saved PNG file.
        """
        if len(shap_values) == 0 or len(words) == 0:
            logger.warning(f"Empty SHAP values for clause {clause_id}. Skipping plot.")
            return None

        # Select top N words by absolute SHAP value
        top_indices = np.argsort(np.abs(shap_values))[-top_n:][::-1]
        top_words   = [words[i] for i in top_indices]
        top_vals    = shap_values[top_indices]

        colors = ["#C0392B" if v > 0 else "#2980B9" for v in top_vals]

        fig, ax = plt.subplots(figsize=(10, 6), facecolor="#0D1B2A")
        ax.set_facecolor("#0D1B2A")

        bars = ax.barh(range(len(top_words)), top_vals, color=colors, edgecolor="none")
        ax.set_yticks(range(len(top_words)))
        ax.set_yticklabels(top_words, fontsize=10, color="#F0E68C")
        ax.set_xlabel("SHAP Value (Token Attribution)", color="#F0E68C", fontsize=11)
        ax.set_title(
            f"Token-Level SHAP: {clause_type}\n(Clause ID: {clause_id[:8]}...)",
            color="#F0E68C", fontsize=13, pad=12,
        )
        ax.tick_params(colors="#F0E68C")
        ax.spines["bottom"].set_color("#F0E68C")
        ax.spines["left"].set_color("#F0E68C")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.axvline(0, color="#F0E68C", linewidth=0.8, alpha=0.5)

        # Add legend
        from matplotlib.patches import Patch
        legend = [
            Patch(color="#C0392B", label="Pushes toward Party A"),
            Patch(color="#2980B9", label="Pushes toward Party B"),
        ]
        ax.legend(handles=legend, loc="lower right", facecolor="#0D1B2A",
                  labelcolor="#F0E68C", edgecolor="#F0E68C")

        plt.tight_layout()

        output_path = config.SHAP_OUTPUT_DIR / f"shap_classifier_{clause_id}.png"
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

        logger.info(f"SHAP plot saved: {output_path}")
        return output_path


# 2. POWER IMBALANCE SHAP EXPLAINER

class PowerImbalanceSHAPExplainer:
    """Generates feature-level SHAP explanations for power imbalance scores.

    Features: sentiment, modal_verbs, obligations, assertiveness, length.
    Shows which features drove the clause's power imbalance toward
    Party A or Party B.
    """

    FEATURE_NAMES = [
        "sentiment_score",
        "modal_score",
        "obligation_score",
        "assertiveness_score",
        "length_score",
    ]

    def __init__(self, power_scorer: Optional[PowerImbalanceScorer] = None):
        self.power_scorer = power_scorer or PowerImbalanceScorer()

    def _build_predict_fn(self, base_text: str):
        """Build a prediction function that maps feature values → imbalance score.

        Since we want to explain at the feature level (not token level),
        we perturb individual feature values and observe the imbalance change.

        Args:
            base_text: The clause text being explained.

        Returns:
            Tuple of (predict_fn, background, base_features).
        """
        # Get base feature values
        scores = self.power_scorer.score([base_text])
        base = scores[0]
        base_features = np.array([
            base["sentiment_score"],
            base["modal_score"],
            base["obligation_score"],
            base["assertiveness_score"],
            base["length_score"],
        ])

        def predict_fn(feature_matrix: np.ndarray) -> np.ndarray:
            """Map perturbed feature vectors to predicted imbalance scores.

            Args:
                feature_matrix: Shape (n_samples, n_features).

            Returns:
                1D imbalance scores, shape (n_samples,).
            """
            results = []
            for feature_row in feature_matrix:
                s, m, o, a, ln = feature_row

                party_a_raw = (
                    config.POWER_WEIGHT_SENTIMENT    * (1.0 - s) +
                    config.POWER_WEIGHT_MODAL_VERBS  * m         +
                    config.POWER_WEIGHT_OBLIGATIONS  * o         +
                    config.POWER_WEIGHT_ASSERTIVENESS * a
                )
                party_b_raw = (
                    config.POWER_WEIGHT_SENTIMENT    * s          +
                    config.POWER_WEIGHT_MODAL_VERBS  * (1.0 - m) +
                    config.POWER_WEIGHT_OBLIGATIONS  * (1.0 - o) +
                    config.POWER_WEIGHT_ASSERTIVENESS * (1.0 - a)
                )
                amplifier = 0.8 + 0.4 * ln
                party_a   = float(np.clip(party_a_raw * amplifier * 100, 0, 100))
                party_b   = float(np.clip(party_b_raw * amplifier * 100, 0, 100))
                results.append(party_a - party_b)

            return np.array(results)

        # Background: all-0.5 (neutral feature values)
        background = np.full((1, 5), 0.5)
        return predict_fn, background, base_features

    def explain(self, clause_text: str) -> Tuple[np.ndarray, np.ndarray]:
        """Compute feature-level SHAP values for power imbalance.

        Args:
            clause_text: Raw clause text.

        Returns:
            Tuple of (shap_values, base_features) both as 1D arrays.
        """
        predict_fn, background, base_features = self._build_predict_fn(clause_text)

        explainer   = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(
            base_features.reshape(1, -1),
            nsamples=100,
            silent=True,
        )
        return shap_values[0], base_features

    def plot_and_save(
        self,
        shap_values: np.ndarray,
        base_features: np.ndarray,
        clause_id: str,
    ) -> Path:
        """Generate and save a feature-level SHAP plot as PNG.

        Args:
            shap_values: SHAP values for each feature (1D, len=5).
            base_features: Actual feature values (1D, len=5).
            clause_id: Clause identifier for filename.

        Returns:
            Path to the saved PNG file.
        """
        feature_labels = [
            f"Sentiment\n({base_features[0]:.2f})",
            f"Modal Verbs\n({base_features[1]:.2f})",
            f"Obligations\n({base_features[2]:.2f})",
            f"Assertiveness\n({base_features[3]:.2f})",
            f"Length\n({base_features[4]:.2f})",
        ]
        colors = ["#C0392B" if v > 0 else "#2980B9" for v in shap_values]

        fig, ax = plt.subplots(figsize=(9, 5), facecolor="#0D1B2A")
        ax.set_facecolor("#0D1B2A")

        ax.barh(feature_labels, shap_values, color=colors, edgecolor="none")
        ax.set_xlabel("SHAP Value (→ Party A | ← Party B)", color="#F0E68C", fontsize=11)
        ax.set_title(
            f"Feature Contributions to Power Imbalance\n(Clause: {clause_id[:8]}...)",
            color="#F0E68C", fontsize=13, pad=12,
        )
        ax.tick_params(colors="#F0E68C")
        ax.spines["bottom"].set_color("#F0E68C")
        ax.spines["left"].set_color("#F0E68C")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.axvline(0, color="#F0E68C", linewidth=0.8, alpha=0.5)
        ax.set_yticklabels(feature_labels, color="#F0E68C", fontsize=10)

        from matplotlib.patches import Patch
        legend = [
            Patch(color="#C0392B", label="Favours Party A"),
            Patch(color="#2980B9", label="Favours Party B"),
        ]
        ax.legend(handles=legend, facecolor="#0D1B2A", labelcolor="#F0E68C",
                  edgecolor="#F0E68C", loc="lower right")

        plt.tight_layout()

        output_path = config.SHAP_OUTPUT_DIR / f"shap_power_{clause_id}.png"
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)

        logger.info(f"Power SHAP plot saved: {output_path}")
        return output_path


# 3. UNIFIED EXPLAINABILITY ENGINE

class ExplainabilityEngine:
    """Unified interface for generating both classifier and power SHAP explanations.

    Lazily initialises sub-explainers to avoid loading heavy models unnecessarily.
    """

    def __init__(self):
        self._classifier_explainer: Optional[ClassifierSHAPExplainer] = None
        self._power_explainer:      Optional[PowerImbalanceSHAPExplainer] = None

    @property
    def classifier_explainer(self) -> ClassifierSHAPExplainer:
        """Lazy-load classifier SHAP explainer."""
        if self._classifier_explainer is None:
            self._classifier_explainer = ClassifierSHAPExplainer()
        return self._classifier_explainer

    @property
    def power_explainer(self) -> PowerImbalanceSHAPExplainer:
        """Lazy-load power imbalance SHAP explainer."""
        if self._power_explainer is None:
            self._power_explainer = PowerImbalanceSHAPExplainer()
        return self._power_explainer

    def explain_clause(
        self,
        clause_text: str,
        clause_id: str,
        clause_type: Optional[str] = None,
    ) -> Dict:
        """Generate both classifier and power SHAP explanations for a clause.

        Args:
            clause_text: Raw clause text.
            clause_id: Unique clause identifier.
            clause_type: Target clause type for classifier explanation.
                         If None, uses the highest-probability predicted type.

        Returns:
            Dict with:
                'classifier_shap_path': Path to classifier SHAP PNG (or None)
                'power_shap_path': Path to power SHAP PNG
                'classifier_shap_values': list of (word, shap_value) pairs
                'power_shap_values': dict of feature → shap_value
        """
        result: Dict = {
            "classifier_shap_path": None,
            "power_shap_path": None,
            "classifier_shap_values": [],
            "power_shap_values": {},
        }

        # --- Classifier SHAP ---
        try:
            if clause_type is None:
                # Predict and use top clause type
                pred = self.classifier_explainer.classifier.predict_single(clause_text)
                if pred["clause_types"]:
                    clause_type = pred["clause_types"][0]

            if clause_type:
                shap_vals, words = self.classifier_explainer.explain(
                    clause_text, clause_type
                )
                png_path = self.classifier_explainer.plot_and_save(
                    shap_vals, words, clause_id, clause_type
                )
                result["classifier_shap_path"] = png_path.as_posix() if png_path else None
                result["classifier_shap_values"] = [
                    {"word": w, "shap_value": float(v)}
                    for w, v in zip(words, shap_vals)
                ]
        except Exception as exc:
            logger.warning(f"Classifier SHAP failed for {clause_id}: {exc}")

        # --- Power Imbalance SHAP ---
        try:
            power_vals, base_feats = self.power_explainer.explain(clause_text)
            png_path = self.power_explainer.plot_and_save(power_vals, base_feats, clause_id)
            result["power_shap_path"] = png_path.as_posix() if png_path else None
            result["power_shap_values"] = {
                name: float(val)
                for name, val in zip(
                    PowerImbalanceSHAPExplainer.FEATURE_NAMES, power_vals
                )
            }
        except Exception as exc:
            logger.warning(f"Power SHAP failed for {clause_id}: {exc}")

        return result

    def explain_contract(
        self, contract_id: str, max_clauses: int = 10
    ) -> List[Dict]:
        """Generate SHAP explanations for up to max_clauses clauses in a contract.

        Limited to max_clauses to keep generation time reasonable.

        Args:
            contract_id: Contract identifier.
            max_clauses: Maximum number of clauses to explain.

        Returns:
            List of explanation dicts (one per explained clause).
        """
        from api.database import Clause, SessionLocal, create_tables
        create_tables()

        with SessionLocal() as session:
            clauses = (
                session.query(Clause)
                .filter(Clause.contract_id == contract_id)
                .limit(max_clauses)
                .all()
            )

        explanations = []
        for clause in clauses:
            logger.info(f"Explaining clause {clause.clause_id}...")
            exp = self.explain_clause(
                clause_text=clause.clause_text,
                clause_id=clause.clause_id,
                clause_type=clause.clause_type.split("|")[0] if clause.clause_type else None,
            )

            # Persist SHAP plot paths to database
            with SessionLocal() as session:
                db_clause = session.get(Clause, clause.clause_id)
                if db_clause and exp.get("classifier_shap_path"):
                    db_clause.shap_plot_path = exp["classifier_shap_path"]
                    session.commit()

            exp["clause_id"] = clause.clause_id
            explanations.append(exp)

        return explanations
