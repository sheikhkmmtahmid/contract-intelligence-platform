"""
anomaly_detector.py — Dual-signal anomaly detection engine.

Two complementary detectors are trained on Legal-RoBERTa-large [CLS] embeddings
of market-standard clause language:

  1. Isolation Forest — tree-based, no labels needed, fast scoring
  2. Shallow Autoencoder — reconstruction error captures structural anomalies
     that Isolation Forest misses (complementary false-positive reduction)

Both scores are min-max normalised to [0, 100] and fused via weighted average.
Any clause with combined score > ANOMALY_FLAG_THRESHOLD (default 70) is flagged.

Architecture:
  Autoencoder: 1024 → 256 → 64 → 256 → 1024
  Activations: ReLU (hidden layers), no activation on output
  Loss:         MSELoss

Usage:
    python src/anomaly_detewhactor.py --mode train
    python src/anomaly_detector.py --mode score --clause_id <id>
    python src/anomaly_detector.py --mode evaluate
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from api.database import Clause, SessionLocal, create_tables
from src.data_pipeline import EmbeddingGenerator

logger.remove()
logger.add(config.LOGS_DIR / "anomaly_detector.log", rotation="10 MB", level="DEBUG")
logger.add(sys.stderr, level="INFO")


# 1. AUTOENCODER MODEL

class ShallowAutoencoder(nn.Module):
    """Shallow autoencoder for Legal-RoBERTa-large embedding reconstruction.

    Learns to reconstruct "normal" clause embeddings. At inference time,
    high reconstruction error (MSE) indicates the clause deviates from
    market-standard language patterns seen during training.

    Architecture:
        Encoder: 1024 → 256 → 64   (ReLU activations)
        Decoder: 64   → 256 → 1024 (ReLU hidden, linear output)

    Args:
        input_dim: Legal-RoBERTa-large [CLS] embedding dimension (1024).
        hidden_dims: Bottleneck sizes [256, 64] matching config.
    """

    def __init__(
        self,
        input_dim: int = config.EMBEDDING_DIM,
        hidden_dims: List[int] = config.AUTOENCODER_HIDDEN_DIMS,
    ):
        super().__init__()

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers += [nn.Linear(prev_dim, h_dim), nn.ReLU()]
            prev_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder (reverse the hidden dims)
        decoder_layers = []
        for h_dim in reversed(hidden_dims[:-1]):
            decoder_layers += [nn.Linear(prev_dim, h_dim), nn.ReLU()]
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))  # linear output
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode then decode the input embedding.

        Args:
            x: Input tensor of shape (batch, 768).

        Returns:
            Reconstructed tensor of shape (batch, 768).
        """
        return self.decoder(self.encoder(x))

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-sample MSE reconstruction error.

        Args:
            x: Input tensor of shape (batch, 768).

        Returns:
            MSE error tensor of shape (batch,).
        """
        recon = self.forward(x)
        return torch.mean((x - recon) ** 2, dim=1)


# 2. SCORE NORMALISER

class AnomalyScoreNormaliser:
    """Min-max normalises raw anomaly scores to [0, 100].

    Isolation Forest returns decision_function scores (lower = more anomalous).
    Autoencoder returns MSE (higher = more anomalous).
    Both are normalised to a common 0–100 scale where 100 = most anomalous.

    The normalisation parameters are fit on the training set and saved to
    disk via joblib for use at inference time.
    """

    def __init__(self):
        self.if_scaler  = MinMaxScaler()
        self.ae_scaler  = MinMaxScaler()
        self._fitted    = False

    def fit(
        self,
        if_scores: np.ndarray,
        ae_scores: np.ndarray,
    ) -> None:
        """Fit normalisation scalers on training-set anomaly scores.

        Args:
            if_scores: Raw Isolation Forest decision scores (1D array).
                       More negative = more anomalous.
            ae_scores: Raw autoencoder reconstruction errors (1D array).
                       Higher = more anomalous.
        """
        # Invert IF scores so higher = more anomalous, consistent with AE
        if_inverted = -if_scores.reshape(-1, 1)
        self.if_scaler.fit(if_inverted)
        self.ae_scaler.fit(ae_scores.reshape(-1, 1))
        self._fitted = True

    def transform(
        self,
        if_scores: np.ndarray,
        ae_scores: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Transform raw scores to 0–100 scale.

        Args:
            if_scores: Raw Isolation Forest scores.
            ae_scores: Raw autoencoder MSE scores.

        Returns:
            Tuple of (if_normalised, ae_normalised), each in [0, 100].
        """
        if not self._fitted:
            raise RuntimeError("Normaliser must be fit before transform.")
        if_norm = self.if_scaler.transform(-if_scores.reshape(-1, 1)).flatten() * 100
        ae_norm = self.ae_scaler.transform(ae_scores.reshape(-1, 1)).flatten() * 100
        return np.clip(if_norm, 0, 100), np.clip(ae_norm, 0, 100)

    def combined_score(
        self,
        if_scores: np.ndarray,
        ae_scores: np.ndarray,
    ) -> np.ndarray:
        """Fuse normalised scores via weighted average.

        Args:
            if_scores: Raw Isolation Forest scores.
            ae_scores: Raw autoencoder MSE scores.

        Returns:
            Combined anomaly risk scores in [0, 100].
        """
        if_norm, ae_norm = self.transform(if_scores, ae_scores)
        return (
            config.ANOMALY_IF_WEIGHT * if_norm
            + config.ANOMALY_AE_WEIGHT * ae_norm
        )


# 3. TRAINER

class AnomalyDetectorTrainer:
    """Trains Isolation Forest and Autoencoder on training-set embeddings.

    The training set is assumed to represent "normal" market-standard
    clause language. Both detectors learn the distribution of normal
    embeddings and flag deviations at inference time.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        lr_checkpoint = str(config.CLASSIFIER_LEGALROBERTA_DIR / "best")
        self.embedding_gen = EmbeddingGenerator(model_name=lr_checkpoint)

    def _load_train_embeddings(self) -> np.ndarray:
        """Load training clause texts and generate Legal-RoBERTa-large embeddings.

        Returns:
            NumPy array of shape (N, 1024).
        """
        csv_path = config.PROCESSED_DIR / "train.csv"
        if not csv_path.exists():
            raise FileNotFoundError("train.csv not found. Run data_pipeline.py first.")

        df = pd.read_csv(csv_path)
        texts = df["clause_text"].fillna("").tolist()
        logger.info(f"Generating embeddings for {len(texts)} training clauses...")
        embeddings = self.embedding_gen.embed(texts)

        # Cache embeddings for faster reruns
        cache_path = config.PROCESSED_DIR / "train_embeddings.npy"
        np.save(str(cache_path), embeddings)
        logger.info(f"Embeddings cached to {cache_path}")
        return embeddings

    def _load_cached_embeddings(self) -> Optional[np.ndarray]:
        """Load cached embeddings if available.

        Returns:
            NumPy array of embeddings or None if cache doesn't exist.
        """
        cache_path = config.PROCESSED_DIR / "train_embeddings.npy"
        if cache_path.exists():
            logger.info("Loading cached embeddings...")
            return np.load(str(cache_path))
        return None

    def train_isolation_forest(
        self, embeddings: np.ndarray
    ) -> IsolationForest:
        """Fit an Isolation Forest on training embeddings.

        Args:
            embeddings: NumPy array of shape (N, 768).

        Returns:
            Fitted IsolationForest instance.
        """
        logger.info("Training Isolation Forest...")
        iforest = IsolationForest(
            contamination=config.ISOLATION_FOREST_CONTAMINATION,
            n_estimators=200,
            max_samples="auto",
            random_state=config.RANDOM_SEED,
            n_jobs=-1,
        )
        iforest.fit(embeddings)
        return iforest

    def train_autoencoder(self, embeddings: np.ndarray) -> ShallowAutoencoder:
        """Train the shallow autoencoder on training embeddings.

        Args:
            embeddings: NumPy array of shape (N, 768).

        Returns:
            Trained ShallowAutoencoder instance.
        """
        logger.info("Training Shallow Autoencoder...")
        model = ShallowAutoencoder().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.AUTOENCODER_LR)
        criterion = nn.MSELoss()

        tensor = torch.tensor(embeddings, dtype=torch.float32)
        dataset = TensorDataset(tensor)
        loader = DataLoader(
            dataset,
            batch_size=config.AUTOENCODER_BATCH_SIZE,
            shuffle=True,
        )

        model.train()
        for epoch in range(1, config.AUTOENCODER_EPOCHS + 1):
            epoch_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(self.device)
                recon = model(batch)
                loss  = criterion(recon, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            if epoch % 5 == 0 or epoch == 1:
                logger.info(f"AE Epoch {epoch}/{config.AUTOENCODER_EPOCHS} — Loss: {avg_loss:.6f}")

        return model

    def train(self, use_cache: bool = True) -> Dict:
        """Full training pipeline: embed → fit IF → train AE → normalise → save.

        Args:
            use_cache: If True, reuse cached embeddings if available.

        Returns:
            Dict with training statistics.
        """
        # Load or generate embeddings
        embeddings = None
        if use_cache:
            embeddings = self._load_cached_embeddings()
        if embeddings is None:
            embeddings = self._load_train_embeddings()

        # Train both detectors
        iforest = self.train_isolation_forest(embeddings)
        autoencoder = self.train_autoencoder(embeddings)

        # Compute training-set scores for normalisation
        if_scores = iforest.decision_function(embeddings)
        ae_scores  = self._compute_ae_scores(autoencoder, embeddings)

        normaliser = AnomalyScoreNormaliser()
        normaliser.fit(if_scores, ae_scores)

        combined = normaliser.combined_score(if_scores, ae_scores)
        n_flagged = int((combined > config.ANOMALY_FLAG_THRESHOLD).sum())

        logger.info(f"Training-set anomaly stats:")
        logger.info(f"  Mean combined score: {combined.mean():.2f}")
        logger.info(f"  Flagged anomalous:   {n_flagged} / {len(embeddings)}")

        # Persist all components
        self._save_all(iforest, autoencoder, normaliser)

        return {
            "n_training_samples": len(embeddings),
            "n_flagged_in_train": n_flagged,
            "mean_combined_score": float(combined.mean()),
            "std_combined_score":  float(combined.std()),
        }

    @torch.no_grad()
    def _compute_ae_scores(
        self, model: ShallowAutoencoder, embeddings: np.ndarray
    ) -> np.ndarray:
        """Compute reconstruction errors for a set of embeddings.

        Args:
            model: Trained ShallowAutoencoder.
            embeddings: NumPy array of shape (N, 768).

        Returns:
            MSE reconstruction errors as a 1D NumPy array.
        """
        model.eval()
        tensor = torch.tensor(embeddings, dtype=torch.float32).to(self.device)
        errors = model.reconstruction_error(tensor)
        return errors.cpu().numpy()

    def _save_all(
        self,
        iforest: IsolationForest,
        autoencoder: ShallowAutoencoder,
        normaliser: AnomalyScoreNormaliser,
    ) -> None:
        """Persist all trained components to the anomaly_detector directory.

        Args:
            iforest: Fitted Isolation Forest.
            autoencoder: Trained autoencoder.
            normaliser: Fitted score normaliser.
        """
        config.ANOMALY_DIR.mkdir(parents=True, exist_ok=True)

        # Isolation Forest via pickle
        with open(config.ANOMALY_DIR / "isolation_forest.pkl", "wb") as f:
            pickle.dump(iforest, f)

        # Autoencoder state dict
        torch.save(
            autoencoder.state_dict(),
            config.ANOMALY_DIR / "autoencoder.pt",
        )

        # Normaliser via pickle
        with open(config.ANOMALY_DIR / "normaliser.pkl", "wb") as f:
            pickle.dump(normaliser, f)

        logger.info(f"All anomaly detector components saved to {config.ANOMALY_DIR}")


# 4. INFERENCE INTERFACE

class AnomalyDetectorInference:
    """Loads trained anomaly detector components and scores new clauses.

    Provides both batch and single-clause scoring interfaces.
    Loads models lazily on first call for memory efficiency.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._iforest:    Optional[IsolationForest]      = None
        self._autoencoder: Optional[ShallowAutoencoder]  = None
        self._normaliser:  Optional[AnomalyScoreNormaliser] = None
        self._emb_gen:     Optional[EmbeddingGenerator]  = None

    def _ensure_loaded(self) -> None:
        """Lazy-load all model components on first use."""
        if self._iforest is not None:
            return

        if_path   = config.ANOMALY_DIR / "isolation_forest.pkl"
        ae_path   = config.ANOMALY_DIR / "autoencoder.pt"
        norm_path = config.ANOMALY_DIR / "normaliser.pkl"

        for p in [if_path, ae_path, norm_path]:
            if not p.exists():
                raise FileNotFoundError(
                    f"Anomaly detector component not found: {p}. "
                    "Run: python src/anomaly_detector.py --mode train"
                )

        # Custom unpickler: normaliser.pkl and isolation_forest.pkl were saved
        # when anomaly_detector.py ran as __main__, so pickle stored classes as
        # __main__.AnomalyScoreNormaliser etc.  Remap to the correct module.
        class _Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                _remap = {
                    "AnomalyScoreNormaliser": AnomalyScoreNormaliser,
                    "ShallowAutoencoder":     ShallowAutoencoder,
                }
                if name in _remap:
                    return _remap[name]
                return super().find_class(module, name)

        with open(if_path, "rb") as f:
            self._iforest = _Unpickler(f).load()

        self._autoencoder = ShallowAutoencoder().to(self.device)
        self._autoencoder.load_state_dict(torch.load(ae_path, map_location=self.device, weights_only=True))
        self._autoencoder.eval()

        with open(norm_path, "rb") as f:
            self._normaliser = _Unpickler(f).load()

        self._emb_gen = EmbeddingGenerator(model_name=config.LEGAL_ROBERTA_MODEL)
        logger.info("Anomaly detector components loaded successfully")

    def score(self, texts: List[str]) -> List[Dict]:
        """Compute anomaly scores for a list of clause texts.

        Args:
            texts: List of clause text strings.

        Returns:
            List of dicts, each with:
                'if_score': raw Isolation Forest decision score
                'ae_score': raw autoencoder MSE
                'if_normalised': IF score on 0-100 scale
                'ae_normalised': AE score on 0-100 scale
                'combined_score': fused anomaly risk score (0-100)
                'is_anomalous': bool flag
        """
        self._ensure_loaded()

        # Generate embeddings
        embeddings = self._emb_gen.embed(texts)

        # Isolation Forest scores
        if_scores = self._iforest.decision_function(embeddings)

        # Autoencoder reconstruction errors
        with torch.no_grad():
            tensor = torch.tensor(embeddings, dtype=torch.float32).to(self.device)
            ae_scores = self._autoencoder.reconstruction_error(tensor).cpu().numpy()

        # Normalise and fuse
        if_norm, ae_norm = self._normaliser.transform(if_scores, ae_scores)
        combined = self._normaliser.combined_score(if_scores, ae_scores)

        results = []
        for i in range(len(texts)):
            results.append(
                {
                    "if_score":       float(if_scores[i]),
                    "ae_score":       float(ae_scores[i]),
                    "if_normalised":  float(if_norm[i]),
                    "ae_normalised":  float(ae_norm[i]),
                    "combined_score": float(combined[i]),
                    "is_anomalous":   bool(combined[i] > config.ANOMALY_FLAG_THRESHOLD),
                }
            )
        return results

    def score_single(self, text: str) -> Dict:
        """Score a single clause text.

        Args:
            text: Clause text string.

        Returns:
            Anomaly score dict for the single clause.
        """
        return self.score([text])[0]

    def score_and_store(self, contract_id: str) -> List[Dict]:
        """Score all unscored clauses for a contract and persist results.

        Args:
            contract_id: Contract identifier to score.

        Returns:
            List of score dicts for all clauses in the contract.
        """
        self._ensure_loaded()
        create_tables()

        with SessionLocal() as session:
            clauses = (
                session.query(Clause)
                .filter(
                    Clause.contract_id == contract_id,
                    Clause.anomaly_score.is_(None),
                )
                .all()
            )

        if not clauses:
            logger.info(f"No unscored clauses found for contract {contract_id}")
            return []

        texts = [c.clause_text for c in clauses]
        scores = self.score(texts)

        with SessionLocal() as session:
            for clause, score_dict in zip(clauses, scores):
                db_clause = session.get(Clause, clause.clause_id)
                if db_clause:
                    db_clause.anomaly_score = score_dict["combined_score"]
                    db_clause.is_anomalous  = score_dict["is_anomalous"]
            session.commit()

        logger.info(
            f"Scored {len(clauses)} clauses for contract {contract_id}. "
            f"Anomalous: {sum(1 for s in scores if s['is_anomalous'])}"
        )
        return scores


# 5. EVALUATOR

class AnomalyDetectorEvaluator:
    """Evaluates anomaly detection on test set using precision@k and recall@k.

    Since CUAD has no ground-truth anomaly labels, we treat clauses with
    unusual structure (very short, entirely numeric, or duplicate patterns)
    as proxy anomalies and report detection rates.
    """

    def __init__(self):
        self.inference = AnomalyDetectorInference()

    def evaluate(self, k: int = 50) -> Dict:
        """Compute precision@k and recall@k using proxy anomaly labels.

        Args:
            k: Top-k anomalous clauses to evaluate.

        Returns:
            Dict with precision_at_k, recall_at_k, and k.
        """
        csv_path = config.PROCESSED_DIR / "test.csv"
        if not csv_path.exists():
            raise FileNotFoundError("test.csv not found.")

        df = pd.read_csv(csv_path).fillna("")
        texts = df["clause_text"].tolist()

        logger.info(f"Scoring {len(texts)} test clauses for anomaly evaluation...")
        scores = self.inference.score(texts)
        combined = np.array([s["combined_score"] for s in scores])

        # Proxy: flag clauses with < 15 words or > 95th percentile length as anomalous
        word_counts = np.array([len(t.split()) for t in texts])
        p95 = np.percentile(word_counts, 95)
        proxy_labels = ((word_counts < 15) | (word_counts > p95)).astype(int)

        # Rank by combined score (descending) and check top-k
        top_k_idx = np.argsort(combined)[::-1][:k]
        top_k_labels = proxy_labels[top_k_idx]

        precision_at_k = top_k_labels.mean()
        total_anomalies = proxy_labels.sum()
        recall_at_k = top_k_labels.sum() / max(total_anomalies, 1)

        results = {
            "anomaly_detector": {
                "precision_at_k": float(precision_at_k),
                "recall_at_k":    float(recall_at_k),
                "k":              k,
                "test_size":      len(texts),
                "proxy_anomalies": int(total_anomalies),
                "note": (
                    "Evaluated using proxy anomaly labels (extreme clause length). "
                    "No ground-truth anomaly labels exist in CUAD."
                ),
            }
        }

        existing = {}
        if config.EVAL_RESULTS_PATH.exists():
            with open(config.EVAL_RESULTS_PATH) as f:
                existing = json.load(f)
        existing.update(results)
        with open(config.EVAL_RESULTS_PATH, "w") as f:
            json.dump(existing, f, indent=2)

        logger.info(f"Anomaly Precision@{k}: {precision_at_k:.4f}")
        logger.info(f"Anomaly Recall@{k}:    {recall_at_k:.4f}")
        return results


# CLI ENTRY POINT

def main():
    """Command-line interface for anomaly detector training and scoring."""
    parser = argparse.ArgumentParser(description="Contract Anomaly Detector")
    parser.add_argument("--mode", choices=["train", "score", "evaluate"], required=True)
    parser.add_argument("--contract_id", type=str, help="Contract ID to score")
    parser.add_argument("--use_cache", action="store_true", default=True)
    args = parser.parse_args()

    if args.mode == "train":
        trainer = AnomalyDetectorTrainer()
        stats = trainer.train(use_cache=args.use_cache)
        print(f"\nTraining complete:")
        print(f"  Training samples: {stats['n_training_samples']}")
        print(f"  Flagged in train: {stats['n_flagged_in_train']}")
        print(f"  Mean score:       {stats['mean_combined_score']:.2f}")

    elif args.mode == "score":
        if not args.contract_id:
            parser.error("--contract_id required for --mode score")
        inference = AnomalyDetectorInference()
        scores = inference.score_and_store(args.contract_id)
        n_anomalous = sum(1 for s in scores if s["is_anomalous"])
        print(f"\nScored {len(scores)} clauses. Anomalous: {n_anomalous}")

    elif args.mode == "evaluate":
        evaluator = AnomalyDetectorEvaluator()
        results = evaluator.evaluate()
        r = results["anomaly_detector"]
        print(f"\nPrecision@{r['k']}: {r['precision_at_k']:.4f}")
        print(f"Recall@{r['k']}:    {r['recall_at_k']:.4f}")


if __name__ == "__main__":
    main()
