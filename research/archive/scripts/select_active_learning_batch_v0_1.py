from __future__ import annotations

import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


PROJECT_ROOT = Path(__file__).resolve().parents[1]

POOL_PATH = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "industrial_safety_annotation_pool.csv"
)

RESOLVED_PATH = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "resolved_annotations_v0.1.csv"
)

EXPERIMENT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "baseline_transformer_embedding_v0.1"
)

HYBRID_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "hybrid_tfidf_transformer_v0.1"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "active_learning_v0.1"
)

OUTPUT_PATH = OUTPUT_DIR / "active_learning_batch_01.csv"
SCORES_PATH = OUTPUT_DIR / "active_learning_ranked_pool.csv"
SUMMARY_PATH = OUTPUT_DIR / "active_learning_summary.json"

TRANSFORMER_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

RANDOM_STATE = 42
BATCH_SIZE = 8
MAX_SEQ_LENGTH = 256

TARGET_SIZE = 30

# Ranking weights. Disagreement and uncertainty are deliberately
# emphasized over any single model's confidence.
WEIGHT_DISAGREEMENT = 0.45
WEIGHT_UNCERTAINTY = 0.35
WEIGHT_MODEL_SPREAD = 0.20


def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_text(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def find_column(
    df: pd.DataFrame,
    candidates: list[str],
    description: str,
) -> str:
    for column in candidates:
        if column in df.columns:
            return column

    raise ValueError(
        f"Could not find {description}. "
        f"Tried {candidates}. Available columns: {list(df.columns)}"
    )


def load_pool() -> pd.DataFrame:
    if not POOL_PATH.exists():
        raise FileNotFoundError(
            f"Annotation pool not found:\n{POOL_PATH}"
        )

    df = pd.read_csv(POOL_PATH)

    report_id_column = find_column(
        df,
        ["report_id", "id"],
        "report ID column",
    )

    description_column = find_column(
        df,
        ["description", "Description"],
        "narrative/description column",
    )

    out = df.copy()

    if report_id_column != "report_id":
        out = out.rename(
            columns={report_id_column: "report_id"}
        )

    if description_column != "description":
        out = out.rename(
            columns={description_column: "description"}
        )

    out["report_id"] = (
        out["report_id"].astype(str).str.strip()
    )
    out["description"] = normalize_text(
        out["description"]
    )

    out = out[out["description"].ne("")].copy()

    # One narrative per report ID.
    out = out.drop_duplicates(
        subset=["report_id"],
        keep="first",
    ).reset_index(drop=True)

    # Remove exact duplicate narratives as an additional safety
    # measure. The first record is kept to retain the original ID.
    out = out.drop_duplicates(
        subset=["description"],
        keep="first",
    ).reset_index(drop=True)

    return out


def load_resolved_ids() -> set[str]:
    if not RESOLVED_PATH.exists():
        raise FileNotFoundError(
            f"Resolved annotation file not found:\n{RESOLVED_PATH}"
        )

    df = pd.read_csv(RESOLVED_PATH)

    report_id_column = find_column(
        df,
        ["report_id", "id"],
        "resolved report ID column",
    )

    ids = (
        df[report_id_column]
        .astype(str)
        .str.strip()
    )

    return set(ids)


def load_locked_test_ids() -> set[str]:
    """Also exclude the existing locked split IDs when available."""
    split_dir = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "sif_splits_v0.1"
    )

    ids: set[str] = set()

    for filename in [
        "train.csv",
        "validation.csv",
        "test.csv",
    ]:
        path = split_dir / filename

        if not path.exists():
            continue

        df = pd.read_csv(path)

        if "report_id" in df.columns:
            ids.update(
                df["report_id"]
                .astype(str)
                .str.strip()
            )

    return ids


def load_encoder() -> SentenceTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder = SentenceTransformer(
        TRANSFORMER_MODEL,
        device=device,
    )
    encoder.max_seq_length = MAX_SEQ_LENGTH

    return encoder


def encode(
    encoder: SentenceTransformer,
    texts: list[str],
) -> np.ndarray:
    return encoder.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)


def minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)

    lo = float(np.min(values))
    hi = float(np.max(values))

    if hi - lo < 1e-12:
        return np.zeros_like(values)

    return (values - lo) / (hi - lo)


def safe_probability_from_model(
    classifier: LogisticRegression,
    X,
) -> np.ndarray:
    return classifier.predict_proba(X)[:, 1]


def fit_full_models(
    texts: list[str],
    labels: np.ndarray,
    embeddings: np.ndarray,
) -> tuple[
    TfidfVectorizer,
    LogisticRegression,
    LogisticRegression,
]:
    # -----------------------------
    # TF-IDF model
    # -----------------------------
    tfidf = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.98,
        sublinear_tf=True,
        max_features=10000,
    )

    X_tfidf = tfidf.fit_transform(texts)

    tfidf_classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_STATE,
        solver="liblinear",
    )

    tfidf_classifier.fit(
        X_tfidf,
        labels,
    )

    # -----------------------------
    # Transformer embedding model
    # -----------------------------
    embedding_classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_STATE,
        solver="liblinear",
    )

    embedding_classifier.fit(
        embeddings,
        labels,
    )

    return (
        tfidf,
        tfidf_classifier,
        embedding_classifier,
    )


def main() -> None:
    set_seed()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pool = load_pool()

    resolved_ids = load_resolved_ids()
    locked_ids = load_locked_test_ids()

    excluded_ids = resolved_ids | locked_ids

    candidates = pool[
        ~pool["report_id"].isin(excluded_ids)
    ].copy()

    if candidates.empty:
        raise RuntimeError(
            "No unlabeled narratives remain after exclusions."
        )

    resolved = pd.read_csv(RESOLVED_PATH)

    label_column = find_column(
        resolved,
        [
            "final_sif_potential",
            "sif_potential",
            "resolved_sif_potential",
        ],
        "resolved SIF label column",
    )

    description_column = find_column(
        resolved,
        ["description", "Description"],
        "resolved description column",
    )

    resolved_train = resolved[
        [
            description_column,
            label_column,
        ]
    ].copy()

    resolved_train = resolved_train.rename(
        columns={
            description_column: "description",
            label_column: "label",
        }
    )

    resolved_train["description"] = normalize_text(
        resolved_train["description"]
    )
    resolved_train["label"] = (
        resolved_train["label"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    resolved_train = resolved_train[
        resolved_train["label"].isin(["YES", "NO"])
    ].copy()

    if len(resolved_train) < 10:
        raise RuntimeError(
            f"Only {len(resolved_train)} resolved binary "
            "annotations are available."
        )

    y = (
        resolved_train["label"]
        .map({"NO": 0, "YES": 1})
        .astype(int)
        .to_numpy()
    )

    train_texts = (
        resolved_train["description"]
        .tolist()
    )

    candidate_texts = (
        candidates["description"].tolist()
    )

    print("=" * 78)
    print("RAKSHAK ACTIVE LEARNING BATCH SELECTOR v0.1")
    print("=" * 78)
    print(f"Resolved labels        : {len(resolved_train)}")
    print(f"Resolved YES           : {int((y == 1).sum())}")
    print(f"Resolved NO            : {int((y == 0).sum())}")
    print(f"Original pool records  : {len(pool)}")
    print(f"Excluded resolved IDs  : {len(resolved_ids)}")
    print(f"Excluded split IDs     : {len(locked_ids)}")
    print(f"Candidate narratives   : {len(candidates)}")
    print(f"Target annotation size : {TARGET_SIZE}")
    print()

    # --------------------------------------------------------
    # Transformer embeddings
    # --------------------------------------------------------

    print("-" * 78)
    print("LOADING TRANSFORMER")
    print("-" * 78)

    encoder = load_encoder()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Model : {TRANSFORMER_MODEL}")
    print(f"Device: {device}")
    print()
    print("Encoding resolved narratives...")

    train_embeddings = encode(
        encoder,
        train_texts,
    )

    print("Encoding candidate narratives...")

    candidate_embeddings = encode(
        encoder,
        candidate_texts,
    )

    # --------------------------------------------------------
    # Fit models on all resolved labels.
    # These models are for candidate selection only.
    # --------------------------------------------------------

    (
        tfidf,
        tfidf_classifier,
        embedding_classifier,
    ) = fit_full_models(
        train_texts,
        y,
        train_embeddings,
    )

    candidate_tfidf = tfidf.transform(
        candidate_texts
    )

    tfidf_probability = safe_probability_from_model(
        tfidf_classifier,
        candidate_tfidf,
    )

    transformer_probability = safe_probability_from_model(
        embedding_classifier,
        candidate_embeddings,
    )

    # --------------------------------------------------------
    # Hybrid probability
    #
    # Do NOT pretend this is a calibrated production probability.
    # For selection, we use the mean of model probabilities as an
    # ensemble score.
    # --------------------------------------------------------

    hybrid_probability = (
        0.5 * tfidf_probability
        + 0.5 * transformer_probability
    )

    # Model disagreement: high when the two models disagree.
    disagreement = np.abs(
        tfidf_probability
        - transformer_probability
    )

    # Hybrid uncertainty: highest near 0.50.
    uncertainty = (
        1.0
        - 2.0 * np.abs(
            hybrid_probability - 0.50
        )
    )

    uncertainty = np.clip(
        uncertainty,
        0.0,
        1.0,
    )

    # Candidate spread is another useful measure:
    # how far the strongest and weakest model score diverge.
    model_spread = disagreement.copy()

    # --------------------------------------------------------
    # Ranking score
    # --------------------------------------------------------

    ranking_score = (
        WEIGHT_DISAGREEMENT * minmax(disagreement)
        + WEIGHT_UNCERTAINTY * minmax(uncertainty)
        + WEIGHT_MODEL_SPREAD * minmax(model_spread)
    )

    scored = candidates[
        ["report_id", "description"]
    ].copy()

    # Retain original pool metadata where available.
    for column in [
        "Data",
        "Countries",
        "Local",
        "Industry Sector",
        "Accident Level",
        "Potential Accident Level",
        "Genre",
        "Employee or Third Party",
        "Critical Risk",
    ]:
        if column in candidates.columns:
            scored[column] = candidates[column].values

    scored["tfidf_probability"] = tfidf_probability
    scored["transformer_probability"] = (
        transformer_probability
    )
    scored["hybrid_probability"] = hybrid_probability
    scored["model_disagreement"] = disagreement
    scored["boundary_uncertainty"] = uncertainty
    scored["ranking_score"] = ranking_score

    # Human-readable selection reason.
    reasons = []

    for i in range(len(scored)):
        tfidf_p = float(tfidf_probability[i])
        transformer_p = float(transformer_probability[i])
        hybrid_p = float(hybrid_probability[i])
        disagreement_i = float(disagreement[i])

        if (
            tfidf_p >= 0.50
            and transformer_p < 0.50
        ):
            reason = (
                "TF-IDF YES / Transformer NO disagreement"
            )
        elif (
            tfidf_p < 0.50
            and transformer_p >= 0.50
        ):
            reason = (
                "TF-IDF NO / Transformer YES disagreement"
            )
        elif abs(hybrid_p - 0.50) <= 0.08:
            reason = (
                "Near decision boundary"
            )
        elif disagreement_i >= 0.20:
            reason = (
                "Strong model disagreement"
            )
        else:
            reason = (
                "High combined uncertainty"
            )

        reasons.append(reason)

    scored["selection_reason"] = reasons

    scored = scored.sort_values(
        [
            "ranking_score",
            "model_disagreement",
            "boundary_uncertainty",
        ],
        ascending=False,
    ).reset_index(drop=True)

    scored["active_learning_rank"] = (
        np.arange(len(scored)) + 1
    )

    # --------------------------------------------------------
    # Stratified selection to avoid getting 30 near-identical
    # narratives. We first take high-scoring disagreements, then
    # fill with high-uncertainty cases while limiting exact
    # Critical Risk duplication when that metadata exists.
    # --------------------------------------------------------

    selected_rows = []

    max_same_risk = 4

    critical_risk_column = (
        "Critical Risk"
        if "Critical Risk" in scored.columns
        else None
    )

    risk_counts: dict[str, int] = {}

    # Pass 1: prioritize strong model disagreement.
    disagreement_candidates = scored.sort_values(
        "model_disagreement",
        ascending=False,
    )

    for _, row in disagreement_candidates.iterrows():
        if len(selected_rows) >= TARGET_SIZE:
            break

        if critical_risk_column is not None:
            risk = str(
                row[critical_risk_column]
            ).strip()

            if risk_counts.get(risk, 0) >= max_same_risk:
                continue
        else:
            risk = ""

        selected_rows.append(row)
        risk_counts[risk] = (
            risk_counts.get(risk, 0) + 1
        )

    # Pass 2: fill with overall ranking.
    if len(selected_rows) < TARGET_SIZE:
        selected_ids = {
            str(row["report_id"])
            for row in selected_rows
        }

        for _, row in scored.iterrows():
            if len(selected_rows) >= TARGET_SIZE:
                break

            report_id = str(row["report_id"])

            if report_id in selected_ids:
                continue

            if critical_risk_column is not None:
                risk = str(
                    row[critical_risk_column]
                ).strip()

                if risk_counts.get(risk, 0) >= max_same_risk:
                    continue
            else:
                risk = ""

            selected_rows.append(row)
            selected_ids.add(report_id)
            risk_counts[risk] = (
                risk_counts.get(risk, 0) + 1
            )

    # Pass 3: if the risk cap prevented filling the batch,
    # take remaining highest-ranked candidates.
    if len(selected_rows) < TARGET_SIZE:
        selected_ids = {
            str(row["report_id"])
            for row in selected_rows
        }

        for _, row in scored.iterrows():
            if len(selected_rows) >= TARGET_SIZE:
                break

            report_id = str(row["report_id"])

            if report_id in selected_ids:
                continue

            selected_rows.append(row)
            selected_ids.add(report_id)

    batch = pd.DataFrame(
        selected_rows
    ).reset_index(drop=True)

    batch["batch_id"] = "active_learning_batch_01"

    # Reorder important columns first.
    front_columns = [
        "batch_id",
        "active_learning_rank",
        "report_id",
        "description",
        "tfidf_probability",
        "transformer_probability",
        "hybrid_probability",
        "model_disagreement",
        "boundary_uncertainty",
        "ranking_score",
        "selection_reason",
    ]

    ordered_columns = [
        column
        for column in front_columns
        if column in batch.columns
    ]

    ordered_columns.extend(
        [
            column
            for column in batch.columns
            if column not in ordered_columns
        ]
    )

    batch = batch[ordered_columns]

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    batch.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    scored.to_csv(
        SCORES_PATH,
        index=False,
        encoding="utf-8",
    )

    top_reason_counts = (
        batch["selection_reason"]
        .value_counts()
        .to_dict()
    )

    summary = {
        "experiment": "active_learning_v0.1",
        "model": {
            "transformer": TRANSFORMER_MODEL,
            "tfidf": "word unigrams+bigrams",
            "classifier": "LogisticRegression",
            "device": device,
        },
        "resolved_records": int(len(resolved_train)),
        "resolved_yes": int((y == 1).sum()),
        "resolved_no": int((y == 0).sum()),
        "original_pool_records": int(len(pool)),
        "excluded_resolved_ids": int(len(resolved_ids)),
        "excluded_locked_split_ids": int(len(locked_ids)),
        "candidate_records": int(len(candidates)),
        "selected_records": int(len(batch)),
        "selection_weights": {
            "disagreement": WEIGHT_DISAGREEMENT,
            "uncertainty": WEIGHT_UNCERTAINTY,
            "model_spread": WEIGHT_MODEL_SPREAD,
        },
        "risk_cap": max_same_risk,
        "selection_reason_counts": {
            str(key): int(value)
            for key, value in top_reason_counts.items()
        },
        "warning": (
            "Scores are model-selection signals, not calibrated "
            "SIF probabilities. Human annotation remains the source "
            "of truth."
        ),
    }

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Console
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("ACTIVE LEARNING BATCH SELECTED")
    print("=" * 78)
    print(f"Candidates scored : {len(scored)}")
    print(f"Selected batch    : {len(batch)}")
    print()
    print(
        batch[
            [
                "active_learning_rank",
                "report_id",
                "tfidf_probability",
                "transformer_probability",
                "hybrid_probability",
                "model_disagreement",
                "boundary_uncertainty",
                "ranking_score",
                "selection_reason",
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 78)
    print("FILES")
    print("=" * 78)
    print(f"Annotation batch : {OUTPUT_PATH}")
    print(f"Full ranking     : {SCORES_PATH}")
    print(f"Summary          : {SUMMARY_PATH}")

    print()
    print("=" * 78)
    print("ACTIVE LEARNING SELECTION COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
