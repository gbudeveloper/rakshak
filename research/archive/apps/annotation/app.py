from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "annotations"

CANDIDATE_PATH = ANNOTATIONS_DIR / "gold_set_candidates.csv"
GOLD_PATH = ANNOTATIONS_DIR / "gold_set_v0.1.csv"

CALIBRATION_02_PATH = ANNOTATIONS_DIR / "calibration_batch_02.csv"
CALIBRATION_02_RESULT_PATH = ANNOTATIONS_DIR / "calibration_batch_02_annotated.csv"

CALIBRATION_03_PATH = ANNOTATIONS_DIR / "calibration_batch_03.csv"

CALIBRATION_03_RESULT_PATH = ANNOTATIONS_DIR / "calibration_batch_03_annotated.csv"

ACTIVE_LEARNING_DIR = ANNOTATIONS_DIR / "active_learning_v0.1"
ACTIVE_LEARNING_PATH = ACTIVE_LEARNING_DIR / "active_learning_batch_01_annotation.csv"
ACTIVE_LEARNING_RESULT_PATH = (
    ACTIVE_LEARNING_DIR / "active_learning_batch_01_annotated.csv"
)


# ============================================================
# Configuration
# ============================================================

st.set_page_config(
    page_title="RAKSHAK SIF Annotation",
    page_icon="🛡️",
    layout="wide",
)


# ============================================================
# Required columns
# ============================================================

REQUIRED_COLUMNS = {
    "report_id",
    "description",
    "department",
    "site",
    "location",
    "hazards",
}


ANNOTATION_DEFAULTS = {
    "annotation_status": "pending",
    "sif_potential": "",
    "sif_reason": "",
    "evidence_text": "",
    "hazard_notes": "",
    "exposure_notes": "",
    "consequence_notes": "",
    "barrier_notes": "",
    "annotator_confidence": "",
    "annotation_version": "v0.1",
    "annotation_round": "pilot_01",
    "annotator_id": "",
    "annotated_at_utc": "",
}


# ============================================================
# Helpers
# ============================================================


def load_dataset(path: Path) -> pd.DataFrame:
    """Load an annotation dataset and ensure required columns exist."""

    if not path.exists():
        raise FileNotFoundError(f"Annotation dataset not found:\n{path}")

    df = pd.read_csv(path)

    for column, default in ANNOTATION_DEFAULTS.items():
        if column not in df.columns:
            df[column] = default

    missing = REQUIRED_COLUMNS.difference(df.columns)

    if missing:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(sorted(missing))
        )

    return df


def save_dataset(df: pd.DataFrame) -> None:
    """Persist annotations to the selected batch output file."""

    output_path: Path = st.session_state["annotation_output_path"]

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = output_path.with_suffix(".tmp.csv")

    df.to_csv(
        temp_path,
        index=False,
        encoding="utf-8",
    )

    temp_path.replace(output_path)


def value_as_text(value: object) -> str:
    """Safely display scalar/list values."""

    if isinstance(value, list):
        return ", ".join(str(x) for x in value)

    if pd.isna(value):
        return ""

    return str(value)


def is_completed(row: pd.Series) -> bool:
    return value_as_text(row.get("annotation_status", "")).lower() == "completed"


def get_next_pending_index(
    df: pd.DataFrame,
    current_index: int,
) -> int | None:

    indices = list(df.index)

    for idx in indices:
        if idx <= current_index:
            continue

        if not is_completed(df.loc[idx]):
            return idx

    for idx in indices:
        if idx >= current_index:
            continue

        if not is_completed(df.loc[idx]):
            return idx

    return None


def get_previous_index(
    df: pd.DataFrame,
    current_index: int,
) -> int | None:

    indices = list(df.index)

    for idx in reversed(indices):
        if idx >= current_index:
            continue

        return idx

    if indices:
        return indices[-1]

    return None


def reset_record_state(batch_name: str) -> None:
    """Reset record navigation when the user changes annotation batches."""

    state_key = "active_annotation_batch"

    previous_batch = st.session_state.get(state_key)

    if previous_batch != batch_name:
        st.session_state[state_key] = batch_name

        for key in [
            "record_index",
            "record_selector",
        ]:
            st.session_state.pop(key, None)


def initialize_state(df: pd.DataFrame) -> None:

    if df.empty:
        raise ValueError("The selected annotation dataset is empty.")

    if "record_index" not in st.session_state:
        pending = [idx for idx in df.index if not is_completed(df.loc[idx])]

        st.session_state.record_index = pending[0] if pending else df.index[0]

    if "annotator_id" not in st.session_state:
        st.session_state.annotator_id = "annotator_01"


# ============================================================
# Dataset selection
# ============================================================

st.sidebar.header("Annotation batch")

batch_options = {
    "Gold Set v0.1": {
        "input": GOLD_PATH,
        "output": GOLD_PATH,
        "round": "pilot_01",
    },
    "Calibration Batch 02": {
        "input": CALIBRATION_02_PATH,
        "output": CALIBRATION_02_RESULT_PATH,
        "round": "pilot_02",
    },
    "Calibration Batch 03": {
        "input": CALIBRATION_03_PATH,
        "output": CALIBRATION_03_RESULT_PATH,
        "round": "v0.2_calibration",
    },
    "Active Learning Batch 01": {
        "input": ACTIVE_LEARNING_PATH,
        "output": ACTIVE_LEARNING_RESULT_PATH,
        "round": "active_learning_01",
    },
}

selected_batch = st.sidebar.selectbox(
    "Dataset",
    options=list(batch_options.keys()),
    key="annotation_batch_selector",
)

batch_config = batch_options[selected_batch]

input_path: Path = batch_config["input"]
output_path: Path = batch_config["output"]
annotation_round: str = batch_config["round"]

reset_record_state(selected_batch)

st.session_state["annotation_output_path"] = output_path
st.session_state["annotation_round"] = annotation_round


# ============================================================
# Load or initialize selected dataset
# ============================================================

try:
    # Calibration batches use separate persistent working files.
    if selected_batch != "Gold Set v0.1" and not output_path.exists():
        source_df = load_dataset(input_path)

        source_df["annotation_round"] = annotation_round

        if selected_batch == "Calibration Batch 03":
            source_df["annotation_version"] = "v0.2"
        elif selected_batch == "Active Learning Batch 01":
            source_df["annotation_version"] = "v0.3"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        source_df.to_csv(
            output_path,
            index=False,
            encoding="utf-8",
        )

    df = load_dataset(output_path)

    # Correct historical/default annotation-round values.
    # Do not overwrite completed records.
    if "annotation_round" not in df.columns:
        df["annotation_round"] = annotation_round

    df.loc[
        df["annotation_status"].fillna("").astype(str).str.strip().eq("pending"),
        "annotation_round",
    ] = annotation_round

    if selected_batch == "Calibration Batch 03":
        df.loc[
            df["annotation_status"].fillna("").astype(str).str.strip().eq("pending"),
            "annotation_version",
        ] = "v0.2"

    if selected_batch == "Active Learning Batch 01":
        df.loc[
            df["annotation_status"].fillna("").astype(str).str.strip().eq("pending"),
            "annotation_version",
        ] = "v0.3"

except Exception as exc:
    st.error(str(exc))
    st.stop()


initialize_state(df)


# ============================================================
# Header
# ============================================================

st.title("🛡️ RAKSHAK — SIF Annotation")

st.caption(f"Blind human annotation • {selected_batch} • {annotation_round}")


# ============================================================
# Progress
# ============================================================

completed_count = int(
    df["annotation_status"].fillna("").astype(str).str.lower().eq("completed").sum()
)

total_count = len(df)

progress = completed_count / total_count if total_count else 0

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Completed",
        f"{completed_count} / {total_count}",
    )

with col2:
    st.metric(
        "Remaining",
        total_count - completed_count,
    )

with col3:
    st.progress(
        progress,
        text=f"{progress:.0%}",
    )


# ============================================================
# Sidebar settings / rubric
# ============================================================

with st.sidebar:

    st.header("Annotation settings")

    annotator_id = st.text_input(
        "Annotator ID",
        value=st.session_state.annotator_id,
        key="annotator_id_input",
    )

    st.session_state.annotator_id = annotator_id.strip() or "annotator_01"

    st.divider()

    st.header("Decision rubric")

    st.markdown("""
**YES**

Credible serious/fatal potential exists.

**NO**

Evidence is sufficient to conclude serious/fatal potential is not credible.

**UNCERTAIN**

The report does not contain enough evidence for a defensible YES/NO decision.
""")

    st.divider()

    st.info(
        "Accident Level and Potential Accident Level are intentionally "
        "hidden during annotation to reduce label leakage."
    )

    if selected_batch == "Calibration Batch 03":
        st.success("Protocol v0.2 is active for this calibration batch.")
    if selected_batch == "Active Learning Batch 01":
        st.success(
            "Protocol v0.3 is active. Model scores are hidden during annotation."
        )


# ============================================================
# Current record
# ============================================================

current_index = st.session_state.record_index

if current_index not in df.index:
    st.session_state.record_index = df.index[0]
    st.rerun()

row = df.loc[current_index]


# ============================================================
# Record navigation
# ============================================================

nav_left, nav_middle, nav_right = st.columns([1, 4, 1])

with nav_left:

    if st.button(
        "← Previous",
        use_container_width=True,
        key="previous_record_button",
    ):
        previous = get_previous_index(
            df,
            current_index,
        )

        if previous is not None:
            st.session_state.record_index = previous
            st.rerun()


with nav_middle:

    indices = df.index.tolist()

    selected = st.selectbox(
        "Current report",
        options=indices,
        index=indices.index(current_index),
        format_func=lambda idx: (
            f"{df.loc[idx, 'report_id']} — "
            f"{'DONE' if is_completed(df.loc[idx]) else 'PENDING'}"
        ),
        key=f"record_selector_{selected_batch}",
    )

    if selected != current_index:
        st.session_state.record_index = selected
        st.rerun()


with nav_right:

    if st.button(
        "Next →",
        use_container_width=True,
        key="next_record_button",
    ):
        next_index = get_next_pending_index(
            df,
            current_index,
        )

        if next_index is not None:
            st.session_state.record_index = next_index
            st.rerun()


# ============================================================
# Report context
# ============================================================

st.divider()

st.subheader(f"Report: {row['report_id']}")

context_left, context_right = st.columns(2)

with context_left:

    st.markdown(f"**Industry:** {value_as_text(row['department'])}")

    st.markdown(f"**Site:** {value_as_text(row['site'])}")

    st.markdown(f"**Location:** {value_as_text(row['location'])}")


with context_right:

    st.markdown(f"**Risk / hazard:** {value_as_text(row['hazards'])}")


st.subheader("Safety report narrative")

st.info(value_as_text(row["description"]))


# ============================================================
# Existing annotation values
# ============================================================

existing_label = value_as_text(row.get("sif_potential", ""))

existing_reason = value_as_text(row.get("sif_reason", ""))

existing_evidence = value_as_text(row.get("evidence_text", ""))

existing_hazard = value_as_text(row.get("hazard_notes", ""))

existing_exposure = value_as_text(row.get("exposure_notes", ""))

existing_consequence = value_as_text(row.get("consequence_notes", ""))

existing_barrier = value_as_text(row.get("barrier_notes", ""))

existing_confidence = value_as_text(row.get("annotator_confidence", ""))


# ============================================================
# Annotation form
# ============================================================

st.divider()

st.subheader("SIF assessment")

label_options = [
    "YES",
    "NO",
    "UNCERTAIN",
]

label = st.radio(
    "SIF potential",
    options=label_options,
    index=(
        label_options.index(existing_label) if existing_label in label_options else None
    ),
    horizontal=True,
    key=f"sif_label_{selected_batch}_{current_index}",
)

reason = st.text_area(
    "Why?",
    value=existing_reason,
    height=100,
    placeholder=("Explain briefly why the narrative supports " "your decision."),
    key=f"sif_reason_{selected_batch}_{current_index}",
)

evidence = st.text_area(
    "Evidence from the report",
    value=existing_evidence,
    height=130,
    placeholder=("Copy the exact sentence or phrase that supports " "your assessment."),
    key=f"evidence_text_{selected_batch}_{current_index}",
)

annotation_left, annotation_right = st.columns(2)

with annotation_left:

    hazard_notes = st.text_area(
        "Hazard / energy",
        value=existing_hazard,
        height=100,
        placeholder=("What hazardous energy, condition, or source is present?"),
        key=f"hazard_notes_{selected_batch}_{current_index}",
    )

    exposure_notes = st.text_area(
        "Exposure",
        value=existing_exposure,
        height=100,
        placeholder=("How was a person exposed to the hazard?"),
        key=f"exposure_notes_{selected_batch}_{current_index}",
    )

    consequence_notes = st.text_area(
        "Potential consequence",
        value=existing_consequence,
        height=100,
        placeholder=("What serious/fatal consequence could credibly occur?"),
        key=f"consequence_notes_{selected_batch}_{current_index}",
    )


with annotation_right:

    barrier_notes = st.text_area(
        "Barrier / control condition",
        value=existing_barrier,
        height=100,
        placeholder=(
            "Present, effective, degraded, absent, failed, " "bypassed, or unknown."
        ),
        key=f"barrier_notes_{selected_batch}_{current_index}",
    )

    confidence_options = [
        "",
        "HIGH",
        "MEDIUM",
        "LOW",
    ]

    confidence = st.selectbox(
        "Annotator confidence",
        options=confidence_options,
        index=(
            confidence_options.index(existing_confidence)
            if existing_confidence in confidence_options
            else 0
        ),
        key=f"annotator_confidence_{selected_batch}_{current_index}",
    )


# ============================================================
# Save
# ============================================================

st.divider()

save_col, next_col = st.columns(2)

with save_col:

    save_clicked = st.button(
        "💾 Save Annotation",
        type="primary",
        use_container_width=True,
        key="save_annotation_button",
    )


with next_col:

    save_next_clicked = st.button(
        "💾 Save & Next",
        use_container_width=True,
        key="save_next_button",
    )


if save_clicked or save_next_clicked:

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    errors: list[str] = []

    if label not in {
        "YES",
        "NO",
        "UNCERTAIN",
    }:
        errors.append("Select YES, NO, or UNCERTAIN.")

    if not reason.strip():
        errors.append("Provide a reason.")

    if label in {"YES", "UNCERTAIN"} and not evidence.strip():
        errors.append("Evidence is required for YES or UNCERTAIN.")

    if not confidence:
        errors.append("Select annotator confidence.")

    if errors:

        for error in errors:
            st.error(error)

        st.stop()

    # --------------------------------------------------------
    # Persist
    # --------------------------------------------------------

    df.loc[
        current_index,
        "annotation_status",
    ] = "completed"

    df.loc[
        current_index,
        "sif_potential",
    ] = label

    df.loc[
        current_index,
        "sif_reason",
    ] = reason.strip()

    df.loc[
        current_index,
        "evidence_text",
    ] = evidence.strip()

    df.loc[
        current_index,
        "hazard_notes",
    ] = hazard_notes.strip()

    df.loc[
        current_index,
        "exposure_notes",
    ] = exposure_notes.strip()

    df.loc[
        current_index,
        "consequence_notes",
    ] = consequence_notes.strip()

    df.loc[
        current_index,
        "barrier_notes",
    ] = barrier_notes.strip()

    df.loc[
        current_index,
        "annotator_confidence",
    ] = confidence

    df.loc[
        current_index,
        "annotator_id",
    ] = st.session_state.annotator_id

    df.loc[
        current_index,
        "annotation_version",
    ] = "v0.1"

    df.loc[
        current_index,
        "annotation_round",
    ] = annotation_round

    df.loc[
        current_index,
        "annotated_at_utc",
    ] = datetime.now(timezone.utc).isoformat()

    save_dataset(df)

    st.success(f"Saved annotation for {row['report_id']}.")

    if save_next_clicked:

        next_index = get_next_pending_index(
            df,
            current_index,
        )

        if next_index is not None:
            st.session_state.record_index = next_index
            st.rerun()

        else:
            st.balloons()
            st.success("All records in this annotation batch are complete.")


# ============================================================
# Completion summary
# ============================================================

st.divider()

done = df[df["annotation_status"].fillna("").astype(str).str.lower().eq("completed")]

if not done.empty:

    st.subheader("Current annotation distribution")

    counts = (
        done["sif_potential"]
        .value_counts()
        .reindex(
            ["YES", "NO", "UNCERTAIN"],
            fill_value=0,
        )
    )

    summary_left, summary_right = st.columns(2)

    with summary_left:
        st.dataframe(
            counts.rename("count"),
            use_container_width=True,
        )

    with summary_right:

        confidence_counts = (
            done["annotator_confidence"]
            .value_counts()
            .reindex(
                ["HIGH", "MEDIUM", "LOW"],
                fill_value=0,
            )
        )

        st.dataframe(
            confidence_counts.rename("count"),
            use_container_width=True,
        )


st.caption(f"Persistent annotation file: {output_path}")
