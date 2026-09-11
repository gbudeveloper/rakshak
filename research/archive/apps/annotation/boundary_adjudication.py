from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Expected layout:
# <project_root>/
#   apps/annotation/boundary_adjudication.py
#   data/annotations/boundary_adjudication_v0.3.csv
if not (PROJECT_ROOT / "data").exists():
    raise RuntimeError(
        "Could not resolve the project root. "
        "Expected this script under <project_root>/apps/annotation/."
    )

INPUT_PATH = PROJECT_ROOT / "data" / "annotations" / "boundary_adjudication_v0.3.csv"

OUTPUT_PATH = INPUT_PATH

st.set_page_config(
    page_title="RAKSHAK Boundary Adjudication v0.3",
    page_icon="🛡️",
    layout="wide",
)


REQUIRED_COLUMNS = {
    "report_id",
    "description",
    "sif_potential",
    "sif_reason",
    "evidence_text",
    "hazard_notes",
    "exposure_notes",
    "consequence_notes",
    "barrier_notes",
    "adjudication_status",
    "adjudication_label",
    "adjudication_reason",
    "adjudication_notes",
    "adjudication_confidence",
    "adjudicator_id",
    "adjudicated_at_utc",
}


def clean(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def load_data() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    df = pd.read_csv(INPUT_PATH)

    for column, default in {
        "adjudication_status": "pending",
        "adjudication_label": "",
        "adjudication_reason": "",
        "adjudication_notes": "",
        "adjudication_confidence": "",
        "adjudicator_id": "",
        "adjudicated_at_utc": "",
    }.items():
        if column not in df.columns:
            df[column] = default

    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))

    return df


def save_data(df: pd.DataFrame) -> None:
    temp = OUTPUT_PATH.with_suffix(".tmp.csv")
    df.to_csv(temp, index=False, encoding="utf-8")
    temp.replace(OUTPUT_PATH)


def is_done(row: pd.Series) -> bool:
    return clean(row["adjudication_status"]).lower() == "completed"


def next_pending(df: pd.DataFrame, current: int) -> int | None:
    for idx in df.index:
        if idx > current and not is_done(df.loc[idx]):
            return idx

    for idx in df.index:
        if idx < current and not is_done(df.loc[idx]):
            return idx

    return None


df = load_data()

if "active_index" not in st.session_state:
    pending = [idx for idx in df.index if not is_done(df.loc[idx])]
    st.session_state.active_index = pending[0] if pending else df.index[0]

if "adjudicator_id" not in st.session_state:
    st.session_state.adjudicator_id = "adjudicator_01"

st.title("🛡️ RAKSHAK — Boundary Adjudication v0.3")
st.caption("Final review of boundary cases from the 40-report annotation pilot.")

completed = int(
    df["adjudication_status"].fillna("").astype(str).str.lower().eq("completed").sum()
)

total = len(df)

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Completed", f"{completed} / {total}")
with c2:
    st.metric("Remaining", total - completed)
with c3:
    st.progress(
        completed / total if total else 0,
        text=f"{completed / total:.0%}" if total else "0%",
    )

with st.sidebar:
    st.header("Adjudicator")

    aid = st.text_input(
        "Adjudicator ID",
        value=st.session_state.adjudicator_id,
        key="adjudicator_id_widget",
    )
    st.session_state.adjudicator_id = aid.strip() or "adjudicator_01"

    st.divider()

    st.header("v0.3 rules")

    st.markdown("""
**YES**

The reported exposure itself has a credible SIF pathway.

**NO**

The described exposure does not have a credible SIF pathway.

**UNCERTAIN**

A missing material fact could change the YES/NO decision.

### Counterfactual restraint

Do not invent a larger quantity, concentration, pressure,
speed, height, duration, energy, or exposure area.

### Barrier rule

Record the observed barrier condition. Do not write corrective
actions as though they were observed facts.
""")

    st.divider()

    st.warning(
        "The existing pilot label and reasoning are shown for adjudication. "
        "Source Accident Level and Potential Accident Level remain hidden."
    )

idx = st.session_state.active_index
if idx not in df.index:
    st.session_state.active_index = df.index[0]
    st.rerun()

row = df.loc[idx]

indices = df.index.tolist()

selected = st.selectbox(
    "Boundary case",
    options=indices,
    index=indices.index(idx),
    format_func=lambda i: (
        f"{df.loc[i, 'report_id']} — " f"{'DONE' if is_done(df.loc[i]) else 'PENDING'}"
    ),
    key="boundary_case_selector",
)

if selected != idx:
    st.session_state.active_index = selected
    st.rerun()

st.subheader(f"Report: {row['report_id']}")

left, right = st.columns(2)

with left:
    st.markdown(f"**Previous label:** {clean(row['sif_potential'])}")
    st.markdown(f"**Industry:** {clean(row.get('department', ''))}")
    st.markdown(f"**Site:** {clean(row.get('site', ''))}")

with right:
    st.markdown(
        f"**Boundary reason:** {clean(row.get('boundary_selection_reason', ''))}"
    )
    st.markdown(
        f"**Selection rule:** `{clean(row.get('boundary_selection_rule', ''))}`"
    )

st.divider()
st.subheader("Original narrative")
st.info(clean(row["description"]))

st.subheader("Previous annotation")

prev_left, prev_right = st.columns(2)

with prev_left:
    st.markdown("**Previous reasoning**")
    st.write(clean(row["sif_reason"]))

    st.markdown("**Previous evidence**")
    st.write(clean(row["evidence_text"]))

with prev_right:
    st.markdown("**Previous hazard**")
    st.write(clean(row["hazard_notes"]))

    st.markdown("**Previous exposure**")
    st.write(clean(row["exposure_notes"]))

    st.markdown("**Previous consequence**")
    st.write(clean(row["consequence_notes"]))

    st.markdown("**Previous barrier**")
    st.write(clean(row["barrier_notes"]))

st.divider()
st.subheader("v0.3 adjudication")

checks = [
    st.checkbox(
        "I identified the actual hazard / energy.",
        key=f"hazard_check_{idx}",
    ),
    st.checkbox(
        "I identified the human exposure.",
        key=f"exposure_check_{idx}",
    ),
    st.checkbox(
        "I assessed the consequence from the described event.",
        key=f"consequence_check_{idx}",
    ),
    st.checkbox(
        "I assessed the observed barrier condition only.",
        key=f"barrier_check_{idx}",
    ),
    st.checkbox(
        "I did not invent unsupported material changes.",
        key=f"counterfactual_check_{idx}",
    ),
]

label_options = ["YES", "NO", "UNCERTAIN"]
previous_adj_label = clean(row["adjudication_label"]).upper()

label = st.radio(
    "Adjudicated SIF potential",
    options=label_options,
    index=(
        label_options.index(previous_adj_label)
        if previous_adj_label in label_options
        else None
    ),
    horizontal=True,
    key=f"adjudication_label_{idx}",
)

reason = st.text_area(
    "Adjudication reason",
    value=clean(row["adjudication_reason"]),
    height=110,
    placeholder="Explain hazard → exposure → consequence.",
    key=f"adjudication_reason_{idx}",
)

notes = st.text_area(
    "Adjudication notes",
    value=clean(row["adjudication_notes"]),
    height=100,
    placeholder="Explain what changed or why the original label is retained.",
    key=f"adjudication_notes_{idx}",
)

confidence_options = ["", "HIGH", "MEDIUM", "LOW"]
previous_conf = clean(row["adjudication_confidence"]).upper()

confidence = st.selectbox(
    "Adjudication confidence",
    confidence_options,
    index=(
        confidence_options.index(previous_conf)
        if previous_conf in confidence_options
        else 0
    ),
    key=f"adjudication_confidence_{idx}",
)

save, save_next = st.columns(2)

with save:
    save_clicked = st.button(
        "Save Adjudication",
        type="primary",
        use_container_width=True,
        key="save_adjudication",
    )

with save_next:
    save_next_clicked = st.button(
        "Save & Next",
        use_container_width=True,
        key="save_next_adjudication",
    )

if save_clicked or save_next_clicked:

    errors = []

    if label not in label_options:
        errors.append("Choose YES, NO, or UNCERTAIN.")

    if not reason.strip():
        errors.append("Provide an adjudication reason.")

    if not confidence:
        errors.append("Select confidence.")

    if not all(checks):
        errors.append("Complete every v0.3 checklist item.")

    if errors:
        for error in errors:
            st.error(error)
        st.stop()

    df.loc[idx, "adjudication_status"] = "completed"
    df.loc[idx, "adjudication_label"] = label
    df.loc[idx, "adjudication_reason"] = reason.strip()
    df.loc[idx, "adjudication_notes"] = notes.strip()
    df.loc[idx, "adjudication_confidence"] = confidence
    df.loc[idx, "adjudicator_id"] = st.session_state.adjudicator_id
    df.loc[idx, "adjudicated_at_utc"] = datetime.now(timezone.utc).isoformat()
    df.loc[idx, "annotation_version"] = "v0.3"

    save_data(df)

    st.success(f"Saved adjudication for {row['report_id']}.")

    if save_next_clicked:
        nxt = next_pending(df, idx)
        if nxt is not None:
            st.session_state.active_index = nxt
            st.rerun()
        else:
            st.balloons()
            st.success("All boundary cases are adjudicated.")

done = df[df["adjudication_status"].fillna("").astype(str).str.lower().eq("completed")]

if not done.empty:
    st.divider()
    st.subheader("Adjudication distribution")

    st.dataframe(
        done["adjudication_label"]
        .value_counts()
        .reindex(
            ["YES", "NO", "UNCERTAIN"],
            fill_value=0,
        )
        .rename("count"),
        use_container_width=True,
    )
