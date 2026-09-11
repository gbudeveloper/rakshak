from __future__ import annotations

from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

FILES = [
    ROOT / "data" / "annotations" / "gold_set_v0.1.csv",
    ROOT / "data" / "annotations" / "calibration_batch_02_annotated.csv",
    ROOT / "data" / "annotations" / "calibration_batch_03_annotated.csv",
]

OUTPUT = ROOT / "data" / "annotations" / "boundary_adjudication_v0.3.csv"

def clean(v: object) -> str:
    if pd.isna(v):
        return ""
    return str(v).strip()

def norm(v: object) -> str:
    return re.sub(r"\s+", " ", clean(v).lower())

def main() -> None:
    frames = []
    for path in FILES:
        if path.exists():
            frames.append(pd.read_csv(path))

    if not frames:
        raise FileNotFoundError("No annotation files found.")

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates("report_id", keep="last")

    df = df[
        df["annotation_status"].fillna("").astype(str).str.lower().eq("completed")
    ].copy()

    text = (
        df["description"].fillna("").astype(str).map(norm)
        + " "
        + df["sif_reason"].fillna("").astype(str).map(norm)
    )

    # Boundary signals are for REVIEW SELECTION only.
    # They do not determine the correct label.
    rules = [
        ("existing_uncertain", "Existing UNCERTAIN label; verify that missing information is materially decision-changing.",
         df["sif_potential"].fillna("").astype(str).str.upper().eq("UNCERTAIN")),

        ("chemical_counterfactual", "Chemical case whose reasoning may rely on changed concentration/quantity/contact assumptions.",
         text.str.contains(
             r"higher concentration|larger volume|greater exposure|longer contact|worse-case|worst-case|more hazardous",
             regex=True, na=False)),

        ("effective_barrier", "High-energy exposure with an explicit protective barrier that may affect the SIF decision.",
         text.str.contains(
             r"protection system acted|protection system|immediately|isolat|lockout|guard|shield",
             regex=True, na=False)),

        ("low_energy_cut", "Low-energy cut/puncture/blade case; verify NO boundary.",
         text.str.contains(
             r"\bfinger\b|glass|puncture|guillotine|mesh|cut",
             regex=True, na=False)),

        ("short_or_unclear_fall", "Fall case where height/mechanism needs bounded-consequence review.",
         text.str.contains(
             r"\b1\s*m\b|\b1m\b|\b2\.5\s*m\b|\b2\.5m\b|\bfall\b|scaffold|ladder",
             regex=True, na=False)),

        ("strong_yes_boundary", "YES case worth checking for unsupported counterfactual escalation.",
         df["sif_potential"].fillna("").astype(str).str.upper().eq("YES")
         & text.str.contains(
             r"could have|could potentially|larger|greater|different|similar release",
             regex=True, na=False)),
    ]

    chosen = []
    used = set()

    for name, reason, mask in rules:
        candidates = df.loc[mask].copy()
        candidates = candidates.sort_values("report_id")

        for _, row in candidates.iterrows():
            rid = str(row["report_id"])
            if rid in used:
                continue
            row = row.copy()
            row["boundary_selection_rule"] = name
            row["boundary_selection_reason"] = reason
            chosen.append(row)
            used.add(rid)
            if len(chosen) >= 10:
                break

        if len(chosen) >= 10:
            break

    if len(chosen) < 10:
        remaining = df[~df["report_id"].astype(str).isin(used)].sort_values("report_id")
        for _, row in remaining.iterrows():
            row = row.copy()
            row["boundary_selection_rule"] = "manual_fill"
            row["boundary_selection_reason"] = "Additional completed case for final protocol check."
            chosen.append(row)
            if len(chosen) >= 10:
                break

    result = pd.DataFrame(chosen).head(10).copy()

    # Adjudication fields.
    result["adjudication_status"] = "pending"
    result["adjudication_label"] = ""
    result["adjudication_reason"] = ""
    result["adjudication_notes"] = ""
    result["adjudication_confidence"] = ""
    result["adjudicator_id"] = ""
    result["adjudicated_at_utc"] = ""
    result["annotation_version"] = "v0.3"

    # Keep only fields useful for review plus source audit fields.
    preferred = [
        "report_id", "description", "sif_potential", "sif_reason",
        "evidence_text", "hazard_notes", "exposure_notes",
        "consequence_notes", "barrier_notes", "annotator_confidence",
        "annotation_round", "annotation_version",
        "boundary_selection_rule", "boundary_selection_reason",
        "adjudication_status", "adjudication_label",
        "adjudication_reason", "adjudication_notes",
        "adjudication_confidence", "adjudicator_id",
        "adjudicated_at_utc",
    ]
    result = result[[c for c in preferred if c in result.columns]]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False, encoding="utf-8")

    print("=" * 68)
    print("RAKSHAK BOUNDARY ADJUDICATION v0.3")
    print("=" * 68)
    print(f"Selected: {len(result)}")
    print()
    print(result[
        ["report_id", "sif_potential", "boundary_selection_rule"]
    ].to_string(index=False))
    print()
    print(f"Output: {OUTPUT}")

if __name__ == "__main__":
    main()
