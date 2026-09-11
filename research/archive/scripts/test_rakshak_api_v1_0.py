"""
RAKSHAK API smoke / consistency test v1.0

Run while uvicorn is running:
    python scripts\test_rakshak_api_v1_0.py

Checks:
- GET /health
- GET /model-info
- POST /predict
- POST /predict/batch
- API prediction against the exact frozen-model reference for 3 locked-test IDs

No model retraining.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import requests

ROOT=Path(__file__).resolve().parents[1]
BASE_URL="http://127.0.0.1:8000"
REFERENCE=ROOT/"experiments"/"rakshak_inference_validation_v1.2"/"final_refit_inference_consistency.csv"
LABELS=ROOT/"data"/"annotations"/"resolved_annotations_v0.2.csv"

TEST_IDS=[
    "industrial_safety_000422",
    "industrial_safety_000364",
    "industrial_safety_000145",
]


def get_json(method, url, **kwargs):
    response=requests.request(method,url,timeout=60,**kwargs)
    response.raise_for_status()
    return response.json()


def main():
    health=get_json("GET",f"{BASE_URL}/health")
    print("HEALTH:",json.dumps(health,indent=2))

    assert health["status"]=="ok"
    assert health["model_input_dimension"]==422

    info=get_json("GET",f"{BASE_URL}/model-info")
    print("\nMODEL INFO:",json.dumps(info,indent=2))

    assert info["model_version"]=="rakshak_final_v1.2"
    assert info["review_policy"]=="v1.4"
    assert info["evidence_engine"]=="v1.5"
    assert info["model_input_dimension"]==422

    labels=pd.read_csv(LABELS)
    labels["report_id"]=labels["report_id"].astype(str)
    labels["description"]=labels["description"].fillna("").astype(str)

    reference=pd.read_csv(REFERENCE)
    reference["report_id"]=reference["report_id"].astype(str)

    selected=(
        reference[reference["report_id"].isin(TEST_IDS)]
        .merge(labels[["report_id","description"]],on="report_id",how="left",validate="one_to_one")
    )

    records=[
        {
            "report_id":row.report_id,
            "description":row.description,
        }
        for row in selected.itertuples()
    ]

    batch=get_json(
        "POST",
        f"{BASE_URL}/predict/batch",
        json={"records":records},
    )

    api_records=batch["records"]
    print("\nBATCH RESULT:")
    for item in api_records:
        print(
            item["report_id"],
            item["sif_precursor_probability"],
            item["review"]["priority"],
        )

    api_map={x["report_id"]:x for x in api_records}

    diffs=[]
    for row in selected.itertuples():
        expected=float(row.final_refit_probability)
        actual=float(api_map[row.report_id]["sif_precursor_probability"])
        diffs.append(abs(expected-actual))

    max_diff=max(diffs)
    print(f"\nMAX API/MODEL ABSOLUTE DIFFERENCE: {max_diff:.12g}")

    assert max_diff < 1e-5, (
        f"API prediction mismatch: max diff {max_diff}"
    )

    # Single-record endpoint.
    one=records[0]
    single=get_json(
        "POST",
        f"{BASE_URL}/predict",
        json=one,
    )
    batch_prob=float(api_map[one["report_id"]]["sif_precursor_probability"])
    single_prob=float(single["sif_precursor_probability"])

    print(
        f"SINGLE ENDPOINT DIFFERENCE: {abs(batch_prob-single_prob):.12g}"
    )

    assert abs(batch_prob-single_prob) < 1e-5

    # Governance fields must be present.
    assert single["governance"]["autonomous_sif_decision"] is False
    assert single["governance"]["requires_hsse_review"] is True

    print("\n" + "="*80)
    print("RAKSHAK API SMOKE TEST: PASS")
    print("="*80)


if __name__=="__main__":
    main()
