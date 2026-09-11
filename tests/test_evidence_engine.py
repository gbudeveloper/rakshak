from scripts.evidence_engine_v1_5 import analyze


def test_evidence_schema_and_electrical_pathway():
    text = (
        "Worker contacted an energized electrical conductor and received an electric shock."
    )
    result = analyze(text)

    required = {
        "hazards",
        "exposures",
        "pathways",
        "hazard_evidence",
        "exposure_evidence",
        "pathway_evidence",
        "hazard_signal",
        "exposure_signal",
        "pathway_signal",
        "complete_pathway",
        "sif_context_status",
    }
    assert required.issubset(result)
    assert result["hazard_signal"] == 1
    assert result["exposure_signal"] == 1
    assert result["pathway_signal"] == 1
    assert result["complete_pathway"] == 1


def test_evidence_engine_does_not_return_autonomous_verdict():
    result = analyze("A worker inspected a machine during routine cleaning.")
    assert "sif_verdict" not in result
