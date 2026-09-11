from scripts.rakshak_reviewer_policy_v1_4 import policy


def row(p, hazard=0, exposure=0, pathway=0, complete=0):
    return {
        "sif_precursor_probability": p,
        "hazard_signal": hazard,
        "exposure_signal": exposure,
        "pathway_signal": pathway,
        "complete_pathway": complete,
    }


def test_policy_priority_boundaries():
    assert policy(row(0.95)) == ("P1", "VERY_HIGH_MODEL_SCORE")
    assert policy(row(0.80, complete=1)) == (
        "P1", "HIGH_MODEL_PLUS_COMPLETE_PATHWAY"
    )
    assert policy(row(0.80)) == ("P2", "HIGH_MODEL_SCORE_REVIEW")
    assert policy(row(0.60, exposure=1)) == (
        "P2", "MODERATE_HIGH_MODEL_PLUS_EXPLICIT_MECHANISM"
    )
    assert policy(row(0.60)) == ("P3", "MODERATE_MODEL_SCORE")
    assert policy(row(0.20, hazard=1)) == (
        "P4", "LOW_MODEL_PLUS_PRECURSOR_SIGNAL"
    )
    assert policy(row(0.20)) == ("P5", "LOW_MODEL_NO_PRECURSOR_SIGNAL")


def test_complete_pathway_escalates_low_score():
    assert policy(row(0.20, complete=1)) == (
        "P1", "MODEL_MISS_PLUS_COMPLETE_PATHWAY"
    )
