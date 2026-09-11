from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "resolved_annotations_v0.2.csv"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "experiments"
    / "safety_information_extraction_v0.1"
)


# ============================================================
# RAKSHAK Safety Ontology v0.1
#
# This is deliberately evidence-first:
#   - extract only concepts supported by source text
#   - return the exact evidence span
#   - do not infer an unmentioned exposure/consequence
#   - this is an extraction layer, NOT a SIF decision engine
#
# The ontology can later be moved into:
#   ontology/hazards/
#   ontology/barriers/
#   ontology/consequences/
#   ontology/activities/
# ============================================================


@dataclass(frozen=True)
class OntologyItem:
    key: str
    label: str
    patterns: tuple[str, ...]


HAZARDS = (
    OntologyItem(
        "electrical_energy",
        "Electrical energy",
        (
            r"\belectri(c|cal)\b",
            r"\benergized\b",
            r"\barc[\s-]?flash\b",
            r"\belectrocution\b",
            r"\belectric shock\b",
            r"\bpower cord\b",
            r"\b440\s*v\b",
            r"\bhigh voltage\b",
        ),
    ),
    OntologyItem(
        "mechanical_energy",
        "Mechanical / rotational energy",
        (
            r"\brotat(ing|ion)\b",
            r"\bfan\b",
            r"\bpropeller\b",
            r"\bmachin(e|ery)\b",
            r"\bcrush(ing|ed)?\b",
            r"\bentangle(d|ment)?\b",
            r"\bamputation\b",
        ),
    ),
    OntologyItem(
        "pressure_process_energy",
        "Pressure / process energy",
        (
            r"\bpressure\b",
            r"\bpressurized\b",
            r"\bpressurised\b",
            r"\bstored energy\b",
            r"\brelease\b",
            r"\brupture(d)?\b",
            r"\bpiping\b",
            r"\bflange\b",
        ),
    ),
    OntologyItem(
        "thermal_energy",
        "Thermal / hot material",
        (
            r"\bhot gas\b",
            r"\bhot pulp\b",
            r"\bthermal\b",
            r"\bheat\b",
            r"\bhot\b",
            r"\bburn(ed|s)?\b",
            r"\bscald(ing|ed)?\b",
        ),
    ),
    OntologyItem(
        "gravity_fall_energy",
        "Gravity / fall-from-height energy",
        (
            r"\bfall\b",
            r"\bfell\b",
            r"\bfalling\b",
            r"\bheights?\b",
            r"\bheight\b",
            r"\bvertical opening\b",
            r"\bchimney\b",
            r"\bladder\b",
            r"\bplatform\b",
        ),
    ),
    OntologyItem(
        "ground_rock_energy",
        "Ground / rockfall energy",
        (
            r"\brockfall\b",
            r"\bfalling rock\b",
            r"\brock\b",
            r"\bslab\b",
            r"\bunstable ground\b",
            r"\bunstable roof\b",
            r"\bground\s*fall\b",
            r"\bground movement\b",
            r"\broof\b",
        ),
    ),
    OntologyItem(
        "mobile_equipment_energy",
        "Mobile equipment / vehicle movement",
        (
            r"\btruck\b",
            r"\bloader\b",
            r"\bwheel[\s-]?loader\b",
            r"\bvehicle\b",
            r"\bmobile equipment\b",
            r"\btire\b",
            r"\broadway\b",
            r"\brun[\s-]?over\b",
        ),
    ),
    OntologyItem(
        "chemical_exposure",
        "Chemical substance / exposure",
        (
            r"\bchemical\b",
            r"\bpaint\b",
            r"\bsolvent\b",
            r"\btoxic\b",
            r"\bcorrosive\b",
            r"\bsubstance\b",
        ),
    ),
    OntologyItem(
        "suspended_load",
        "Suspended / falling load",
        (
            r"\bsuspended load\b",
            r"\bfalling load\b",
            r"\blift(ed|ing)?\b",
            r"\bcrane\b",
            r"\bhook\b",
            r"\bpump\b.*\bfall(s|en)?\b",
        ),
    ),
    OntologyItem(
        "fire_explosion",
        "Fire / explosion energy",
        (
            r"\bfire\b",
            r"\bflame\b",
            r"\bexplosion\b",
            r"\bexplosive\b",
            r"\bignit(e|ion|ed)\b",
            r"\bigniting\b",
        ),
    ),
    OntologyItem(
        "biological_sting",
        "Biological / animal hazard",
        (
            r"\bbees?\b",
            r"\bsting\b",
            r"\bbites?\b",
            r"\bsnake\b",
            r"\bvenomous\b",
            r"\binsect\b",
        ),
    ),
)


EXPOSURES = (
    OntologyItem(
        "direct_worker_exposure",
        "Worker directly exposed",
        (
            r"\bworker\b.*\bexpos(ed|ure)\b",
            r"\bdirectly exposed\b",
            r"\bwas exposed\b",
            r"\bwere exposed\b",
            r"\bin the path\b",
            r"\bin the line of fire\b",
            r"\bstanding\b.*\bnear\b",
            r"\bworking\b.*\bnear\b",
        ),
    ),
    OntologyItem(
        "body_part_contact",
        "Direct contact with body part",
        (
            r"\bhand\b.*\bcontact\b",
            r"\bhis hand\b",
            r"\bher hand\b",
            r"\bface\b",
            r"\beye\b",
            r"\barm\b",
            r"\bleg\b",
            r"\bhead\b",
            r"\bfoot\b",
            r"\bfinger\b",
        ),
    ),
    OntologyItem(
        "line_of_fire",
        "Line-of-fire exposure",
        (
            r"\bline of fire\b",
            r"\bin the path\b",
            r"\bdirection of (the )?flow\b",
            r"\bstruck by\b",
            r"\bprojected\b",
            r"\bprojection\b",
        ),
    ),
    OntologyItem(
        "fall_exposure",
        "Fall exposure",
        (
            r"\bworking at\b.*\bheight\b",
            r"\bheight of\b",
            r"\bnear\b.*\bopening\b",
            r"\badjacent to\b.*\bopening\b",
            r"\bclimbs?\b.*\bheight\b",
            r"\bfalls?\b",
            r"\bfell\b",
        ),
    ),
    OntologyItem(
        "mobile_equipment_exposure",
        "Pedestrian / mobile-equipment interaction",
        (
            r"\bon foot\b",
            r"\bcross(ed|ing)? the roadway\b",
            r"\bloader\b.*\bpassed\b",
            r"\bvehicle\b.*\bworker\b",
            r"\bnear\b.*\btruck\b",
        ),
    ),
)


CONSEQUENCES = (
    OntologyItem(
        "fatality",
        "Fatality",
        (
            r"\bfatal(ity)?\b",
            r"\bdeath\b",
            r"\bdied\b",
            r"\bdead\b",
        ),
    ),
    OntologyItem(
        "serious_injury",
        "Serious / life-altering injury",
        (
            r"\bserious injury\b",
            r"\bsevere injury\b",
            r"\blife[\s-]?altering\b",
            r"\btraumatic injury\b",
            r"\bfracture(d|s)?\b",
            r"\bamputation\b",
            r"\bcrushing\b",
        ),
    ),
    OntologyItem(
        "burn_thermal_injury",
        "Burn / thermal injury",
        (
            r"\bburn(ed|s)?\b",
            r"\bscald(ed|ing)?\b",
            r"\bthermal injury\b",
        ),
    ),
    OntologyItem(
        "electrical_injury",
        "Electrical injury",
        (
            r"\belectric shock\b",
            r"\belectrocution\b",
            r"\bcardiac arrest\b",
            r"\barc[\s-]?flash\b",
        ),
    ),
    OntologyItem(
        "crush_amputation",
        "Crushing / amputation",
        (
            r"\bcrush(ing|ed)?\b",
            r"\bamputation\b",
            r"\btraumatic amputation\b",
            r"\brun[\s-]?over\b",
        ),
    ),
    OntologyItem(
        "struck_by",
        "Struck-by injury",
        (
            r"\bstruck by\b",
            r"\bhit\b",
            r"\bimpact(ed)?\b",
            r"\bfalling rock\b",
            r"\bproject(ed|ile)\b",
        ),
    ),
    OntologyItem(
        "minor_injury",
        "Minor injury",
        (
            r"\bminor\b",
            r"\bsmall burn\b",
            r"\bcut\b",
            r"\bscratch\b",
            r"\bswelling\b",
            r"\brash\b",
            r"\birritation\b",
        ),
    ),
)


BARRIERS = (
    OntologyItem(
        "isolation_loto",
        "Isolation / lockout-tagout",
        (
            r"\blockout\b",
            r"\btagout\b",
            r"\bLOTO\b",
            r"\bisolation\b",
            r"\bde-energiz(ed|ation)\b",
            r"\bdeenergiz(ed|ation)\b",
        ),
    ),
    OntologyItem(
        "guarding",
        "Machine guarding",
        (
            r"\bguard(ing|ed)?\b",
            r"\bmachine protection\b",
            r"\bprotective guard\b",
        ),
    ),
    OntologyItem(
        "fall_protection",
        "Fall prevention / fall protection",
        (
            r"\bfall protection\b",
            r"\bfall prevention\b",
            r"\bguardrail\b",
            r"\bhandrail\b",
            r"\bsafety harness\b",
            r"\blifeline\b",
        ),
    ),
    OntologyItem(
        "exclusion_barricade",
        "Exclusion / barricading",
        (
            r"\bbarricad(e|ed|ing)\b",
            r"\bexclusion zone\b",
            r"\bmarked with tape\b",
            r"\bcones\b",
            r"\bcontrolled access\b",
        ),
    ),
    OntologyItem(
        "pressure_isolation",
        "Pressure isolation / depressurization",
        (
            r"\bdepressuriz(e|ation|ed)\b",
            r"\bdepressurise\b",
            r"\bpressure verification\b",
            r"\bbleed(ing)?\b",
            r"\bdrain(ing)?\b",
            r"\bpositive isolation\b",
        ),
    ),
    OntologyItem(
        "ground_control",
        "Ground control / scaling",
        (
            r"\bground control\b",
            r"\bground-control\b",
            r"\bscaling\b",
            r"\bground support\b",
            r"\broof support\b",
        ),
    ),
    OntologyItem(
        "electrical_protection",
        "Electrical protection",
        (
            r"\babsence-of-voltage\b",
            r"\babsence of voltage\b",
            r"\barc[\s-]?flash PPE\b",
            r"\belectrical isolation\b",
            r"\bprotection system\b",
            r"\bPPE\b",
        ),
    ),
    OntologyItem(
        "work_procedure",
        "Procedure / work control",
        (
            r"\bprocedure\b",
            r"\bpermit\b",
            r"\bJSA\b",
            r"\bjob safety analysis\b",
            r"\bwork permit\b",
        ),
    ),
)


ACTIVITIES = (
    OntologyItem(
        "loading_drilling",
        "Drilling / loading",
        (
            r"\bloading drills\b",
            r"\bdrill(ing)?\b",
            r"\bloader\b",
            r"\bproduction drills\b",
        ),
    ),
    OntologyItem(
        "shotcrete",
        "Shotcrete",
        (
            r"\bshotcrete\b",
            r"\bshotcrete casting\b",
            r"\bshotcrete launch\b",
        ),
    ),
    OntologyItem(
        "painting",
        "Painting",
        (
            r"\bpaint(ing|ed)?\b",
            r"\bpainting\b",
        ),
    ),
    OntologyItem(
        "washing_cleaning",
        "Washing / cleaning",
        (
            r"\bwash(ing|ed)?\b",
            r"\bclean(ing|ed)?\b",
            r"\bcleaning\b",
        ),
    ),
    OntologyItem(
        "assembly_installation",
        "Assembly / installation",
        (
            r"\bassembl(y|ing)\b",
            r"\binstall(ation|ing)\b",
            r"\bassembling\b",
        ),
    ),
    OntologyItem(
        "inspection",
        "Inspection",
        (
            r"\binspect(ion|ing|ed)\b",
            r"\binspection\b",
        ),
    ),
)


@dataclass
class Evidence:
    ontology_key: str
    label: str
    text: str
    start: int
    end: int
    sentence: str
    match_type: str
    confidence: str


@dataclass
class ExtractionResult:
    report_id: str
    narrative: str
    hazards: list[Evidence]
    exposures: list[Evidence]
    consequences: list[Evidence]
    barriers: list[Evidence]
    activities: list[Evidence]
    summary: dict


def sentence_spans(text: str) -> list[tuple[int, int, str]]:
    """
    Lightweight sentence segmentation that works reasonably well
    on the source industrial narratives without adding another
    NLP dependency.
    """
    spans = []

    # Split on sentence-ending punctuation or line breaks, while
    # retaining offsets for exact evidence extraction.
    pattern = re.compile(
        r"[^.!?\n]+(?:[.!?]+|$)",
        flags=re.MULTILINE,
    )

    for match in pattern.finditer(text):
        raw = match.group(0)
        left_trimmed = raw.lstrip()
        offset = len(raw) - len(left_trimmed)

        start = match.start() + offset
        end = match.end()

        sentence = text[start:end].strip()

        if sentence:
            spans.append(
                (
                    start,
                    end,
                    sentence,
                )
            )

    return spans


def compile_patterns(
    ontology: Iterable[OntologyItem],
) -> dict[str, list[re.Pattern[str]]]:
    return {
        item.key: [
            re.compile(
                pattern,
                flags=re.IGNORECASE,
            )
            for pattern in item.patterns
        ]
        for item in ontology
    }


def confidence_for_match(
    item: OntologyItem,
    evidence: str,
    sentence: str,
) -> str:
    """
    Conservative lexical confidence:
      HIGH   = more specific safety phrase or direct event wording
      MEDIUM = common domain cue
      LOW    = broad/ambiguous cue

    This is NOT a calibrated probability.
    """
    lower = evidence.lower()

    strong_tokens = {
        "arc flash",
        "electric shock",
        "electrocution",
        "line of fire",
        "struck by",
        "fatality",
        "death",
        "lockout",
        "tagout",
        "positive isolation",
        "ground control",
        "fall protection",
        "vertical opening",
        "suspended load",
        "ground fall",
    }

    if any(token in lower for token in strong_tokens):
        return "HIGH"

    if len(evidence.split()) >= 2:
        return "MEDIUM"

    return "LOW"


def extract_category(
    text: str,
    ontology: tuple[OntologyItem, ...],
) -> list[Evidence]:
    pattern_map = compile_patterns(ontology)
    sentence_data = sentence_spans(text)

    matches: list[Evidence] = []

    for item in ontology:
        for sent_start, sent_end, sentence in sentence_data:
            for pattern in pattern_map[item.key]:
                found = pattern.search(sentence)
                if not found:
                    continue

                local_start = found.start()
                local_end = found.end()

                absolute_start = (
                    sent_start + local_start
                )
                absolute_end = (
                    sent_start + local_end
                )

                evidence_text = text[
                    absolute_start:absolute_end
                ]

                matches.append(
                    Evidence(
                        ontology_key=item.key,
                        label=item.label,
                        text=evidence_text,
                        start=absolute_start,
                        end=absolute_end,
                        sentence=sentence,
                        match_type="lexical",
                        confidence=confidence_for_match(
                            item,
                            evidence_text,
                            sentence,
                        ),
                    )
                )

                # One evidence hit per item/sentence is sufficient.
                break

            # Avoid duplicate evidence for one ontology item in
            # multiple patterns within the same sentence.
            if any(
                m.ontology_key == item.key
                and m.start >= sent_start
                and m.end <= sent_end
                for m in matches
            ):
                continue

    # De-duplicate exact spans.
    unique: dict[
        tuple[str, int, int],
        Evidence,
    ] = {}

    for match in matches:
        key = (
            match.ontology_key,
            match.start,
            match.end,
        )

        existing = unique.get(key)

        if existing is None:
            unique[key] = match
            continue

        confidence_rank = {
            "LOW": 1,
            "MEDIUM": 2,
            "HIGH": 3,
        }

        if (
            confidence_rank[match.confidence]
            > confidence_rank[existing.confidence]
        ):
            unique[key] = match

    return list(
        sorted(
            unique.values(),
            key=lambda evidence: (
                evidence.start,
                evidence.end,
            ),
        )
    )


def deduplicate_semantically_same_hits(
    evidence: list[Evidence],
) -> list[Evidence]:
    """
    Remove overlapping evidence for the same ontology key.
    Keep the longest/highest-confidence span.
    """
    confidence_rank = {
        "LOW": 1,
        "MEDIUM": 2,
        "HIGH": 3,
    }

    best: dict[str, Evidence] = {}

    for item in evidence:
        previous = best.get(item.ontology_key)

        if previous is None:
            best[item.ontology_key] = item
            continue

        old_score = (
            confidence_rank[previous.confidence],
            len(previous.text),
        )
        new_score = (
            confidence_rank[item.confidence],
            len(item.text),
        )

        if new_score > old_score:
            best[item.ontology_key] = item

    return sorted(
        best.values(),
        key=lambda item: item.start,
    )


def to_dict_evidence(items: list[Evidence]) -> list[dict]:
    return [asdict(item) for item in items]


def extract_narrative(
    report_id: str,
    narrative: str,
) -> ExtractionResult:
    hazards = deduplicate_semantically_same_hits(
        extract_category(
            narrative,
            HAZARDS,
        )
    )

    exposures = deduplicate_semantically_same_hits(
        extract_category(
            narrative,
            EXPOSURES,
        )
    )

    consequences = deduplicate_semantically_same_hits(
        extract_category(
            narrative,
            CONSEQUENCES,
        )
    )

    barriers = deduplicate_semantically_same_hits(
        extract_category(
            narrative,
            BARRIERS,
        )
    )

    activities = deduplicate_semantically_same_hits(
        extract_category(
            narrative,
            ACTIVITIES,
        )
    )

    high_risk_hazards = [
        item
        for item in hazards
        if item.ontology_key
        in {
            "electrical_energy",
            "mechanical_energy",
            "pressure_process_energy",
            "thermal_energy",
            "gravity_fall_energy",
            "ground_rock_energy",
            "mobile_equipment_energy",
            "suspended_load",
            "fire_explosion",
        }
    ]

    direct_exposure = [
        item
        for item in exposures
        if item.ontology_key
        in {
            "direct_worker_exposure",
            "line_of_fire",
            "fall_exposure",
            "mobile_equipment_exposure",
        }
    ]

    severe_consequences = [
        item
        for item in consequences
        if item.ontology_key
        in {
            "fatality",
            "serious_injury",
            "burn_thermal_injury",
            "electrical_injury",
            "crush_amputation",
            "struck_by",
        }
    ]

    summary = {
        "hazard_count": len(hazards),
        "exposure_count": len(exposures),
        "consequence_count": len(consequences),
        "barrier_count": len(barriers),
        "activity_count": len(activities),
        "high_risk_hazard_count": len(
            high_risk_hazards
        ),
        "direct_exposure_count": len(
            direct_exposure
        ),
        "severe_consequence_signal_count": len(
            severe_consequences
        ),
        "has_fatality_wording": any(
            item.ontology_key == "fatality"
            for item in consequences
        ),
        "has_direct_exposure_signal": bool(
            direct_exposure
        ),
        "has_barrier_signal": bool(barriers),
    }

    return ExtractionResult(
        report_id=report_id,
        narrative=narrative,
        hazards=hazards,
        exposures=exposures,
        consequences=consequences,
        barriers=barriers,
        activities=activities,
        summary=summary,
    )


def load_input(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Input not found:\n{path}"
        )

    df = pd.read_csv(path)

    if "report_id" not in df.columns:
        raise ValueError(
            "Input requires 'report_id'."
        )

    description_column = None

    for candidate in [
        "description",
        "Description",
        "narrative",
    ]:
        if candidate in df.columns:
            description_column = candidate
            break

    if description_column is None:
        raise ValueError(
            "Input requires description/narrative column."
        )

    return df.rename(
        columns={
            description_column: "description",
        }
    ).copy()


def write_jsonl(
    results: list[ExtractionResult],
    path: Path,
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for result in results:
            payload = {
                "report_id": result.report_id,
                "narrative": result.narrative,
                "hazards": to_dict_evidence(
                    result.hazards
                ),
                "exposures": to_dict_evidence(
                    result.exposures
                ),
                "consequences": to_dict_evidence(
                    result.consequences
                ),
                "barriers": to_dict_evidence(
                    result.barriers
                ),
                "activities": to_dict_evidence(
                    result.activities
                ),
                "summary": result.summary,
            }

            handle.write(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                )
                + "\n"
            )


def write_flat_csv(
    results: list[ExtractionResult],
    path: Path,
) -> None:
    rows = []

    for result in results:
        def labels(items: list[Evidence]) -> str:
            return "; ".join(
                sorted(
                    {
                        item.label
                        for item in items
                    }
                )
            )

        def evidence(items: list[Evidence]) -> str:
            return " || ".join(
                item.text
                for item in items
            )

        rows.append(
            {
                "report_id": result.report_id,
                "hazards": labels(result.hazards),
                "hazard_evidence": evidence(
                    result.hazards
                ),
                "exposures": labels(result.exposures),
                "exposure_evidence": evidence(
                    result.exposures
                ),
                "consequences": labels(
                    result.consequences
                ),
                "consequence_evidence": evidence(
                    result.consequences
                ),
                "barriers": labels(result.barriers),
                "barrier_evidence": evidence(
                    result.barriers
                ),
                "activities": labels(
                    result.activities
                ),
                "activity_evidence": evidence(
                    result.activities
                ),
                **result.summary,
            }
        )

    pd.DataFrame(rows).to_csv(
        path,
        index=False,
        encoding="utf-8",
    )


def print_case(result: ExtractionResult) -> None:
    print()
    print("-" * 78)
    print(f"REPORT: {result.report_id}")
    print("-" * 78)
    print(result.narrative)
    print()

    sections = [
        ("HAZARDS", result.hazards),
        ("EXPOSURES", result.exposures),
        ("CONSEQUENCES", result.consequences),
        ("BARRIERS", result.barriers),
        ("ACTIVITIES", result.activities),
    ]

    for title, items in sections:
        print(title)

        if not items:
            print("  - none detected")
            continue

        for item in items:
            print(
                f"  - {item.label} "
                f"[{item.confidence}] "
                f'→ "{item.text}"'
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evidence-first RAKSHAK safety information extractor."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only first N records; 0 means all.",
    )

    args = parser.parse_args()

    df = load_input(args.input)

    if args.limit > 0:
        df = df.head(args.limit).copy()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 78)
    print("RAKSHAK SAFETY INFORMATION EXTRACTION v0.1")
    print("=" * 78)
    print(f"Input   : {args.input}")
    print(f"Records : {len(df)}")
    print()
    print(
        "Method  : evidence-first ontology + lexical extraction"
    )
    print(
        "Important: extracted evidence is not a SIF decision."
    )

    results = []

    for row in df.itertuples(index=False):
        report_id = str(
            getattr(row, "report_id")
        )

        narrative = str(
            getattr(row, "description")
        )

        result = extract_narrative(
            report_id=report_id,
            narrative=narrative,
        )

        results.append(result)

    jsonl_path = (
        args.output_dir
        / "safety_extraction.jsonl"
    )

    csv_path = (
        args.output_dir
        / "safety_extraction_flat.csv"
    )

    summary_path = (
        args.output_dir
        / "summary.json"
    )

    write_jsonl(
        results,
        jsonl_path,
    )

    write_flat_csv(
        results,
        csv_path,
    )

    category_counts = {
        "hazard_evidence": sum(
            len(result.hazards)
            for result in results
        ),
        "exposure_evidence": sum(
            len(result.exposures)
            for result in results
        ),
        "consequence_evidence": sum(
            len(result.consequences)
            for result in results
        ),
        "barrier_evidence": sum(
            len(result.barriers)
            for result in results
        ),
        "activity_evidence": sum(
            len(result.activities)
            for result in results
        ),
    }

    summary = {
        "version": "safety_information_extraction_v0.1",
        "input": str(
            args.input.relative_to(PROJECT_ROOT)
            if args.input.is_relative_to(PROJECT_ROOT)
            else args.input
        ),
        "records_processed": len(results),
        "category_evidence_counts": category_counts,
        "records_with_direct_exposure_signal": sum(
            result.summary[
                "has_direct_exposure_signal"
            ]
            for result in results
        ),
        "records_with_fatality_wording": sum(
            result.summary[
                "has_fatality_wording"
            ]
            for result in results
        ),
        "records_with_barrier_signal": sum(
            result.summary[
                "has_barrier_signal"
            ]
            for result in results
        ),
        "limitations": [
            "Lexical extraction can miss paraphrases.",
            "Broad words such as 'fall', 'rock', 'face', or 'worker' can be ambiguous.",
            "Evidence extraction must be reviewed before being treated as a structured safety fact.",
            "This layer does not decide SIF potential and does not infer an unmentioned consequence.",
        ],
    }

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Print up to first five examples for manual inspection.
    print()
    print(
        f"Saved {len(results)} structured extraction records."
    )

    for result in results[:5]:
        print_case(result)

    print()
    print("=" * 78)
    print("OUTPUTS")
    print("=" * 78)
    print(f"JSONL   : {jsonl_path}")
    print(f"CSV     : {csv_path}")
    print(f"Summary : {summary_path}")
    print()
    print("=" * 78)
    print("EXTRACTION COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
