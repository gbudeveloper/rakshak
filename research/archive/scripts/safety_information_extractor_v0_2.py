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
    / "safety_information_extraction_v0.2"
)


@dataclass(frozen=True)
class Rule:
    key: str
    label: str
    patterns: tuple[str, ...]
    priority: int = 1


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


# ============================================================
# SAFETY ONTOLOGY v0.2
#
# v0.1 was intentionally conservative but too literal. For
# example, it missed "440V electrical board", "blade ... cuts",
# and "injured / imprisoned between equipment".
#
# v0.2 therefore uses:
#   1) high-value multi-word phrases first
#   2) contextual object/action phrases
#   3) broader fallbacks
#   4) stronger confidence for specific phrases
#
# It still returns source evidence and does not infer SIF labels.
# ============================================================


HAZARDS = (
    Rule(
        "electrical_energy",
        "Electrical energy",
        (
            r"\b440\s*v\b",
            r"\b\d+\s*v\b.*\b(panel|board|equipment|installation)\b",
            r"\belectric(al)? board\b",
            r"\belectr(ical|ic)\s+equipment\b",
            r"\benerg(ized|ise|ize|izing)\b",
            r"\barc[\s-]?flash\b",
            r"\bphase[\s-]+to[\s-]+ground\b",
            r"\belectric shock\b",
            r"\belectrocution\b",
            r"\bpower cord\b",
            r"\bhigh voltage\b",
            r"\bthermomagnetic\b",
        ),
        5,
    ),
    Rule(
        "mechanical_energy",
        "Mechanical / moving equipment energy",
        (
            r"\brotat(ing|ion)\b",
            r"\bmoving (equipment|machine|part)\b",
            r"\bmachine\b",
            r"\bmechanical\b",
            r"\bfan\b",
            r"\bpropeller\b",
            r"\bblade\b",
            r"\bcutter blade\b",
            r"\bcrush(ing|ed)?\b",
            r"\bpinch(ed|ing)?\b",
            r"\bentangle(d|ment)?\b",
            r"\brun[\s-]?over\b",
        ),
        4,
    ),
    Rule(
        "pressure_process_energy",
        "Pressure / process energy",
        (
            r"\bpressurized\b",
            r"\bpressurised\b",
            r"\bpressure\b",
            r"\bstored pressure\b",
            r"\bstored energy\b",
            r"\brupture(d)?\b",
            r"\brelease(d)?\b",
            r"\bflange\b",
            r"\bpiping\b",
            r"\bpipe\b",
            r"\bprocess line\b",
        ),
        3,
    ),
    Rule(
        "thermal_energy",
        "Thermal / hot material",
        (
            r"\bhot gas\b",
            r"\bhot pulp\b",
            r"\bhot material\b",
            r"\bthermal\b",
            r"\bheated\b",
            r"\bheat\b",
            r"\bburn(ed|s)?\b",
            r"\bscald(ing|ed)?\b",
            r"\bfurnace\b",
            r"\bmolten\b",
        ),
        5,
    ),
    Rule(
        "gravity_fall_energy",
        "Gravity / fall-from-height energy",
        (
            r"\bfall from height\b",
            r"\bfalling from\b",
            r"\bfalls?\s+\w+\s+meters?\b",
            r"\bfell\b",
            r"\bfalls?\b",
            r"\bheight of \d+\s*(m|meters?)\b",
            r"\bat an approximate height\b",
            r"\bvertical opening\b",
            r"\bchimney\b",
            r"\bopening\b",
        ),
        4,
    ),
    Rule(
        "ground_rock_energy",
        "Ground / rockfall energy",
        (
            r"\bfalling rock\b",
            r"\brockfall\b",
            r"\brock block\b",
            r"\bstone slab\b",
            r"\bslab\b.*\bdetached\b",
            r"\bunstable (ground|roof|rock)\b",
            r"\bground movement\b",
            r"\bground fall\b",
            r"\broof\b.*\bdetach(ed|ment)\b",
            r"\brock\b.*\bdetach(ed|ment)\b",
        ),
        5,
    ),
    Rule(
        "mobile_equipment_energy",
        "Mobile equipment / vehicle movement",
        (
            r"\bwheel[\s-]?loader\b",
            r"\bloader\b",
            r"\btruck\b",
            r"\bvehicle\b",
            r"\bmobile equipment\b",
            r"\btire\b",
            r"\broadway\b",
            r"\bcross(ed|ing)?\b.*\broad\b",
            r"\bpassed\b.*\bworker\b",
        ),
        4,
    ),
    Rule(
        "chemical_exposure",
        "Chemical substance / exposure",
        (
            r"\bpaint\b",
            r"\bsolvent\b",
            r"\bchemical\b",
            r"\btoxic\b",
            r"\bcorrosive\b",
            r"\bchemical substance\b",
        ),
        3,
    ),
    Rule(
        "suspended_load",
        "Suspended / falling load",
        (
            r"\bsuspended load\b",
            r"\bfalling load\b",
            r"\blift(ed|ing)?\b.*\bload\b",
            r"\bcrane\b",
            r"\bhook\b.*\bload\b",
            r"\bpump\b.*\bfall(s|en|ing)?\b",
            r"\bobject\b.*\bfell\b",
        ),
        3,
    ),
    Rule(
        "fire_explosion",
        "Fire / explosion energy",
        (
            r"\bfire\b",
            r"\bflame\b",
            r"\bexplosion\b",
            r"\bexplosive\b",
            r"\bignit(e|ion|ed|ing)\b",
            r"\bflash\b",
        ),
        4,
    ),
    Rule(
        "biological_hazard",
        "Biological / animal hazard",
        (
            r"\bbee(s)?\b",
            r"\bsting\b",
            r"\bstung\b",
            r"\bbite(s)?\b",
            r"\bvenomous\b",
            r"\bsnake\b",
        ),
        3,
    ),
)


EXPOSURES = (
    Rule(
        "direct_contact",
        "Direct bodily contact / exposure",
        (
            r"\b(operator|employee|worker|collaborator)\b.*\bcausing\b.*\b(injury|burn|cut)\b",
            r"\b(worker|employee|operator)\b.*\bwas exposed\b",
            r"\bdirectly exposed\b",
            r"\bexposed to\b",
            r"\breached\b.*\boperator\b",
            r"\breaches?\b.*\bworker\b",
        ),
        5,
    ),
    Rule(
        "body_part_exposure",
        "Body part exposed / contacted",
        (
            r"\b(hit|struck|touch(es|ed)?|graz(e|ed|ing)|rub(s|bed|bing))\b.*\b(hand|arm|leg|face|eye|head|finger|foot)\b",
            r"\b(hand|arm|leg|face|eye|head|finger|foot)\b.*\b(hit|struck|burn|cut|contact|rubbed|grazed)\b",
            r"\bcausing (?:a|an)\s+(?:cut|burn|injury)\b",
            r"\breaches?\b.*\bface\b",
        ),
        5,
    ),
    Rule(
        "line_of_fire",
        "Line-of-fire exposure",
        (
            r"\bline of fire\b",
            r"\bin the path\b",
            r"\bin the direction of\b",
            r"\bstruck by\b",
            r"\bprojected\b",
            r"\bprojection\b",
            r"\bbetween\b.*\b(equipment|machine|vehicle)\b.*\band\b",
        ),
        5,
    ),
    Rule(
        "fall_exposure",
        "Fall exposure",
        (
            r"\bworking\b.*\b(at|from)\b.*\bheight\b",
            r"\bheight of \d+\s*(m|meters?)\b",
            r"\bclimbs?\b.*\b(\d+|height)\b",
            r"\badjacent to\b.*\b(opening|chimney)\b",
            r"\bnear\b.*\b(opening|chimney)\b",
            r"\bfalls?\b",
            r"\bfell\b",
        ),
        4,
    ),
    Rule(
        "mobile_equipment_exposure",
        "Pedestrian / mobile-equipment interaction",
        (
            r"\bon foot\b",
            r"\bcross(ed|ing)?\b.*\broad\b",
            r"\bloader\b.*\bworker\b",
            r"\bvehicle\b.*\bworker\b",
            r"\bnear\b.*\btruck\b",
            r"\bbetween\b.*\b(equipment|machine|vehicle)\b",
        ),
        4,
    ),
    Rule(
        "material_exposure",
        "Material / substance exposure",
        (
            r"\breached?\b.*\b(face|eye|skin)\b",
            r"\bpaint\b.*\bface\b",
            r"\bhot gas\b.*\bface\b",
            r"\bhot pulp\b.*\b(employee|worker|operator)\b",
            r"\bcontact\b.*\b(chemical|paint|gas)\b",
        ),
        3,
    ),
)


CONSEQUENCES = (
    Rule(
        "fatality",
        "Fatality",
        (
            r"\bfatality\b",
            r"\bfatal\b",
            r"\bdeath\b",
            r"\bdied\b",
            r"\bdead\b",
            r"\bpassed away\b",
        ),
        6,
    ),
    Rule(
        "serious_injury",
        "Serious / potentially serious injury",
        (
            r"\bserious injury\b",
            r"\bsevere injury\b",
            r"\blife[\s-]?altering\b",
            r"\btraumatic injury\b",
            r"\bmajor injury\b",
            r"\bfracture(d|s)?\b",
            r"\bamputation\b",
            r"\bcrushing\b",
            r"\btrapped\b",
            r"\bimprisoned\b",
        ),
        5,
    ),
    Rule(
        "electrical_injury",
        "Electrical injury",
        (
            r"\belectric shock\b",
            r"\belectrocution\b",
            r"\barc[\s-]?flash\b.*\b(injury|reach|operator)\b",
            r"\bflash\b.*\breach(es|ed)?\b.*\boperator\b",
        ),
        5,
    ),
    Rule(
        "thermal_injury",
        "Burn / thermal injury",
        (
            r"\bburn(ed|s)?\b",
            r"\bscald(ed|ing)?\b",
            r"\bthermal injury\b",
            r"\bhot gas\b.*\b(injury|face|employee)\b",
        ),
        4,
    ),
    Rule(
        "struck_by",
        "Struck-by injury",
        (
            r"\bstruck by\b",
            r"\bhit\b",
            r"\bimpacted?\b",
            r"\bimpact(s|ed)?\b.*\b(arm|leg|head|worker)\b",
            r"\bgrazing\b",
            r"\bgrazed\b",
            r"\bblade\b.*\b(cut|injur)\b",
        ),
        4,
    ),
    Rule(
        "crush_amputation",
        "Crushing / amputation",
        (
            r"\bcrush(ing|ed)?\b",
            r"\bamputation\b",
            r"\btrapped\b",
            r"\bimprisoned\b",
            r"\b(run[\s-]?over|run over)\b",
        ),
        5,
    ),
    Rule(
        "cut_laceration",
        "Cut / laceration",
        (
            r"\bcut(s)?\b",
            r"\blacerat(ed|ion)\b",
            r"\bblunt cut\b",
            r"\bsmall cuts?\b",
            r"\bcut of\b",
        ),
        3,
    ),
    Rule(
        "minor_injury",
        "Minor injury / irritation",
        (
            r"\blittle trauma\b",
            r"\bminor\b",
            r"\bscratch\b",
            r"\bswelling\b",
            r"\birritation\b",
            r"\brash\b",
        ),
        2,
    ),
)


BARRIERS = (
    Rule(
        "isolation_loto",
        "Isolation / lockout-tagout",
        (
            r"\blockout\b",
            r"\btagout\b",
            r"\bLOTO\b",
            r"\block\b.*\bremoved\b",
            r"\bisolation\b",
            r"\bde-energiz(ed|ation|e|ing)\b",
        ),
        5,
    ),
    Rule(
        "guarding",
        "Machine guarding",
        (
            r"\bguard(ing|ed)?\b",
            r"\bmachine protection\b",
            r"\bprotective guard\b",
        ),
        4,
    ),
    Rule(
        "fall_protection",
        "Fall prevention / protection",
        (
            r"\bfall protection\b",
            r"\bfall prevention\b",
            r"\bguardrail\b",
            r"\bhandrail\b",
            r"\bsafety harness\b",
            r"\blifeline\b",
        ),
        5,
    ),
    Rule(
        "exclusion_control",
        "Exclusion / barricading / separation",
        (
            r"\bbarricad(e|ed|ing)\b",
            r"\bexclusion zone\b",
            r"\bmarked with tape\b",
            r"\bcones\b",
            r"\bcontrolled access\b",
            r"\bseparation\b",
        ),
        4,
    ),
    Rule(
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
        5,
    ),
    Rule(
        "ground_control",
        "Ground control / scaling",
        (
            r"\bground control\b",
            r"\bground-control\b",
            r"\bscaling\b",
            r"\bground support\b",
            r"\broof support\b",
        ),
        5,
    ),
    Rule(
        "electrical_protection",
        "Electrical protection",
        (
            r"\babsence[- ]of[- ]voltage\b",
            r"\babsence of voltage\b",
            r"\barc[\s-]?flash PPE\b",
            r"\belectrical isolation\b",
            r"\bprotection system\b",
            r"\bprotective equipment\b",
            r"\bgloves?\b",
            r"\bPPE\b",
        ),
        3,
    ),
    Rule(
        "procedure_work_control",
        "Procedure / work control",
        (
            r"\bprocedure\b",
            r"\bpermit\b",
            r"\bJSA\b",
            r"\bjob safety analysis\b",
            r"\bwork permit\b",
        ),
        2,
    ),
)


ACTIVITIES = (
    Rule(
        "drilling",
        "Drilling",
        (
            r"\bdrill(ing)?\b",
            r"\bloading drills\b",
            r"\bproduction drills\b",
            r"\bdrill holes\b",
        ),
        5,
    ),
    Rule(
        "shotcrete",
        "Shotcrete",
        (
            r"\bshotcrete\b",
            r"\bshotcrete casting\b",
            r"\bshotcrete launch\b",
        ),
        5,
    ),
    Rule(
        "painting",
        "Painting",
        (
            r"\bpaint(ing|ed)?\b",
            r"\bpainting\b",
        ),
        4,
    ),
    Rule(
        "washing_cleaning",
        "Washing / cleaning",
        (
            r"\bwash(ing|ed)?\b",
            r"\bclean(ing|ed)?\b",
            r"\bcleaning\b",
        ),
        4,
    ),
    Rule(
        "assembly_installation",
        "Assembly / installation",
        (
            r"\bassembl(y|ing)\b",
            r"\binstall(ation|ing)\b",
            r"\binstallation\b",
        ),
        3,
    ),
    Rule(
        "inspection",
        "Inspection",
        (
            r"\binspect(ion|ing|ed)\b",
            r"\binspection\b",
        ),
        3,
    ),
    Rule(
        "adjustment_tightening",
        "Adjustment / tightening",
        (
            r"\badjustment\b",
            r"\btighten(ing)?\b",
            r"\bloos(en|ing)\b",
        ),
        3,
    ),
)


def compile_rules(
    rules: Iterable[Rule],
) -> dict[str, list[tuple[Rule, re.Pattern[str]]]]:
    return {
        rule.key: [
            (
                rule,
                re.compile(
                    pattern,
                    flags=re.IGNORECASE,
                ),
            )
            for pattern in rule.patterns
        ]
        for rule in rules
    }


def sentence_spans(
    text: str,
) -> list[tuple[int, int, str]]:
    pattern = re.compile(
        r"[^.!?\n]+(?:[.!?]+|$)",
        flags=re.MULTILINE,
    )

    spans = []

    for match in pattern.finditer(text):
        raw = match.group(0)
        leading = len(raw) - len(raw.lstrip())
        start = match.start() + leading
        end = match.end()
        sentence = text[start:end].strip()

        if sentence:
            spans.append(
                (start, end, sentence)
            )

    return spans


def confidence(
    rule: Rule,
    evidence_text: str,
) -> str:
    lower = evidence_text.lower()

    strong = (
        len(evidence_text.split()) >= 3
        or rule.priority >= 5
        or any(
            phrase in lower
            for phrase in [
                "440 v",
                "arc flash",
                "electric shock",
                "lock removed",
                "ground control",
                "fall protection",
                "line of fire",
                "fatal",
                "death",
                "imprisoned",
                "struck by",
            ]
        )
    )

    if strong:
        return "HIGH"

    if rule.priority >= 3 or len(evidence_text.split()) >= 2:
        return "MEDIUM"

    return "LOW"


def extract_category(
    text: str,
    rules: tuple[Rule, ...],
) -> list[Evidence]:
    compiled = compile_rules(rules)
    spans = sentence_spans(text)

    hits: list[Evidence] = []

    for rule in rules:
        rule_hits = []

        for sentence_start, sentence_end, sentence in spans:
            for current_rule, pattern in compiled[rule.key]:
                match = pattern.search(sentence)

                if not match:
                    continue

                start = (
                    sentence_start
                    + match.start()
                )
                end = (
                    sentence_start
                    + match.end()
                )

                evidence_text = text[start:end]

                rule_hits.append(
                    Evidence(
                        ontology_key=current_rule.key,
                        label=current_rule.label,
                        text=evidence_text,
                        start=start,
                        end=end,
                        sentence=sentence,
                        match_type="contextual_lexical",
                        confidence=confidence(
                            current_rule,
                            evidence_text,
                        ),
                    )
                )

                break

        # Prefer highest-quality evidence for each ontology key.
        if rule_hits:
            rule_hits.sort(
                key=lambda item: (
                    {"HIGH": 3, "MEDIUM": 2, "LOW": 1}[
                        item.confidence
                    ],
                    item.end - item.start,
                ),
                reverse=True,
            )
            hits.append(rule_hits[0])

    return sorted(
        hits,
        key=lambda item: (
            item.start,
            -item.end,
        ),
    )


def extract_narrative(
    report_id: str,
    narrative: str,
) -> ExtractionResult:
    hazards = extract_category(
        narrative,
        HAZARDS,
    )
    exposures = extract_category(
        narrative,
        EXPOSURES,
    )
    consequences = extract_category(
        narrative,
        CONSEQUENCES,
    )
    barriers = extract_category(
        narrative,
        BARRIERS,
    )
    activities = extract_category(
        narrative,
        ACTIVITIES,
    )

    summary = {
        "hazard_count": len(hazards),
        "exposure_count": len(exposures),
        "consequence_count": len(consequences),
        "barrier_count": len(barriers),
        "activity_count": len(activities),
        "high_confidence_hazards": sum(
            item.confidence == "HIGH"
            for item in hazards
        ),
        "high_confidence_exposures": sum(
            item.confidence == "HIGH"
            for item in exposures
        ),
        "high_confidence_consequences": sum(
            item.confidence == "HIGH"
            for item in consequences
        ),
        "has_fatality_evidence": any(
            item.ontology_key == "fatality"
            for item in consequences
        ),
        "has_direct_exposure_evidence": any(
            item.ontology_key
            in {
                "direct_contact",
                "body_part_exposure",
                "line_of_fire",
                "fall_exposure",
                "mobile_equipment_exposure",
                "material_exposure",
            }
            for item in exposures
        ),
        "has_severe_consequence_signal": any(
            item.ontology_key
            in {
                "fatality",
                "serious_injury",
                "electrical_injury",
                "thermal_injury",
                "crush_amputation",
            }
            for item in consequences
        ),
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
            "Input must contain report_id."
        )

    description_column = None

    for column in [
        "description",
        "Description",
        "narrative",
    ]:
        if column in df.columns:
            description_column = column
            break

    if description_column is None:
        raise ValueError(
            "Input must contain description/narrative."
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
            handle.write(
                json.dumps(
                    {
                        "report_id": result.report_id,
                        "narrative": result.narrative,
                        "hazards": [
                            asdict(item)
                            for item in result.hazards
                        ],
                        "exposures": [
                            asdict(item)
                            for item in result.exposures
                        ],
                        "consequences": [
                            asdict(item)
                            for item in result.consequences
                        ],
                        "barriers": [
                            asdict(item)
                            for item in result.barriers
                        ],
                        "activities": [
                            asdict(item)
                            for item in result.activities
                        ],
                        "summary": result.summary,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def write_flat_csv(
    results: list[ExtractionResult],
    path: Path,
) -> None:
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

    rows = []

    for result in results:
        rows.append(
            {
                "report_id": result.report_id,
                "hazards": labels(result.hazards),
                "hazard_evidence": evidence(result.hazards),
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
                "activities": labels(result.activities),
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
    print(
        f"REPORT: {result.report_id}"
    )
    print("-" * 78)
    print(result.narrative)
    print()

    for title, items in [
        ("HAZARDS", result.hazards),
        ("EXPOSURES", result.exposures),
        ("CONSEQUENCES", result.consequences),
        ("BARRIERS", result.barriers),
        ("ACTIVITIES", result.activities),
    ]:
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
    parser = argparse.ArgumentParser()

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
        default=5,
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
    print("RAKSHAK SAFETY INFORMATION EXTRACTION v0.2")
    print("=" * 78)
    print(f"Input   : {args.input}")
    print(f"Records : {len(df)}")
    print()
    print(
        "Method  : contextual evidence-first safety ontology"
    )
    print(
        "Important: extracted evidence is not a SIF decision."
    )

    results = []

    for row in df.itertuples(index=False):
        result = extract_narrative(
            str(row.report_id),
            str(row.description),
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

    summary = {
        "version": "safety_information_extraction_v0.2",
        "records_processed": len(results),
        "total_evidence": {
            "hazards": sum(
                len(result.hazards)
                for result in results
            ),
            "exposures": sum(
                len(result.exposures)
                for result in results
            ),
            "consequences": sum(
                len(result.consequences)
                for result in results
            ),
            "barriers": sum(
                len(result.barriers)
                for result in results
            ),
            "activities": sum(
                len(result.activities)
                for result in results
            ),
        },
        "limitations": [
            "Rule-based evidence extraction can still miss paraphrases.",
            "Evidence does not establish SIF potential by itself.",
            "A detected term can still be contextually irrelevant and must be reviewed.",
            "This version is intended as a structured-information baseline before a learned extraction model.",
        ],
    }

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "Generated structured extraction for "
        f"{len(results)} records."
    )

    for result in results:
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
