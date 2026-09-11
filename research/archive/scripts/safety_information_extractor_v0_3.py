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
    / "safety_information_extraction_v0.3"
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
# v0.3 DESIGN
#
# v0.2 improved recall but exposed a precision problem:
#   - "opening" -> fall hazard
#   - "flash" -> fire/explosion hazard
#   - "gloves" -> electrical protection
#   - "flange" -> pressure hazard
#   - "hit" -> struck-by consequence
#
# v0.3 therefore:
#   1) removes several broad single-token rules
#   2) prioritizes multi-word contextual phrases
#   3) separates actual event consequence from hazard cues
#   4) recognizes caught-between / crush mechanisms
#   5) treats PPE only as a barrier when safety-function context exists
#
# This remains an evidence extraction baseline, not a SIF engine.
# ============================================================


HAZARDS = (
    Rule(
        "electrical_energy",
        "Electrical energy",
        (
            r"\b\d+\s*v\b.*\b(electric|panel|board|equipment|installation)\b",
            r"\belectric(al)?\s+(board|panel|equipment|installation)\b",
            r"\benergiz(e|ed|ing|ation)\b",
            r"\barc[\s-]?flash\b",
            r"\bphase\s*(to|-)\s*ground\b",
            r"\belectric shock\b",
            r"\belectrocution\b",
            r"\bhigh voltage\b",
            r"\bpower cord\b",
        ),
        6,
    ),
    Rule(
        "mechanical_energy",
        "Mechanical / moving equipment energy",
        (
            r"\brotat(ing|ion)\b",
            r"\bmoving\s+(equipment|machine|part)\b",
            r"\bcutter blade\b",
            r"\bblade\b.*\b(detached|released|slip|hit|cut)\b",
            r"\bmachine\b.*\b(started|move|moving)\b",
            r"\bmechanical\b",
            r"\bpinch(ed|ing)?\b",
            r"\bcrush(ing|ed)?\b",
            r"\bentangle(d|ment)?\b",
        ),
        5,
    ),
    Rule(
        "pressure_process_energy",
        "Pressure / process energy",
        (
            r"\bpressurized\b",
            r"\bpressurised\b",
            r"\bstored pressure\b",
            r"\bpressure\b.*\b(release|rupture|line|piping)\b",
            r"\bprocess line\b",
            r"\bpipe\b.*\b(pressure|release|rupture)\b",
            r"\bpiping\b.*\b(pressure|release|rupture)\b",
        ),
        5,
    ),
    Rule(
        "thermal_energy",
        "Thermal / hot material",
        (
            r"\bhot gas\b",
            r"\bhot pulp\b",
            r"\bhot material\b",
            r"\bthermal\b",
            r"\bhigh temperature\b",
            r"\bburn(ed|s)?\b.*\b(hot|gas|material|thermal)\b",
            r"\bfurnace\b.*\b(gas|heat|hot)\b",
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
            r"\bheight of \d+\s*(m|meters?)\b",
            r"\bat an approximate height\b",
            r"\bvertical opening\b",
            r"\bopen shaft\b",
            r"\bchimney\b",
            r"\bfell\b.*\b\d+\s*(m|meters?)\b",
        ),
        6,
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
            r"\brock\b.*\bdetached\b",
            r"\broof\b.*\bdetached\b",
        ),
        6,
    ),
    Rule(
        "mobile_equipment_energy",
        "Mobile equipment / vehicle movement",
        (
            r"\bwheel[\s-]?loader\b",
            r"\bmobile equipment\b",
            r"\bvehicle\b.*\b(worker|employee|operator)\b",
            r"\bloader\b.*\b(worker|employee|operator)\b",
            r"\btire\b.*\b(worker|employee|operator)\b",
            r"\bvehicle\b.*\bmovement\b",
        ),
        5,
    ),
    Rule(
        "chemical_exposure",
        "Chemical substance / exposure",
        (
            r"\bpaint\b.*\b(face|eye|skin|exposure|contact)\b",
            r"\bchemical\b.*\b(exposure|contact|release)\b",
            r"\bsolvent\b.*\b(exposure|contact)\b",
            r"\bcorrosive\b.*\b(exposure|contact)\b",
        ),
        4,
    ),
    Rule(
        "suspended_load",
        "Suspended / falling load",
        (
            r"\bsuspended load\b",
            r"\bfalling load\b",
            r"\bcrane\b.*\b(load|hook|lift)\b",
            r"\bload\b.*\b(fell|falling|dropped)\b",
            r"\bobject\b.*\b(fell|falling|dropped)\b",
        ),
        5,
    ),
    Rule(
        "fire_explosion",
        "Fire / explosion energy",
        (
            r"\bexplosion\b",
            r"\bexplosive\b",
            r"\bfire\b.*\b(event|ignition|explosion)\b",
            r"\bignit(e|ed|ion|ing)\b",
        ),
        5,
    ),
    Rule(
        "biological_hazard",
        "Biological / animal hazard",
        (
            r"\bbees?\b",
            r"\bstung\b",
            r"\bsting\b",
            r"\bbites?\b",
            r"\bvenomous\b",
            r"\bsnake\b",
        ),
        3,
    ),
)


EXPOSURES = (
    Rule(
        "direct_exposure",
        "Direct worker exposure",
        (
            r"\breaches?\b.*\b(worker|operator|employee|collaborator)\b",
            r"\breach(?:es|ed)?\s+the\s+(operator|worker|employee)\b",
            r"\bcausing the injury\b",
            r"\bcausing.*\binjury\b",
            r"\bdirectly exposed\b",
            r"\bexposed to\b",
        ),
        6,
    ),
    Rule(
        "body_part_contact",
        "Body part contact",
        (
            r"\b(grazing|grazed|hit|struck|touch(ed)?|rub(bed|bing)?)\b.*\b(hand|arm|leg|face|eye|head|finger|foot|thumb)\b",
            r"\b(hand|arm|leg|face|eye|head|finger|foot|thumb)\b.*\b(grazed|hit|struck|burn|cut|contact|rubbed)\b",
            r"\b(right|left)\s+(hand|arm|leg|forearm|thumb|finger)\b.*\b(cut|hit|struck|grazed)\b",
        ),
        6,
    ),
    Rule(
        "caught_between",
        "Caught-between / pinch exposure",
        (
            r"\bimprisoned between\b",
            r"\btrapped between\b",
            r"\bcaught between\b",
            r"\bbetween\b.*\b(equipment|machine|vehicle|wall|rim)\b.*\band\b",
        ),
        6,
    ),
    Rule(
        "line_of_fire",
        "Line-of-fire exposure",
        (
            r"\bline of fire\b",
            r"\bin the path of\b",
            r"\bin the direction of\b",
            r"\bproject(ed|ile|ion)\b.*\b(worker|operator|employee)\b",
            r"\bstruck by\b",
        ),
        6,
    ),
    Rule(
        "fall_exposure",
        "Fall exposure",
        (
            r"\bworking\b.*\bheight of \d+\s*(m|meters?)\b",
            r"\bworking\b.*\b(at|from)\b.*\bheight\b",
            r"\bnear\b.*\b(vertical opening|chimney|shaft)\b",
            r"\badjacent to\b.*\b(opening|chimney|shaft)\b",
            r"\bfell\b.*\bheight\b",
        ),
        5,
    ),
    Rule(
        "mobile_equipment_exposure",
        "Pedestrian / mobile-equipment interaction",
        (
            r"\bon foot\b.*\b(loader|truck|vehicle)\b",
            r"\bcross(ed|ing)?\b.*\broad\b.*\b(loader|truck|vehicle)\b",
            r"\bworker\b.*\bbetween\b.*\b(equipment|machine|vehicle)\b",
            r"\boperator\b.*\bbetween\b.*\b(equipment|machine|vehicle)\b",
        ),
        5,
    ),
    Rule(
        "material_contact",
        "Material / substance contact",
        (
            r"\bpaint\b.*\bface\b",
            r"\bpaint\b.*\bskin\b",
            r"\bhot gas\b.*\bface\b",
            r"\bhot pulp\b.*\b(employee|worker|operator)\b",
            r"\bchemical\b.*\b(contact|exposure)\b",
        ),
        5,
    ),
)


CONSEQUENCES = (
    Rule(
        "fatality",
        "Fatality",
        (
            r"\bfatality\b",
            r"\bdeath\b",
            r"\bdied\b",
            r"\bdead\b",
        ),
        7,
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
            r"\btrapped\b",
            r"\bimprisoned\b",
        ),
        6,
    ),
    Rule(
        "crush_amputation",
        "Crushing / amputation",
        (
            r"\bcrush(ing|ed)?\b",
            r"\bamputation\b",
            r"\btrapped between\b",
            r"\bcaught between\b",
            r"\bimprisoned between\b",
        ),
        6,
    ),
    Rule(
        "electrical_injury",
        "Electrical injury",
        (
            r"\belectric shock\b",
            r"\belectrocution\b",
            r"\barc[\s-]?flash\b.*\b(injury|operator|employee|worker)\b",
            r"\bflash\b.*\breach(es|ed)?\b.*\boperator\b",
        ),
        6,
    ),
    Rule(
        "thermal_injury",
        "Burn / thermal injury",
        (
            r"\bburn(ed|s)?\b",
            r"\bscald(ed|ing)?\b",
            r"\bthermal injury\b",
        ),
        4,
    ),
    Rule(
        "struck_by",
        "Struck-by injury",
        (
            r"\bstruck by\b",
            r"\bwas hit\b",
            r"\bhit (him|her|the worker|the employee)\b",
            r"\bimpacted?\b.*\b(arm|leg|head|worker|employee)\b",
            r"\bgrazing\b.*\b(hand|arm|finger)\b",
            r"\bgrazed\b.*\b(hand|arm|finger)\b",
        ),
        4,
    ),
    Rule(
        "cut_laceration",
        "Cut / laceration",
        (
            r"\bcut(s)?\b.*\b(hand|finger|arm|thumb)\b",
            r"\bcut of\b.*\b(hand|finger|arm|thumb)\b",
            r"\blacerat(ed|ion)\b",
            r"\bblunt cut\b",
            r"\bsmall cuts?\b",
        ),
        4,
    ),
    Rule(
        "minor_injury",
        "Minor injury / irritation",
        (
            r"\blittle trauma\b",
            r"\bminor injury\b",
            r"\bswelling\b",
            r"\birritation\b",
            r"\brash\b",
        ),
        3,
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
            r"\bde-energiz(e|ed|ation|ing)\b",
        ),
        7,
    ),
    Rule(
        "guarding",
        "Machine guarding",
        (
            r"\bguard(ing|ed)?\b.*\bmachine\b",
            r"\bmachine protection\b",
            r"\bprotective guard\b",
        ),
        6,
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
        6,
    ),
    Rule(
        "exclusion_control",
        "Exclusion / barricading / separation",
        (
            r"\bbarricad(e|ed|ing)\b",
            r"\bexclusion zone\b",
            r"\bmarked with tape\b",
            r"\bcontrolled access\b",
            r"\bseparation\b.*\b(worker|equipment|vehicle)\b",
            r"\bcones\b.*\bwarning\b",
        ),
        5,
    ),
    Rule(
        "pressure_isolation",
        "Pressure isolation / depressurization",
        (
            r"\bdepressuriz(e|ation|ed)\b",
            r"\bdepressurise\b",
            r"\bpressure verification\b",
            r"\bpositive isolation\b",
            r"\bbleed(ing)?\b.*\bline\b",
            r"\bdrain(ing)?\b.*\bline\b",
        ),
        6,
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
        6,
    ),
    Rule(
        "electrical_protection",
        "Electrical protection",
        (
            r"\belectrical isolation\b",
            r"\babsence[- ]of[- ]voltage\b",
            r"\babsence of voltage\b",
            r"\barc[\s-]?flash PPE\b",
            r"\belectrical protective equipment\b",
            r"\belectrical protection\b",
        ),
        7,
    ),
    # PPE is only recognized as a safety barrier when it is
    # explicitly linked to a safety function. A bare "gloves"
    # hit is intentionally NOT enough.
    Rule(
        "personal_protective_equipment",
        "Personal protective equipment",
        (
            r"\bprotective equipment\b",
            r"\bPPE\b",
            r"\bsafety gloves\b",
            r"\bprotective gloves\b",
            r"\bgloves\b.*\b(protection|protective)\b",
        ),
        3,
    ),
    Rule(
        "procedure_work_control",
        "Procedure / work control",
        (
            r"\bprocedure\b",
            r"\bpermit\b.*\bwork\b",
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
            r"\binstall(ation|ing)\b",
            r"\bassembly\b",
            r"\bassembling\b",
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

    very_specific = [
        "440v",
        "arc flash",
        "lock removed",
        "imprisoned between",
        "trapped between",
        "caught between",
        "vertical opening",
        "fall from height",
        "fall protection",
        "ground control",
        "electric shock",
        "electrocution",
        "fatality",
        "death",
        "line of fire",
    ]

    if (
        rule.priority >= 6
        or any(
            phrase in lower
            for phrase in very_specific
        )
    ):
        return "HIGH"

    if (
        rule.priority >= 4
        or len(evidence_text.split()) >= 2
    ):
        return "MEDIUM"

    return "LOW"


def extract_category(
    text: str,
    rules: tuple[Rule, ...],
) -> list[Evidence]:
    compiled = compile_rules(rules)
    spans = sentence_spans(text)

    hits = []

    for rule in rules:
        candidates = []

        for sentence_start, sentence_end, sentence in spans:
            for current_rule, pattern in compiled[rule.key]:
                match = pattern.search(sentence)

                if not match:
                    continue

                absolute_start = (
                    sentence_start + match.start()
                )
                absolute_end = (
                    sentence_start + match.end()
                )

                candidates.append(
                    Evidence(
                        ontology_key=current_rule.key,
                        label=current_rule.label,
                        text=text[
                            absolute_start:absolute_end
                        ],
                        start=absolute_start,
                        end=absolute_end,
                        sentence=sentence,
                        match_type="contextual_lexical",
                        confidence=confidence(
                            current_rule,
                            text[
                                absolute_start:absolute_end
                            ],
                        ),
                    )
                )
                break

        if candidates:
            candidates.sort(
                key=lambda item: (
                    {"HIGH": 3, "MEDIUM": 2, "LOW": 1}[
                        item.confidence
                    ],
                    len(item.text),
                ),
                reverse=True,
            )
            hits.append(candidates[0])

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

    # --------------------------------------------------------
    # Precision guardrails.
    #
    # A word such as "flash" should not automatically become
    # fire/explosion when the text explicitly describes an
    # electrical phase-to-ground flash.
    # --------------------------------------------------------

    lower = narrative.lower()

    if (
        "phase to ground" in lower
        or "phase-to-ground" in lower
        or "440v" in lower
    ):
        hazards = [
            item
            for item in hazards
            if item.ontology_key != "fire_explosion"
        ]

    # A bare "opening" must not generate fall hazard.
    if not re.search(
        r"\b(vertical opening|open shaft|chimney|fall from height|height of \d+)",
        narrative,
        flags=re.IGNORECASE,
    ):
        hazards = [
            item
            for item in hazards
            if item.ontology_key != "gravity_fall_energy"
        ]

    # A flange or pipe alone is insufficient evidence of pressure.
    if not re.search(
        r"\b(pressurized|pressurised|stored pressure|pressure release|"
        r"pressure rupture|process line|piping.*(pressure|release|rupture)|"
        r"pipe.*(pressure|release|rupture))\b",
        narrative,
        flags=re.IGNORECASE,
    ):
        hazards = [
            item
            for item in hazards
            if item.ontology_key
            != "pressure_process_energy"
        ]

    # A glove alone is not evidence of a specific barrier.
    if not re.search(
        r"\b(safety gloves|protective gloves|gloves.*(protection|protective)|"
        r"protective equipment|PPE)\b",
        narrative,
        flags=re.IGNORECASE,
    ):
        barriers = [
            item
            for item in barriers
            if item.ontology_key
            != "personal_protective_equipment"
        ]

    # --------------------------------------------------------
    # Derived summary signals. These are feature flags only;
    # they do not determine SIF.
    # --------------------------------------------------------

    summary = {
        "hazard_count": len(hazards),
        "exposure_count": len(exposures),
        "consequence_count": len(consequences),
        "barrier_count": len(barriers),
        "activity_count": len(activities),
        "has_fatality_evidence": any(
            item.ontology_key == "fatality"
            for item in consequences
        ),
        "has_caught_between_evidence": any(
            item.ontology_key == "caught_between"
            for item in exposures
        ),
        "has_direct_exposure_evidence": any(
            item.ontology_key
            in {
                "direct_exposure",
                "body_part_contact",
                "caught_between",
                "line_of_fire",
                "fall_exposure",
                "mobile_equipment_exposure",
                "material_contact",
            }
            for item in exposures
        ),
        "has_severe_consequence_signal": any(
            item.ontology_key
            in {
                "fatality",
                "serious_injury",
                "crush_amputation",
                "electrical_injury",
                "thermal_injury",
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
                f'  - {item.label} [{item.confidence}] '
                f'→ "{item.text}"'
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAKSHAK evidence-first safety extractor v0.3"
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
    print("RAKSHAK SAFETY INFORMATION EXTRACTION v0.3")
    print("=" * 78)
    print(f"Input   : {args.input}")
    print(f"Records : {len(df)}")
    print()
    print(
        "Method  : contextual evidence-first extraction "
        "with precision guardrails"
    )
    print(
        "Important: extraction is not a SIF decision."
    )

    results = []

    for row in df.itertuples(index=False):
        results.append(
            extract_narrative(
                str(row.report_id),
                str(row.description),
            )
        )

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
        "version": "safety_information_extraction_v0.3",
        "records_processed": len(results),
        "total_evidence": {
            "hazards": sum(
                len(r.hazards)
                for r in results
            ),
            "exposures": sum(
                len(r.exposures)
                for r in results
            ),
            "consequences": sum(
                len(r.consequences)
                for r in results
            ),
            "barriers": sum(
                len(r.barriers)
                for r in results
            ),
            "activities": sum(
                len(r.activities)
                for r in results
            ),
        },
        "design_goal": (
            "Improve precision over v0.2 by requiring contextual "
            "evidence for hazards and barriers and explicitly "
            "handling caught-between mechanisms."
        ),
        "limitations": [
            "Still rule-based; paraphrases can be missed.",
            "A detected evidence span can still require human review.",
            "Does not infer missing consequence severity.",
            "Does not decide SIF potential.",
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
        f"Generated structured extraction for {len(results)} records."
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
