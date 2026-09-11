# RAKSHAK SIF Annotation Protocol v0.1

## Purpose

Create a human-adjudicated Gold Set for identifying whether a safety
report contains credible Serious Injury & Fatality (SIF) potential.

## Primary label

Each report receives exactly one label:

- YES
- NO
- UNCERTAIN

## Core question

Based only on the evidence contained in the report:

Could the described event or exposure realistically have resulted
in a serious injury or fatality?

This is a potential-consequence judgment, not an actual-injury judgment.

## YES

Use YES when the narrative provides credible evidence of an exposure
or event that could realistically produce serious injury or fatality.

Typical evidence categories include:

- person in line of fire
- uncontrolled or stored energy
- vehicle/mobile-equipment interaction
- significant fall exposure
- suspended-load exposure
- electrical exposure
- pressure or chemical release
- confined-space exposure
- critical barrier failure
- other credible high-energy exposure

These are evidence categories, not keyword rules.

## NO

Use NO when the report contains enough information to conclude that
serious/fatal potential is not credible.

## UNCERTAIN

Use UNCERTAIN when the available narrative is insufficient to make
a defensible YES/NO decision.

Examples:

- critical context is missing
- exposure cannot be established
- possible consequence is ambiguous
- barrier/context information is insufficient

## Evidence

Record the exact portion of the narrative that supports the decision.

Avoid writing only a general interpretation when exact evidence exists.

## Reason

Write a concise explanation of why the evidence supports the label.

## Hazard notes

Record the relevant hazardous exposure, energy source, or unsafe situation.

## Consequence notes

Describe the credible serious/fatal consequence.

## Barrier notes

Describe barriers when identifiable:

- present/effective
- degraded
- absent
- bypassed
- failed
- unknown

Do not invent barriers that are not supported by the narrative.

## Confidence

Annotator confidence:

- HIGH — evidence is clear
- MEDIUM — evidence is reasonably clear but some uncertainty remains
- LOW — limited or ambiguous evidence

## Important distinctions

Actual outcome != potential consequence

Potential Accident Level != SIF potential

SIF potential != barrier status

An unlabeled report is not automatically NO.

## Blind-labeling rule

During SIF annotation, the following source fields must not be shown
to the annotator:

- Accident Level
- Potential Accident Level

They remain in the underlying dataset for later analysis.

## Quality rule

Do not use keyword matching as the basis for a label.

The decision must consider:

1. exposure
2. credible consequence
3. available evidence
4. barrier condition where identifiable
