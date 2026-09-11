# RAKSHAK SIF Annotation Protocol v0.3

## Purpose
Create a defensible human-adjudicated label for whether a safety report contains credible Serious Injury and Fatality (SIF) potential.

## Primary label
Exactly one:
- YES
- NO
- UNCERTAIN

## Core decision
Based only on the evidence contained in the report:

> Could the described exposure itself, or a tightly bounded consequence of that exposure, credibly result in a serious injury or fatality?

Do not invent material facts.

## Decision sequence

### 1. Hazard / energy
Identify the hazard, energy source, or hazardous condition supported by the narrative.

### 2. Human exposure
Identify the actual or credibly imminent human exposure.

A hazard with no credible human exposure is not automatically SIF.

### 3. Consequence pathway
Assess whether the described exposure has a credible pathway to a serious, permanently disabling, or fatal consequence.

### 4. Barrier condition
Record only barrier/control information supported by the narrative.

Use:
- effective / present
- degraded
- absent
- bypassed
- failed
- unknown / not stated

Do not turn this field into a corrective-action recommendation.

## Counterfactual restraint
A potential consequence may be considered only when it is a realistic consequence of the described mechanism.

Do NOT create a hypothetical event by arbitrarily changing:
- concentration
- quantity
- pressure
- speed
- height
- energy
- duration
- exposure area
- number of exposed people

## YES
Use YES when the reported exposure itself has a credible SIF pathway.

Typical examples include:
- uncontrolled high-energy electrical exposure
- line-of-fire exposure to moving/heavy equipment
- significant fall-from-height exposure
- uncontrolled ground/rock fall exposure
- suspended-load exposure
- molten-metal exposure
- uncontrolled pressure/process release
- other clearly high-energy exposure with credible serious consequence

These are categories, not keyword rules.

## NO
Use NO when the narrative provides sufficient evidence that the described exposure does not have a credible SIF pathway.

Do not use NO merely because the actual injury was minor.

## UNCERTAIN
Use UNCERTAIN when a missing material fact could reasonably change the YES/NO decision.

Examples:
- chemical concentration/material properties are unknown and materially affect consequence severity;
- amount/exposure duration is unclear;
- fall mechanism or height is insufficiently described;
- hazardous energy state cannot be established;
- the narrative is materially incomplete.

## Evidence
Copy exact supporting text from the report where practical.

## Reason
The reason should focus on:
hazard → exposure → consequence

Avoid unsupported hypothetical escalation.

## Hazard notes
Describe the observed hazard/energy.

## Exposure notes
Describe how the person was exposed.

## Potential consequence
Describe the credible consequence of the reported exposure.

## Barrier notes
Describe observed barrier/control condition only.

Separate recommendations from annotation.

## Confidence
HIGH = evidence clearly supports the decision.
MEDIUM = decision is reasonably supported, but a meaningful ambiguity remains.
LOW = important evidence is limited or ambiguous.

## Source-label blindness
Do not use Accident Level or Potential Accident Level to make the SIF decision.

## Conceptual separation
Actual outcome != potential consequence
Potential Accident Level != SIF potential
Hazard != exposure
Barrier condition != corrective action

## Annotation rounds
Pilot 01: pre-v0.2
Pilot 02: pre-v0.2
Calibration Batch 03: v0.2
Boundary adjudication: v0.3
