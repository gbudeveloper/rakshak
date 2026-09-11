# RAKSHAK SIF Annotation Protocol v0.2

## 1. Objective

Determine whether the safety report contains credible Serious Injury
and Fatality (SIF) potential.

The annotation is based on the evidence contained in the narrative.

## 2. Primary label

Exactly one:

- YES
- NO
- UNCERTAIN

## 3. Core decision question

Could the event or exposure described in the report, without inventing
material facts that are absent from the report, credibly result in a
serious injury or fatality?

## 4. Decision sequence

### Step 1 — Hazard / energy

Identify the hazardous source, condition, or energy.

Examples:

- electrical
- mechanical
- pressure
- thermal
- chemical
- gravitational
- mobile equipment
- suspended load
- confined-space
- ground instability

Do not infer a hazard that is not supported by the narrative.

### Step 2 — Exposure

Identify how a person was exposed.

Examples:

- directly struck
- inside line of fire
- contacted energized equipment
- beneath unstable material
- exposed to release
- inside moving-equipment zone

A hazard without credible human exposure is not automatically SIF.

### Step 3 — Potential consequence

Determine whether the described exposure has a credible pathway to:

- fatality
- permanent/life-altering injury
- other serious injury

## 5. Counterfactual restraint

The annotator may consider a realistic potential consequence, but must
not introduce unsupported material changes.

Do not arbitrarily increase:

- quantity
- concentration
- pressure
- energy
- height
- speed
- duration
- body area exposed
- number of exposed workers

unless supported by the report.

Example:

Bad:
"The acid could have been much more concentrated."

Better:
"The reported facial exposure to corrosive acid presents credible
potential for severe eye or tissue injury."

## 6. YES

Use YES when:

1. credible hazard/energy is present;
2. a person is credibly exposed; and
3. the described exposure itself has a credible serious/fatal pathway.

## 7. NO

Use NO when the narrative provides enough evidence that the described
exposure does not present credible SIF potential.

Minor injury alone does not determine NO.

## 8. UNCERTAIN

Use UNCERTAIN when a missing fact materially affects the SIF decision.

Examples:

- critical exposure details are missing;
- consequence depends on an unknown material property;
- fall height/mechanism is unknown;
- whether hazardous energy was present is unclear;
- narrative is too incomplete for a defensible judgment.

Do not use UNCERTAIN merely because the case is difficult.

## 9. Barrier annotation

Record what is supported by the report.

Use:

- effective/present
- degraded
- absent
- bypassed
- failed
- unknown/not stated

Do not turn the barrier field into a recommended corrective-action plan.

## 10. Evidence

Evidence must be grounded in the source narrative.

Prefer exact text spans.

## 11. Reason

Reason should be concise and explain:

hazard → exposure → potential consequence

## 12. Confidence

### HIGH

The evidence clearly supports the decision.

### MEDIUM

The decision is reasonably supported, but meaningful uncertainty remains.

### LOW

Evidence is limited or ambiguous.

## 13. Source-label blindness

Do not use:

- Accident Level
- Potential Accident Level

as the SIF decision.

These are source fields, not the Gold Set target.

## 14. Separation of concepts

Actual outcome != potential consequence

Potential Accident Level != SIF potential

Hazard != exposure

Barrier condition != corrective action

## 15. Quality rule

When a reasonable YES and NO interpretation both depend on a missing
material fact, use UNCERTAIN.

## 16. Annotation rounds

Pilot 01:
pre-v0.2

Pilot 02:
pre-v0.2

New annotations:
v0.2
