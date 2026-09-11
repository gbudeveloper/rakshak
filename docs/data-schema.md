# RAKSHAK Canonical Safety Report Schema

The canonical schema is the internal contract used by all dataset adapters,
preprocessing, ML, evaluation and API components.

Source datasets must be converted into this schema before model training.

## Identity

- `report_id`
  - string
  - globally unique internal identifier

- `source`
  - string
  - dataset/source name

- `report_type`
  - string
  - e.g. incident, near_miss, unsafe_act, unsafe_condition, observation

## Time and location

- `timestamp`
  - datetime or null

- `site`
  - string or null

- `location`
  - string or null

- `department`
  - string or null

- `activity`
  - string or null

- `asset`
  - string or null

## Narrative

- `description`
  - original safety-event narrative

## Safety semantics

- `unsafe_act`
  - boolean or null

- `unsafe_condition`
  - boolean or null

- `near_miss`
  - boolean or null

- `hi_po`
  - boolean or null

- `hazards`
  - list of strings

- `exposure`
  - list of strings

- `potential_consequences`
  - list of strings

- `barriers`
  - list of strings

- `barrier_status`
  - list of strings

- `life_saving_rules`
  - list of strings

## Outcomes and source labels

- `actual_outcome`
  - string or null

- `potential_accident_level`
  - string or null
  - source-specific field
  - NOT equivalent to `sif_potential`

- `sif_potential`
  - `yes`, `no`, `uncertain`, or null
  - gold human/adjudicated target
  - must not be inferred automatically from unrelated labels

## Annotation metadata

- `annotation_confidence`
  - float or null

- `annotation_version`
  - string or null
