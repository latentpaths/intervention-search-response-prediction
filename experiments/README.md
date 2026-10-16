# Recorded experiments

The three studies separate successful intervention search from prediction of
untried interventions under an eight-assay budget. This directory holds retained
evidence and frozen numerical references. Fresh runs and regenerated figures go
to the ignored `outputs/` directory.

Start with the [paper](../paper.pdf) for the argument,
[methods](../docs/methods.md) for the scientific protocol, or the
[reproduction guide](../docs/reproduction.md) for commands and integrity checks.

## Simulation baselines

Six deterministic procedures are evaluated on 232 tasks from 39 published Boolean
networks, with 1,392 episodes. Controlled comparisons change the predictor,
assay-selection rule, or final-choice rule while holding the other components
fixed where stated.

| File | Contents |
| --- | --- |
| [tasks.json](simulation-baselines/tasks.json) | Task manifest |
| [episodes.jsonl](simulation-baselines/episodes.jsonl) | Recorded deterministic episodes |
| [reference-summary.json](simulation-baselines/reference-summary.json) | Frozen aggregate results |

The [model notice](../data/boolean-models/README.md) records source attribution,
selection, and license terms.

## Simulation notes

Paired LLM runs compare carried notes and biological versus anonymous node labels.
The later observation-note control is reported separately from the original
comparison; three smoke episodes are excluded from the primary analysis.

| File | Contents |
| --- | --- |
| [task-selection.json](simulation-notes/task-selection.json) | Named, anonymous, and smoke task selections |
| [episodes.jsonl](simulation-notes/episodes.jsonl) | Recorded LLM episodes |
| [inference-calls.jsonl](simulation-notes/inference-calls.jsonl) | Exact model requests and responses |
| [reference-summary.json](simulation-notes/reference-summary.json) | Frozen note analysis |
| [reference-comparisons.json](simulation-notes/reference-comparisons.json) | Matched deterministic controls and search/prediction associations |
| [reference-readout-sensitivity.json](simulation-notes/reference-readout-sensitivity.json) | Sensitivity to probes that directly clamp scored readouts |

The comparison analysis also reads deterministic baseline episodes. Readout
sensitivity reconstructs tasks from the Boolean models. These files are inputs
to the analyses, rather than independent experiments.

## Measured replay

Retrospective replay uses six T-cell cytokine screens and K562 CRISPRa
combinations. The primary note comparison fixes assay identities and observations;
adaptive acquisition and shuffled feedback are secondary diagnostics. Each dataset
uses five fixed splits, which overlap and are not independent biological replicates.

| File | Contents |
| --- | --- |
| [baseline-episodes.json](measured-replay/baseline-episodes.json) | Deterministic reference episodes |
| [paired-blocks.jsonl](measured-replay/paired-blocks.jsonl) | Paired LLM episodes |
| [inference-ledger.jsonl](measured-replay/inference-ledger.jsonl) | Requests, responses, and budget reservations |
| [protocol.json](measured-replay/protocol.json) | Frozen scientific parameters and original code/input hashes |
| [frozen-at.json](measured-replay/frozen-at.json) | Protocol freeze timestamp and hash |
| [hardware.json](measured-replay/hardware.json) | Original reporting-host record |
| [reference-summary.json](measured-replay/reference-summary.json) | Frozen measured analysis |

The [data notice](../data/README.md) links the biological sources and prepared
inputs. The [reproduction guide](../docs/reproduction.md) explains preserved
historical identifiers, validation boundaries, and inference-ledger handling.

## Validate and reproduce

From the repository root, after `uv sync --locked`:

```sh
uv run --locked isrp validate
uv run --locked isrp reproduce
```

Validation checks the [checksum catalogue](../checksums.json) before scientific
audits. Reproduction checks numerical equality against the five frozen summaries
and writes summaries and eight figures to `outputs/reproduction/`. Neither command
makes inference calls.
