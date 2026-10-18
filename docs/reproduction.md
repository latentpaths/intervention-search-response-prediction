# Reproduction and recorded evidence

Run commands from the checkout root or set `ISRP_ROOT` to it. `isrp` also accepts
`--root` for explicit checkout selection. Importing scientific objects and asking
for CLI help do not require a checkout.

```sh
uv sync --locked
uv run --locked isrp validate
uv run --locked isrp reproduce --output outputs/reproduction
```

These commands are offline after installation. Validation checks retained bytes,
frozen scientific inputs, task views, saved predictions, histories, scores, and
spending ledgers. Reproduction requires exact numerical equality with all five frozen
summaries. [Methods](methods.md) defines the quantities and comparisons.

## Inputs and studies

`data/boolean-models/` contains pinned model files, their manifest, and source
license. `data/tcell-cytokine-screens/responses.json.gz` contains losslessly
compressed response JSON. `data/k562-combinations/` contains `pooled-counts.npz`
and `metadata.json`. [Data sources and rights](../data/README.md) describes them.
`paths.py` maps historical input names to these paths without rewriting records.

Recorded experiments live in three directories under `experiments/`:

| Study | Files and purpose |
| --- | --- |
| `simulation-baselines` | `tasks.json`: task manifest; `episodes.jsonl`: deterministic episodes; `reference-summary.json`: frozen aggregate |
| `simulation-notes` | `task-selection.json`: named, anonymous, and smoke selections; `episodes.jsonl`: LLM episodes; `inference-calls.jsonl`: exact requests/responses; `reference-summary.json`: note analysis; `reference-comparisons.json`: matched controls and associations; `reference-readout-sensitivity.json`: direct-clamp diagnostic |
| `measured-replay` | `baseline-episodes.json`: deterministic episodes; `paired-blocks.jsonl`: paired LLM episodes; `inference-ledger.jsonl`: requests, responses, and reservations; `reference-summary.json`: frozen analysis |
| `measured-replay` protocol records | `protocol.json`: original scientific and code freeze; `frozen-at.json`: timestamp and protocol hash; `hardware.json`: reporting-host record |

The simulation-note comparison and sensitivity references use deterministic control
episodes from `simulation-baselines`. Sensitivity also reconstructs tasks from the
Boolean models. The references remain beside the note study because they test its
interpretation. Original IDs, prompts, selection salts, seeds, and schema keys stay
unchanged, including source filenames embedded in frozen records.

## Generated results

Complete reproduction writes one directory per study under `outputs/reproduction/`.
Each has `summary.json` and `figures/`. Simulation notes additionally has
`comparisons.json` and `readout-sensitivity.json`. `validation.json` records the
shared audit outcome. The measured summary retains the historical reporting-host
field; a standalone analysis records its current reporting host separately from retained
hardware details.

| Study | Figure files |
| --- | --- |
| `simulation-baselines` | `learning-curves.pdf`, `component-effects.pdf` |
| `simulation-notes` | `probe-scores.pdf`, `note-effects.pdf`, `node-label-effects.pdf` |
| `measured-replay` | `prediction-gains.pdf`, `search-progress.pdf`, `cost-and-latency.pdf` |

All generated outputs default to ignored `outputs/`. `--input` selects a recorded
study directory, `--output` selects a generated directory, `--data` selects the
prepared data root, and `--models` selects the Boolean model directory. Writers
reject destinations overlapping retained inputs, the checksum catalogue, project
metadata, source, tests, or the submitted PDF, including paths through symlinks.

### Deterministic runs and standalone analyses

```sh
uv run --locked isrp run --output outputs/simulation-baselines
uv run --locked isrp figures --input experiments/simulation-baselines
uv run --locked python -m intervention_search_response_prediction.simulation.note_analysis --input experiments/simulation-notes
uv run --locked python -m intervention_search_response_prediction.simulation.comparisons --input experiments/simulation-notes
uv run --locked python -m intervention_search_response_prediction.simulation.sensitivity --input experiments/simulation-notes
uv run --locked python -m intervention_search_response_prediction.measured.run classical --output outputs/measured-replay
uv run --locked python -m intervention_search_response_prediction.measured.analysis --input experiments/measured-replay
```

Fresh runs default to `outputs/<study>/`; analyses default to
`outputs/analysis/<study>/`. Comparison and sensitivity commands read baseline
controls from the checkout. Sensitivity accepts `--models`, `--root`, and
`--include-posthoc`; the latter adds the later observation-note control to the
original four prompt/label groups. It changes the analysis, not the frozen reference.
The fresh Boolean task manifest, episode log, and summary reproduce the retained
files byte for byte. Measured deterministic records agree scientifically, while
wall times reflect the current execution.

Both audits can also run directly:

```sh
uv run --locked python -m intervention_search_response_prediction.simulation.audit
uv run --locked python -m intervention_search_response_prediction.measured.audit --input experiments/measured-replay
```

## File integrity and scientific validation

`experiments/checksums.json` catalogues 62 public files: source models and notices, compact
measured inputs, recorded experiments, frozen summaries, and the measured protocol.
Validation rejects missing files and checksum mismatches before either scientific
audit runs. Its report separates public file integrity from the simulation and
measured scientific checks. It does not verify which historical source code
executed the original experiments.

The original measured protocol retains its historical code hashes unchanged.
Scientific validation compares its input hashes, parameters, and all 35 task views
with current reconstructions, then independently checks recorded predictions,
paired histories, scores, and spending. The full reproduction command additionally
checks exact equality with the five reference summaries.

Compressed JSON retains its original decompressed checksum. Five recoverable raw
downloads are omitted; the measured audit reports their expected locations under
each dataset's `raw/` directory and verifies hashes when available. Compact inputs
suffice for all 2,542 retained measured audit checks.

Fresh measured runs freeze current package hashes as `runtime_sha256`, separately
from the original protocol's `code_sha256`. Changed code or scientific inputs
refuse to resume a fresh run; use a new output directory.

## Explicit inference and resume rules

Set `PRIME_API_KEY` in the shell or authenticate the Prime CLI. `.env.example`
documents the variable; `.env` is not loaded automatically. Only the following
explicit inference operations spend credits:

```sh
uv run --locked python -m intervention_search_response_prediction.simulation.inference select
uv run --locked python -m intervention_search_response_prediction.simulation.inference smoke
uv run --locked python -m intervention_search_response_prediction.simulation.inference run
uv run --locked python -m intervention_search_response_prediction.simulation.inference posthoc
uv run --locked python -m intervention_search_response_prediction.measured.run smoke
uv run --locked python -m intervention_search_response_prediction.measured.run freeze
uv run --locked python -m intervention_search_response_prediction.measured.run llm --workers 4
```

`select` and `freeze` do not invoke inference. Simulation commands share
`outputs/simulation-notes/` and retain the original $50 estimated-cost cap.
Measured commands share `outputs/measured-replay/`, with $5 for smoke checks and
$45 for scored inference. Override either destination with `--output`.

Run one measured inference process at a time. Its workers share a locked,
persistent ledger that reserves a whole paired block before launch. Requests are
logged before sending; unanswered attempts keep conservative charges after restart.
Preserve the protocol, request ledger, reservations, and episodes together. Resume
reuses only identical request hashes and complete episodes. Returned charges above
the bound stop execution. The recorded inference budget uses $2/million input and
$10/million output tokens. No training or GPU rental is included.

Timing distinguishes original uncached API latency, decision latency, episode wall
time, and replay wall time. Cached response speed does not demonstrate the protocol's
less-than-one-second reference-decision or less-than-30-second median LLM-decision
targets. The recorded run used six concurrent independent episodes; the current
CLI defaults to four. Primary fixed blocks precede adaptive, then shuffled blocks.

## Preparing upstream data

Preprocessing is separate from offline reproduction. It downloads source files and
streams the large K562 count matrix:

```sh
uv sync --locked --extra measured
uv run --locked --extra measured python -m intervention_search_response_prediction.measured.data all
```

The default is `outputs/prepared-data/`; `--output` changes it. Its dataset folders
match `data/`, so `--data outputs/prepared-data` selects freshly prepared inputs.
The T-cell workbook requires optional `openpyxl`. Source URLs and checksums remain
with prepared data; see [methods](methods.md) for exclusions and normalization.
