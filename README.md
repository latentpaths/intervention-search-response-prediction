<p>
  <a href="https://www.latentpaths.com/"><img src="https://raw.githubusercontent.com/latentpaths/.github/main/assets/latent-paths-wordmark.png" alt="Latent Paths" width="280"></a>
</p>

# Intervention search and response prediction

Can an agent find a successful intervention without learning to predict untried
interventions? Under an eight-assay budget, these experiments separate search
success from held-out response prediction. Controlled Boolean-network comparisons
show that improving one outcome can leave the other unchanged or make it worse.

[Paper](paper.pdf) · [Project](https://latentpaths.com/projects/intervention-search-response-prediction) · [OpenReview](https://openreview.net/forum?id=iHkXEsYah3) · [Experiments and evidence](experiments/README.md)

*Separating Intervention Search from Response Prediction in Closed-Loop Biological
Agents* — Ayush Bodade. Accepted spotlight, AgenticLS workshop at NeurIPS 2026.

[Reproduce results](#reproduce-the-results) · [Run new experiments](#deterministic-runs-and-standalone-analyses) ·
[Inference and resume rules](#explicit-inference-and-resume-rules) · [Develop](#development)

## Reproduce the results

With Python 3.12 and [uv](https://docs.astral.sh/uv/):

```sh
git clone https://github.com/latentpaths/intervention-search-response-prediction.git
cd intervention-search-response-prediction
uv sync --locked
uv run --locked isrp validate
uv run --locked isrp reproduce --output outputs/reproduction
```

After installation, both commands run offline without inference credentials.
Validation checks retained file hashes and re-scores the recorded studies.
Reproduction checks five summaries against frozen references and writes eight
figures to `outputs/reproduction/`. Retained inputs remain unchanged.

Run commands from the checkout root, set `ISRP_ROOT`, or pass `--root` to select
it explicitly. Importing scientific objects and asking for CLI help do not
require a checkout. Validation also checks frozen scientific inputs, task views,
saved predictions, histories, scores, and spending ledgers.

## What is measured

A procedure chooses assays, observes responses, and selects a final intervention.
Forecasts of reserved interventions are scored before and after the eight assays.
Search success measures the final intervention; prediction error measures held-out
forecasts. A permutation gap separately tests whether forecasts align with the
correct intervention identities.

On **232 tasks from 39 Boolean networks**, controlled component changes give:

| Change | Held fixed | Result (95% network-bootstrap interval) |
| --- | --- | --- |
| Mean predictor → ridge regression | Graph-ranked assays; best-observed final choice | Brier error changes by −0.056 [−0.064, −0.049]; search success stays at 61.6% |
| Best-observed → predicted-best final choice | Graph-ranked assays; ridge forecasts | Search success changes by −5.0 percentage points [−9.4, −0.7]; forecasts stay identical |

In the LLM simulations, hypothesis notes improve intervention alignment on named
tasks, but the effect weakens with hidden gene names or exclusion of action
fallbacks. In retrospective replay of six human T-cell screens and a selected
eight-gene K562 panel, hypothesis notes do not consistently reduce final prediction
error against observation notes given identical assay histories. Replay splits
overlap; they are not independent biological replicates. These results do not
establish prospective wet-lab discovery or a general benefit from hypothesis notes.

The [experiment guide](experiments/README.md) defines the protocol, scores,
comparisons, uncertainty, and retained evidence. The
[data notice](data/README.md) identifies source measurements and their rights.

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
explicit inference operations require explicit configuration:

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
with prepared data; see [experiment protocol](experiments/README.md#measured-replay) for exclusions and normalization.

## Development

```sh
uv sync --locked --group dev
uv run --locked python -m unittest discover -s tests -v
uv run --locked ruff check src tests
uv run --locked ruff format --check src tests
```

## Edit the manuscript

The ignored author-local `paper/README.md` owns manuscript build, visual review,
and candidate-asset export instructions. Those tools do not run inference.
Public reproduction does not require manuscript sources or LaTeX. The root
`paper.pdf` remains the submitted paper until an explicit manuscript update.

## Citation

```bibtex
@misc{bodade2026intervention,
  author       = {Ayush Bodade},
  title        = {Separating Intervention Search from Response Prediction in Closed-Loop Biological Agents},
  year         = {2026},
  howpublished = {Agentic AI for Biological Discovery (AgenticLS), NeurIPS 2026 Workshop},
  note         = {Accepted spotlight; non-archival workshop paper},
  url          = {https://latentpaths.com/projects/intervention-search-response-prediction}
}
```

Cite the original biological data sources when using their measurements; see the
[data notice](data/README.md). Prime Intellect provided inference credits.

Code and software documentation use [MIT](LICENSE#software-mit-license). The paper, original figures,
and original research results use [CC BY 4.0](LICENSE#research-material-creative-commons-attribution-40-international). Third-party
material retains its existing terms. Contact [Ayush Bodade](mailto:ayush@latentpaths.com).
