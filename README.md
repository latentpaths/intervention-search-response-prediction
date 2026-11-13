# Intervention search and response prediction

Can an agent find a successful intervention without learning to predict untried
interventions? Under an eight-assay budget, these experiments separate search
success from held-out response prediction. Controlled Boolean-network comparisons
show that improving one outcome can leave the other unchanged or make it worse.

[Paper](paper.pdf) · [Reproduce](#reproduce-the-results) · [Protocol](#experiment-protocol)

*Separating Intervention Search from Response Prediction in Closed-Loop Biological
Agents* — Ayush Bodade. Accepted spotlight, AgenticLS workshop at NeurIPS 2026.

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

The [experiment guide](#experiment-protocol) defines the protocol, scores,
comparisons, uncertainty, and retained evidence. The
[data notice](data/README.md) identifies source measurements and their rights.

## More commands

<details>
<summary>Inspect results, run experiments, and manage outputs</summary>

Run commands from the checkout root, set `ISRP_ROOT`, or pass `--root` to select
it explicitly. Importing scientific objects and asking for CLI help do not
require a checkout. Validation also checks frozen scientific inputs, task views,
saved predictions, histories, scores, and spending ledgers.

### Generated results

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

#### Deterministic runs and standalone analyses

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

### File integrity and scientific validation

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

### Explicit inference and resume rules

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

### Preparing upstream data

Preprocessing is separate from offline reproduction. It downloads source files and
streams the large K562 count matrix:

```sh
uv sync --locked --extra measured
uv run --locked --extra measured python -m intervention_search_response_prediction.measured.data all
```

The default is `outputs/prepared-data/`; `--output` changes it. Its dataset folders
match `data/`, so `--data outputs/prepared-data` selects freshly prepared inputs.
The T-cell workbook requires optional `openpyxl`. Source URLs and checksums remain
with prepared data; see [experiment protocol](#measured-replay) for exclusions and normalization.

</details>

## Experiment protocol

<details>
<summary>Read the protocol, metrics, and retained-record reference</summary>

### Episode and information access

An agent receives the allowed interventions and disclosed context. It predicts a
reserved set of interventions, called probes, before and after eight assays. Each
assay forecast is committed before its response is revealed. Probe outcomes stay
hidden throughout. Where a search objective exists, the agent also submits a final
intervention. Boolean update rules are hidden; measured replay reveals only recorded
responses and never imputes a missing outcome.

A procedure has three components: an assay-selection rule, a response predictor,
and a final-choice rule. Changing one need not change the others.

| Component | Rules evaluated |
| --- | --- |
| Assay selection | Uniformly random untried interventions; graph-ranked candidates in simulation; maximum ridge posterior variance, with simulation restricted to graph-ranked candidates |
| Response prediction | Constant probability 1/2 in simulation; observed mean; ridge regression; summed disclosed single-perturbation responses for K562 |
| Final choice | Best observed intervention; in simulation, the allowed intervention with the highest predicted target response |

LLM conditions receive the same task information and preceding assay observations.
All predict responses. **No note** carries no separate note between calls;
**Observation note** summarizes observations; **Hypothesis note** maintains and
revises a response hypothesis. The original simulation comparison changes the
full procedure, including the assays it acquires. The primary measured comparison
fixes assay identities and observations across the two note conditions. Biological
node names versus opaque IDs change simulated labels, preserving response rules.

Historical keys remain in the evidence: `history` means No note, `predictive`
means Hypothesis note in simulation, and `summary` means Observation note.
`classical` identifies deterministic reference procedures. `discrimination` stores
the score described below as the permutation gap. Dataset identifiers `schmidt`
and `norman` identify the T-cell and K562 sources, respectively.

### Scores and aggregation

Prediction error is mean squared error (MSE) across held-out probes and readouts,
in squared assay units. For binary simulated outcomes it is the Brier score.
Initial minus final error measures improvement after eight assays.

The permutation gap is expected MSE after randomly reassigning forecasts among
probe identities minus actual MSE. For each readout it equals twice the population
covariance of predictions and outcomes; the reported value averages readouts.
Constant forecasts have zero gap. A positive gap indicates correct assignment to
intervention identities, but rescaling forecasts can increase it while worsening
MSE. It is neither a proper score nor evidence of a recovered mechanism.

Simulation search success means the final intervention reaches the specified
readout bits. T-cell search scores final measured enrichment, the cumulative best
observed enrichment, and regret: maximum enrichment in the candidate pool minus
that of the final choice. K562 has no optimization target and reports prediction
only. Its response MSE equals additive-residual MSE algebraically; residual
permutation gap and additive-reference error provide the interaction comparison.

Simulation summaries aggregate within network, with network-level uncertainty.
Measured results retain each screen's five split values, mean, sample standard
deviation, and full range. Splits overlap and are not five independent biological
experiments. Different screens and units are reported separately; individual-cell
bootstrapping would not estimate biological uncertainty here.

### Simulation studies

`simulation-baselines` evaluates six deterministic procedures on 232 tasks from
39 published Boolean models, using three seeds for 1,392 episodes. Controlled
comparisons change the predictor, acquisition rule, or final-choice rule while
holding the other components fixed. Model selection and source attribution are
in the [model notice](data/boolean-models/README.md).

`simulation-notes` retains the original named/anonymous paired prompt comparison,
three excluded smoke episodes, and a later observation-note control. The primary
analysis excludes smoke and reports the post hoc control separately. Its comparison
analysis also consumes deterministic episodes from `simulation-baselines`: it
matches task IDs for reference comparisons and evaluates associations between
search and prediction. The readout sensitivity analysis uses both episode logs
and Boolean task definitions, excluding probes that directly clamp the scored
readout; its frozen reference covers the original four prompt/label groups.

These controls show that a changed predictor can improve held-out prediction
without changing search success, and that a changed final-choice rule can change
search without changing predictions. The hypothesis-note permutation-gap effect
in simulation weakens in sensitivity checks.

### Measured replay

The [data notice](data/README.md) gives source links, attribution, and rights.
The frozen protocol specifies five seeds, an eight-assay budget, and a primary
comparison of final probe MSE for Hypothesis note minus Observation note under
identical random assay histories. Adaptive acquisition and shuffled feedback are
secondary diagnostics. Format checks use a separate seed and are excluded.

#### T-cell cytokine screens

Six IL2/IFNG sheets from Schmidt et al. remain separate: primary CD4 IL2 and CD8
IFNG screens under CRISPRa and CRISPRi, and supplementary CD4 IL2 and IFNG CRISPRa
screens. TNF is excluded. Outcomes are published gene-level log2 guide enrichment
in cytokine-high versus cytokine-low bins, rather than cytokine concentration or
RNA abundance. All finite effects with at least two guides are eligible, without
significance or effect-size filtering. Each seed samples 160 genes uniformly:
128 assay/final candidates and 32 disjoint probes. Direct cytokine-gene
perturbations remain eligible and are identified in action lists.

The workbook pools donor measurements and lacks donor-level effect columns.
Empty replicate-ID lists record that absence; guide counts are not biological
sample sizes. HGNC gene-family annotations are matched by exact symbol. Missing
annotations give an intercept-only feature.

#### K562 combinations

Norman et al.'s filtered UMI matrix is aggregated into perturbation-level sums.
Retained cells have good coverage, one assigned cell, and matching duplicated
guide identities. Unsupported combinations and combinations without both eligible
singles are excluded. Disclosed singles and scored combinations require at least
30 cells. GEM groups are technical processing groups, not independent donors.

Normalize each perturbation's pooled counts per million, apply log2(1+CPM), then
subtract the similarly normalized non-targeting control. This differs from the
mean of individually normalized cells. The eight readouts have the greatest
variance across disclosed single-response changes, with Ensembl-ID tie breaks.
Disclosed singles also determine feature scaling; combination outcomes determine
neither selection nor scaling. All eligible singles are disclosed. Each of five
seeded splits reserves ceil(20%) of combinations as probes; eight from the
remainder are revealed sequentially.

#### Predictors, notes, and fallback

Mean forecasts start at zero, then use observed means. Ridge uses a Gaussian
linear posterior with unit observation variance and prior precision 2. T-cell
features are a constant and unit-normalized multi-hot gene-family membership.
K562 features are a constant, the sum of constituent single-response changes,
and their absolute difference, scaled with disclosed singles. K562 ridge learns
residuals around summed-single predictions. An unmodified sum is an additional
reference. Uncertainty selection maximizes posterior latent variance with fixed
hash tie breaks. Deterministic final choices maximize observed enrichment.

LLM prompts share the frozen model, temperature zero, no reasoning effort,
response schema, 600-character note limit, and 1,536 output-token ceiling. Each
episode has an initial probe call, eight assay calls, a final probe call, and a
T-cell final-action call: ten calls for K562 and eleven for T cells. Probe calls
never update the acting note. Fixed histories disclose action/value pairs without
condition-specific forecasts. Shuffled feedback permutes the eight fixed assay
outcomes across assay identities without revealing probe outcomes.

One schema-only repair is allowed per malformed response. After a second failure,
forecasts use ridge trained only on visible history, assay choices use the first
allowed ID, and final choices maximize observed visible response. Failures and
repairs remain included; no significance-based rerunning or exclusion occurs.
Only complete paired blocks enter paired analysis; incomplete requests and costs
remain in the ledger. [The working guide](#explicit-inference-and-resume-rules) describes spending, timing,
integrity checks, and resume rules.

After the primary comparisons, analysis added Systema's matching-mean reference
for K562: half the sum of constituent single-response changes. Its coefficient
0.5 is fixed rather than fitted on probes. This is a post hoc comparator, separate
from the frozen scored protocol.

### Scope and prior work

The measured comparison does not show a consistent final-error advantage for
Hypothesis note across screens. It concerns these historical screens and the
selected eight-gene K562 panel. Published outcomes may occur in LLM training data;
there is no contamination-free claim. These results establish neither prospective
wet-lab discovery nor a general benefit from hypothesis notes.

| Prior work | Relevant comparison and claim boundary |
| --- | --- |
| [BioDiscoveryAgent](https://proceedings.iclr.cc/paper_files/paper/2025/hash/4252dc94531833029000f85dc5fac792-Abstract-Conference.html) | Evaluates sequential perturbation selection for hits, including T-cell screens. Here, held-out forecasts before/after assays are scored alongside search; experimental design and these datasets are established. |
| [Systema](https://www.nature.com/articles/s41587-025-02777-8) | Examines common response components that can inflate prediction scores. Here, identity alignment is measured alongside MSE; permutation covariance is a diagnostic. |
| [GEARS](https://www.nature.com/articles/s41587-023-01905-6) | Predicts single/multigene responses and interactions. This replay uses disclosed singles, small assay budgets, and additive references; it neither trains nor claims to outperform GEARS. |
| [Gupta et al.](https://aclanthology.org/2025.findings-emnlp.838/) | Provides feedback-permutation ablations and optimization references. Shuffled feedback and separation of prior knowledge from feedback are established ideas. |
| [Ahlmann-Eltze et al.](https://www.nature.com/articles/s41592-025-02772-6) | Shows the strength of simple linear perturbation predictors. Mean, linear, and additive references constrain interpretation here; a small LLM contrast establishes no broad superiority. |

The contribution is the joint evaluation protocol and controlled comparisons,
without an exhaustive priority claim for individual metrics or controls.

### Retained files

`data/boolean-models/` contains pinned model files, their manifest, and source
license. `data/tcell-cytokine-screens/responses.json.gz` contains losslessly
compressed response JSON. `data/k562-combinations/` contains `pooled-counts.npz`
and `metadata.json`. [Data sources and rights](data/README.md) describes them.
`paths.py` maps historical input names to these paths without rewriting records.



#### Simulation baselines

| File | Contents |
| --- | --- |
| [tasks.json](experiments/simulation-baselines/tasks.json) | Task manifest |
| [episodes.jsonl](experiments/simulation-baselines/episodes.jsonl) | Recorded deterministic episodes |
| [reference-summary.json](experiments/simulation-baselines/reference-summary.json) | Frozen aggregate results |

#### Simulation notes

| File | Contents |
| --- | --- |
| [task-selection.json](experiments/simulation-notes/task-selection.json) | Named, anonymous, and smoke task selections |
| [episodes.jsonl](experiments/simulation-notes/episodes.jsonl) | Recorded LLM episodes |
| [inference-calls.jsonl](experiments/simulation-notes/inference-calls.jsonl) | Exact model requests and responses |
| [reference-summary.json](experiments/simulation-notes/reference-summary.json) | Frozen note analysis |
| [reference-comparisons.json](experiments/simulation-notes/reference-comparisons.json) | Matched deterministic controls and search/prediction associations |
| [reference-readout-sensitivity.json](experiments/simulation-notes/reference-readout-sensitivity.json) | Sensitivity to probes that directly clamp scored readouts |

#### Measured replay

| File | Contents |
| --- | --- |
| [baseline-episodes.json](experiments/measured-replay/baseline-episodes.json) | Deterministic reference episodes |
| [paired-blocks.jsonl](experiments/measured-replay/paired-blocks.jsonl) | Paired LLM episodes |
| [inference-ledger.jsonl](experiments/measured-replay/inference-ledger.jsonl) | Requests, responses, and budget reservations |
| [protocol.json](experiments/measured-replay/protocol.json) | Frozen scientific parameters and original code/input hashes |
| [frozen-at.json](experiments/measured-replay/frozen-at.json) | Protocol freeze timestamp and hash |
| [hardware.json](experiments/measured-replay/hardware.json) | Original reporting-host record |
| [reference-summary.json](experiments/measured-replay/reference-summary.json) | Frozen measured analysis |

The simulation-note comparison and sensitivity references use deterministic control
episodes from `simulation-baselines`. Sensitivity also reconstructs tasks from the
Boolean models. The references remain beside the note study because they test its
interpretation. Original IDs, prompts, selection salts, seeds, and schema keys stay
unchanged, including source filenames embedded in frozen records.

</details>

## Development

```sh
uv sync --locked --group dev
uv run --locked python -m unittest discover -s tests -v
uv run --locked ruff check src tests
uv run --locked ruff format --check src tests
```

## Citation

<!-- BEGIN generated citations -->

```bibtex
@inproceedings{bodadepaperbodade2026intervention,
  author = {Bodade, Ayush},
  title = {{Separating Intervention Search from Response Prediction in Closed-Loop Biological Agents}},
  year = {2026},
  month = {09},
  day = {30},
  date = {2026-09-30},
  booktitle = {Agentic AI for Biological Discovery (AgenticLS), NeurIPS 2026 workshop},
  url = {https://doi.org/10.5281/zenodo.24196494},
  doi = {10.5281/zenodo.24196494},
  note = {Accepted spotlight; Non-archival workshop paper}
}
```

<!-- END generated citations -->

Cite the original biological data sources when using their measurements; see the
[data notice](data/README.md). Prime Intellect provided inference credits.

Code and software documentation use [MIT](LICENSE#software-mit-license). The paper, original figures,
and original research results use [CC BY 4.0](LICENSE#research-material-creative-commons-attribution-40-international). Third-party
material retains its existing terms. Contact [Ayush Bodade](mailto:ayush@latentpaths.com).
