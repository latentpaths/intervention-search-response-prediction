# Separating Intervention Search from Response Prediction in Closed-Loop Biological Agents

Ayush Bodade · Independent Researcher · [ayush@latentpaths.com](mailto:ayush@latentpaths.com)

**Accepted as a spotlight at the [Agentic AI for Biological Discovery (AgenticLS) workshop](https://agenticls.github.io/), NeurIPS 2026.**

December 11, 2026 · Sydney, Australia

[Paper (PDF)](paper.pdf) · [Project website](https://latentpaths.com/projects/intervention-search-response-prediction) · [Workshop](https://agenticls.github.io/) · [OpenReview paper](https://openreview.net/forum?id=iHkXEsYah3) · [BibTeX](CITATION.bib)

Does finding a successful intervention show that an agent learned to predict
interventions it has not tried? This repository measures those outcomes
separately under an **eight-assay budget**. It contains Boolean-network
simulations, retrospective replay of published biological measurements, recorded
model calls, and the evidence needed to reproduce the paper's comparisons.

## Main findings

Each procedure chooses interventions and observes their responses. Forecasts of
reserved, untried interventions are scored before and after eight assays; the
final intervention is scored independently. Three outcomes answer different
questions:

- **Search:** does the final intervention achieve the target?
- **Prediction error:** how closely do held-out forecasts match measured responses?
- **Intervention alignment:** do forecasts match the correct intervention identities
  better than randomly reassigned forecasts? The permutation gap measures this
  alignment and must be interpreted alongside prediction error.

Controlled comparisons on **232 tasks from 39 published Boolean networks** show
why these outcomes need separate measurement:

| Change | Held fixed | Observed effect (95% network-bootstrap interval) |
| --- | --- | --- |
| Replace the mean predictor with ridge regression | Graph-ranked assays and best-observed final choice | Held-out Brier error changes by −0.056 [−0.064, −0.049]; search success stays at 61.6% |
| Replace best-observed final choice with predicted-best | Graph-ranked assays and ridge forecasts | Search success changes by −5.0 percentage points [−9.4, −0.7]; forecasts stay identical |

The random-assay mean predictor has the lowest error among the six deterministic
procedures (Brier score 0.174), despite giving the same forecast for every
intervention on a readout and therefore having zero permutation gap.

In the LLM simulation study, a hypothesis note improves intervention alignment on
named tasks, but the effect weakens when gene names are hidden or runs requiring
action fallbacks are excluded. In replay of **six human T-cell cytokine screens
and K562 CRISPRa combinations**, hypothesis notes do not consistently lower final
prediction error relative to observation notes given identical assay histories.

The measured comparisons concern historical screens and a selected eight-gene
K562 panel. Their five splits overlap and are not independent biological
replicates. The results establish neither prospective wet-lab discovery nor a
general benefit from hypothesis notes. The [paper](paper.pdf) and
[methods](docs/methods.md) give the full comparisons and claim boundaries.

## Quick start: reproduce the reported results

Use Python 3.12 and [uv](https://docs.astral.sh/uv/):

```sh
git clone https://github.com/stochi0/intervention-search-response-prediction.git
cd intervention-search-response-prediction
uv sync --locked
uv run --locked isrp validate
uv run --locked isrp reproduce
```

After installation, validation and reproduction run offline without inference
credentials. Validation checks retained file hashes and independently re-scores
the recorded studies. Reproduction checks five summaries against frozen
references and writes eight figures under the ignored `outputs/reproduction/`
directory.

| Output | Contents |
| --- | --- |
| `validation.json` | File-integrity and scientific-audit results |
| `simulation-baselines/` | Summary, learning curves, and controlled component comparisons |
| `simulation-notes/` | Summary, paired comparisons, readout sensitivity, and note/label figures |
| `measured-replay/` | Summary, prediction gains, search progress, and cost/latency figures |

See the [reproduction guide](docs/reproduction.md) for individual analysis commands,
input/output options, integrity checks, and resume rules. Run
`uv run --locked isrp --help` for the command overview.

## Three studies

| Study | Question | Recorded evidence |
| --- | --- | --- |
| [Simulation baselines](experiments/simulation-baselines/) | How do assay selection, prediction, and final choice affect the outcomes? | 39 models, 232 tasks, 1,392 deterministic episodes |
| [Simulation notes](experiments/simulation-notes/) | How do carried notes and node labels affect LLM performance? | Paired prompt comparisons, a post hoc observation-note control, and inference logs |
| [Measured replay](experiments/measured-replay/) | Does the note instruction improve predictions given identical measured histories? | Six T-cell screens, K562 combinations, deterministic references, and paired LLM blocks |

The [evidence index](experiments/README.md) links the inputs, recorded calls,
frozen protocol, and reference summaries for each study.
[Methods](docs/methods.md) defines the task information, forecasts, scores,
comparisons, and uncertainty.

## Fresh experiments

Rerun the deterministic Boolean experiments without inference credentials:

```sh
uv run --locked isrp run
```

This writes fresh results to `outputs/simulation-baselines/`. The
[reproduction guide](docs/reproduction.md#deterministic-runs-and-standalone-analyses)
also documents deterministic measured replay and standalone analyses.

Fresh LLM experiments consume inference credits through
[Prime Intellect](https://www.primeintellect.ai/). Set `PRIME_API_KEY` in your
shell or authenticate the Prime CLI; see [`.env.example`](.env.example).
The `.env` file is not loaded automatically. Use the
[inference and resume instructions](docs/reproduction.md#explicit-inference-and-resume-rules)
for the required preparation commands, output directories, spending limits, and
ledger handling. Provider changes can make fresh responses differ from the
recorded evidence.

Prepared inputs are included for offline reproduction. Downloading and
preprocessing upstream data is a separate optional step, documented under
[preparing upstream data](docs/reproduction.md#preparing-upstream-data).

## Repository guide

| Location | Contents |
| --- | --- |
| [paper.pdf](paper.pdf) | Submitted manuscript, including appendices |
| [docs/methods.md](docs/methods.md) | Scientific protocol, scores, and limits |
| [docs/reproduction.md](docs/reproduction.md) | Reproduction, fresh runs, inference, and preprocessing |
| [experiments/README.md](experiments/README.md) | Index of recorded studies and reference summaries |
| [data/README.md](data/README.md) | Prepared inputs, source links, attribution, and rights |
| [data/boolean-models/README.md](data/boolean-models/README.md) | Boolean-model source, selection, and license notice |
| [checksums.json](checksums.json) | SHA-256 catalogue of retained inputs and study records |
| [src/intervention_search_response_prediction/](src/intervention_search_response_prediction/) | Simulation, measured replay, statistics, integrity checks, and CLI |
| [tests/](tests/) | Scientific behavior, evidence integrity, and command checks |
| [.github/workflows/checks.yml](.github/workflows/checks.yml) | Offline CI checks and reproduction |
| [LICENSE-MIT](LICENSE-MIT) · [LICENSE-CC-BY-4.0](LICENSE-CC-BY-4.0) | MIT software license and CC BY 4.0 research-content terms |
| `outputs/` | Ignored regenerated results and fresh runs |

## Development

```sh
uv sync --locked --group dev
uv run --locked python -m unittest discover -s tests -v
uv run --locked ruff check src tests
uv run --locked ruff format --check src tests
```

CI also runs offline reproduction. Scientific objects and CLI help work without
a checkout; evidence-reading commands locate one when needed.

## Citation

If you use this code, evaluation protocol, or recorded evidence, cite the
accompanying workshop paper. Download [CITATION.bib](CITATION.bib) or use the
machine-readable [CITATION.cff](CITATION.cff). Cite the original biological data
sources when using their measurements; attribution and source links are in the
[data notice](data/README.md).

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

## Acknowledgment, license, and contact

The author thanks Prime Intellect for providing inference credits used in this
study.

Code and software documentation are licensed under [MIT](LICENSE-MIT).
The paper, original figures, and original research results are licensed under
[CC BY 4.0](LICENSE-CC-BY-4.0). Third-party material retains its existing terms.
Third-party models and biological source measurements retain their existing
terms; see [data/README.md](data/README.md).
For research questions, contact
[ayush@latentpaths.com](mailto:ayush@latentpaths.com).
