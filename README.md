# Intervention search and response prediction

Can an agent find a successful intervention without learning to predict untried
interventions? Under an eight-assay budget, these experiments separate search
success from held-out response prediction. Controlled Boolean-network comparisons
show that improving one outcome can leave the other unchanged or make it worse.

[Paper](paper.pdf) · [Protocol and evidence](experiments/README.md) · [Command reference](docs/reproduction.md)

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

The [experiment guide](experiments/README.md) defines the protocol, scores,
comparisons, uncertainty, and retained evidence. The
[data notice](data/README.md) identifies source measurements and their rights.

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
  url = {https://openreview.net/forum?id=iHkXEsYah3},
  note = {Accepted spotlight; Non-archival workshop paper}
}
```

<!-- END generated citations -->

Cite the original biological data sources when using their measurements; see the
[data notice](data/README.md). Prime Intellect provided inference credits.

Code and software documentation use [MIT](LICENSE#software-mit-license). The paper, original figures,
and original research results use [CC BY 4.0](LICENSE#research-material-creative-commons-attribution-40-international). Third-party
material retains its existing terms. Contact [Ayush Bodade](mailto:ayush@latentpaths.com).
