# Intervention search and response prediction

Does finding a successful intervention also improve predictions about untried
interventions? These experiments measure search success and held-out response
prediction separately in Boolean networks and retrospective biological replay.
Improving one outcome does not consistently improve the other.

[Paper on Zenodo](https://doi.org/10.5281/zenodo.24196494) · [Experiment reference](docs/reference.md)

## Reproduce

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/latentpaths/intervention-search-response-prediction.git
cd intervention-search-response-prediction
uv sync --locked
uv run --locked isrp validate
uv run --locked isrp reproduce --output outputs/reproduction
```

After installation, reproduction runs offline without credentials or LaTeX.
It checks retained study measurements and frozen references and regenerates statistics and figures.
See the [reference](docs/reference.md) for results, protocols, output files and
fresh experiment commands. Fresh inference runs consume provider credits.

## Development

```sh
uv sync --locked --group dev
uv run --locked python -m unittest discover -s tests -v
uv run --locked --group dev ruff check src tests
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
