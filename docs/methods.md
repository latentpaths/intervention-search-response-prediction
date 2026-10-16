# Methods

The question is whether a procedure improves predictions of untried interventions
while searching for a desired response. Search success and held-out prediction
are scored separately. The experiments test these outcomes in Boolean simulations
and retrospective replay of published measurements.

## Episode and information access

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

## Scores and aggregation

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

## Simulation studies

`simulation-baselines` evaluates six deterministic procedures on 232 tasks from
39 published Boolean models, using three seeds for 1,392 episodes. Controlled
comparisons change the predictor, acquisition rule, or final-choice rule while
holding the other components fixed. Model selection and source attribution are
in the [model notice](../data/boolean-models/README.md).

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

## Measured replay

The [data notice](../data/README.md) gives source links, attribution, and rights.
The frozen protocol specifies five seeds, an eight-assay budget, and a primary
comparison of final probe MSE for Hypothesis note minus Observation note under
identical random assay histories. Adaptive acquisition and shuffled feedback are
secondary diagnostics. Format checks use a separate seed and are excluded.

### T-cell cytokine screens

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

### K562 combinations

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

### Predictors, notes, and fallback

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
remain in the ledger. [Reproduction](reproduction.md) describes spending, timing,
integrity checks, and resume rules.

After the primary comparisons, analysis added Systema's matching-mean reference
for K562: half the sum of constituent single-response changes. Its coefficient
0.5 is fixed rather than fitted on probes. This is a post hoc comparator, separate
from the frozen scored protocol.

## Scope and prior work

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
