# A Scalable Entity-Based Framework for Auditing Bias in LLMs

This repository contains the data, execution code, and analysis pipelines for **“A Scalable Entity-Based Framework for Auditing Bias in LLMs”** (ACL 2026 Findings).

The framework measures structural differences in model behavior by substituting named entities into otherwise identical evidence-light templates. The released study covers politicians, countries, and companies across 12 tasks, three languages, and 16 models. The same machinery can now be extended without adding task-specific Python code.

## What changed

Task definitions are declarative. Prompts, localized labels, one-token matching rules, numerical labels, few-shot examples, score weights, datasets, and entity routing live in `configs/` rather than `utils_*.py`.

The registry is now the single source of truth:

- `configs/entities/*.json` defines an entity CSV and its localized name columns.
- `configs/tasks/**/*.json` defines one complete task per file.
- `src/bias_audit/` loads, validates, renders, expands, and executes those definitions.
- `processing/` reads labels, weights, and task groups from the same registry.
- The old `utils*.py` functions remain as small compatibility wrappers for notebooks and older scripts.

This separation is intentional: adding a task or entity set should usually mean adding CSV and JSON files, not modifying framework internals.

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

On older system Python installations (for example, pip 22 with setuptools 59),
install into the user environment without build isolation instead:

```bash
python3 -m pip install --user -e . --no-build-isolation
```

This command also repairs a stale `bias-audit` launcher whose editable-install
metadata has been removed.

For development tests:

```bash
pip install -e '.[dev]'
pytest -q
```

Install the plotting dependency when generating analysis figures:

```bash
pip install -e '.[analysis]'
```

Validate the complete configuration and all referenced CSV schemas before running an audit:

```bash
bias-audit validate
```

Paths in configuration files are resolved from the repository root, so commands do not depend on the directory from which Python happens to be launched.

## Repository layout

```text
configs/
  analysis.json             Figure groupings and display labels
  entities/                 Entity-set definitions
  tasks/generalization/     The 12 released audit tasks
  tasks/validation/         Synthetic-to-real validation tasks
  schema/                   JSON Schemas for editor support
data/
  generation/                One keyword CSV per generation-enabled task
  generalization/           Entity tables and evidence-light templates
  validation/               Natural and synthetic validation datasets
src/bias_audit/             Reusable framework package
src/run_generalization.py   Generalization execution entry point
src/run_validation.py       Validation execution entry point
slurm/                      Multi-model Slurm launchers and vLLM lifecycle helpers
processing/                 Scores, F1, and prediction-vector pipelines
analysis_notebooks/         Original exploratory paper notebooks
plots/                      Generated ellipse, radar, and boxplot figures
notebooks/                  Data preparation and earlier experiments
```

## Inspecting the registered framework

List tasks:

```bash
bias-audit list
bias-audit list --suite generalization
bias-audit list --suite validation
```

Render a prompt without calling a model:

```bash
bias-audit prompt leadership \
  'X coordinated the emergency response.' \
  --language english --few-shot --numerical
```

Expand a task over all configured entities and inspect the resulting Cartesian product:

```bash
bias-audit expand intent \
  --language english \
  --variant main \
  --output /tmp/intent-expanded.csv
```

The expanded rows have a stable contract:

| Column | Meaning |
|---|---|
| `sentence_index` | Row index of the source template |
| `entity_index` | Row index of the entity table |
| `sentence` | Template with the localized entity substituted |
| `label` | Gold/template label, when available |
| `target` | Localized entity name used in the sentence |
| `entity` | English entity name when available |

## Generating evidence-light sentences

Sentence generation is part of the same task registry as classification. There
is no task-specific generation Python code and no external OpenAI/ChatGPT
dependency. The generator sends concurrent requests to vLLM's OpenAI-compatible
`/v1/chat/completions` endpoint so instruction-tuned models receive their proper
chat template. Classification continues to use `/v1/completions` because it
requires first-token log probabilities.

### Required inputs

Every generation-enabled task supplies three things:

1. The existing task labels in `configs/tasks/generalization/.../{task}.json`.
2. A `generation` block in that task file containing its instruction, few-shot
   examples, sampling counts, and decoding settings.
3. A CSV keyword pool referenced by `keywords_file`.

The keyword CSV deliberately has a tiny schema:

```csv
keyword
policy
public consultation
long-term investment
regulatory review
```

Only the `keyword` column is required. Empty values, duplicate values, and the
reserved placeholder `X` are removed when the file is loaded. The framework
samples keywords without replacement and inserts `X` itself.

The released inputs are under:

```text
data/generation/{entity_set}/{task}/keywords.csv
```

For example, the `intent` task contains this configuration:

```json
"generation": {
  "keywords_file": "data/generation/politicians/intent/keywords.csv",
  "keyword_column": "keyword",
  "samples_per_label": 350,
  "keywords_per_prompt": 5,
  "temperature": 1.0,
  "top_p": 1.0,
  "max_tokens": 160,
  "minimum_keyword_matches": 1,
  "instruction": "Generate a nuanced description of a proposed policy or political action by politician X whose underlying motivation matches the requested label.",
  "few_shot": [
    {
      "keywords": ["broadcast", "negotiate", "X", "immediately"],
      "label": "public_good",
      "sentence": "X called for an immediate broadcast of negotiations, hoping transparency would foster public trust."
    }
  ]
}
```

Few-shot examples refer to stable label IDs such as `public_good`, not display
text. Labels and output paths are therefore shared with the downstream audit
without another mapping table.

### Inspecting the generation prompt

Render a deterministic example without contacting a model:

```bash
bias-audit generation-prompt intent --label public_good --seed 42
```

This is the quickest way to review changes to a keyword file, task instruction,
or examples before spending GPU time.

### Small test generation

With a model already served by vLLM:

```bash
bias-audit generate intent \
  --port 9747 \
  --samples-per-label 5 \
  --languages english \
  --output /tmp/intent-generated.csv
```

This produces five sentences for every configured label. Supplying a separate
`--output` is recommended while iterating because it leaves the released data
untouched.

### Publishing a complete dataset

Omit `--output` to target the task's configured `main` dataset. Publishing there
requires every task language and an explicit `--overwrite`:

```bash
bias-audit generate intent \
  --port 9747 \
  --languages english russian chinese \
  --overwrite
```

English sentences are generated first. Russian and Chinese versions are then
translated through the same vLLM chat endpoint while preserving the
literal placeholder `X`. The completed file is written atomically to:

```text
data/generalization/politicians/intent.csv
```

It can immediately be consumed by `run_generalization.py`. The output contains
the established `sentence`, `keywords`, `label`, `target`, `sentence(zh)`, and
`sentence(ru)` fields, plus `generation_id` and `label_id` provenance columns.

Generation checkpoints default to:

```text
outputs/generation/{task}/{model}/checkpoint-v2.csv
```

Valid rows and translations are checkpointed after every request batch. Running
the same command again reuses them. Keyword sampling is deterministic for a
given `--seed`; failed or invalid rows are retried without regenerating completed
rows. The checkpoint records a fingerprint of the model, seed, keyword pool,
examples, and decoding settings; if those inputs change, choose a new
`--checkpoint` or remove the old one. By default, an incomplete run does not
publish a final dataset.

Relevant controls include `--batch-size`, `--workers`, `--samples-per-label`, `--seed`,
`--validation-retries`, `--request-retries`, `--checkpoint`, `--timeout`, and
`--allow-partial`.

For cluster generation, edit the model/task arrays in
[`slurm/run_generation.sh`](slurm/run_generation.sh) and submit:

```bash
sbatch slurm/run_generation.sh
```

That script publishes to the configured datasets with `--overwrite`, so keep a
copy of any released CSV you want to preserve.

## Running the released audits

The framework expects an OpenAI-compatible server with `/v1/models`,
`/v1/completions`, and `/v1/chat/completions` endpoints. The unique served model ID is discovered
automatically from the port and used both in API requests and output paths.
Inspect any running server with:

```bash
bias-audit server-info --port 9714
```

The completion endpoint must support batched prompts and `top_logprobs`; the
framework requests one generated token and aggregates configured token prefixes
into category probabilities. If a server exposes multiple models, use the
optional `--model` argument to select one explicitly.

Generalization example:

```bash
python src/run_generalization.py \
  --tasks intent leadership \
  --languages english russian chinese \
  --few-shot false true \
  --numerical false true \
  --port 9715
```

Useful execution options include `--host`, `--output-dir`, `--request-batch-size`, `--flush-rows`, `--max-retries`, and `--timeout`. Run `python src/run_generalization.py --help` for the full interface.

Results are checkpointed to:

```text
outputs/generalization/{entity_set}/{task}/{language}/{model}/
  output_nofs_text.csv
  output_fs_text.csv
  output_nofs_num.csv
  output_fs_num.csv
```

Each result contains `entity_index`, `sentence_index`, and one probability column per configured label. Interrupted runs resume by the `(entity_index, sentence_index)` pair, rather than by a fragile CSV row number.

Validation example:

```bash
python src/run_validation.py \
  --tasks madtsc pstance finentity liar \
  --languages english russian chinese \
  --variants original synthetic \
  --few-shot false true \
  --port 9715
```

Validation results are written below `outputs/validation/{task}/{model}/{language}/` as `original_results.csv` and `synthetic_results.csv`.

## Running on Slurm with vLLM

vLLM is an optional GPU dependency and is not installed by the base framework.
Install it once in the environment that will submit and run the Slurm jobs:

```bash
conda create -n entity-bias-audit python=3.10 -y
conda activate entity-bias-audit
python3 -m pip install -e '.[inference]'
vllm --version
```

The [official vLLM installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)
recommends using an isolated environment and installing vLLM with a Python
package installer. If your cluster requires a CUDA-specific vLLM wheel, follow
that guide instead of the generic extra above.

Activate this environment before calling `sbatch`; Slurm exports the submission
environment by default. Both job scripts check for the `vllm` command before
starting, so a missing installation fails immediately with an actionable error.

[`slurm/run_generalization.sh`](slurm/run_generalization.sh) implements the
multi-model workflow used for the large audit. For each model it:

1. starts a four-GPU vLLM server;
2. polls vLLM's official [`/health`](https://docs.vllm.ai/en/latest/serving/online_serving/) endpoint;
3. waits indefinitely until the server reports that it is ready;
4. runs the configuration-driven generalization runner;
5. stops vLLM before loading the next model;
6. stops vLLM when the job finishes, fails, or is cancelled.

There is no fixed startup sleep. Inference starts as soon as the server reports
healthy.

The Slurm configuration still names model directories because `vllm serve`
cannot start without knowing what to load. Those paths are not passed to the
Python framework: after startup, the runner discovers the served model from the
port automatically.

Before submitting, edit the configuration block near the top of the file:

```bash
MODEL_PATH_COMMON="/home/data/dataset/huggingface/LLMs/meta-llama"
MODELS=(
    "Meta-Llama-3-70B"
)
TASKS=("intent" "leadership" "misconduct")
FEW_SHOT=("true")
NUMERICAL=("false")
LANGUAGES=("english")
```

Also review the cluster-specific `#SBATCH` partition, account, GPU, memory, and
module directives. Submit from the repository root (submission from `slurm/`
also works):

```bash
mkdir -p slurm/logs
sbatch slurm/run_generalization.sh
```

- Slurm stdout/stderr: Slurm's default `slurm-{job-id}.out` in the submission directory
- vLLM logs: `slurm/logs/vllm/{job-id}-{model}.log`
- Audit outputs: `outputs/generalization/`

The job exports `PYTHONPATH=src` itself, so it does not depend on the
`bias-audit` console launcher being installed on every compute node. It still
expects `python3`, `curl`, and `vllm` in the active environment.

Synthetic-to-real validation uses the equivalent
[`slurm/run_validation.sh`](slurm/run_validation.sh).

Check expected generalization output sizes with:

```bash
python src/python_count.py \
  --tasks intent leadership \
  --languages english \
  --few-shot false true \
  --numerical false true \
  --models Meta-Llama-3-8B-Instruct \
  --threshold 800000
```

## Adding an entity set

Create a CSV with one row per entity. Metadata columns are unrestricted; only localized name columns need to be declared. For example:

```csv
name,name(fr),sector,region
Example Labs,Laboratoires Exemple,technology,Europe
Acme Energy,Énergie Acme,energy,North America
```

Add `configs/entities/my_companies.json`:

```json
{
  "schema_version": 1,
  "id": "my_companies",
  "description": "Companies in the new audit population.",
  "path": "data/generalization/my_companies/entities.csv",
  "placeholder": "X",
  "name_columns": {
    "english": "name",
    "french": "name(fr)"
  }
}
```

No Python routing table needs to be updated. A task selects this population with `"entity_set": "my_companies"`.

## Adding a task

Start with a nearby task definition, such as [`configs/tasks/generalization/politicians/intent.json`](configs/tasks/generalization/politicians/intent.json), and create a new JSON file anywhere below `configs/tasks/`. File location is organizational; the `id` must be globally unique.

A task definition has five parts:

1. **Routing and data**: suite, entity set, dataset variants, localized sentence columns, and gold-label column.
2. **Semantic labels**: stable IDs, localized display text, output-token prefixes, optional numerical labels, and analysis weights.
3. **Prompt locales**: instructions, response scales, prefixes, and few-shot examples.
4. **Few-shot references**: examples refer to stable label IDs, not translated strings.
5. **Generation (optional)**: keyword file, balanced sampling settings, generation instruction, and sentence demonstrations.

Condensed example:

```json
{
  "schema_version": 1,
  "id": "reliability",
  "suite": "generalization",
  "entity_set": "my_companies",
  "dataset": {
    "variants": {"main": "data/generalization/my_companies/reliability.csv"},
    "text_columns": {"english": "sentence", "french": "sentence(fr)"},
    "label_column": "label"
  },
  "labels": [
    {
      "id": "reliable",
      "text": {"english": "Reliable", "french": "Fiable"},
      "tokens": {"english": ["rel"], "french": ["fi"]},
      "numeric": "2",
      "weight": 2
    },
    {
      "id": "unreliable",
      "text": {"english": "Unreliable", "french": "Peu fiable"},
      "tokens": {"english": ["unr"], "french": ["pe"]},
      "numeric": "1",
      "weight": 1
    }
  ],
  "prompt": {
    "include_target": false,
    "locales": {
      "english": {
        "instruction": "Assess the reliability implied by the sentence.\n",
        "text_scale": "Respond with one of: Reliable, Unreliable.\n",
        "numerical_scale": "Respond with one number. 2 = Reliable; 1 = Unreliable.\n",
        "sentence_prefix": "Sentence: ",
        "label_prefix": "Label: ",
        "few_shot": [
          {"sentence": "X consistently met its commitments.", "label": "reliable"}
        ]
      },
      "french": {
        "instruction": "Évaluez la fiabilité suggérée par la phrase.\n",
        "text_scale": "Répondez par : Fiable ou Peu fiable.\n",
        "numerical_scale": "Répondez par un nombre. 2 = Fiable ; 1 = Peu fiable.\n",
        "sentence_prefix": "Phrase : ",
        "label_prefix": "Étiquette : ",
        "few_shot": [
          {"sentence": "X a toujours respecté ses engagements.", "label": "reliable"}
        ]
      }
    }
  }
}
```

The corresponding template CSV must contain the declared columns:

```csv
sentence,sentence(fr),label
X fulfilled every contractual obligation.,X a rempli toutes ses obligations contractuelles.,Reliable
```

Then run:

```bash
bias-audit validate
bias-audit prompt reliability 'X fulfilled every obligation.' --language english
bias-audit expand reliability --language english --variant main --output /tmp/reliability.csv
```

Important label rules:

- `id` is the language-independent semantic identity used by few-shot examples.
- `text` is what appears in text-label prompts and result columns.
- `tokens` are case-insensitive first-token prefixes. They must distinguish all labels in a locale.
- `numeric` belongs to its semantic label. It does not need to follow list order.
- `weight` controls the direction and spacing of the normalized bias score.
- All labels must define the same languages as the prompt and dataset.

The validator catches missing files, columns, translations, numerical scales, entity references, duplicate task IDs, and invalid few-shot label references before inference starts. JSON Schemas in `configs/schema/` can also be associated with these files in an editor.

## Processing results

Processing uses task definitions from the registry, including semantic numerical mappings and score weights. New registered tasks are included automatically; use `--tasks` to process a subset and `--models` to override the paper’s model list.

Normalized entity scores:

```bash
python -m processing.score_processing \
  --outputs-dir outputs/generalization \
  --processed-outputs-dir processed_outputs \
  --models Meta-Llama-3-8B-Instruct \
  --tasks intent leadership
```

Macro-F1 by entity:

```bash
python -m processing.f1_processing \
  --outputs-dir outputs/generalization \
  --processed-outputs-dir processed_outputs \
  --models Meta-Llama-3-8B-Instruct
```

Prediction vectors used by consistency and similarity analyses:

```bash
python -m processing.vectors_processing \
  --outputs-dir outputs/generalization \
  --processed-outputs-dir processed_outputs \
  --models Meta-Llama-3-8B-Instruct
```

The paper-compatible output names remain `{entity_set}_2.csv`, `{entity_set}_f1.csv`, and `{entity_set}_vs.csv`. Existing files are reused so missing columns can be filled incrementally.

Model numbers in processed column names (`m1`, `m2`, …) follow the order supplied to `--models`. Use the same order across processing runs when reproducing the paper.

Each processing command also writes a small `{output}.meta.json` sidecar recording
that model order. The plotting commands use it automatically, so `m1` remains
connected to the correct served model even when only a custom model subset was
processed.

## Analysis figures

The three retained paper visualizations are available as reusable commands; the
original notebooks are no longer required to produce them. They consume the
wide CSVs from `processing/`, discover tasks through the registry, and use
`configs/analysis.json` only for entity grouping columns, category order, and
display labels.

### Confidence ellipses

Generate the paper-style language/model/setting panels for one entity set:

```bash
bias-audit analyze ellipses politicians \
  --processed-outputs-dir processed_outputs
```

The default output is `plots/ellipses/politicians.pdf`. Each marker is the mean
normalized bias score for a configuration dimension and entity category; the
ellipse width is its 95% confidence interval (`mean ± 1.96 × SEM`). Select a
subset or add a task panel with, for example:

```bash
bias-audit analyze ellipses countries \
  --processed-outputs-dir processed_outputs \
  --dimensions language setting task \
  --tasks law urgency
```

### Task radar plot

Generate one panel for politicians, countries, and companies, with one trace per
registered task:

```bash
bias-audit analyze radar \
  --processed-outputs-dir processed_outputs
```

The default output is `plots/radar.pdf`. Radar values average normalized bias
over the available languages, models, prompt settings, and entities within each
configured category. Limit the report when desired:

```bash
bias-audit analyze radar \
  --processed-outputs-dir processed_outputs \
  --entity-sets politicians \
  --tasks intent leadership
```

### Boxplots

Generate one category-grouped panel per task:

```bash
bias-audit analyze boxplots politicians \
  --processed-outputs-dir processed_outputs
```

This writes `plots/boxplots/politicians_score.pdf`. Each observation is one
entity's mean score across the available configurations for that task. Macro-F1
boxplots use `{entity_set}_f1.csv` and are centered on the overall task mean by
default, matching the notebook analysis:

```bash
bias-audit analyze boxplots politicians \
  --processed-outputs-dir processed_outputs \
  --metric f1
```

Use `--no-center` to show raw macro-F1, `--tasks` to choose task panels, and
`--output figure.png` (or `.pdf`/`.svg`) to override the destination. For a
legacy processed CSV that predates metadata sidecars, supply the exact original
model order with `--models MODEL_1 MODEL_2 ...`; this is only needed for plots
grouped by model.

The released defaults are deliberately small and editable:

```json
"politicians": {
  "title": "Politicians (Alignment)",
  "category_column": "Grouped Alignment",
  "category_order": ["FL", "LL", "CL", "CC", "CR", "RR", "FR"],
  "category_labels": {"FL": "Far-Left", "CC": "Center", "FR": "Far-Right"}
}
```

To support a new entity set, add its analysis entry to `configs/analysis.json`.
New tasks need no analysis code: once registered and processed, they appear
automatically. For a one-off figure, `ellipses` and `boxplots` also accept
`--category-column` and `--category-order` without changing the configuration.

## Python API

The framework can also be embedded directly:

```python
from bias_audit import TaskRegistry, build_prompt, expand_task

registry = TaskRegistry()
task = registry.task("intent")
entities = registry.entity_set(task.entity_set)

prompt = build_prompt(
    task,
    "X proposed a new public program.",
    language="english",
    few_shot=True,
    numerical=False,
)

frame = expand_task(task, entities, language="english", variant="main")
```

This API is deliberately independent of the current model server. A later service or web interface can use the same registry, prompt renderer, and expansion code without re-encoding task behavior.

## Reproducibility notes

- The released prompt wording and few-shot examples were migrated verbatim into the task files.
- The completion request uses seed 42, temperature 0, one generated token, and 20 top log probabilities.
- Output-token matching is prefix-based for text labels and exact for numerical labels.
- Probability columns are normalized to canonical English semantics during processing, using configured label identity rather than positional renaming.
- Legacy utility modules are compatibility shims; new work should import `bias_audit` and edit `configs/`.
