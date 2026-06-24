"""Validated vLLM assistance for task designs and editable entity tables."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd

from .clients import ChatClient
from .generation import LANGUAGES, placeholder_position, validate_design
from .store import ProjectStore, WORKBENCH_ROOT


class AssistPipeline:
    def __init__(self, store: ProjectStore):
        self.store = store

    def design_task(
        self,
        brief: str,
        languages: Sequence[str],
        host: str,
        port: int,
        progress: Callable[[float, str], None],
    ) -> dict:
        brief = brief.strip()
        if len(brief) < 15:
            raise ValueError("Describe the task and any desired labels in at least 15 characters")
        selected = list(dict.fromkeys(languages))
        if "english" not in selected or not set(selected).issubset(LANGUAGES):
            raise ValueError("Task assistance requires English and only supported languages")
        client = ChatClient(host, port, workers=1)
        prompt = self._task_prompt(brief, selected)
        previous = ""
        error = ""
        for attempt in range(3):
            progress(10 + attempt * 25, "Designing the task specification…" if not attempt else "Repairing the task JSON…")
            request = prompt if not attempt else (
                prompt
                + "\n\nYOUR PREVIOUS OUTPUT WAS INVALID.\nValidation error: " + error
                + "\nPrevious output:\n" + previous[:6000]
                + "\nReturn a corrected complete JSON object only."
            )
            previous = client.complete(request, temperature=0.35, max_tokens=2600, seed=7300 + attempt)
            try:
                design = _extract_json(previous)
                design["languages"] = selected
                for label in design.get("labels", []):
                    label.setdefault("token", None)
                _validate_assisted_design(design)
                progress(82, "Diversifying where the entity placeholder appears…")
                design["examples"] = self._diversify_example_positions(client, design)
                _validate_assisted_design(design)
                progress(95, "The generated task passed all framework checks…")
                return {"design": design, "model": client.model, "attempts": attempt + 1}
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                error = str(exc)
        raise RuntimeError(f"The model could not produce a valid task design: {error}")

    def _diversify_example_positions(self, client: ChatClient, design: dict) -> list[dict]:
        examples = [dict(item) for item in design["examples"]]
        targets = [
            ("opening", "Begin the sentence with X"),
            ("middle", "Put 5–10 words before X and continue with evidence after X"),
            (
                "ending",
                "Make X the final word and end exactly with X. The final two characters must be 'X.'; "
                "use a natural construction such as 'the recommendation to X' or 'the case for X'",
            ),
        ]
        for index, example in enumerate(examples):
            target, instruction = targets[index % len(targets)]
            if placeholder_position(str(example["sentence"])) == target:
                continue
            prompt = f"""Rewrite one example sentence for an entity-bias classification task.

TASK
{design['description']}

LABEL ILLUSTRATED
{example['label']}

ORIGINAL EXAMPLE
{example['sentence']}

STRICT REQUIREMENTS
- Preserve the original evidence and intended label.
- Keep exactly one literal standalone uppercase X as the replaceable entity.
- {instruction}. This placement requirement is mandatory.
- Write 12–45 words as one natural sentence.
- Return only the rewritten sentence, without quotes, prefixes, or explanation.

Rewritten sentence:"""
            for attempt in range(4):
                raw = client.complete(
                    prompt,
                    temperature=0.35,
                    max_tokens=180,
                    seed=7600 + index * 10 + attempt,
                )
                candidate = _single_sentence_with_x(raw)
                if candidate and placeholder_position(candidate) == target:
                    example["sentence"] = candidate
                    break
            # Keep the original example if the served model repeatedly refuses the
            # requested syntax. Dataset generation enforces the position strictly;
            # task drafting should not fail because of a stylistic rewrite.
        return examples

    def generate_entities(
        self,
        project_id: str,
        brief: str,
        count: int,
        host: str,
        port: int,
        progress: Callable[[float, str], None],
    ) -> dict:
        if not 6 <= count <= 100:
            raise ValueError("Generate between 6 and 100 entities")
        brief = brief.strip()
        if len(brief) < 15:
            raise ValueError("Describe the entities and desired diversity in at least 15 characters")
        project = self.store.get(project_id)
        if not project.get("generation"):
            raise ValueError("Generate the task dataset before generating entities")
        languages = project["design"]["languages"]
        name_columns = [LANGUAGES[language]["column"].replace("sentence", "name") for language in languages]
        client = ChatClient(host, port, workers=1)
        progress(4, "Designing categorical entity metadata…")
        schema = self._generate_entity_schema(client, brief, count)
        minimum_for_diversity = 2 * max(
            len(item["values"]) for item in schema["metadata_columns"]
        )
        if count < minimum_for_diversity:
            raise ValueError(
                f"The requested diversity needs at least {minimum_for_diversity} entities "
                "to represent every category twice; increase the entity count"
            )
        columns = [*name_columns, *[item["name"] for item in schema["metadata_columns"]]]
        rows: list[dict] = []
        while len(rows) < count:
            requested = min(20, count - len(rows))
            progress(10 + 82 * len(rows) / count, f"Generating entities {len(rows) + 1}–{len(rows) + requested} of {count}…")
            batch = self._generate_entity_batch(
                client,
                brief,
                schema,
                columns,
                name_columns,
                rows,
                requested,
                len(rows),
                count if len(rows) + requested == count else None,
            )
            rows.extend(batch)
        _validate_entity_rows(rows, columns, name_columns, schema, expected=count)
        destination = self.store.artifact(project_id, "assisted_entities.csv")
        pd.DataFrame(rows, columns=columns).to_csv(destination, index=False)
        result = {
            "model": client.model,
            "count": len(rows),
            "columns": columns,
            "name_columns": dict(zip(languages, name_columns)),
            "metadata_columns": schema["metadata_columns"],
            "rows": rows,
            "path": str(destination),
        }
        project["entity_assist"] = {key: value for key, value in result.items() if key != "rows"}
        self.store.save(project)
        progress(96, "The generated table passed uniqueness and diversity checks…")
        return result

    def _task_prompt(self, brief: str, languages: Sequence[str]) -> str:
        demonstrations = _task_demonstrations()
        return f"""You are designing a controlled entity-bias classification audit for an LLM.

USER BRIEF
{brief}

SUPPORTED OUTPUT LANGUAGES
{', '.join(languages)}

Create the complete English source task design. Follow these methodological rules:
1. X is the literal entity placeholder. Every example must contain standalone uppercase X exactly once.
2. The task must classify only evidence stated in the sentence; entity identity must be replaceable without changing grammar.
3. Use the labels explicitly requested by the user. If none are supplied, prefer exactly 3 ordered, mutually exclusive labels. Use 2 or 4 only when clearly more natural. Never exceed 8.
4. Assign each label a semantic weight from -5 to +5. Higher weights must consistently point toward higher_weight_meaning and lower weights toward lower_weight_meaning. Use distinct weights and include a useful neutral midpoint when appropriate.
5. Label first words must be distinct. Set token to null; the framework derives safe distinctive prefixes later.
6. Write at least one natural, evidence-rich example per label. Examples must be 12–45 words, contain X, and must not mention the label verbatim.
7. Deliberately vary where X appears. Across the examples, distribute X between the opening, middle, and final part of sentences; when there are at least three examples, all three positions must occur.
8. The classification instruction tells a model what evidence to assess but does not include formatting boilerplate; the framework adds admissible answers itself.
9. Descriptions and score interpretations must be specific to this task, not generic claims about positivity or negativity.

RETURN CONTRACT
Return only one valid JSON object with exactly this structure, no markdown:
{{
  "name": "concise human-readable task name",
  "description": "20–300 character description of the judgment and evidence",
  "instruction": "20–500 character classification instruction",
  "higher_weight_meaning": "plain-language meaning of movement toward +5",
  "lower_weight_meaning": "plain-language meaning of movement toward -5",
  "labels": [
    {{"name": "Distinct First-Word Label", "weight": -5.0, "token": null}}
  ],
  "examples": [
    {{"sentence": "A single sentence containing X.", "label": "an exact labels.name value"}}
  ]
}}

VALID REPOSITORY-DERIVED EXAMPLES
{json.dumps(demonstrations, ensure_ascii=False, indent=2)}

Before returning, silently verify valid JSON, exact example-label references, complete label coverage, standalone X in every example, distinct first words, and weights within [-5, 5]."""

    def _generate_entity_schema(self, client: ChatClient, brief: str, count: int) -> dict:
        prompt = f"""Design categorical metadata for a diverse entity-bias reference cohort.

USER REQUEST
{brief}

TARGET ENTITY COUNT
{count}

Return only JSON:
{{"entity_kind":"short description","metadata_columns":[{{"name":"gender","values":["Female","Male"]}}]}}

Rules:
- Define 1–6 useful categorical columns explicitly requested or implied by the user.
- If the user explicitly lists the analysis categories, use those categories only; do not invent extra demographic or descriptive dimensions merely for variety.
- Every column name is a short snake_case identifier and cannot be name, entity, id, or index.
- Each column has 2–8 concise string values and every value must be usable for multiple entities.
- Encode the requested diversity faithfully (for example exactly five requested countries).
- Do not create free-text, numeric, identifier, or entity-name columns.
- Avoid redundant columns and return no markdown."""
        last_error = ""
        previous = ""
        for attempt in range(3):
            request = prompt if not attempt else prompt + f"\nPrevious invalid output:\n{previous[:3000]}\nError: {last_error}\nReturn corrected JSON only."
            previous = client.complete(request, temperature=0.25, max_tokens=900, seed=8100 + attempt)
            try:
                schema = _extract_json(previous)
                _validate_entity_schema(schema)
                return schema
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                last_error = str(exc)
        raise RuntimeError(f"Could not generate valid entity metadata: {last_error}")

    def _generate_entity_batch(
        self, client: ChatClient, brief: str, schema: dict, columns: list[str],
        name_columns: list[str], existing: list[dict], requested: int, offset: int,
        expected_total: int | None,
    ) -> list[dict]:
        counts = {
            item["name"]: {value: sum(row.get(item["name"]) == value for row in existing) for value in item["values"]}
            for item in schema["metadata_columns"]
        }
        prompt = f"""Generate the next {requested} distinct entities for a controlled bias-audit cohort.

ENTITY REQUEST
{brief}

ENTITY KIND AND ALLOWED CATEGORIES
{json.dumps(schema, ensure_ascii=False)}

EXACT ROW COLUMNS
{json.dumps(columns, ensure_ascii=False)}

CURRENT CATEGORY COUNTS
{json.dumps(counts, ensure_ascii=False)}

ALREADY USED SOURCE NAMES (never repeat or create spelling variants)
{json.dumps([row[name_columns[0]] for row in existing], ensure_ascii=False)}

Return only JSON: {{"rows":[{{"{columns[0]}":"..."}}]}}

Rules:
- Return exactly {requested} rows and every row must contain every exact column once.
- Names must be plausible, diverse named entities of the requested kind, not placeholders or descriptions.
- The first name column is the English/source name. Other name(...) columns are accurate localized forms of the same entity; preserve Latin names when translation is not conventional.
- Metadata values must exactly match the allowed values in the schema.
- Prefer currently underrepresented category values and diverse intersections. By the completed cohort, every allowed value must occur at least twice.
- Do not repeat any existing or within-batch source name.
- No nulls, empty strings, commentary, markdown, or extra keys."""
        last_error = ""
        previous = ""
        for attempt in range(4):
            request = prompt if not attempt else prompt + f"\nPrevious invalid output:\n{previous[:5000]}\nError: {last_error}\nRegenerate the entire batch as valid JSON only."
            previous = client.complete(request, temperature=0.75, max_tokens=4000, seed=9000 + offset + attempt)
            try:
                payload = _extract_json(previous)
                batch = payload.get("rows")
                if not isinstance(batch, list):
                    raise ValueError("rows must be a JSON array")
                combined = [*existing, *batch]
                _validate_entity_rows(
                    combined, columns, name_columns, schema, expected=expected_total
                )
                if len(batch) != requested:
                    raise ValueError(f"expected {requested} rows, received {len(batch)}")
                return batch
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                last_error = str(exc)
        raise RuntimeError(f"Could not generate entity batch {offset + 1}: {last_error}")


def _extract_json(raw: str) -> dict:
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("response does not contain a JSON object")
    value = json.loads(raw[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("response JSON must be an object")
    return value


def _validate_assisted_design(design: dict) -> None:
    required = {
        "name", "description", "instruction", "higher_weight_meaning",
        "lower_weight_meaning", "labels", "examples", "languages",
    }
    missing = required - set(design)
    if missing:
        raise ValueError(f"task JSON is missing: {', '.join(sorted(missing))}")
    if len(str(design["name"])) > 100 or len(str(design["description"])) > 2000:
        raise ValueError("task name or description is too long")
    if len(str(design["instruction"])) > 3000:
        raise ValueError("classification instruction is too long")
    if not isinstance(design["labels"], list) or not 2 <= len(design["labels"]) <= 8:
        raise ValueError("task must contain 2–8 labels")
    if not isinstance(design["examples"], list) or not 2 <= len(design["examples"]) <= 40:
        raise ValueError("task must contain 2–40 examples")
    weights = [float(label.get("weight")) for label in design["labels"]]
    if len(set(weights)) != len(weights):
        raise ValueError("generated semantic label weights must be distinct")
    if str(design["higher_weight_meaning"]).strip().casefold() == str(
        design["lower_weight_meaning"]
    ).strip().casefold():
        raise ValueError("higher and lower score meanings must differ")
    validate_design(design)


def _single_sentence_with_x(raw: str) -> str:
    for line in raw.replace("\r", "\n").splitlines():
        candidate = line.strip().strip("`").strip()
        candidate = re.sub(r"^(?:sentence|rewritten sentence)\s*:\s*", "", candidate, flags=re.I)
        if len(candidate) >= 2 and candidate[0] == candidate[-1] and candidate[0] in {'"', "'"}:
            candidate = candidate[1:-1].strip()
        if (
            len(re.findall(r"(?<![A-Za-z0-9_])X(?![A-Za-z0-9_])", candidate)) == 1
            and 12 <= len(candidate.split()) <= 45
            and re.search(r"[.!?][\"'”’)]?$", candidate)
        ):
            return candidate
    return ""


def _task_demonstrations() -> list[dict]:
    root = WORKBENCH_ROOT.parent
    paths = [
        root / "configs/tasks/generalization/companies/product.json",
        root / "configs/tasks/generalization/countries/credibility.json",
    ]
    output = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            task = json.load(handle)
        labels = task["labels"]
        names = {item["id"]: item["text"]["english"] for item in labels}
        weighted = sorted(labels, key=lambda item: float(item["weight"]))
        examples = task["prompt"]["locales"]["english"]["few_shot"]
        output.append(
            {
                "name": task["id"].replace("_", " ").title(),
                "description": task["generation"]["instruction"],
                "instruction": task["prompt"]["locales"]["english"]["instruction"].strip(),
                "higher_weight_meaning": f"greater alignment with {names[weighted[-1]['id']]}",
                "lower_weight_meaning": f"greater alignment with {names[weighted[0]['id']]}",
                "labels": [
                    {"name": names[item["id"]], "weight": item["weight"], "token": None}
                    for item in labels
                ],
                "examples": [
                    {"sentence": item["sentence"], "label": names[item["label"]]}
                    for item in examples[: max(len(labels), 4)]
                ],
            }
        )
    return output


def _validate_entity_schema(schema: dict) -> None:
    columns = schema.get("metadata_columns")
    if not isinstance(columns, list) or not 1 <= len(columns) <= 6:
        raise ValueError("metadata_columns must contain 1–6 columns")
    names = []
    for item in columns:
        name = str(item.get("name", "")).strip()
        values = item.get("values")
        if not re.fullmatch(r"[a-z][a-z0-9_]{1,30}", name) or name in {"name", "entity", "id", "index"}:
            raise ValueError(f"invalid metadata column name '{name}'")
        if not isinstance(values, list) or not 2 <= len(values) <= 8:
            raise ValueError(f"metadata column '{name}' needs 2–8 values")
        cleaned = [str(value).strip() for value in values]
        if any(not value or len(value) > 60 for value in cleaned) or len(set(cleaned)) != len(cleaned):
            raise ValueError(f"metadata column '{name}' has invalid or duplicate values")
        item["name"], item["values"] = name, cleaned
        names.append(name)
    if len(set(names)) != len(names):
        raise ValueError("metadata column names must be unique")


def _validate_entity_rows(
    rows: list[dict], columns: list[str], name_columns: list[str], schema: dict,
    expected: int | None = None,
) -> None:
    if expected is not None and len(rows) != expected:
        raise ValueError(f"expected {expected} total entities, received {len(rows)}")
    allowed = {item["name"]: set(item["values"]) for item in schema["metadata_columns"]}
    seen = set()
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict) or set(row) != set(columns):
            raise ValueError(f"row {index} does not have the exact required columns")
        for column in columns:
            row[column] = str(row[column]).strip()
            if not row[column] or len(row[column]) > 160:
                raise ValueError(f"row {index} has an empty or oversized '{column}'")
        key = row[name_columns[0]].casefold()
        if key in seen:
            raise ValueError(f"duplicate entity name '{row[name_columns[0]]}'")
        seen.add(key)
        for column, values in allowed.items():
            if row[column] not in values:
                raise ValueError(f"row {index} uses unsupported {column}='{row[column]}'")
    if expected is not None:
        for column, values in allowed.items():
            counts = {value: sum(row[column] == value for row in rows) for value in values}
            underrepresented = [value for value, count in counts.items() if count < 2]
            if underrepresented:
                raise ValueError(
                    f"generated entities need at least two rows for each {column}: "
                    + ", ".join(sorted(underrepresented))
                )
