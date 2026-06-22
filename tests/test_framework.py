from pathlib import Path
import tempfile
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

import pandas as pd

from bias_audit import TaskRegistry
from bias_audit.analysis import normalize_probability_columns
from bias_audit.data import expand_frames, substitute_entity
from bias_audit.prompts import build_prompt, category_tokens
from bias_audit.generation import (
    TextGenerationClient,
    build_generation_jobs,
    generate_dataset,
    clean_generated_text,
    load_keywords,
    render_generation_prompt,
    validate_generated_sentence,
)
from bias_audit.inference import CompletionClient
from bias_audit.plotting import (
    load_analysis_spec,
    metric_long_frame,
    validate_analysis_spec,
)
from bias_audit.server import discover_model, list_served_models, model_output_name


def test_registry_validates_repository_data():
    registry = TaskRegistry()
    registry.validate(check_data=True)
    assert len(registry.tasks(suite="generalization")) == 12
    assert len(registry.tasks(suite="validation")) == 5


def test_numerical_labels_follow_semantics_not_column_position():
    task = TaskRegistry().task("leadership")
    assert category_tokens(task, "english", numerical=True) == {
        "4": ["4"],
        "3": ["3"],
        "2": ["2"],
        "1": ["1"],
    }
    prompt = build_prompt(task, "X acted.", numerical=True)
    assert "4 = Excellent Leadership" in prompt


def test_localized_and_numeric_outputs_normalize_to_canonical_labels():
    task = TaskRegistry().task("leadership")
    frame = pd.DataFrame(
        {
            "entity_index": [0],
            "sentence_index": [0],
            "4": [0.8],
            "3": [0.1],
            "2": [0.05],
            "1": [0.05],
        }
    )
    normalized = normalize_probability_columns(frame, task, "english", numerical=True)
    assert normalized["Excellent Leadership"].iloc[0] == 0.8
    assert normalized["Catastrophic Leadership"].iloc[0] == 0.05


def test_hashtag_substitution_compacts_entity_name():
    assert substitute_entity("I support #TeamX and X.", "Jane Doe") == "I support #TeamJaneDoe and Jane Doe."


def test_expansion_uses_configured_columns():
    registry = TaskRegistry()
    task = registry.task("intent")
    entity_set = registry.entity_set("politicians")
    templates = pd.DataFrame(
        {"sentence": ["X proposed a plan."], "sentence(zh)": ["X提出计划。"], "sentence(ru)": ["X предложил план."], "label": ["Public Good"]}
    )
    entities = pd.DataFrame({"name": ["Jane Doe"], "name(zh)": ["简·多伊"], "name(ru)": ["Джейн Доу"]})
    expanded = expand_frames(task, entity_set, templates, entities, "english", show_progress=False)
    assert expanded.loc[0, "sentence"] == "Jane Doe proposed a plan."
    assert expanded.loc[0, "target"] == "Jane Doe"


def test_every_generalization_task_has_valid_generation_inputs():
    registry = TaskRegistry()
    for task in registry.tasks(suite="generalization"):
        assert task.generation is not None
        keywords = load_keywords(task.generation)
        assert len(keywords) >= task.generation.keywords_per_prompt
        assert {example.label for example in task.generation.few_shot} == {
            label.id for label in task.labels
        }


def test_analysis_configuration_and_processed_column_adapter():
    registry = TaskRegistry()
    analysis = load_analysis_spec()
    validate_analysis_spec(analysis, registry)
    task = registry.task("intent")
    frame = pd.DataFrame(
        {
            "Grouped Alignment": ["FL", "CC"],
            "score_en_s1_m1_intent": [0.1, -0.1],
            "score_ru_s2_m2_intent": [0.2, -0.2],
            "p_en_s1_m1_intent": [0.0, 0.0],
        }
    )
    long = metric_long_frame(
        frame,
        "score",
        [task],
        "Grouped Alignment",
        {1: "First Model", 2: "Second Model"},
    )
    assert len(long) == 4
    assert set(long["language"]) == {"en", "ru"}
    assert set(long["setting"]) == {1, 2}
    assert set(long["model"]) == {"First Model", "Second Model"}
    assert set(long["task"]) == {"intent"}


def test_generation_prompt_uses_stable_label_and_keyword_contract():
    task = TaskRegistry().task("intent")
    keywords = load_keywords(task.generation)
    jobs = build_generation_jobs(task, keywords, seed=42, samples_per_label=1)
    prompt = render_generation_prompt(task, jobs[0])
    assert "Desired label: Public Good" in prompt
    assert "literal placeholder X" in prompt
    assert "REQUEST" in prompt


def test_generation_validation_requires_placeholder_and_keywords():
    assert validate_generated_sentence("X introduced a carefully reviewed policy proposal.", ["policy"], 1) is None
    assert validate_generated_sentence("这是关于X的翻译句子。", minimum_words=0) is None
    assert validate_generated_sentence("A carefully reviewed policy proposal was introduced.", ["policy"], 1)
    assert validate_generated_sentence("X introduced an unrelated carefully reviewed proposal.", ["policy"], 1)


def test_generation_parser_extracts_sentence_from_completion_artifacts():
    raw = (
        "πισ ContentType\n\n\n"
        "X incorporated public consultation into a carefully reviewed policy.\n\n"
        "Keywords: X, unrelated, continuation\n"
        "Sentence: X should not select this echoed prompt."
    )
    assert clean_generated_text(raw) == (
        "X incorporated public consultation into a carefully reviewed policy."
    )


def test_generation_engine_checkpoints_and_resumes():
    class FakeClient:
        def __init__(self):
            self.calls = 0
            self.model = "fake-model"

        def complete(self, prompts, **options):
            self.calls += 1
            outputs = []
            for index, prompt in enumerate(prompts):
                if prompt.startswith("Translate the following sentence"):
                    outputs.append(f"这是关于 X 的第 {index} 个翻译句子。")
                    continue
                request = prompt.rsplit("REQUEST", 1)[-1]
                keyword_line = next(line for line in request.splitlines() if line.startswith("Keywords:"))
                keyword = next(
                    value.strip()
                    for value in keyword_line.removeprefix("Keywords:").split(",")
                    if value.strip() != "X"
                )
                outputs.append(
                    f"X incorporated {keyword} into a carefully structured proposal number {self.calls}-{index}."
                )
            return outputs

    task = TaskRegistry().task("intent")
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        checkpoint = directory / "checkpoint.csv"
        first_output = directory / "first.csv"
        first_client = FakeClient()
        generate_dataset(
            task,
            first_client,
            first_output,
            checkpoint,
            languages=["english", "chinese"],
            samples_per_label=1,
            batch_size=2,
        )
        generated = pd.read_csv(first_output)
        assert len(generated) == len(task.labels)
        assert set(generated["label_id"]) == {label.id for label in task.labels}
        assert generated["sentence(zh)"].str.contains("X").all()

        resumed_client = FakeClient()
        generate_dataset(
            task,
            resumed_client,
            directory / "resumed.csv",
            checkpoint,
            languages=["english", "chinese"],
            samples_per_label=1,
            batch_size=2,
        )
        assert resumed_client.calls == 0


def test_model_is_discovered_from_server_port():
    model_id = "/models/provider/Example-Model"

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            assert self.path == "/v1/models"
            body = json.dumps(
                {"object": "list", "data": [{"id": model_id, "root": model_id}]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_port
        assert discover_model("127.0.0.1", port) == model_id
        assert list_served_models("127.0.0.1", port)[0]["root"] == model_id
        assert model_output_name(model_id) == "Example-Model"
        assert CompletionClient(host="127.0.0.1", port=port).model == model_id
        assert TextGenerationClient(host="127.0.0.1", port=port).model == model_id
    finally:
        server.shutdown()
        thread.join()
