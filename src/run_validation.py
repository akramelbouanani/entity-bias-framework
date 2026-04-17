#!/usr/bin/env python3
"""
Run batched classification experiments (original vs. synthetic data) against a
local model completion API and write per-row category probability outputs to CSV.

This script is intended as an experiment driver and integrity check: it expands
each task into datasets, builds prompts (optionally few-shot), queries the model
in batches with retries, and saves incremental outputs so runs can be resumed
after interruption.
"""

import argparse
import time
import os
import math
import pandas as pd
from tqdm import tqdm
import requests
from utils_validation import *

tqdm.pandas()

# ------------------------------------------------------------------
# Model inference
# ------------------------------------------------------------------

def call_classification_model(
    prompts,
    model_config,
    port=9715,
    category_tokens=None,
    max_retries=3,
    delay=2,
    debug=False,
):
    """
    Call a local completion API to classify one or more prompts into predefined
    categories using token-level log probabilities.

    Parameters
    ----------
    prompts : str or list[str]
        Prompt or list of prompts to classify.
    model_config : dict
        Configuration passed to the API request body (must include "model").
    port : int, default=9715
        Port where the local API is listening.
    category_tokens : dict[str, list[str]]
        Mapping from category name to representative tokens for that category.
    max_retries : int, default=3
        Maximum number of request retries upon failure.
    delay : float, default=2
        Delay in seconds between retries.
    debug : bool, default=False
        If True, return the raw `top_logprobs` dict for the first completion,
        without aggregating probabilities.

    Returns
    -------
    dict or list[dict]
        If `prompts` is a single string, returns a dict mapping category -> probability.
        If `prompts` is a list, returns a list of such dicts aligned to the input order.
        If all retries fail, returns NaN probabilities.
    """
    categories = list(category_tokens.keys())

    single_input = False
    if isinstance(prompts, str):
        prompts = [prompts]
        single_input = True

    payload = model_config.copy()
    payload.update({
        "prompt": prompts,
        "seed": 42,
        "max_tokens": 1,
        "temperature": 0,
        "top_p": 1,
        "n": 1,
        "logprobs": 20
    })

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"http://localhost:{port}/v1/completions",
                json=payload,
                timeout=60
            )
            resp.raise_for_status()
            data = resp.json()

            outputs = []
            for choice in data["choices"]:
                top_logprobs = choice.get("logprobs", {}).get("top_logprobs", [{}])[0]

                if debug:
                    return top_logprobs

                category_logprobs = {c: [] for c in categories}

                for token, logprob in top_logprobs.items():
                    t = token.strip().lower()
                    for c, toks in category_tokens.items():
                        if any(t.startswith(tok.lower()) for tok in toks):
                            category_logprobs[c].append(logprob)

                aggregated = {
                    c: logsumexp(v) if v else -1e9
                    for c, v in category_logprobs.items()
                }
                unnorm = {c: math.exp(v) for c, v in aggregated.items()}
                total = sum(unnorm.values())
                probs = (
                    {c: v / total for c, v in unnorm.items()}
                    if total > 0 else
                    {c: float("nan") for c in categories}
                )
                outputs.append(probs)

            return outputs[0] if single_input else outputs

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(delay)

    nan = {c: float("nan") for c in categories}
    return nan if single_input else [nan for _ in prompts]

# ------------------------------------------------------------------
# Batch classification
# ------------------------------------------------------------------

def batch_classification(
    df,
    model_config,
    port,
    task,
    language,
    few_shot=False,
    batch_request_size=500,
    batch_size=5000,
    kind="original",
    model=None
):
    """
    Run batched classification over a dataframe and append results to CSV.

    The function supports resuming: if the output CSV already exists, it reloads
    previously saved rows with non-NaN category probabilities and skips those
    indices.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with at least: "sentence", "target", "sentence_index", "entity_index".
    model_config : dict
        Configuration passed to the API request body (must include "model").
    port : int
        Port where the local API is listening.
    task : str
        Task name used to select prompt templates and category tokens.
    language : str
        Language identifier used by prompt/token utilities.
    few_shot : bool, default=False
        Whether to generate few-shot prompts.
    batch_request_size : int, default=500
        Number of prompts per API request.
    batch_size : int, default=5000
        Number of processed rows buffered before writing an interim CSV append.
    kind : {"original", "synthetic"}, default="original"
        Dataset variant to label outputs and build the output path.
    model : str or None, default=None
        Model name used to build the output path.

    Returns
    -------
    None
        Results are written incrementally to disk.
    """
    output_csv = get_output_path(task, kind, language, model)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    category_tokens = get_category_tokens(task, language)
    categories = list(category_tokens.keys())

    df = df.copy()
    processed_ids = set()

    if os.path.exists(output_csv):
        done = pd.read_csv(output_csv)
        done = done.dropna(subset=categories)
        processed_ids = set(done.index)
        print(f"Resuming — {len(processed_ids)} rows already processed.")

    to_run = df.drop(index=processed_ids).reset_index()
    print(f"Processing {len(to_run)} rows ({kind})")

    results = []

    for i in tqdm(range(0, len(to_run), batch_request_size), desc=f"{task} | {kind}"):
        batch = to_run.iloc[i:i + batch_request_size]

        prompts = [
            build_prompt(
                sentence=row["sentence"],
                task=task,
                target=row["target"],
                language=language,
                few_shot=few_shot
            )
            for _, row in batch.iterrows()
        ]

        batch_probs = call_classification_model(
            prompts=prompts,
            model_config=model_config,
            port=port,
            category_tokens=category_tokens
        )

        for (_, row), probs in zip(batch.iterrows(), batch_probs):
            results.append({
                "sentence_index": row["sentence_index"],
                "entity_index": row["entity_index"],
                **probs
            })

        if len(results) >= batch_size:
            pd.DataFrame(results).to_csv(
                output_csv,
                mode="a",
                index=False,
                header=not os.path.exists(output_csv)
            )
            results = []

    if results:
        pd.DataFrame(results).to_csv(
            output_csv,
            mode="a",
            index=False,
            header=not os.path.exists(output_csv)
        )

# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_arguments():
    """
    Parse command-line arguments for running the classification experiments.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments including tasks, few-shot flags, port, model info,
        and languages.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--few_shot", nargs="+", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--languages", nargs="+", default=["english"])
    return parser.parse_args()

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_arguments()

    tasks = args.tasks
    few_shot_list = [fs.lower() == "true" for fs in args.few_shot]
    languages = args.languages
    port = args.port

    model_config = {
        "model": os.path.join(args.model_path, args.model)
    }

    model = args.model

    for language in languages:
        for task in tasks:
            for kind in ["original", "synthetic"]:
                df = expand_task(task, language, kind)
                for few_shot in few_shot_list:
                    batch_classification(
                        df=df,
                        model_config=model_config,
                        port=port,
                        task=task,
                        language=language,
                        few_shot=few_shot,
                        kind=kind,
                        batch_request_size=1000,
                        batch_size=8000,
                        model=model
                    )
