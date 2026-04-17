#!/usr/bin/env python3
"""
Run batched classification experiments against a local model API and save per-row
category probability outputs to CSV.

This script is designed as an integrity/throughput driver for the experiment
pipeline: it expands tasks into datasets, builds prompts (optionally few-shot and
optionally using numerical labels), queries the model in batches with retries,
and writes incremental results to disk so runs can be resumed after interruption.
"""
import argparse
import time
import pandas as pd
import itertools
import os
import math
from tqdm import tqdm
from utils import *
import requests
tqdm.pandas()

RISK_PATH = '../data/companies/financial_risk.csv'
PRODUCT_PATH = '../data/companies/product_rating.csv'
VIOLATION_PATH = '../data/companies/regulatory_sentences.csv'
COMPANIES_PATH = '../data/companies/generated_companies.csv'

# -------------------------------
# Model inference
# -------------------------------

def call_classification_model(prompts, model_config, port=9715, category_tokens=None, max_retries=3, delay=2, debug=False, numerical_labels=False):
    """
    Call a local language model API to classify one or multiple prompts into
    predefined categories using token-level log probabilities.

    Parameters
    ----------
    prompts : str or list[str]
        A single prompt or a list of prompts to classify.
    model_config : dict
        Configuration dictionary passed to the API (must include the model path/name).
    port : int, default=9715
        Port on which the local completion API is running.
    category_tokens : dict[str, list[str]]
        Mapping from each category name to the list of tokens representing that category.
    max_retries : int, default=3
        Maximum number of retry attempts in case of request failure.
    delay : float, default=2
        Delay (in seconds) between retry attempts.
    debug : bool, default=False
        If True, returns the raw top_logprobs dictionary instead of aggregated probabilities.
    numerical_labels : bool, default=False
        If True, matches category tokens exactly; otherwise uses prefix-based matching.

    Returns
    -------
    dict or list[dict]
        If a single prompt is provided, returns a dictionary mapping category
        names to normalized probabilities. If multiple prompts are provided,
        returns a list of such dictionaries (one per prompt). In case of total
        failure, returns NaN probabilities.
    """

    categories = list(category_tokens.keys())
    if categories is None or category_tokens is None:
        raise ValueError("categories and category_tokens must be provided")

    # Normalize input to list
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

    # Retry loop
    for attempt in range(max_retries):
        try:
            resp = requests.post(f"http://localhost:{port}/v1/completions", json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()

            outputs = []
            for choice in data["choices"]:
                top_logprobs = choice.get("logprobs", {}).get("top_logprobs", [{}])[0]
                
                if debug:
                    return top_logprobs

                # Aggregate logprobs per category
                category_logprobs = {c: [] for c in categories}

                for token, logprob in top_logprobs.items():
                    t = token.strip().lower() 
                    for c, tokens in category_tokens.items():
                        if numerical_labels:
                            if any(tok.strip().lower() == t for tok in tokens):
                                category_logprobs[c].append(logprob)
                        else:
                            #if any(tok.strip().lower() in t for tok in tokens):
                            if any(t.startswith(tok.strip().lower()) for tok in tokens):
                                category_logprobs[c].append(logprob)

                # Compute probabilities
                aggregated = {c: logsumexp(logs) if logs else -1e9 for c, logs in category_logprobs.items()}
                unnorm = {c: math.exp(v) for c, v in aggregated.items()}
                total = sum(unnorm.values())
                probs = {c: v / total for c, v in unnorm.items()} if total > 0 else {c: float('nan') for c in categories}
                outputs.append(probs)

            return outputs[0] if single_input else outputs

        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            time.sleep(delay)

    # On failure
    nan_dict = {c: float('nan') for c in categories}
    return nan_dict if single_input else [nan_dict for _ in prompts]


# -------------------------------
# Batch classification
# -------------------------------


def batch_classification(
    df,
    model_config,
    port=9715,
    batch_request_size=500,
    batch_size=200,
    task=None,
    few_shot=False,
    numerical_labels=False,
    entities="companies",
    language="english"

):
    """
    Classifies entries in a dataframe using a specified prompt generator and model function.
    Saves partial results to CSV to allow resuming.
    
    Parameters:
        df: pd.DataFrame with at least a 'sentence' column
        model_config: dict, model configuration for API
        create_prompt_fn: function, takes a string and returns a prompt
        categories: list of category names
        category_tokens: dict mapping category -> list of tokens
        port: int, API port
        batch_request_size: int, number of prompts per API call
        batch_size: int, number of results before saving interim CSV
        output_csv: str or None, path to save CSV
    """
    output_csv = get_output_path(entities,
                                 task,
                                 language,
                                 os.path.basename(model_config.get("model", "unknown")),
                                 few_shot, numerical_labels)

    category_tokens = get_category_tokens(task, entities, language=language, numerical=numerical_labels)
    categories = list(category_tokens.keys())

    df = df.copy()
    processed_ids = set()

    # Resume from previous results
    if output_csv and os.path.exists(output_csv):
        done = pd.read_csv(output_csv)
        done = done.dropna(subset=categories)
        processed_ids = set(done.index)
        print(f"Resuming — {len(processed_ids)} valid results already saved.")

    to_run = df.drop(index=processed_ids).reset_index()
    print(f"Processing {len(to_run)} remaining rows...")

    results = []

    # Batch inference
    for i in tqdm(range(0, len(to_run), batch_request_size), desc="Classification (batches)"):
        batch = to_run.iloc[i:i+batch_request_size]

        # Build batch prompts using modular function
        batch_prompts = [build_prompt(s,
                                      task=task,
                                      entities=entities,
                                      language=language,
                                      few_shot=few_shot,
                                      numerical_labels=numerical_labels)
                        for s in batch['sentence']]

        # Call classifier for whole batch
        batch_probs = call_classification_model(
            prompts=batch_prompts,
            model_config=model_config,
            port=port,
            category_tokens=category_tokens,
            numerical_labels=numerical_labels
        )

        # Append aligned results
        for (row_idx, row), probs in zip(batch.iterrows(), batch_probs):
            results.append({
                'entity_index': row['entity_index'],
                'sentence_index': row['sentence_index'],
                **probs
            })

        # Interim save
        if len(results) >= batch_size and output_csv:
            temp = pd.DataFrame(results)
            temp.to_csv(
                output_csv,
                mode='a',
                index=False,
                header=not os.path.exists(output_csv)
            )
            results = []

    # Final save
    if output_csv and results:
        temp = pd.DataFrame(results).dropna()
        temp.to_csv(output_csv, mode='a', index=False, header=not os.path.exists(output_csv))


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', nargs='+', required=True, help="List of task names.")
    parser.add_argument('--few_shot', nargs='+', required=True, help="List of True/False values for few-shot.")
    parser.add_argument('--numerical', nargs='+', required=True, help="List of True/False values for numerical labels.")
    parser.add_argument('--port', type=int, required=True, help="Port to use for generation.")
    parser.add_argument('--model_path', type=str, required=False, help="Path to the model.")
    parser.add_argument('--model', type=str, required=True, help="Model name.")
    parser.add_argument('--languages', nargs='+', default=["english"], help="List of languages.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # Prepare parameters

    tasks = args.tasks
    few_shot_list = [fs.lower() == 'true' for fs in args.few_shot]
    numerical_list = [nl.lower() == 'true' for nl in args.numerical]
    languages = args.languages
    port = args.port
    model_config = {"model": f"{args.model_path}/{args.model}"}

    # Prepare data

    for language in languages:

        expanded = expand_all_tasks(tasks, language=language)
        
        for task in tasks:
            df = expanded[task]
            entities = get_entities(task)
            for few_shot, numerical_labels in itertools.product(few_shot_list, numerical_list):
                batch_classification(
                    df=df,
                    model_config=model_config,
                    batch_size=8000,
                    batch_request_size=1000,
                    port=port,
                    task=task,
                    few_shot=few_shot,
                    numerical_labels=numerical_labels,
                    entities=entities,
                    language=language
                )