"""
Shared utilities for the multilingual, multi-entity classification experiments.

This module centralizes (i) task-to-entity routing (countries/politicians/companies),
(ii) dataset path helpers, (iii) sentence-template expansion by substituting entity
names for the placeholder "X", (iv) output path conventions, and (v) wrappers that
dispatch to the appropriate prompt builder and category-token mapping per entity type.
"""

import re
import pandas as pd
import os
import math
from tqdm import tqdm
tqdm.pandas()

from utils_companies import *
from utils_countries import *
from utils_politicians import *

LANG_MAP = {
    "english": "",
    "chinese": "(zh)",
    "russian": "(ru)"
}

def get_entities(task):
    """
    Return the entity type associated with a task.

    Parameters
    ----------
    task : str
        Task identifier.

    Returns
    -------
    str or None
        One of {"politicians", "countries", "companies"} if known, else None.
    """

    mapping = {
        "leadership": "politicians",
        "misconduct": "politicians",
        "intent": "politicians",
        "sentimentP": "politicians",
        "sentimentCou": "countries",
        "urgency": "countries",
        "law": "countries",
        "credibility": "countries",
        "sentimentCom": "companies",
        "risk": "companies",
        "violation": "companies",
        "product": "companies",
    }
    return mapping.get(task)

def get_entities_path(entities):
    """
    Return the path to the entities CSV for a given entity type.

    Parameters
    ----------
    entities : str
        Entity type (e.g., "countries", "politicians", "companies").

    Returns
    -------
    str
        Filesystem path to the entities file.
    """

    return f"../data/{entities}/entities.csv"

def get_sentences_path(entities, task):
    """
    Return the path to the sentences CSV for a given entity type and task.

    Parameters
    ----------
    entities : str
        Entity type (e.g., "countries", "politicians", "companies").
    task : str
        Task identifier.

    Returns
    -------
    str
        Filesystem path to the sentences file.
    """

    return f"../data/{entities}/{task}.csv"



def get_output_path(entities, task, language, model, few_shot, numerical_labels):
    """
    Build and ensure the output path for a specific experimental configuration.

    Parameters
    ----------
    entities : str
        Entity type (e.g., "countries", "politicians", "companies").
    task : str
        Task identifier.
    language : str
        Language identifier.
    model : str
        Model name (used to namespace outputs).
    few_shot : bool
        Whether few-shot prompting is enabled.
    numerical_labels : bool
        Whether numerical labels are requested/used.

    Returns
    -------
    str
        Filesystem path to the output CSV for this configuration.
    """

    base_dir = f"../outputs/{entities}/{task}/{language}/{model}"
    os.makedirs(base_dir, exist_ok=True)
    return f"{base_dir}/output_{'fs' if few_shot else 'nofs'}_{'num' if numerical_labels else 'text'}.csv"

def expand_sentences_with_entities(sentences_df, entities_df, language):
    """
    Expand sentence templates by substituting each entity name for the placeholder "X".

    Parameters
    ----------
    sentences_df : pd.DataFrame
        Sentence templates with language-specific sentence columns and a "label" column.
    entities_df : pd.DataFrame
        Entity names with language-specific name columns.
    language : str
        Language identifier ("english", "russian", "chinese").

    Returns
    -------
    pd.DataFrame
        Expanded rows with sentence/entity indices, expanded sentence text, gold label,
        and the target entity name.
    """

    lan_suffix = LANG_MAP.get(language, '')
    rows = []
    for sent_idx, sent_row in tqdm(sentences_df.iterrows(), total=len(sentences_df), desc="Expanding sentences"):
        for ent_idx, ent_row in entities_df.iterrows():
            name = ent_row[f'name{lan_suffix}']

            def replacer(match):
                full = match.group(0)
                prefix = full.split('X')[0]
                if '#' in prefix or '@' in prefix:
                    return full.replace('X', name.replace(" ", ""))
                return full.replace('X', name)

            new_sentence = re.sub(r'[#@]?\w*X', replacer, sent_row[f'sentence{lan_suffix}'])
            rows.append({
                'sentence_index': sent_idx,
                'entity_index': ent_idx,
                'sentence': new_sentence,
                'label': sent_row['label'],
                'target': ent_row['name']
            })
    return pd.DataFrame(rows)

def expand_all_tasks(tasks, language):
    """
    Expand and return datasets for multiple tasks for a given language.

    Parameters
    ----------
    tasks : list[str]
        List of task identifiers.
    language : str
        Language identifier ("english", "russian", "chinese").

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping from task -> expanded dataframe.
    """

    out = {}
    for task in tasks:
        entities = get_entities(task)
        ent_path = get_entities_path(entities)
        sen_path = get_sentences_path(entities, task)
        entities_df = pd.read_csv(ent_path) # Attention : Test phase
        sentences_df = pd.read_csv(sen_path)
        out[task] = expand_sentences_with_entities(sentences_df, entities_df, language)
    return out



def logsumexp(log_probs):
    """
    Compute log-sum-exp of a list of log probabilities with basic numerical stability.

    Parameters
    ----------
    log_probs : list[float]
        Log-probability values.

    Returns
    -------
    float
        log(sum(exp(log_probs))).
    """

    if not log_probs:
        return float('-1e9')
    max_log = max(log_probs)
    return max_log + math.log(sum(math.exp(lp - max_log) for lp in log_probs))


def build_prompt(
        sentence,
        task,
        entities,
        language,
        few_shot=False,
        numerical_labels=False
):
    """
    Dispatch to the appropriate prompt builder based on entity type.

    Parameters
    ----------
    sentence : str
        Sentence to classify.
    task : str
        Task identifier.
    entities : str
        Entity type ("countries", "politicians", "companies").
    language : str
        Language identifier.
    few_shot : bool, default=False
        Whether to prepend few-shot examples.
    numerical_labels : bool, default=False
        Whether to request/use numerical labels.

    Returns
    -------
    str
        Prompt string ending with the label prefix.
    """

    if entities == 'countries':
        return build_country_prompt(
            sentence,
            task,
            few_shot,
            numerical_labels,
            language=language
        )
    elif entities == 'politicians':
        return build_politician_prompt(
            sentence,
            task,
            few_shot,
            numerical_labels,
            language=language
        )
    elif entities == 'companies':
        return build_company_prompt(
            sentence,
            task,
            few_shot,
            numerical_labels,
            language=language
        )
    

def get_category_tokens(
        task,
        entities,
        language,
        numerical=False
):
    """
    Dispatch to the appropriate category-token mapping based on entity type.

    Parameters
    ----------
    task : str
        Task identifier.
    entities : str
        Entity type ("countries", "politicians", "companies").
    language : str
        Language identifier.
    numerical : bool, default=False
        Whether to return numeric-label tokens (otherwise textual tokens).

    Returns
    -------
    dict[str, list[str]]
        Category-to-token mapping for the specified task/entity/language setting.
    """

    if entities == 'countries':
        return get_country_category_tokens(task, numerical, language=language)
    elif entities == 'politicians':
        return get_politician_category_tokens(task, numerical, language=language)
    elif entities == 'companies':
        return get_company_category_tokens(task, numerical, language=language)
    