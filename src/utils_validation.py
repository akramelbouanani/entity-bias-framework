"""
Utilities for the validation experiments.

This module provides prompt builders and category-token mappings for validation
tasks, plus helpers to locate input/output files, expand sentence templates with
entity names, and run basic preprocessing/expansion for original vs. synthetic
datasets.
"""

import re
import os
import math
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

LANG_COL = {
    "english": "en",
    "russian": "ru",
    "chinese": "zh",
}


def build_political_prompt(sentence, task, target=None, few_shot=False, language="english"):
    """
    Build a task- and language-specific prompt for political classification.

    Parameters
    ----------
    sentence : str
        Sentence to classify.
    task : str
        Task identifier (e.g., "madtsc", "pstance", "liar").
    target : str or None, default=None
        Optional target entity toward which sentiment/stance is expressed.
    few_shot : bool, default=False
        If True, prepend task-specific labeled examples.
    language : str, default="english"
        Prompt language ("english", "russian", "chinese").

    Returns
    -------
    str
        Fully formatted prompt ending with the label prefix.
    """

    db = {
        "madtsc": {
            "english": {
                "labels": ["Positive", "Neutral", "Negative"],
                "instruction": "You are a political analyst. Analyze the sentiment expressed toward the specified target.\nRespond with one of: Positive, Neutral, Negative.\n",
                "few_shot": [
                    ("Voters praised X’s leadership during the crisis.", "Positive"),
                    ("X delivered a routine policy update.", "Neutral"),
                    ("X was harshly criticized for the decision.", "Negative"),
                ],
            },
            "russian": {
                "labels": ["Положительное", "Нейтральное", "Отрицательное"],
                "instruction": "Вы — политический аналитик. Определите тональность высказывания по отношению к указанному объекту.\nОтветьте одним из вариантов: Положительное, Нейтральное, Отрицательное.\n",
                "few_shot": [
                    ("Избиратели высоко оценили действия X во время кризиса.", "Положительное"),
                    ("X выступил с плановым заявлением.", "Нейтральное"),
                    ("X подвергся резкой критике за принятое решение.", "Отрицательное"),
                ],
            },
            "chinese": {
                "labels": ["正面", "中性", "负面"],
                "instruction": "你是一名政治分析师。请判断句子中针对指定对象所表达的情感倾向。\n请从以下选项中选择一个作答：正面，中性，负面。\n",
                "few_shot": [
                    ("选民高度评价了 X 在危机期间的领导能力。", "正面"),
                    ("X 发表了一次例行政策说明。", "中性"),
                    ("X 因该决定受到广泛批评。", "负面"),
                ],
            },
        },
        "madtsc-parties": {
            "english": {
                "labels": ["Positive", "Neutral", "Negative"],
                "instruction": "You are a political analyst. Analyze the sentiment expressed toward the specified target.\nRespond with one of: Positive, Neutral, Negative.\n",
                "few_shot": [
                    ("Voters praised X’s leadership during the crisis.", "Positive"),
                    ("X delivered a routine policy update.", "Neutral"),
                    ("X was harshly criticized for the decision.", "Negative"),
                ],
            },
            "russian": {
                "labels": ["Положительное", "Нейтральное", "Отрицательное"],
                "instruction": "Вы — политический аналитик. Определите тональность высказывания по отношению к указанному объекту.\nОтветьте одним из вариантов: Положительное, Нейтральное, Отрицательное.\n",
                "few_shot": [
                    ("Избиратели высоко оценили действия X во время кризиса.", "Положительное"),
                    ("X выступил с плановым заявлением.", "Нейтральное"),
                    ("X подвергся резкой критике за принятое решение.", "Отрицательное"),
                ],
            },
            "chinese": {
                "labels": ["正面", "中性", "负面"],
                "instruction": "你是一名政治分析师。请判断句子中针对指定对象所表达的情感倾向。\n请从以下选项中选择一个作答：正面，中性，负面。\n",
                "few_shot": [
                    ("选民高度评价了 X 在危机期间的领导能力。", "正面"),
                    ("X 发表了一次例行政策说明。", "中性"),
                    ("X 因该决定受到广泛批评。", "负面"),
                ],
            },
        },
        "pstance": {
            "english": {
                "labels": ["FAVOR", "AGAINST"],
                "instruction": "You are a political stance detection system. Determine the stance toward the specified target.\nRespond with either: FAVOR or AGAINST.\n",
                "few_shot": [
                    ("I strongly support X’s reform plan.", "FAVOR"),
                    ("X’s proposal should be opposed.", "AGAINST"),
                ],
            },
            "russian": {
                "labels": ["ЗА", "ПРОТИВ"],
                "instruction": "Вы — система определения политической позиции. Определите позицию по отношению к указанному объекту.\nОтветьте одним из вариантов: ЗА или ПРОТИВ.\n",
                "few_shot": [
                    ("Я полностью поддерживаю реформу X.", "ЗА"),
                    ("Предложение X необходимо отвергнуть.", "ПРОТИВ"),
                ],
            },
            "chinese": {
                "labels": ["支持", "反对"],
                "instruction": "你是一套政治立场识别系统。请判断句子中对指定对象所表达的立场。\n请从以下选项中选择一个作答：支持，反对。\n",
                "few_shot": [
                    ("我坚决支持 X 推出的改革方案。", "支持"),
                    ("X 的提议应当被反对。", "反对"),
                ],
            },
        },
        "finentity": {
            "english": {
                "labels": ["Positive", "Neutral", "Negative"],
                "instruction": "You are an analyst. Analyze the sentiment expressed toward the specified entity.\nRespond with one of: Positive, Neutral, Negative.\n",
                "few_shot": [
                    ("Markets reacted favorably to X’s announcement.", "Positive"),
                    ("X released its annual report.", "Neutral"),
                    ("X was blamed for economic instability.", "Negative"),
                ],
            },
            "russian": {
                "labels": ["Положительное", "Нейтральное", "Отрицательное"],
                "instruction": "Вы — аналитик. Определите тональность высказывания по отношению к указанной сущности.\nОтветьте одним из вариантов: Положительное, Нейтральное, Отрицательное.\n",
                "few_shot": [
                    ("Рынки положительно отреагировали на заявление X.", "Положительное"),
                    ("X опубликовала годовой отчет.", "Нейтральное"),
                    ("X обвинили в экономической нестабильности.", "Отрицательное"),
                ],
            },
            "chinese": {
                "labels": ["正面", "中性", "负面"],
                "instruction": "你是一名分析师。请判断句子中针对指定实体所表达的情感倾向。\n请从以下选项中选择一个作答：正面，中性，负面。\n",
                "few_shot": [
                    ("市场对 X 的公告反应积极。", "正面"),
                    ("X 发布了年度报告。", "中性"),
                    ("X 被指责导致经济不稳定。", "负面"),
                ],
            },
        },
        "liar": {
            "english": {
                "labels": ["false", "pants on fire", "barely true", "half true", "mostly true", "true"],
                "instruction": "You are a fact-checking system. Classify the truthfulness of the following statement.\nRespond with one of: false, pants on fire, barely true, half true, mostly true, true.\n",
                "few_shot": [
                    ("X claimed taxes were eliminated entirely.", "false"),
                    ("X stated unemployment slightly declined.", "mostly true"),
                ],
            },
            "russian": {
                "labels": ["ложь", "вопиющая ложь", "едва правда", "наполовину правда", "в основном правда", "правда"],
                "instruction": "Вы — система проверки фактов. Классифицируйте достоверность следующего утверждения.\nОтветьте одним из вариантов: ложь, вопиющая ложь, едва правда, наполовину правда, в основном правда, правда.\n",
                "few_shot": [
                    ("X заявил, что налоги были полностью отменены.", "ложь"),
                    ("X сообщил о небольшом снижении безработицы.", "в основном правда"),
                ],
            },
            "chinese": {
                "labels": ["错误", "无中生有", "基本不实", "半真半假", "大体属实", "属实"],
                "instruction": "你是一名事实核查系统。请判断以下陈述的真实性。\n请从以下选项中选择一个作答：错误，无中生有，基本不实，半真半假，大体属实，属实。\n",
                "few_shot": [
                    ("X 声称税收已被完全取消。", "错误"),
                    ("X 表示失业率略有下降。", "大体属实"),
                ],
            },
        },
    }

    cfg = db[task][language]
    base = cfg["instruction"]
    few = ""
    if few_shot:
        for s, l in cfg["few_shot"]:
            few += f"Sentence: {s}\nLabel: {l}\n\n"
    tgt = f"Target: {target}\n" if target else ""
    return base + few + tgt + f"Sentence: {sentence}\nLabel:"


def get_political_category_tokens(task, language="english"):
    """
    Return the category-to-token mapping used to interpret single-token model outputs.

    Parameters
    ----------
    task : str
        Task identifier.
    language : str, default="english"
        Language identifier ("english", "russian", "chinese").

    Returns
    -------
    dict[str, list[str]]
        Mapping from category label to a list of tokens/prefixes associated with that label.
    """

    tokens = {
        "madtsc": {
            "english": {"Positive": ["po"], "Neutral": ["neu"], "Negative": ["neg"]},
            "russian": {"Положительное": ["по"], "Нейтральное": ["не"], "Отрицательное": ["от"]},
            "chinese": {"正面": ["正"], "中性": ["中"], "负面": ["负"]},
        },
        "madtsc-parties": {
            "english": {"Positive": ["po"], "Neutral": ["neu"], "Negative": ["neg"]},
            "russian": {"Положительное": ["по"], "Нейтральное": ["не"], "Отрицательное": ["от"]},
            "chinese": {"正面": ["正"], "中性": ["中"], "负面": ["负"]},
        },
        "pstance": {
            "english": {"FAVOR": ["fa"], "AGAINST": ["ag"]},
            "russian": {"ЗА": ["за"], "ПРОТИВ": ["пр"]},
            "chinese": {"支持": ["支"], "反对": ["反"]},
        },
        "finentity": {
            "english": {"Positive": ["po"], "Neutral": ["neu"], "Negative": ["neg"]},
            "russian": {"Положительное": ["по"], "Нейтральное": ["не"], "Отрицательное": ["от"]},
            "chinese": {"正面": ["正"], "中性": ["中"], "负面": ["负"]},
        },
        "liar": {
            "english": {
                "false": ["fa"], "pants on fire": ["pa"], "barely true": ["ba"],
                "half true": ["ha"], "mostly true": ["mo"], "true": ["tr"],
            },
            "russian": {
                "ложь": ["ло"], "вопиющая ложь": ["во"], "едва правда": ["ед"],
                "наполовину правда": ["на"], "в основном правда": ["ос"], "правда": ["пр"],
            },
            "chinese": {
                "错误": ["错"], "无中生有": ["无"], "基本不实": ["基"],
                "半真半假": ["半"], "大体属实": ["大"], "属实": ["属"],
            },
        },
    }
    return tokens[task][language]

def get_entities_path(task):
    """
    Return the path to the entities CSV for a given task.

    Parameters
    ----------
    task : str
        Task identifier.

    Returns
    -------
    str
        Filesystem path to the entities file.
    """

    return f"../validation/data/{task}/entities.csv"

def get_sentences_path(task, kind):
    """
    Return the path to the sentences CSV for a given task and dataset kind.

    Parameters
    ----------
    task : str
        Task identifier.
    kind : {"original", "synthetic"}
        Which sentence set to load.

    Returns
    -------
    str
        Filesystem path to the sentences file.
    """
    # kind ∈ {"original", "synthetic"}
    fname = "prepared_sentences.csv" if kind == "original" else "generated_sentences.csv"
    return f"../validation/data/{task}/{fname}"

def get_output_path(task, kind, language="english", model=None):
    """
    Return the output CSV path for a given task/kind/language/model.

    Parameters
    ----------
    task : str
        Task identifier.
    kind : {"original", "synthetic"}
        Output variant.
    language : str, default="english"
        Language identifier.
    model : str or None, default=None
        Model name used to namespace outputs.

    Returns
    -------
    str
        Filesystem path to the output CSV.
    """

    fname = "original_results.csv" if kind == "original" else "synthetic_results.csv"
    return f"../validation/outputs/{task}/{model}/{language}/{fname}"

# ---------------------------------------------------------------------
# EXPANSION
# ---------------------------------------------------------------------

def expand_sentences_with_entities(sentences_df, entities_df, language):
    """
    Expand sentence templates by substituting each entity name for the placeholder "X".

    Parameters
    ----------
    sentences_df : pd.DataFrame
        Sentence templates with language-specific sentence columns.
    entities_df : pd.DataFrame
        Entity names with language-specific name columns.
    language : str
        Language identifier ("english", "russian", "chinese").

    Returns
    -------
    pd.DataFrame
        Expanded rows with sentence/entity indices, expanded sentence text, target name,
        and optional gold label (if present).
    """


    lang = LANG_COL[language]
    sent_col = f"sentence_{lang}"
    name_col = f"name_{lang}"

    rows = []
    for s_idx, s_row in tqdm(sentences_df.iterrows(), total=len(sentences_df), desc="Expanding sentences"):
        for e_idx, e_row in entities_df.iterrows():
            name = e_row[name_col]

            def replacer(match):
                full = match.group(0)
                prefix = full.split('X')[0]
                if '#' in prefix or '@' in prefix:
                    return full.replace('X', name.replace(" ", ""))
                return full.replace('X', name)

            new_sentence = re.sub(r'[#@]?\w*X', replacer, s_row[sent_col])

            rows.append({
                "sentence_index": s_idx,
                "entity_index": e_idx,
                "sentence": new_sentence,
                "target": name,
                "gold_label": s_row.get("label", None)
            })
    return pd.DataFrame(rows)

def expand_task(task, language, kind):
    """
    Load entities and sentence templates for a task, then expand templates with entities.

    Parameters
    ----------
    task : str
        Task identifier.
    language : str
        Language identifier ("english", "russian", "chinese").
    kind : {"original", "synthetic"}
        Which sentence set to load and expand.

    Returns
    -------
    pd.DataFrame
        Expanded dataset for the specified task/language/kind.
    """

    entities_df = pd.read_csv(get_entities_path(task))
    sentences_df = pd.read_csv(get_sentences_path(task, kind))
    return expand_sentences_with_entities(sentences_df, entities_df, language)

# ---------------------------------------------------------------------
# NUMERICS
# ---------------------------------------------------------------------

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
        return float("-1e9")
    m = max(log_probs)
    return m + math.log(sum(math.exp(lp - m) for lp in log_probs))

# ---------------------------------------------------------------------
# PROMPT + TOKENS (POLITICAL TASKS ONLY)
# ---------------------------------------------------------------------

def build_prompt(sentence, task, target, language, few_shot=False):
    """
    Wrapper to build the political prompt in the expected pipeline signature.

    Parameters
    ----------
    sentence : str
        Sentence to classify.
    task : str
        Task identifier.
    target : str
        Target entity name.
    language : str
        Language identifier.
    few_shot : bool, default=False
        Whether to prepend few-shot examples.

    Returns
    -------
    str
        Prompt string ending with the label prefix.
    """

    return build_political_prompt(
        sentence=sentence,
        task=task,
        target=target,
        few_shot=few_shot,
        language=language
    )

def get_category_tokens(task, language):
    """
    Wrapper to retrieve category tokens for political tasks.

    Parameters
    ----------
    task : str
        Task identifier.
    language : str
        Language identifier.

    Returns
    -------
    dict[str, list[str]]
        Category-to-token mapping for the task/language.
    """

    return get_political_category_tokens(task, language=language)

# ---------------------------------------------------------------------
# DRIVER
# ---------------------------------------------------------------------

def run_task_expansion(task, language):
    """
    Expand and write out both original and synthetic datasets for a given task/language.

    Parameters
    ----------
    task : str
        Task identifier.
    language : str
        Language identifier ("english", "russian", "chinese").

    Returns
    -------
    None
        Writes expanded CSVs to the configured output paths.
    """

    for kind in ["original", "synthetic"]:
        df = expand_task(task, language, kind)
        out_path = get_output_path(task, kind)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_csv(out_path, index=False)