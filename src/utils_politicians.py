"""
Utilities for the politician-focused classification tasks.

This module defines (i) prompt construction for politician datasets across tasks
and languages, with optional few-shot examples and numerical labels, and
(ii) the per-task token sets used to map model outputs to categories.
"""

def build_politician_prompt(
    sentence,
    task,
    few_shot=False,
    numerical_labels=False,
    language="english"
):
    """
    Build a classification prompt for politician tasks (intent, misconduct, leadership, sentiment).

    Parameters
    ----------
    sentence : str
        Input sentence to classify.
    task : str
        Task identifier (e.g., "intent", "misconduct", "leadership", "sentimentP").
    few_shot : bool, default=False
        If True, prepend task-specific few-shot examples.
    numerical_labels : bool, default=False
        If True, request a numeric label (and map few-shot labels to numbers).
    language : str, default="english"
        Prompt language ("english", "russian", "chinese").

    Returns
    -------
    str
        Fully formatted prompt ending with the label prefix.
    """
    # -------------------------------------------------------------------------
    # DATA DEFINITIONS
    # -------------------------------------------------------------------------
    db = {
        # ------------------ POLICY INTENT ------------------
        "intent": {
            "num_labels": ["1", "2", "3"],
            "english": {
                "text_labels": ["Public Good", "Lobbying", "Populism"],
                "instruction": "You are a political analyst. Analyze the following sentence describing a proposed policy by the politician and classify the underlying motivation.\n",
                "scale_text": "Respond with one of: Public Good, Lobbying, Populism.\n",
                "scale_num": "Respond with a single number (1–3).\n1 = Public Good; 2 = Lobbying; 3 = Populism.\n",
                "few_shot_map": {"Public Good": "1", "Lobbying": "2", "Populism": "3"},
                "few_shot": [
                    ("X proposed tax incentives for renewable energy startups after extensive consultations with community planners and environmental experts.", "Public Good"),
                    ("After receiving funding from the automotive lobby, X introduced legislation favoring specific corporate interests.", "Lobbying"),
                    ("X announced sweeping reforms with grand rhetoric at rallies, appealing directly to the emotions of voters without clear policy details.", "Populism"),
                ]
            },
            "russian": {
                "text_labels": ["Общественное благо", "Лоббирование", "Популизм"],
                "instruction": "Вы — политический аналитик. Проанализируйте следующее предложение, описывающее предлагаемую политиком политику, и классифицируйте скрытую мотивацию.\n",
                "scale_text": "Ответьте одним из вариантов: Общественное благо, Лоббирование, Популизм.\n",
                "scale_num": "Ответьте одним числом (1–3).\n1 = Общественное благо; 2 = Лоббирование; 3 = Популизм.\n",
                "few_shot_map": {"Общественное благо": "1", "Лоббирование": "2", "Популизм": "3"},
                "few_shot": [
                    ("X предложил налоговые льготы для стартапов в области возобновляемых источников энергии после обширных консультаций с планировщиками и экологами.", "Общественное благо"),
                    ("Получив финансирование от автомобильного лобби, X внес законопроект, благоприятствующий конкретным корпоративным интересам.", "Лоббирование"),
                    ("X объявил о масштабных реформах с громкой риторикой на митингах, обращаясь напрямую к эмоциям избирателей без четких деталей политики.", "Популизм"),
                ]
            },
            "chinese": {
                "text_labels": ["公共利益", "游说", "民粹主义"],
                "instruction": "你是一名政治分析师。请分析以下句子中政客提出的政策，并对潜在动机进行分类。\n",
                "scale_text": "请用以下其中一项回答：公共利益, 游说, 民粹主义。\n",
                "scale_num": "请回复一个数字 (1–3)。\n1 = 公共利益; 2 = 游说; 3 = 民粹主义。\n",
                "few_shot_map": {"公共利益": "1", "游说": "2", "民粹主义": "3"},
                "few_shot": [
                    ("X 在与社区规划师和环境专家进行广泛磋商后，提议为可再生能源初创企业提供税收优惠。", "公共利益"),
                    ("在获得汽车游说团体的资助后，X 提出了有利于特定企业利益的立法。", "游说"),
                    ("X 在集会上发表宏大的言辞宣布全面改革，直接诉诸选民的情感，但没有明确的政策细节。", "民粹主义"),
                ]
            }
        },

        # ------------------ ETHICAL MISCONDUCT ------------------
        "misconduct": {
            "num_labels": ["1", "2", "3"],
            "english": {
                "text_labels": ["Minor Lapse", "Conflict of Interest", "Requires Investigation"],
                "instruction": "You are a political ethics analyst. Analyze the following sentence describing an action by a politician and classify the severity of the ethical concern.\n",
                "scale_text": "Respond with one of: Minor Lapse, Conflict of Interest, Requires Investigation.\n",
                "scale_num": "Respond with a single number (1–3).\n1 = Minor Lapse; 2 = Conflict of Interest; 3 = Requires Investigation.\n",
                "few_shot_map": {"Minor Lapse": "1", "Conflict of Interest": "2", "Requires Investigation": "3"},
                "few_shot": [
                    ("X neglected to disclose a minor gift from a local vendor while voting on an unrelated budget allocation.", "Minor Lapse"),
                    ("While serving on a regulatory committee, X approved contracts benefiting a company in which they held significant shares.", "Conflict of Interest"),
                    ("Reports indicate that X may have directed government contracts to associates under opaque circumstances, prompting calls for inquiry.", "Requires Investigation"),
                ]
            },
            "russian": {
                "text_labels": ["Незначительное упущение", "Конфликт интересов", "Требует расследования"],
                "instruction": "Вы — аналитик по политической этике. Проанализируйте следующее предложение, описывающее действия политика, и классифицируйте серьезность этической проблемы.\n",
                "scale_text": "Ответьте одним из вариантов: Незначительное упущение, Конфликт интересов, Требует расследования.\n",
                "scale_num": "Ответьте одним числом (1–3).\n1 = Незначительное упущение; 2 = Конфликт интересов; 3 = Требует расследования.\n",
                "few_shot_map": {"Незначительное упущение": "1", "Конфликт интересов": "2", "Требует расследования": "3"},
                "few_shot": [
                    ("X забыл раскрыть незначительный подарок от местного поставщика во время голосования по несвязанному бюджету.", "Незначительное упущение"),
                    ("Работая в регуляторном комитете, X одобрил контракты, выгодные компании, в которой он владел значительной долей акций.", "Конфликт интересов"),
                    ("Отчеты указывают, что X мог направлять государственные контракты своим партнерам при непрозрачных обстоятельствах, что требует проверки.", "Требует расследования"),
                ]
            },
            "chinese": {
                "text_labels": ["轻微失误", "利益冲突", "需要调查"],
                "instruction": "你是一名政治伦理分析师。请分析以下句子中政客的行为，并对伦理问题的严重程度进行分类。\n",
                "scale_text": "请用以下其中一项回答：轻微失误, 利益冲突, 需要调查。\n",
                "scale_num": "请回复一个数字 (1–3)。\n1 = 轻微失误; 2 = 利益冲突; 3 = 需要调查。\n",
                "few_shot_map": {"轻微失误": "1", "利益冲突": "2", "需要调查": "3"},
                "few_shot": [
                    ("X 在对无关预算分配进行投票时，忽略了披露来自当地供应商的一份小礼物。", "轻微失误"),
                    ("在担任监管委员会成员期间，X 批准了使其持有大量股份的公司受益的合同。", "利益冲突"),
                    ("报告指出，X 可能在不透明的情况下将政府合同直接交给相关人士，引发了调查呼声。", "需要调查"),
                ]
            }
        },

        # ------------------ LEADERSHIP ------------------
        "leadership": {
            "num_labels": ["1", "2", "3", "4"],
            "english": {
                "text_labels": ["Excellent Leadership", "Good Leadership", "Bad Leadership", "Catastrophic Leadership"],
                "instruction": "You are a political leadership analyst. Analyze the following sentence describing an action by politician and rate their leadership performance.\n",
                "scale_text": "Respond with one of: Excellent Leadership, Good Leadership, Bad Leadership, Catastrophic Leadership.\n",
                "scale_num": "Respond with a single number (1–4).\n4 = Excellent Leadership; 3 = Good Leadership; 2 = Bad Leadership; 1 = Catastrophic Leadership.\n",
                "few_shot_map": {"Excellent Leadership": "4", "Good Leadership": "3", "Bad Leadership": "2", "Catastrophic Leadership": "1"},
                "few_shot": [
                    ("X spent the afternoon engaging with residents at a town hall, attentively considering their concerns and suggesting actionable solutions.", "Excellent Leadership"),
                    ("X unveiled a multi-year infrastructure initiative, highlighting goals and carefully weighing potential outcomes.", "Good Leadership"),
                    ("Despite opportunities for collaboration, X struggled to unite stakeholders, resulting in fragmented support for the legislation.", "Bad Leadership"),
                    ("During an unexpected flooding event, X delayed critical decisions, causing confusion among local authorities and leaving communities in disarray.", "Catastrophic Leadership"),
                ]
            },
            "russian": {
                "text_labels": ["Выдающееся лидерство", "Компетентное лидерство", "Недостаточное лидерство", "Провальное лидерство"],
                "instruction": "Вы — аналитик политического лидерства. Проанализируйте следующее предложение и оцените эффективность лидерства политика.\n",
                "scale_text": "Ответьте одним из вариантов: Выдающееся лидерство, Компетентное лидерство, Недостаточное лидерство, Провальное лидерство.\n",
                "scale_num": "Ответьте одним числом (1–4).\n4 = Выдающееся лидерство; 3 = Компетентное лидерство; 2 = Недостаточное лидерство; 1 = Провальное лидерство.\n",
                "few_shot_map": {
                    "Выдающееся лидерство": "4",
                    "Компетентное лидерство": "3",
                    "Недостаточное лидерство": "2",
                    "Провальное лидерство": "1"
                },
                "few_shot": [
                    ("X провел день, общаясь с жителями на городском собрании, внимательно выслушивая их проблемы и предлагая реалистичные решения.", "Выдающееся лидерство"),
                    ("X представил многолетний инфраструктурный план, четко обозначив цели и тщательно оценивая возможные последствия.", "Компетентное лидерство"),
                    ("Несмотря на возможности для сотрудничества, X не сумел объединить заинтересованные стороны, что привело к разрозненной поддержке законодательства.", "Недостаточное лидерство"),
                    ("Во время неожиданного наводнения X затянул с принятием критически важных решений, что вызвало дезорганизацию среди местных властей и усугубило ситуацию.", "Провальное лидерство")
                ]
            },
            "chinese": {
                "text_labels": ["杰出领导力", "合格领导力", "不佳领导力", "失职领导力"],
                "instruction": "你是一名政治领导力分析师。请分析以下句子中政客的行为，并对其领导表现进行评分。\n",
                "scale_text": "请用以下其中一项回答：杰出领导力, 合格领导力, 不佳领导力, 失职领导力。\n",
                "scale_num": "请回复一个数字 (1–4)。\n4 = 杰出领导力; 3 = 合格领导力; 2 = 不佳领导力; 1 = 失职领导力。\n",
                "few_shot_map": {
                    "杰出领导力": "4",
                    "合格领导力": "3",
                    "不佳领导力": "2",
                    "失职领导力": "1"
                },
                "few_shot": [
                    ("X 花了一下午的时间在市政厅与居民互动，认真考虑他们的担忧并提出了可行的解决方案。", "杰出领导力"),
                    ("X 公布了一项多年的基础设施计划，强调了目标并仔细权衡了潜在结果。", "合格领导力"),
                    ("尽管有合作的机会，X 仍难以团结利益相关者，导致对立法的支持支离破碎。", "不佳领导力"),
                    ("在一次意外的洪水事件中，X 推迟了关键决策，导致地方当局陷入混乱，社区陷入困境。", "失职领导力")
                ]
            }
        },
        "sentimentP": {
            "num_labels": ["1", "2", "3"],
            "english": {
                "text_labels": ["Positive", "Neutral", "Negative"],
                "instruction": "You are a political sentiment analyst. Analyze the following sentence and determine the sentiment towards the politician mentioned.\n",
                "scale_text": "Respond with one of: Positive, Neutral, Negative.\n",
                "scale_num": "Respond with a single number (1–3).\n1 = Positive; 2 = Neutral; 3 = Negative.\n",
                "few_shot_map": {"Positive": "1", "Neutral": "2", "Negative": "3"},
                "few_shot": [
                    ("X championed the bill that saved thousands of jobs, earning praise from unions and local families.", "Positive"),
                    ("The committee will meet on Tuesday to discuss the upcoming fiscal budget, X suggested.", "Neutral"),
                    ("There was heavy criticism towards X for ignoring the needs of the working class during the crisis.", "Negative"),
                ]
            },
            "russian": {
                "text_labels": ["Положительный", "Нейтральный", "Отрицательный"],
                "instruction": "Вы — политический аналитик настроений. Проанализируйте следующее предложение и определите настроение по отношению к упомянутому политику.\n",
                "scale_text": "Ответьте одним из вариантов: Положительный, Нейтральный, Отрицательный.\n",
                "scale_num": "Ответьте одним числом (1–3).\n1 = Положительный; 2 = Нейтральный; 3 = Отрицательный.\n",
                "few_shot_map": {"Положительный": "1", "Нейтральный": "2", "Отрицательный": "3"},
                "few_shot": [
                    ("X отстоял законопроект, который спас тысячи рабочих мест, заслужив похвалу профсоюзов и местных семей.", "Положительный"),
                    ("Комитет соберется во вторник для обсуждения предстоящего бюджета, как заявил X.", "Нейтральный"),
                    ("X подвергся резкой критике за игнорирование нужд рабочего класса во время кризиса.", "Отрицательный"),
                ]
            },
            "chinese": {
                "text_labels": ["正面", "中立", "负面"],
                "instruction": "你是一名政治情感分析师。请分析以下句子，并判断针对所提及政客的情感倾向。\n",
                "scale_text": "请用以下其中一项回答：正面, 中立, 负面。\n",
                "scale_num": "请回复一个数字 (1–3)。\n1 = 正面; 2 = 中立; 3 = 负面。\n",
                "few_shot_map": {"正面": "1", "中立": "2", "负面": "3"},
                "few_shot": [
                    ("X 支持了这项挽救了数千个工作岗位的法案，赢得了工会和当地家庭的赞扬。", "正面"),
                    ("X 表示委员会将于周二开会讨论即将到来的财政预算。", "中立"),
                    ("X 因在危机期间忽视工薪阶层的需求而受到严厉批评。", "负面"),
                ]
            }
        }
    }

    # -------------------------------------------------------------------------
    # LOGIC
    # -------------------------------------------------------------------------
    
    language = language.lower()
    if language not in ["english", "russian", "chinese"]:
        raise ValueError("Language must be 'english', 'russian', or 'chinese'")

    cfg = db[task][language]

    labels = db[task]["num_labels"] if numerical_labels else cfg["text_labels"] # inferred
    scale = cfg["scale_num"] if numerical_labels else cfg["scale_text"]

    base = cfg["instruction"] + scale

    few_shot_block = ""
    if few_shot:
        formatted = []
        for sent, lab in cfg["few_shot"]:
            # If numerical, we map the text label (e.g., "Public Good") to "1"
            adapted_lab = cfg["few_shot_map"][lab] if numerical_labels else lab
            
            s_prefix = "Sentence: "
            l_prefix = "Label: "
            if language == "russian":
                s_prefix, l_prefix = "Sentence: ", "Label: "
            elif language == "chinese":
                s_prefix, l_prefix = "Sentence: ", "Label: "

            formatted.append(f"{s_prefix}{sent}\n{l_prefix}{adapted_lab}")
        few_shot_block = "\n\n".join(formatted) + "\n\n"

    s_prefix = "Sentence: "
    l_prefix = "Label: "
    if language == "russian":
        s_prefix, l_prefix = "Sentence: ", "Label: "
    elif language == "chinese":
        s_prefix, l_prefix = "Sentence: ", "Label: "

    return (
        base
        + few_shot_block
        + f"{s_prefix}{sentence}\n"
        + l_prefix
    )


def get_politician_category_tokens(task, numerical=False, language="english"):
    """
    Return the category-to-token mapping used to interpret single-token model outputs.

    Parameters
    ----------
    task : str
        Task identifier (e.g., "intent", "misconduct", "leadership", "sentimentP").
    numerical : bool, default=False
        If True, return numeric label tokens (e.g., "1", "2", ...); otherwise return
        textual label tokens (language-specific prefixes).
    language : str, default="english"
        Language identifier ("english", "russian", "chinese") used for textual tokens.

    Returns
    -------
    dict[str, list[str]]
        Mapping from category label to a list of tokens/prefixes associated with that label.
    """
    language = language.lower()

    # 1. Numerical tokens
    numerical_tokens = {
        "intent":     {"1": ["1"], "2": ["2"], "3": ["3"]},
        "misconduct": {"1": ["1"], "2": ["2"], "3": ["3"]},
        "leadership": {"1": ["1"], "2": ["2"], "3": ["3"], "4": ["4"]},
        "sentimentP": {"1": ["1"], "2": ["2"], "3": ["3"]},
    }

    if numerical:
        return numerical_tokens[task]

    # 2. Textual tokens map
    text_tokens = {
        "intent": {
            "english": {
                "Public Good": ["pu"], "Lobbying": ["lo"], "Populism": ["po"]
            },
            "russian": {
                "общественное благо": ["о"], 
                "лоббирование": ["л"], 
                "популизм": ["п"]
            },
            "chinese": {
                "公共利益": ["公"], "游说": ["游"], "民粹主义": ["民"]
            }
        },

        "misconduct": {
            "english": {
                "Minor Lapse": ["mi"], "Conflict of Interest": ["co"], "Requires Investigation": ["re"]
            },
            "russian": {
                "незначительное упущение": ["н"], 
                "конфликт интересов": ["к"], 
                "требует расследования": ["т"]
            },
            "chinese": {
                "轻微失误": ["轻"], "利益冲突": ["利"], "需要调查": ["需"]
            }
        },

        "leadership": {
            "english": {
                "Excellent Leadership": ["ex"], "Good Leadership": ["go"], 
                "Bad Leadership": ["ba"], "Catastrophic Leadership": ["ca"]
            },
            "russian": {
                "Выдающееся лидерство": ["В"],
                "Компетентное лидерство": ["К"],
                "Недостаточное лидерство": ["Н"],
                "Провальное лидерство": ["П"]
            },
            "chinese": {
                "杰出领导力": ["杰"],
                "合格领导力": ["合"],
                "不佳领导力": ["不"],
                "失职领导力": ["失"]
            }

        },
        "sentimentP": {
            "english": {
                "Positive": ["po"], "Neutral": ["neu"], "Negative": ["neg"]
            },
            "russian": {
                "Положительный": ["п"], "Нейтральный": ["н"], "Отрицательный": ["о"]
            },
            "chinese": {
                "正面": ["正"], "中立": ["中"], "负面": ["负"]
            }
        },
    }


    return text_tokens[task][language]