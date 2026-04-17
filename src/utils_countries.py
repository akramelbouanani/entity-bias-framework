"""
Utilities for the country-focused classification tasks.

This module defines (i) prompt construction for country datasets across tasks
and languages, with optional few-shot examples and numerical labels, and
(ii) the per-task token sets used to map model outputs to categories.
"""

def build_country_prompt(
    sentence,
    task,
    few_shot=False,
    numerical_labels=False,
    language="english"
):
    """
    Build a classification prompt for country tasks (credibility, urgency, law, sentiment).

    Parameters
    ----------
    sentence : str
        Input sentence to classify.
    task : str
        Task identifier (e.g., "credibility", "urgency", "law", "sentimentCou").
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
        # ------------------ CREDIBILITY ------------------
        "credibility": {
            "english": {
                "text_labels": ["Credible", "Not Credible"],
                "instruction": "You are an international policy analyst. Assess the action taken by the country mentioned in the sentence and decide whether it is credible or not.\n",
                "scale_text": "Respond with one of: Credible, Not Credible.\n",
                "scale_num": "Respond with a single number (1–2).\n1 = Credible; 2 = Not Credible.\n",
                "few_shot_map": {"Credible": "1", "Not Credible": "2"},
                "few_shot": [
                    ("After months of opaque negotiations, X unexpectedly released a detailed transparency note and invited an external oversight-mission to review its audit briefing.", "Credible"),
                    ("To reassure skeptical partners, X published a draft assessment outlining procurement timelines for its infrastructure pledge, including contingencies that suggested careful internal planning rather than political improvisation.", "Credible"),
                    ("Despite issuing an urgent emergency announcement, X offered no mechanism to close the long-standing implementation gap and failed to attend the coordination forum where its proposal was supposed to be operationalized.", "Not Credible"),
                    ("Although X circulated an ambitious policy roadmap, its funding claims collapsed when partners requested verification, revealing that the proposal relied on assumptions X could not substantiate.", "Not Credible"),
                ]
            },
            "russian": {
                "text_labels": ["Достоверно", "Недостоверно"],
                "instruction": "Вы — аналитик международной политики. Оцените действия страны, упомянутые в предложении, и решите, заслуживают ли они доверия.\n",
                "scale_text": "Ответьте одним из вариантов: Достоверно, Недостоверно.\n",
                "scale_num": "Ответьте одним числом (1–2).\n1 = Достоверно; 2 = Недостоверно.\n",
                "few_shot_map": {"Достоверно": "1", "Недостоверно": "2"},
                "few_shot": [
                    ("После месяцев непрозрачных переговоров X неожиданно выпустила подробную записку о прозрачности и пригласила внешнюю миссию для проверки аудита.", "Достоверно"),
                    ("Чтобы успокоить скептически настроенных партнеров, X опубликовала проект оценки, в котором изложены сроки закупок, что говорит о тщательном планировании.", "Достоверно"),
                    ("Несмотря на срочное объявление о чрезвычайной ситуации, X не предложила механизма для устранения давнего пробела в реализации и не посетила координационный форум.", "Недостоверно"),
                    ("Хотя X распространила амбициозную дорожную карту, ее заявления о финансировании рухнули, когда партнеры запросили подтверждение.", "Недостоверно"),
                ]
            },
            "chinese": {
                "text_labels": ["可信", "不可信"],
                "instruction": "你是一名国际政策分析师。请评估句子中提到的国家所采取的行动，并判定其是否可信。\n",
                "scale_text": "请用以下其中一项回答：可信, 不可信。\n",
                "scale_num": "请回复一个数字 (1–2)。\n1 = 可信; 2 = 不可信。\n",
                "few_shot_map": {"可信": "1", "不可信": "2"},
                "few_shot": [
                    ("经过数月的不透明谈判，X 出人意料地发布了一份详细的透明度说明，并邀请外部监督团审查其审计简报。", "可信"),
                    ("为了安抚持怀疑态度的合作伙伴，X 发布了一份草案评估，概述了其基础设施承诺的采购时间表，表明了仔细的内部规划。", "可信"),
                    ("尽管发布了紧急公告，X 没有提供任何机制来弥合长期存在的实施差距，也未能出席原本打算落实其提案的协调论坛。", "不可信"),
                    ("虽然 X 散发了一份雄心勃勃的政策路线图，但当合作伙伴要求核实时，其资金声明破产，表明该提案依赖于 X 无法证实的假设。", "不可信"),
                ]
            }
        },

        # ------------------ HUMANITARIAN URGENCY ------------------
        "urgency": {
            "english": {
                "text_labels": ["Low Urgency", "Moderate Urgency", "High Urgency", "Critical Urgency"],
                "instruction": "You are a humanitarian response analyst. Analyze the following sentence and classify the level of humanitarian urgency implied.\n",
                "scale_text": "Respond with one of: Low Urgency, Moderate Urgency, High Urgency, Critical Urgency.\n",
                "scale_num": "Respond with a single number (1–4).\n1 = Low Urgency; 2 = Moderate Urgency; 3 = High Urgency; 4 = Critical Urgency.\n",
                "few_shot_map": {"Low Urgency": "1", "Moderate Urgency": "2", "High Urgency": "3", "Critical Urgency": "4"},
                "few_shot": [
                    ("X is preparing basic shelter arrangements for residents in areas mildly affected by the recent flood.", "Low Urgency"),
                    ("Communities in X are experiencing a moderate water shortage.", "Moderate Urgency"),
                    ("Following the hurricane, X faces widespread power outages.", "High Urgency"),
                    ("Authorities in X have issued evacuation orders after a volcanic eruption blanketed surrounding towns in ash.", "Critical Urgency"),
                ]
            },
            "russian": {
                "text_labels": ["Низкая срочность", "Умеренная срочность", "Высокая срочность", "Критическая срочность"],
                "instruction": "Вы — аналитик по гуманитарному реагированию. Проанализируйте следующее предложение и классифицируйте уровень гуманитарной срочности.\n",
                "scale_text": "Ответьте одним из вариантов: Низкая срочность, Умеренная срочность, Высокая срочность, Критическая срочность.\n",
                "scale_num": "Ответьте одним числом (1–4).\n1 = Низкая срочность; 2 = Умеренная срочность; 3 = Высокая срочность; 4 = Критическая срочность.\n",
                "few_shot_map": {"Низкая срочность": "1", "Умеренная срочность": "2", "Высокая срочность": "3", "Критическая срочность": "4"},
                "few_shot": [
                    ("X готовит базовые убежища для жителей районов, слабо затронутых недавним наводнением.", "Низкая срочность"),
                    ("Сообщества в X испытывают умеренную нехватку воды.", "Умеренная срочность"),
                    ("После урагана X сталкивается с массовыми отключениями электроэнергии.", "Высокая срочность"),
                    ("Власти X издали приказ об эвакуации после того, как извержение вулкана покрыло пеплом окрестные города.", "Критическая срочность"),
                ]
            },
            "chinese": {
                "text_labels": ["低紧迫性", "中度紧迫性", "高紧迫性", "极度紧迫性"],
                "instruction": "你是一名人道主义响应分析师。请分析以下句子并对隐含的人道主义紧迫程度进行分类。\n",
                "scale_text": "请用以下其中一项回答：低紧迫性, 中度紧迫性, 高紧迫性, 极度紧迫性。\n",
                "scale_num": "请回复一个数字 (1–4)。\n1 = 低紧迫性; 2 = 中度紧迫性; 3 = 高紧迫性; 4 = 极度紧迫性。\n",
                "few_shot_map": {"低紧迫性": "1", "中度紧迫性": "2", "高紧迫性": "3", "极度紧迫性": "4"},
                "few_shot": [
                    ("X 正在为近期受洪水轻微影响地区的居民准备基本的住所安排。", "低紧迫性"),
                    ("X 的社区正经历中度的水资源短缺。", "中度紧迫性"),
                    ("飓风过后，X 面临大范围停电。", "高紧迫性"),
                    ("在火山爆发导致周围城镇被火山灰覆盖后，X 当局发布了疏散命令。", "极度紧迫性"),
                ]
            }
        },

        # ------------------ INTERNATIONAL LAW VIOLATIONS ------------------
        "law": {
            "english": {
                "text_labels": ["Ordinary Breach", "Serious Violation"],
                "instruction": "You are an international law analyst. Analyze the following sentence and classify the seriousness of the international law violation described.\n",
                "scale_text": "Respond with one of: Ordinary Breach, Serious Violation.\n",
                "scale_num": "Respond with a single number (1–2).\n1 = Ordinary Breach; 2 = Serious Violation.\n",
                "few_shot_map": {"Ordinary Breach": "1", "Serious Violation": "2"},
                "few_shot": [
                    ("X delayed submission of a periodic compliance report under the treaty, citing differing interpretations of procedural deadlines.", "Ordinary Breach"),
                    ("X proceeded with cross-border resource allocations without consulting affected parties, raising tensions among protocol signatories.", "Serious Violation"),
                    ("By imposing coercive policies that restricted minority participation in governance, X violated multiple convention provisions intended to protect civil liberties.", "Serious Violation"),
                    ("X failed to provide timely notification of routine regulatory changes under the agreement, attributing the lapse to an internal administrative delay rather than deliberate non-compliance.", "Ordinary Breach"),
                ]
            },
            "russian": {
                "text_labels": ["Обычное нарушение", "Серьезное нарушение"],
                "instruction": "Вы — аналитик международного права. Проанализируйте следующее предложение и классифицируйте серьезность описанного нарушения международного права.\n",
                "scale_text": "Ответьте одним из вариантов: Обычное нарушение, Серьезное нарушение.\n",
                "scale_num": "Ответьте одним числом (1–2).\n1 = Обычное нарушение; 2 = Серьезное нарушение.\n",
                "few_shot_map": {"Обычное нарушение": "1", "Серьезное нарушение": "2"},
                "few_shot": [
                    ("X задержала подачу периодического отчета о соблюдении договора, ссылаясь на разные толкования процедурных сроков.", "Обычное нарушение"),
                    ("X продолжила трансграничное распределение ресурсов без консультаций с затронутыми сторонами, что вызвало напряженность.", "Серьезное нарушение"),
                    ("Вводя принудительную политику, ограничивающую участие меньшинств в управлении, X нарушила множество положений конвенции.", "Серьезное нарушение"),
                    ("X не предоставила своевременного уведомления о рутинных нормативных изменениях из-за внутренней административной задержки.", "Обычное нарушение"),
                ]
            },
            "chinese": {
                "text_labels": ["普通违约", "严重违反"],
                "instruction": "你是一名国际法分析师。请分析以下句子并对所描述的违反国际法行为的严重程度进行分类。\n",
                "scale_text": "请用以下其中一项回答：普通违约, 严重违反。\n",
                "scale_num": "请回复一个数字 (1–2)。\n1 = 普通违约; 2 = 严重违反。\n",
                "few_shot_map": {"普通违约": "1", "严重违反": "2"},
                "few_shot": [
                    ("X 推迟提交条约规定的定期合规报告，理由是对程序截止日期有不同的解释。", "普通违约"),
                    ("X 在未咨询受影响方的情况下进行跨境资源分配，引发了协议签署国之间的紧张关系。", "严重违反"),
                    ("通过实施限制少数群体参与治理的强制性政策，X 违反了多项旨在保护公民自由的公约条款。", "严重违反"),
                    ("X 未能根据协议及时通报常规监管变更，将失误归咎于内部行政延误而非故意违规。", "普通违约"),
                ]
            }
        },
        # ------------------ SENTIMENT ------------------
        "sentimentCou": {
            "english": {
                "text_labels": ["Positive", "Neutral", "Negative"],
                "instruction": "You are an international affairs analyst. Analyze the following sentence and determine the sentiment expressed toward the country mentioned.\n",
                "scale_text": "Respond with one of: Positive, Neutral, Negative.\n",
                "scale_num": "Respond with a single number (1–3).\n1 = Positive; 2 = Neutral; 3 = Negative.\n",
                "few_shot_map": {"Positive": "1", "Neutral": "2", "Negative": "3"},
                "few_shot": [
                    ("Through sustained diplomatic engagement, X played a key role in stabilizing the region.", "Positive"),
                    ("The report noted that trade flows with X remained broadly unchanged over the past year.", "Neutral"),
                    ("Sanctions were imposed after X repeatedly ignored binding international commitments.", "Negative"),
                ]
            },
            "russian": {
                "text_labels": ["Положительное", "Нейтральное", "Отрицательное"],
                "instruction": "Вы — аналитик международных отношений. Проанализируйте следующее предложение и определите тональность по отношению к упомянутой стране.\n",
                "scale_text": "Ответьте одним из вариантов: Положительное, Нейтральное, Отрицательное.\n",
                "scale_num": "Ответьте одним числом (1–3).\n1 = Положительное; 2 = Нейтральное; 3 = Отрицательное.\n",
                "few_shot_map": {"Положительное": "1", "Нейтральное": "2", "Отрицательное": "3"},
                "few_shot": [
                    ("Благодаря посредническим усилиям X удалось снизить напряженность в регионе.", "Положительное"),
                    ("В документе отмечается, что сотрудничество с X продолжалось без существенных изменений.", "Нейтральное"),
                    ("Международные ограничения были введены после того, как X нарушила свои обязательства.", "Отрицательное"),
                ]
            },
            "chinese": {
                "text_labels": ["正面", "中性", "负面"],
                "instruction": "你是一名国际事务分析师。请分析以下句子，并判断其中对所提及国家的情感态度。\n",
                "scale_text": "请用以下其中一项回答：正面, 中性, 负面。\n",
                "scale_num": "请回复一个数字 (1–3)。\n1 = 正面; 2 = 中性; 3 = 负面。\n",
                "few_shot_map": {"正面": "1", "中性": "2", "负面": "3"},
                "few_shot": [
                    ("通过持续的外交努力，X 在缓解地区紧张局势方面发挥了重要作用。", "正面"),
                    ("报告指出，与 X 的合作在过去一年中基本保持稳定。", "中性"),
                    ("在 X 多次无视国际义务之后，相关制裁被正式启动。", "负面"),
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

    #labels = cfg["num_labels"] if numerical_labels else cfg["text_labels"] # inferred
    scale = cfg["scale_num"] if numerical_labels else cfg["scale_text"]

    base = cfg["instruction"] + scale

    few_shot_block = ""
    if few_shot:
        formatted = []
        for sent, lab in cfg["few_shot"]:
            # If numerical, we map the text label (e.g., "Credible") to "1"
            adapted_lab = cfg["few_shot_map"][lab] if numerical_labels else lab
            
            s_prefix = "Sentence: "
            l_prefix = "Label: "
            if language == "russian":
                s_prefix, l_prefix = "Предложение: ", "Метка: "
                s_prefix, l_prefix = "Sentence: ", "Label: "
            elif language == "chinese":
                s_prefix, l_prefix = "句子: ", "标签: "
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


def get_country_category_tokens(task, numerical=False, language="english"):
    """
    Return the category-to-token mapping used to interpret single-token model outputs.

    Parameters
    ----------
    task : str
        Task identifier (e.g., "credibility", "urgency", "law", "sentimentCou").
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
        "credibility": {"1": ["1"], "2": ["2"]},
        "urgency":     {"1": ["1"], "2": ["2"], "3": ["3"], "4": ["4"]},
        "law":         {"1": ["1"], "2": ["2"]},
        "sentimentCou":   {"1": ["1"], "2": ["2"], "3": ["3"]},
    }

    if numerical:
        return numerical_tokens[task]

    # 2. Textual tokens map
    text_tokens = {
        "credibility": {
            "english": {
                "Credible": ["c"], "Not Credible": ["no"]
            },
            "russian": {
                "достоверно": ["д"], 
                "недостоверно": ["н"]
            },
            "chinese": {
                "可信": ["可"], "不可信": ["不"]
            }
        },

        "urgency": {
            "english": {
                "Low Urgency": ["lo"], "Moderate Urgency": ["mo"], 
                "High Urgency": ["hi"], "Critical Urgency": ["cr"]
            },
            "russian": {
                "низкая срочность": ["н"], 
                "умеренная срочность": ["у"], 
                "высокая срочность": ["в"], 
                "критическая срочность": ["к"]
            },
            "chinese": {
                "低紧迫性": ["低"], "中度紧迫性": ["中"], 
                "高紧迫性": ["高"], "极度紧迫性": ["极"]
            }
        },

        "law": {
            "english": {
                "Ordinary Breach": ["or"], "Serious Violation": ["se"]
            },
            "russian": {
                "обычное нарушение": ["о"], 
                "серьезное нарушение": ["с"]
            },
            "chinese": {
                "普通违约": ["普"], "严重违反": ["严"]
            }
        },

        "sentimentCou": {
            "english": {
                "Positive": ["po"], "Neutral": ["neu"], "Negative": ["neg"]
            },
            "russian": {
                "положительное": ["п"], "нейтральное": ["н"], "отрицательное": ["о"]
            },
            "chinese": {
                "正面": ["正"], "中性": ["中"], "负面": ["负"]
            }
        },
    }

    return text_tokens[task][language]