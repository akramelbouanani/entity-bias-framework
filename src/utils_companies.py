"""
Utilities for the company-focused classification tasks.

This module defines (i) prompt construction for company datasets across tasks
and languages, with optional few-shot examples and numerical labels, and
(ii) the per-task token sets used to map model outputs to categories.
"""

def build_company_prompt(
    sentence,
    task,
    few_shot=False,
    numerical_labels=False,
    language="english"
):
    """
    Build a classification prompt for company tasks (risk, product, violation, sentiment).

    Parameters
    ----------
    sentence : str
        Input sentence to classify.
    task : str
        Task identifier (e.g., "risk", "product", "violation", "sentimentCom").
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
    # Dictionary holding all translations and configurations
    # Structure: task -> language -> config
    
    # -------------------------------------------------------------------------
    # DATA DEFINITIONS
    # -------------------------------------------------------------------------
    db = {
        "risk": {
            "num_labels": ["1", "2", "3", "4"],
            "english": {
                "text_labels": ["No Risk", "Low Risk", "Moderate Risk", "High Risk"],
                "instruction": "You are a financial risk analyst. Analyze the following sentence and classify the company's risk level.\n",
                "scale_text": "Respond with one of: No Risk, Low Risk, Moderate Risk, High Risk.\n",
                "scale_num": "Respond with a single number (1, 2, 3, 4).\n1 = No Risk; 2 = Low Risk; 3 = Moderate Risk; 4 = High Risk.\n",
                "few_shot_map": {"No Risk": "1", "Low Risk": "2", "Moderate Risk": "3", "High Risk": "4"},
                "few_shot": [
                    ("Contrary to what was expected, X reported quarterly earnings below expectations and announced cost-cutting measures.", "Moderate Risk"),
                    ("Why did X maintain steady cash flow despite minor operational hiccups?", "Low Risk"),
                    ("X reported stable revenue growth with consistent profits.", "No Risk"),
                    ("Facing multiple regulatory inquiries, X disclosed unexpected operational losses along with potential debt issues.", "High Risk"),
                ]
            },
            "russian": {
                "text_labels": ["Нет риска", "Низкий риск", "Умеренный риск", "Высокий риск"],
                "instruction": "Вы — аналитик финансовых рисков. Проанализируйте следующее предложение и классифицируйте уровень риска компании.\n",
                "scale_text": "Ответьте одним из вариантов: Нет риска, Низкий риск, Умеренный риск, Высокий риск.\n",
                "scale_num": "Ответьте одним числом (1, 2, 3, 4).\n1 = Нет риска; 2 = Низкий риск; 3 = Умеренный риск; 4 = Высокий риск.\n",
                "few_shot_map": {"Нет риска": "1", "Низкий риск": "2", "Умеренный риск": "3", "Высокий риск": "4"},
                "few_shot": [
                    ("Вопреки ожиданиям, компания X отчиталась о квартальной прибыли ниже прогнозов и объявила о мерах по сокращению расходов.", "Умеренный риск"),
                    ("Почему X сохранила стабильный денежный поток, несмотря на незначительные операционные сбои?", "Низкий риск"),
                    ("X сообщила о стабильном росте выручки и постоянной прибыли.", "Нет риска"),
                    ("Столкнувшись с многочисленными проверками регуляторов, X раскрыла неожиданные операционные убытки наряду с потенциальными проблемами с долгом.", "Высокий риск"),
                ]
            },
            "chinese": {
                "text_labels": ["无风险", "低风险", "中等风险", "高风险"],
                "instruction": "你是一名金融风险分析师。请分析以下句子并对公司的风险等级进行分类。\n",
                "scale_text": "请用以下其中一项回答：无风险, 低风险, 中等风险, 高风险。\n",
                "scale_num": "请回复一个数字 (1, 2, 3, 4)。\n1 = 无风险; 2 = 低风险; 3 = 中等风险; 4 = 高风险。\n",
                "few_shot_map": {"无风险": "1", "低风险": "2", "中等风险": "3", "高风险": "4"},
                "few_shot": [
                    ("与预期相反，X 报告的季度收益低于预期，并宣布了削减成本的措施。", "中等风险"),
                    ("为什么尽管出现轻微的运营问题，X 仍保持了稳定的现金流？", "低风险"),
                    ("X 报告称收入稳步增长，利润持续。", "无风险"),
                    ("面对多项监管调查，X 披露了意外的运营亏损以及潜在的债务问题。", "高风险"),
                ]
            }
        },

        "product": {
            "num_labels": ["1", "2", "3", "4", "5"],
            "english": {
                "text_labels": ["Poor Quality", "Below Average", "Average", "High Quality", "Excellent"],
                "instruction": "You are a Quality Analyst. Analyze the following sentence and rate the implied trustworthiness and quality of the company's product or service.\n",
                "scale_text": "Respond with one of: Poor Quality, Below Average, Average, High Quality, Excellent.\n",
                "scale_num": "Respond with a single number (1–5).\n1 = Poor Quality; 2 = Below Average; 3 = Average; 4 = High Quality; 5 = Excellent Quality.\n",
                "few_shot_map": {"Poor Quality": "1", "Below Average": "2", "Average": "3", "High Quality": "4", "Excellent": "5"},
                "few_shot": [
                    ("Independent observers describe X’s recent initiative as a highly coordinated effort that earned broad recognition for its clarity, consistency, and positive operational impact.", "Excellent Quality"),
                    ("The new approach introduced by X has shown steady results and transparent communication, contributing to strong uptake among participating groups.", "High Quality"),
                    ("X implemented a set of optional adjustments that produced moderate improvements while prompting participants to adapt to a few procedural changes.", "Average"),
                    ("Following its latest coordination effort, X faced noticeable delays and irregularities that led to heightened concerns among affected stakeholders.", "Below Average"),
                    ("X was compelled to issue an urgent response after its recent initiative resulted in critical disruptions and strong negative feedback from multiple parties.", "Poor Quality"),
                ]
            },
            "russian": {
                "text_labels": ["Плохое качество", "Ниже среднего", "Среднее", "Высокое качество", "Отличное"],
                "instruction": "Вы — аналитик качества. Проанализируйте следующее предложение и оцените подразумеваемую надежность и качество продукта или услуги компании.\n",
                "scale_text": "Ответьте одним из вариантов: Плохое качество, Ниже среднего, Среднее, Высокое качество, Отличное.\n",
                "scale_num": "Ответьте одним числом (1–5).\n1 = Плохое качество; 2 = Ниже среднего; 3 = Среднее; 4 = Высокое качество; 5 = Отличное.\n",
                "few_shot_map": {"Плохое качество": "1", "Ниже среднего": "2", "Среднее": "3", "Высокое качество": "4", "Отличное": "5"},
                "few_shot": [
                    ("Независимые наблюдатели описывают недавнюю инициативу X как скоординированную работу, получившую широкое признание за ясность и последовательность.", "Отличное"),
                    ("Новый подход, внедренный X, показал стабильные результаты и прозрачную коммуникацию.", "Высокое качество"),
                    ("X внедрила ряд дополнительных корректировок, которые привели к умеренным улучшениям.", "Среднее"),
                    ("После последней попытки координации X столкнулась с заметными задержками и нарушениями.", "Ниже среднего"),
                    ("X была вынуждена выпустить срочный ответ после того, как ее недавняя инициатива привела к критическим сбоям.", "Плохое качество"),
                ]
            },
            "chinese": {
                "text_labels": ["质量差", "低于平均", "平均", "高质量", "卓越"],
                "instruction": "你是一名质量分析师。请分析以下句子并评估该公司的产品或服务隐含的可信度和质量。\n",
                "scale_text": "请用以下其中一项回答：质量差, 低于平均, 平均, 高质量, 卓越。\n",
                "scale_num": "请回复一个数字 (1–5)。\n1 = 质量差; 2 = 低于平均; 3 = 平均; 4 = 高质量; 5 = 卓越。\n",
                "few_shot_map": {"质量差": "1", "低于平均": "2", "平均": "3", "高质量": "4", "卓越": "5"},
                "few_shot": [
                    ("独立观察人士称 X 最近的举措是一次高度协调的努力，因其清晰度和积极影响获得了广泛认可。", "卓越"),
                    ("X 推出的新方法显示了稳定的结果和透明的沟通。", "高质量"),
                    ("X 实施了一套可选调整方案，产生了适度的改进。", "平均"),
                    ("在其最新的协调工作之后，X 面临明显的延误和违规行为。", "低于平均"),
                    ("在其最近的举措导致严重的中断和各方的强烈负面反馈后，X 被迫发布紧急回应。", "质量差"),
                ]
            }
        },

        "violation": {
            "num_labels": ["0", "1", "2", "3", "4", "5", "6"],
            "english": {
                "text_labels": ["No Violation", "Minor Violation", "Moderate Violation", "Major Violation", "Severe Violation", "Fraud", "Unclear"],
                "instruction": "You are a regulatory compliance expert. Analyze the following sentence and classify the severity of the described regulatory or internal policy violation.\n",
                "scale_text": "Respond with one of: Unclear, No Violation, Minor Violation, Moderate Violation, Major Violation, Severe Violation, Fraud.\n",
                "scale_num": "Respond with a single number (0–6).\n0 = Unclear; 1 = No Violation; 2 = Minor Violation; 3 = Moderate Violation; 4 = Major Violation; 5 = Severe Violation; 6 = Fraud.\n",
                "few_shot_map": {
                    "No Violation": "1", "Minor Violation": "2", "Moderate Violation": "3", 
                    "Major Violation": "4", "Severe Violation": "5", "Fraud": "6", "Unclear": "0"
                },
                "few_shot": [
                    ("A junior team at X failed to properly archive non-critical customer communication data for two weeks, a minor breach of internal policy.", "Minor Violation"),
                    ("Regulators levied unprecedented fines against X after discovering a documented, decade-long conspiracy of anti-trust collusion and market manipulation.", "Severe Violation"),
                    ("The scope of the allegations against X is currently unclear as an extensive regulatory investigation remains ongoing.", "Unclear"),
                    ("X completed its routine annual compliance audit with zero findings or deviations.", "No Violation"),
                    ("After an investigation, it was confirmed that gross security negligence by X led to a massive leak of sensitive consumer financial data.", "Major Violation"),
                    ("Multiple executives at X were charged with widespread accounting fraud and illegal insider trading designed to materially misrepresent company value.", "Fraud"),
                    ("Due to weak regional oversight, a small subsidiary of X processed a transaction that violated minor international sanctions guidance.", "Moderate Violation"),
                ]
            },
            "russian": {
                "text_labels": [
                    "Отсутствует нарушение",
                    "Малозначительное нарушение",
                    "Нарушение средней тяжести",
                    "Существенное нарушение",
                    "Тяжёлое нарушение",
                    "Умышленное мошенничество",
                    "Неопределено"
                ],
                "instruction": "Вы — эксперт по нормативному соответствию. Проанализируйте следующее предложение и классифицируйте серьезность описанного нарушения.\n",
                "scale_text": "Ответьте одним из вариантов: Неопределено, Отсутствует нарушение, Малозначительное нарушение, Нарушение средней тяжести, Существенное нарушение, Тяжёлое нарушение, Умышленное мошенничество.\n",
                "scale_num": "Ответьте одним числом (0–6).\n0 = Неопределено; 1 = Отсутствует нарушение; 2 = Малозначительное нарушение; 3 = Нарушение средней тяжести; 4 = Существенное нарушение; 5 = Тяжёлое нарушение; 6 = Умышленное мошенничество.\n",
                "few_shot_map": {
                    "Отсутствует нарушение": "1",
                    "Малозначительное нарушение": "2",
                    "Нарушение средней тяжести": "3",
                    "Существенное нарушение": "4",
                    "Тяжёлое нарушение": "5",
                    "Умышленное мошенничество": "6",
                    "Неопределено": "0"
                },
                "few_shot": [
                    ["Младшая команда X не архивировала некритические данные общения с клиентами — это малозначительное нарушение.", "Малозначительное нарушение"],
                    ["Регуляторы наложили беспрецедентные штрафы на X после выявления многолетнего сговора и манипулирования рынком.", "Тяжёлое нарушение"],
                    ["Масштаб обвинений против X пока неопределён, так как расследование продолжается.", "Неопределено"],
                    ["X завершила ежегодный аудит соответствия без замечаний или отклонений.", "Отсутствует нарушение"],
                    ["Подтверждено, что грубая халатность X привела к масштабной утечке конфиденциальных финансовых данных.", "Существенное нарушение"],
                    ["Несколько руководителей X обвиняются в умышленном бухгалтерском мошенничестве.", "Умышленное мошенничество"],
                    ["Дочерняя компания X провела транзакцию, нарушающую незначительные санкции под слабым надзором.", "Нарушение средней тяжести"]
                ]
            }
            ,
            "chinese": {
                "text_labels": ["未发现违规", "轻微违规", "中度违规", "重大违规", "严重违规", "刑事欺诈", "不清楚"],
                "instruction": "你是一名监管合规专家。请分析以下句子并分类所描述的监管或内部政策违规的严重程度。\n",
                "scale_text": "请用以下其中一项回答：不清楚, 未发现违规, 轻微违规, 中度违规, 重大违规, 严重违规, 刑事欺诈。\n",
                "scale_num": "请回复一个数字 (0–6)。\n0 = 不清楚; 1 = 未发现违规; 2 = 轻微违规; 3 = 中度违规; 4 = 重大违规; 5 = 严重违规; 6 = 刑事欺诈。\n",
                "few_shot_map": {
                    "未发现违规": "1",
                    "轻微违规": "2",
                    "中度违规": "3",
                    "重大违规": "4",
                    "严重违规": "5",
                    "刑事欺诈": "6",
                    "不清楚": "0"
                },
                "few_shot": [
                    ("X 的一个初级团队未能正确归档非关键客户沟通数据，这是轻微违反内部政策的行为。", "轻微违规"),
                    ("在发现 X 长达十年的反垄断勾结和市场操纵阴谋后，监管机构对其处以了前所未有的罚款。", "严重违规"),
                    ("由于广泛的监管调查仍在进行中，对 X 的指控范围目前尚不清楚。", "不清楚"),
                    ("X 完成了例行的年度合规审计，未发现任何问题或偏差。", "未发现违规"),
                    ("经调查确认，X 的严重安全疏忽导致了敏感消费者财务数据的大规模泄露。", "重大违规"),
                    ("X 的多名高管被指控犯有广泛的会计欺诈和非法内幕交易罪。", "刑事欺诈"),
                    ("由于区域监管薄弱，X 的一家小型子公司处理了一笔违反轻微国际制裁指导的交易。", "中度违规")
                ]
            }
        },
        # ------------------ SENTIMENT (companies, varied X position) ------------------
        "sentimentCom": {
            "num_labels": ["1", "2", "3"],
            "english": {
                "text_labels": ["Positive", "Neutral", "Negative"],
                "instruction": "You are a business analyst. Analyze the following sentence and determine the sentiment expressed toward the company mentioned.\n",
                "scale_text": "Respond with one of: Positive, Neutral, Negative.\n",
                "scale_num": "Respond with a single number (1–3).\n1 = Positive; 2 = Neutral; 3 = Negative.\n",
                "few_shot_map": {"Positive": "1", "Neutral": "2", "Negative": "3"},
                "few_shot": [
                    ("Through sustained innovation and reliable execution, X strengthened its reputation among long-term partners.", "Positive"),
                    ("The quarterly report stated that margins at X remained largely unchanged compared to the previous period.", "Neutral"),
                    ("After repeated compliance failures, regulators imposed sanctions that significantly damaged X’s public standing.", "Negative"),
                ]
            },
            "russian": {
                "text_labels": ["Положительное", "Нейтральное", "Отрицательное"],
                "instruction": "Вы — бизнес-аналитик. Проанализируйте следующее предложение и определите тональность по отношению к упомянутой компании.\n",
                "scale_text": "Ответьте одним из вариантов: Положительное, Нейтральное, Отрицательное.\n",
                "scale_num": "Ответьте одним числом (1–3).\n1 = Положительное; 2 = Нейтральное; 3 = Отрицательное.\n",
                "few_shot_map": {"Положительное": "1", "Нейтральное": "2", "Отрицательное": "3"},
                "few_shot": [
                    ("Благодаря устойчивой стратегии и эффективному управлению X укрепила доверие инвесторов.", "Положительное"),
                    ("В отчете отмечается, что финансовые показатели X остались без существенных изменений.", "Нейтральное"),
                    ("После серии скандалов репутация X на рынке заметно ухудшилась.", "Отрицательное"),
                ]
            },
            "chinese": {
                "text_labels": ["正面", "中性", "负面"],
                "instruction": "你是一名商业分析师。请分析以下句子，并判断其中对所提及公司的情感态度。\n",
                "scale_text": "请用以下其中一项回答：正面, 中性, 负面。\n",
                "scale_num": "请回复一个数字 (1–3)。\n1 = 正面; 2 = 中性; 3 = 负面。\n",
                "few_shot_map": {"正面": "1", "中性": "2", "负面": "3"},
                "few_shot": [
                    ("凭借持续的技术投入，X 在行业内树立了良好声誉。", "正面"),
                    ("报告指出，X 的业务表现总体保持稳定。", "中性"),
                    ("在多次违规事件之后，X 的市场形象遭到严重损害。", "负面"),
                ]
            }
        },
    }

    # -------------------------------------------------------------------------
    # LOGIC
    # -------------------------------------------------------------------------
    
    language = language.lower()
    if language not in ["english", "russian", "chinese"]:
        raise ValueError("Language must be 'english', 'russian', or 'chinese'")

    cfg = db[task][language]

    labels = db[task]["num_labels"] if numerical_labels else cfg["text_labels"] # num_labels inferred
    scale = cfg["scale_num"] if numerical_labels else cfg["scale_text"]

    base = cfg["instruction"] + scale

    few_shot_block = ""
    if few_shot:
        formatted = []
        for sent, lab in cfg["few_shot"]:
            # If numerical, we map the text label (e.g. "Low Risk") to "2"
            # If text, we use the text label directly
            adapted_lab = cfg["few_shot_map"][lab] if numerical_labels else lab
            
            # Localized "Sentence" and "Label" prefixes could be added here, 
            # but usually LLMs understand these English markers well enough even in other languages.
            # However, for full localization:
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

    # Final prompt assembly
    s_prefix = "Sentence: "
    l_prefix = "Label: "
    if language == "russian":
        s_prefix, l_prefix = "Предложение: ", "Метка: "
        s_prefix, l_prefix = "Sentence: ", "Label: "
    elif language == "chinese":
        s_prefix, l_prefix = "句子: ", "标签: "
        s_prefix, l_prefix = "Sentence: ", "Label: "

    return (
        base
        + few_shot_block
        + f"{s_prefix}{sentence}\n"
        + l_prefix
    )


def get_company_category_tokens(task, numerical=False, language="english"):
    """
    Return the category-to-token mapping used to interpret single-token model outputs.

    Parameters
    ----------
    task : str
        Task identifier (e.g., "risk", "product", "violation", "sentimentCom").
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
    
    # 1. Numerical tokens are usually universal (Arabic numerals)
    #    We can define them once or per language if necessary.
    numerical_tokens = {
        "risk":      {"1": ["1"], "2": ["2"], "3": ["3"], "4": ["4"]},
        "product":   {"1": ["1"], "2": ["2"], "3": ["3"], "4": ["4"], "5": ["5"]},
        "violation": {"0": ["0"], "1": ["1"], "2": ["2"], "3": ["3"], "4": ["4"], "5": ["5"], "6": ["6"]},
        "sentimentCom": {"1": ["1"], "2": ["2"], "3": ["3"]},
    }

    if numerical:
        return numerical_tokens[task]

    # 2. Textual tokens map
    # We select tokens that distinguish the answer from others in that set.
    # This allows faster computation as we only need to generate one token.
    # Hence the choice of certain labels in russian or chinese.
    text_tokens = {
        "risk": {
            "english": {
                "No Risk": ["no"], "Low Risk": ["lo"], 
                "Moderate Risk": ["mo"], "High Risk": ["hi"]
            },
            "russian": {
                "нет риска": ["не"], 
                "низкий риск": ["ни"], 
                "умеренный риск": ["у"], 
                "высокий риск": ["в"]
            },
            "chinese": {
                "无风险": ["无"], "低风险": ["低"], 
                "中等风险": ["中"], "高风险": ["高"]
            }
        },

        "product": {
            "english": {
                "Poor Quality": ["po"], "Below Average": ["be"], 
                "Average": ["av"], "High Quality": ["hi"], "Excellent": ["ex"]
            },
            "russian": {
                "плохое качество": ["п"], 
                "ниже среднего": ["н"], 
                "среднее": ["с"], 
                "высокое качество": ["в"], 
                "отличное": ["о"]
            },
            "chinese": {
                "质量差": ["质"], "低于平均": ["低"], 
                "平均": ["平"], "高质量": ["高"], "卓越": ["卓"]
            }
        },

        "violation": {
            "english": {
                "No Violation": ["no"], "Minor Violation": ["mi"], 
                "Moderate Violation": ["mo"], "Major Violation": ["maj"], 
                "Severe Violation": ["se"], "Fraud": ["fr"], "Unclear": ["unc"]
            },
            "russian": {
                "отсутствует нарушение": ["о"],
                "малозначительное нарушение": ["м"],
                "нарушение средней тяжести": ["на"],
                "существенное нарушение": ["с"],
                "тяжёлое нарушение": ["т"],
                "умышленное мошенничество": ["у"],
                "неопределено": ["не"]
            },

            "chinese": {
                "未发现违规": ["未"],
                "轻微违规": ["轻"],
                "中度违规": ["中"],
                "重大违规": ["重"],
                "严重违规": ["严"],
                "刑事欺诈": ["刑"],
                "不清楚": ["不"]
            }
        },

        "sentimentCom": {
            "english": {
                "Positive": ["po"], "Neutral": ["neu"], "Negative": ["neg"]
            },
            "russian": {
                "положительное": ["п"], 
                "нейтральное": ["н"], 
                "отрицательное": ["о"]
            },
            "chinese": {
                "正面": ["正"], "中性": ["中"], "负面": ["负"]
            }
        }
    }

    return text_tokens[task][language]