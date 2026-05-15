from langchain_core.prompts import ChatPromptTemplate


ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "你是一个查询路由器。判断用户问题应当走【内部知识库 vectorstore】还是【实时网络搜索 web_search】。\n"
                "- 若问题涉及内部文档、公司制度、产品说明、历史项目、专业领域材料等可能已被索引的内容，输出 vectorstore。\n"
                "- 若问题涉及最新时事、实时数据、当前事件、新近发布的产品/版本等，输出 web_search。\n"
                "只输出一个 JSON：{{\"route\": \"vectorstore\"}} 或 {{\"route\": \"web_search\"}}，不要任何解释。"
            ),
        ),
        ("human", "{question}"),
    ]
)


QUERY_REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "你是一名检索查询改写专家。将用户问题改写为更适合向量检索的版本：\n"
                "- 补全省略的主语/上下文；展开缩写；中英文术语并列；\n"
                "- 保持事实意图，不要扩展为多条问题；\n"
                "- 仅输出改写后的问题，不要前缀、不要解释。"
            ),
        ),
        ("human", "原始问题：{question}"),
    ]
)


QUERY_TRANSFORM_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "之前的检索没找到足够相关的内容。请基于原问题给出一个语义等价但表述不同的检索查询，"
                "可以尝试更具体的术语、不同关键词或上位概念。仅输出新查询本身。"
            ),
        ),
        (
            "human",
            "原问题：{question}\n上一次的查询：{previous}\n",
        ),
    ]
)


DOC_GRADER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "你是文档相关性评估器。判断给定文档是否包含可用于回答用户问题的信息。\n"
                "宽松判断：只要语义相关、即使不能完整回答，也算相关。\n"
                "只输出 JSON：{{\"relevant\": \"yes\"}} 或 {{\"relevant\": \"no\"}}。"
            ),
        ),
        (
            "human",
            "问题：{question}\n\n文档片段：\n{document}",
        ),
    ]
)


GENERATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "你是一名严谨的企业知识助手。仅基于提供的【上下文】回答用户问题。\n"
                "规则：\n"
                "1. 答案必须有上下文依据；上下文未涵盖的部分明确说\"知识库中未提及\"。\n"
                "2. 用中文回答，结构清晰，可以用要点列表。\n"
                "3. 末尾用 [来源 1] [来源 2] 这样的角标标注引用，对应上下文中的序号。"
            ),
        ),
        (
            "human",
            "【上下文】\n{context}\n\n【问题】\n{question}",
        ),
    ]
)


HALLUCINATION_GRADER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "你是事实一致性评估器。判断【答案】中的事实主张是否都能由【上下文】支持。\n"
                "- 若答案中只要存在一条事实没有上下文依据，判 no。\n"
                "- \"知识库中未提及\"这种诚实声明视为有依据。\n"
                "只输出 JSON：{{\"grounded\": \"yes\"}} 或 {{\"grounded\": \"no\"}}。"
            ),
        ),
        (
            "human",
            "上下文：\n{context}\n\n答案：\n{generation}",
        ),
    ]
)


ANSWER_GRADER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "你是答案有用性评估器。判断【答案】是否实质性地解决了【问题】。\n"
                "只输出 JSON：{{\"useful\": \"yes\"}} 或 {{\"useful\": \"no\"}}。"
            ),
        ),
        (
            "human",
            "问题：{question}\n\n答案：\n{generation}",
        ),
    ]
)
