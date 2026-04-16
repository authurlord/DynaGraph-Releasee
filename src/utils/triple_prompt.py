import json
import re

# ==============================================================================
# PROMPT 模板
# 注意：这里我们保留了原始的 JSON 单花括号 {}，不需要双花括号转义。
# 我们使用特殊占位符 <TITLE_PLACEHOLDER> 和 <TEXT_PLACEHOLDER> 来进行替换。
# ==============================================================================
TRIPLE_EXTRACTION_TEMPLATE = """
You are an expert Knowledge Graph Engineer and Linguist specializing in Information Extraction for Multi-hop Question Answering systems.

Your task is to extract high-quality semantic triplets `(Subject, Relation, Object)` from the provided text, given a specific Context Title.

### Guidelines:
1. **Coreference Resolution (CRUCIAL):** The text usually describes the entity mentioned in the "Context Title". Replace pronouns (it, he, she, they, this) with the specific entity names. 
   - *Example:* If Title is "Duilio Coletti" and text says "He directed...", output `["Duilio Coletti", "director", ...]`.
2. **Wikidata-style Relations:** Use standard relations.
   - `directed by` -> `director`
   - `released in` -> `publication date`
   - `born in` -> `place of birth` or `date of birth`
   - `is a` -> `instance of`
3. **Atomic Facts:** Split compound sentences into separate triplets.

### Output Format:
- **STRICTLY JSON ONLY.**
- Do not output markdown code blocks (like ```json).
- Do not output conversational text.
- Format: `{"triplets": [ ["Subject", "Relation", "Object"], ... ]}`

### Examples:

Input Title: I sette dell'Orsa maggiore
Input Text: I sette dell'Orsa maggiore is a 1953 Italian film based on the Raid on Alexandria. It was directed by Duilio Coletti.
Output: {
  "triplets": [
    ["I sette dell'Orsa maggiore", "instance of", "film"],
    ["I sette dell'Orsa maggiore", "publication date", "1953"],
    ["I sette dell'Orsa maggiore", "country of origin", "Italy"],
    ["I sette dell'Orsa maggiore", "director", "Duilio Coletti"]
  ]
}

Input Title: Gonzalo Martínez Ortega
Input Text: Gonzalo Martínez Ortega (27 April 1934 – 2 June 1998) was a Mexican actor and director.
Output: {
  "triplets": [
    ["Gonzalo Martínez Ortega", "date of birth", "27 April 1934"],
    ["Gonzalo Martínez Ortega", "date of death", "2 June 1998"],
    ["Gonzalo Martínez Ortega", "country of citizenship", "Mexico"],
    ["Gonzalo Martínez Ortega", "occupation", "actor"],
    ["Gonzalo Martínez Ortega", "occupation", "director"]
  ]
}

### Current Task:

Input Title: <TITLE_PLACEHOLDER>
Input Text: <TEXT_PLACEHOLDER>
Output:
"""

def generate_triple_extraction_prompt(title: str, text: str) -> str:
    """
    构造用于三元组抽取的 Prompt。
    使用 .replace() 方法避免 JSON 花括号导致的 KeyError。
    """
    # 简单的预处理，去除首尾空白
    clean_title = str(title).strip()
    clean_text = str(text).strip()
    
    # 使用 replace 替换占位符，这是处理包含 JSON 字符串最安全的方法
    prompt = TRIPLE_EXTRACTION_TEMPLATE.replace("<TITLE_PLACEHOLDER>", clean_title)
    prompt = prompt.replace("<TEXT_PLACEHOLDER>", clean_text)
    
    return prompt


def parse_triple_response(llm_output: str) -> list:
    """
    解析 LLM 的输出，提取三元组列表。
    具有鲁棒性：可以处理包含 Markdown 代码块或纯 JSON 的情况。
    """
    try:
        # 1. 清洗字符串：去除 Markdown 代码标记
        cleaned_output = re.sub(r'```json', '', llm_output, flags=re.IGNORECASE)
        cleaned_output = re.sub(r'```', '', cleaned_output)
        cleaned_output = cleaned_output.strip()

        # 2. 尝试寻找 JSON 结构的起始和结束位置
        start_idx = cleaned_output.find('{')
        end_idx = cleaned_output.rfind('}')

        if start_idx != -1 and end_idx != -1:
            json_str = cleaned_output[start_idx : end_idx + 1]
        else:
            json_str = cleaned_output

        # 3. 解析 JSON
        data = json.loads(json_str)

        # 4. 提取 triplets 键
        triplets = []
        if isinstance(data, dict):
            # 优先查找 triplets 键，兼容 knowledge_units 或直接 list
            if "triplets" in data:
                triplets = data["triplets"]
            elif "knowledge_units" in data:
                triplets = data["knowledge_units"]
            else:
                # 尝试获取字典的第一个 value
                values = list(data.values())
                if values and isinstance(values[0], list):
                    triplets = values[0]
        elif isinstance(data, list):
            triplets = data

        # 5. 最终校验格式 (确保是 list of lists 且长度至少为3)
        valid_triplets = []
        if isinstance(triplets, list):
            for t in triplets:
                if isinstance(t, list) and len(t) >= 3:
                    # 转换为字符串，防止由数字引起的类型错误
                    valid_triplets.append([str(item) for item in t[:3]])
        
        return valid_triplets

    except json.JSONDecodeError:
        # 如果解析失败，返回空列表，不中断程序
        return []
    except Exception as e:
        print(f"[Error] Parsing Exception: {e}")
        return []
import json
import re

# ==============================================================================
# PROMPT 模板：Query Expansion 策略
# 核心思想：
# 1. Preserve (保留): 不要修改原问题，防止语义丢失。
# 2. Augment (增强): 在原问题后面追加从 Context 中发现的“桥梁实体”和“关键词”。
# ==============================================================================
QUERY_REWRITE_TEMPLATE = """
You are an expert Query Expansion Agent for a Multi-hop Question Answering system.

### Task
You have an `Original Question` and `Retrieved Contexts` (from the 1st hop).
Your goal is to formulate a **Search Query** for the 2nd hop that solves the missing information.

### Strategy: Expansion instead of Replacement
1. **Analyze:** Identify the "Bridge Entity" in the `Retrieved Contexts` (the person/object/event that connects the question to the answer).
2. **Construct:** The new query must be: `[Original Question] + [Bridge Entity Name] + [Target Keywords]`.
   - **Do NOT** remove the original question.
   - **Do** append specific names found in the text.
   - **Do** append keywords related to the missing information (e.g., "birthplace", "author", "date").

### Examples

**Example 1**
**Original Question:** "This Experts Network sports analysts was inducted into the Pro Football Hall of Fame in 2000 and played in the NFL for how many seasons?"
**Retrieved Contexts:** "...consisting of sports analysts Cris Collinsworth, Boomer Esiason, Howie Long, Cal Ripken Jr..."
**Reasoning:** The question mentions an analyst inducted in 2000. The text lists candidates. 'Howie Long' is a likely candidate (or we should search for all of them).
**Output:** {
  "reasoning": "The text provides specific analyst names. I will append the most likely names to the original question to verify their Hall of Fame status.",
  "expanded_query": "This Experts Network sports analysts was inducted into the Pro Football Hall of Fame in 2000 and played in the NFL for how many seasons? Howie Long Cris Collinsworth Hall of Fame 2000"
}

**Example 2**
**Original Question:** "Who wrote Bring Me Sunshine and where did they write it?"
**Retrieved Contexts:** "Bring Me Sunshine (1994) was ... a tribute to Eric Morecambe..."
**Reasoning:** The text links the song to 'Eric Morecambe'. He is the bridge entity. I need to find the author/writer relative to him.
**Output:** {
  "reasoning": "The context identifies 'Eric Morecambe' as the key entity associated with the song. I will append his name to find the specific writer and location.",
  "expanded_query": "Who wrote Bring Me Sunshine and where did they write it? Eric Morecambe writer author origin"
}

### Current Task

**Original Question:** <QUESTION_PLACEHOLDER>
**Retrieved Contexts:**
<CONTEXT_PLACEHOLDER>

**Output:**
STRICTLY JSON FORMAT.
"""

def generate_rewrite_prompt(question: str, contexts: list) -> str:
    """
    构造用于 Query Expansion 的 Prompt。
    保留原问题，追加关键词。
    """
    formatted_contexts = ""
    for idx, text in enumerate(contexts):
        # 截取前 400 字符，通常足够包含实体信息
        clean_text = str(text).strip()[:400]
        formatted_contexts += f"[Text {idx+1}]: {clean_text}\n"
    
    clean_question = str(question).strip()
    
    # 替换占位符
    prompt = QUERY_REWRITE_TEMPLATE.replace("<QUESTION_PLACEHOLDER>", clean_question)
    prompt = prompt.replace("<CONTEXT_PLACEHOLDER>", formatted_contexts)
    
    return prompt


def parse_rewrite_response(llm_output: str) -> str:
    """
    解析 LLM 输出，返回 expanded_query。
    """
    try:
        # 1. 清洗 Markdown
        cleaned_output = re.sub(r'```json', '', llm_output, flags=re.IGNORECASE)
        cleaned_output = re.sub(r'```', '', cleaned_output)
        cleaned_output = cleaned_output.strip()

        # 2. 定位 JSON
        start_idx = cleaned_output.find('{')
        end_idx = cleaned_output.rfind('}')
        
        if start_idx != -1 and end_idx != -1:
            json_str = cleaned_output[start_idx : end_idx + 1]
            data = json.loads(json_str)
            
            # 优先查找 expanded_query，但也兼容其他常见的 key
            if "expanded_query" in data:
                return data["expanded_query"]
            elif "rewritten_query" in data:
                return data["rewritten_query"]
            elif "query" in data:
                return data["query"]

        # 兜底：如果解析不出 JSON，直接返回清洗后的文本
        return cleaned_output

    except Exception as e:
        print(f"[Error] Rewrite Parsing Failed: {e}")
        # 出错时返回空字符串，由调用方决定是否使用原问题兜底
        return ""