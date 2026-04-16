import json
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import requests
import pandas as pd
from tqdm import tqdm
import faiss
import numpy as np
import torch
import pandas as pd

import json
def faiss_cosine_search_cpu(A, B, k=10):
    """
    使用 Faiss CPU 版本计算 A 和 B 的 Cosine Similarity Top-K。
    
    参数:
        A: numpy array 或 Tensor, shape (m, 1024)
        B: numpy array 或 Tensor, shape (n, 1024)
        k: int, 返回的前 k 个结果
        
    返回:
        topk_indices: numpy array, shape (m, k)
        topk_scores:  numpy array, shape (m, k)
    """
    
    # 1. 格式转换：Faiss 严格要求输入为 float32 的 numpy 数组
    # 如果输入是 Tensor，先转为 numpy
    if isinstance(A, torch.Tensor):
        A = A.detach().cpu().numpy()
    if isinstance(B, torch.Tensor):
        B = B.detach().cpu().numpy()
        
    # 确保是 float32 (Faiss 默认只支持 float32)
    A = np.ascontiguousarray(A.astype('float32'))
    B = np.ascontiguousarray(B.astype('float32'))
    
    d = A.shape[1] # 维度 1024

    # 2. L2 归一化 (In-place 操作)
    # 这一步至关重要，只有归一化后，内积才等于余弦相似度
    faiss.normalize_L2(A)
    faiss.normalize_L2(B)

    # 3. 构建索引
    # IndexFlatIP: 暴力精确内积搜索 (Exact Inner Product)
    index = faiss.IndexFlatIP(d)
    
    # 4. 将 B 添加到索引中 (构建底库)
    index.add(B)
    
    # 5. 搜索 (A 作为查询)
    # D: Distances (分数), I: Indices (索引)
    D, I = index.search(A, k)
    
    return I, D


def row_to_prompt_cot_fixed_demos(row,current_contexts,question_col='question', context_col=None):
    """
    修改版函数：
    1. 使用优化的 Chain-of-Thought (CoT) Prompt 模板。
    2. 移除随机采样，使用硬编码的 3 个固定 Demonstrations。
    3. Demonstrations 的格式已调整，包含推理步骤 (Step-by-Step Reasoning)，以引导模型输出逻辑链。
    4. 对输入 row 不使用 supporting_facts 进行过滤，保留 context 中的全部片段。
    """

    # --- 内部 Helper 函数保持不变，用于解析 Context ---
    def _context_map_from_val(context_val):
        cmap = {}
        if isinstance(context_val, list):
            for item in context_val:
                if isinstance(item, dict):
                    title = str(item.get('title') or item.get('Title') or item.get('name') or '')
                    content = item.get('content') or item.get('text') or item.get('Content') or item.get('texts') or ''
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    title = str(item[0])
                    content = item[1]
                else:
                    title = ''
                    content = item
                if isinstance(content, list):
                    for c in content:
                        cmap.setdefault(title, []).append(str(c))
                else:
                    cmap.setdefault(title, []).append(str(content))
        elif isinstance(context_val, dict):
            for k, v in context_val.items():
                if isinstance(v, list):
                    for vv in v:
                        cmap.setdefault(str(k), []).append(str(vv))
                else:
                    cmap.setdefault(str(k), []).append(str(v))
        else:
            cmap.setdefault('', []).append(str(context_val))
        return cmap

    def _contexts_all_from_row(r):
        cc = context_col
        if cc is None:
            # 自动检测 context 列
            for c in r.index:
                v = r[c]
                if isinstance(v, list) and len(v) > 0 and (isinstance(v[0], (list, tuple, dict)) \
                   or (isinstance(v[0], str) and ('Title' in str(v[0]) or ':' in str(v[0])) == False)):
                    cc = c; break
            if cc is None:
                for c in r.index:
                    if isinstance(r[c], (list, dict, str)):
                        cc = c; break
        context_val_local = r[cc] if cc in r.index else []
        cmap_local = _context_map_from_val(context_val_local)

        ctxs = []
        for t, clist in cmap_local.items():
            cleaned = [str(x).strip() for x in clist if x is not None and str(x).strip() != ""]
            joined = " ".join(cleaned) if cleaned else ""
            ctxs.append((str(t), joined))
        return ctxs

    # --- 1. 获取 Question ---
    if question_col in row.index:
        question = str(row[question_col])
    else:
        question = ''
        for c in row.index:
            if 'question' in str(c).lower():
                question = str(row[c]); break

    # --- 2. 硬编码的 Demonstrations (已格式化为 CoT 风格) ---
    # 注意：这里给 demo 加上了简短的 Step-by-Step Reasoning，以便让模型模仿这种输出格式
    fixed_demos_text = """
--- Demonstration 1 ---
Example Question: Are Quaqtaq Airport and Fort Nelson/Gordon Field Airport located in the same country?
Given Context: Title: Quaqtaq Airport
Content: "Quaqtaq Airport,, is located near Quaqtaq, Quebec, Canada."
Given Context: Title: Fort Nelson/Gordon Field Airport
Content: "Fort Nelson/ Gordon Field Airport,, is located near to Fort Nelson, British Columbia, Canada."

Step-by-Step Reasoning:
1. Quaqtaq Airport is located in Quebec, Canada.
2. Fort Nelson/Gordon Field Airport is located in British Columbia, Canada.
3. Both locations are in Canada.
ANSWER: yes

--- Demonstration 2 ---
Example Question: Who was born first out of Erick Chipeta and Praveen Agrawal?
Given Context: Title: Erick Chipeta
Content: "Erick Chipeta( born 28 June 1990) is a Zimbabwean professional footballer who plays as a defender and midfielder for the Zimbabwe national team."
Given Context: Title: Praveen Agrawal
Content: "Praveen Agrawal( born 20 November 1969) is an Indian politician and a member of the Bharatiya Janata Party."

Step-by-Step Reasoning:
1. Erick Chipeta was born on 28 June 1990.
2. Praveen Agrawal was born on 20 November 1969.
3. 1969 is earlier than 1990, so Praveen Agrawal was born first.
ANSWER: Praveen Agrawal

--- Demonstration 3 ---
Example Question: Who lived longer, Ludovico Sabbatini or Marcelino Manuel Da Graça?
Given Context: Title: Ludovico Sabbatini
Content: "Venerable Ludovico Sabbatini( 30 August 1650 – 11 June 1724) was an Italian priest and religious educator, who was beatified by the Catholic Church in 1765."
Given Context: Title: Marcelino Manuel da Graça
Content: "Marcelino Manuel da Graça( January 25, 1881 or 1884 — January 12, 1960), better known as Charles Manuel\\" Sweet Daddy\\" Grace, or Daddy Grace, was the founder and first bishop of the predominantly African- American denomination the United House of Prayer For All People."

Step-by-Step Reasoning:
1. Ludovico Sabbatini lived from 1650 to 1724 (approx 74 years).
2. Marcelino Manuel da Graça lived from 1881/1884 to 1960 (approx 76-79 years).
3. Marcelino lived longer than Ludovico.
ANSWER: Marcelino Manuel Da Graça
"""

    # --- 3. 获取当前 Row 的所有 Context ---
    # current_contexts = _contexts_all_from_row(row)
    # print(current_contexts)

    # --- 4. 构建最终 Prompt ---
    parts = []
    
    # Header / System Instruction
    parts.append("You are an expert multi-hop reasoning engine. Your goal is to answer complex questions by connecting facts from multiple provided context snippets.")
    parts.append("\n### INSTRUCTIONS:")
    parts.append("1. **Analyze the Question:** Identify the key entity and what relationship is being queried.")
    parts.append("2. **Search & Bridge:** Find the initial entity, identify the connecting link (bridge entity), and locate the final information.")
    parts.append("3. **Filter Noise:** Ignore context snippets that do not contribute to the logical chain.")
    parts.append("4. **Derive Answer:** Combine the facts to form the final answer.")
    
    parts.append("\n### RESPONSE FORMAT:")
    parts.append("- **Step-by-Step Reasoning:** Briefly explain the logical steps (e.g., 1. Found Entity X... 2. Linked to Y... 3. Concluded Z).")
    parts.append("- **ANSWER:** Output ONLY the final answer line starting with 'ANSWER: '.")

    parts.append("\nDEMONSTRATIONS:")
    parts.append(fixed_demos_text)

    parts.append("\nCURRENT TASK:")
    parts.append(f"Question: {question}")
    
    parts.append("Given Contexts:")
    for title, content in current_contexts:
        # 使用 json.dumps 确保 content 中的引号被正确转义，模拟之前 quoted 的格式
        quoted_content = json.dumps(content, ensure_ascii=False)
        if title:
            parts.append(f"Title: {title}\nContent: {quoted_content}\n")
        else:
            parts.append(f"Content: {quoted_content}\n")

    parts.append("\n### YOUR OUTPUT:")
    parts.append("Step-by-Step Reasoning:")

    return "\n".join(parts)


import json

def row_to_prompt_cot_fixed_demos_musique(row, current_contexts, question_col='question', context_col=None):
    """
    修改版函数：
    1. 输入签名与原函数 1 保持完全一致。
    2. Prompt 内容（Demonstrations）已替换为函数 2 的版本（针对 Musique 数据集，Messi/Walker 案例）。
    3. 上下文生成逻辑进行了融合：优先尝试解析 row 中的 paragraphs/decomposition 结构以匹配新 Prompt 的能力，
       如果失败则回退使用 current_contexts。
    """

    # --- 内部 Helper 函数保持不变 (来自函数 1，保留以符合"其余内容不变"的要求) ---
    def _context_map_from_val(context_val):
        cmap = {}
        if isinstance(context_val, list):
            for item in context_val:
                if isinstance(item, dict):
                    title = str(item.get('title') or item.get('Title') or item.get('name') or '')
                    content = item.get('content') or item.get('text') or item.get('Content') or item.get('texts') or ''
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    title = str(item[0])
                    content = item[1]
                else:
                    title = ''
                    content = item
                if isinstance(content, list):
                    for c in content:
                        cmap.setdefault(title, []).append(str(c))
                else:
                    cmap.setdefault(title, []).append(str(content))
        elif isinstance(context_val, dict):
            for k, v in context_val.items():
                if isinstance(v, list):
                    for vv in v:
                        cmap.setdefault(str(k), []).append(str(vv))
                else:
                    cmap.setdefault(str(k), []).append(str(v))
        else:
            cmap.setdefault('', []).append(str(context_val))
        return cmap

    def _contexts_all_from_row(r):
        cc = context_col
        if cc is None:
            for c in r.index:
                v = r[c]
                if isinstance(v, list) and len(v) > 0 and (isinstance(v[0], (list, tuple, dict)) \
                   or (isinstance(v[0], str) and ('Title' in str(v[0]) or ':' in str(v[0])) == False)):
                    cc = c; break
            if cc is None:
                for c in r.index:
                    if isinstance(r[c], (list, dict, str)):
                        cc = c; break
        context_val_local = r[cc] if cc in r.index else []
        cmap_local = _context_map_from_val(context_val_local)
        ctxs = []
        for t, clist in cmap_local.items():
            cleaned = [str(x).strip() for x in clist if x is not None and str(x).strip() != ""]
            joined = " ".join(cleaned) if cleaned else ""
            ctxs.append((str(t), joined))
        return ctxs

    # --- 1. 获取 Question (逻辑来自函数 1) ---
    if question_col in row.index:
        question = str(row[question_col])
    else:
        question = ''
        for c in row.index:
            if 'question' in str(c).lower():
                question = str(row[c]); break

    # --- 2. 硬编码的 Demonstrations (替换为函数 2 的版本) ---
    fixed_demos_text = """
--- Demonstration 1 ---
Example Question: When was the person who Messi's goals in Copa del Rey compared to get signed by Barcelona?

Given Contexts:
Title: FC Barcelona
Content: "Barcelona advanced to the semi-finals of the Copa del Rey, winning the first leg against Getafe 5–2, with a goal from Messi bringing comparison to Diego Maradona's goal of the century, but then lost the second leg 4–0."
Title: FC Barcelona
Content: "In June 1982, Diego Maradona was signed for a world record fee of £5 million from Boca Juniors. In the following season, under coach Luis, Barcelona won the Copa del Rey..."

Step-by-Step Reasoning:
1. The question asks about the person whose goals Messi's Copa del Rey goals were compared to.
2. According to the context, Messi's goal brought comparison to **Diego Maradona** ("Diego Maradona's goal of the century").
3. The question then asks when this person (Diego Maradona) was signed by Barcelona.
4. The context states that **Diego Maradona** was signed in **June 1982**.
ANSWER: June 1982

--- Demonstration 2 ---
Example Question: What was the name of the star of Forced Vengeance in Walter, Texas Ranger?

Given Contexts:
Title: Forced Vengeance
Content: "Dragon's chief of security, Josh Randall (Chuck Norris) goes looking for the head of the syndicate to exact revenge for the murder of his employer, friend and mentor."
Title: Walker, Texas Ranger
Content: "Name Portrayed by ... Cordell Walker Chuck Norris Texas Ranger Main ..."

Step-by-Step Reasoning:
1. First, identify the star of the movie "Forced Vengeance". The context mentions **Chuck Norris** as the star (Josh Randall).
2. Next, find the name of the character played by **Chuck Norris** in the show "Walker, Texas Ranger".
3. The table/context for "Walker, Texas Ranger" lists **Chuck Norris** portraying **Cordell Walker**.
ANSWER: Cordell Walker
"""

    # --- 3. 准备 Context 字符串 (融合逻辑) ---
    # 优先尝试从 row 中获取原始结构化数据 (因为函数 2 的 Prompt 依赖这种格式)
    target_col = context_col if context_col else 'paragraphs'
    raw_data_list = row.get(target_col, [])
    
    formatted_contexts_list = []
    
    # 策略 A: 如果 raw_data_list 是有效的列表且包含字典，使用函数 2 的解析逻辑
    
    for title, content in current_contexts:
        quoted_content = json.dumps(str(content), ensure_ascii=False)
        if title:
            formatted_contexts_list.append(f"Title: {title}\nContent: {quoted_content}")
        else:
            formatted_contexts_list.append(f"Content: {quoted_content}")

    full_context_str = "\n\n".join(formatted_contexts_list)

    # --- 4. 构建最终 Prompt (结构保持不变，内容使用新变量) ---
    parts = []
    
    # Header / System Instruction (使用函数 2 的通用 Header，实际上两者几乎一致)
    parts.append("You are an expert multi-hop reasoning engine. Your goal is to answer complex questions by connecting facts from multiple provided context snippets.")
    parts.append("\n### INSTRUCTIONS:")
    parts.append("1. **Analyze the Question:** Identify the key entity and what relationship is being queried.")
    parts.append("2. **Search & Bridge:** Find the initial entity, identify the connecting link (bridge entity), and locate the final information.")
    parts.append("3. **Filter Noise:** Ignore context snippets that do not contribute to the logical chain.")
    parts.append("4. **Derive Answer:** Combine the facts to form the final answer.")
    
    parts.append("\n### RESPONSE FORMAT:")
    parts.append("- **Step-by-Step Reasoning:** Briefly explain the logical steps (e.g., 1. Found Entity X... 2. Linked to Y... 3. Concluded Z).")
    parts.append("- **ANSWER:** Output ONLY the final answer line starting with 'ANSWER: '.")

    parts.append("\nDEMONSTRATIONS:")
    parts.append(fixed_demos_text)

    # Current Task
    parts.append("\n### CURRENT TASK:")
    parts.append(f"Question: {question}")
    
    parts.append("\nGiven Contexts:")
    parts.append(full_context_str)

    parts.append("\n### YOUR OUTPUT:")
    parts.append("Step-by-Step Reasoning:")

    return "\n".join(parts)




import pandas as pd
import json

def build_mmqa_cot_prompt(
    question, 
    text_list, 
    table_list, 
    image_list, 
    template_path=None
):
    """
    Constructs a Chain-of-Thought prompt for MultimodalQA (MMQA) tasks.
    
    Structure:
    1. System Instructions (Role & Format)
    2. Few-shot Demonstrations (Hardcoded MMQA examples with Text/Table/Image reasoning)
    3. Current Task Contexts (Text + Formatted Tables + Image Captions)
    4. Current Question & Output Trigger

    Args:
        question (str): The question string to be answered.
        text_list (list): List of [title, text] pairs.
                          Example: [['Title A', 'Content A'], ['Title B', 'Content B']]
        table_list (list): List of [title, table_df] pairs.
                           Example: [['Table Title A', df_A], ...]
        image_list (list): List of image caption strings.
        template_path (str, optional): Path to template file (ignored here to use fixed demos).

    Returns:
        str: The fully constructed prompt.
    """

    # --- 1. Helper: Table Formatting Logic ---
    def _format_table(table_data):
        if isinstance(table_data, pd.DataFrame):
            # 转换 DataFrame 为特定文本格式
            headers = 'col : ' + " | ".join(map(str, table_data.columns))
            rows = [
                f'row {i} : ' + " | ".join(map(str, row))
                for i, row in enumerate(table_data.itertuples(index=False))
            ]
            table_body = headers + "\n" + "\n".join(rows)
            return f"/*\n{table_body}\n*/"
        
        elif isinstance(table_data, str):
            # 如果已经是字符串，检查是否需要包裹
            s = table_data.strip()
            if s.startswith("/*") and s.endswith("*/"):
                return s
            return f"/*\n{s}\n*/"
        return ""

    # --- 2. Hardcoded Demonstrations (MMQA Specific) ---
    fixed_demos_text = """
--- Demonstration 1 ---
Example Question: When was the movie that had Ben Piazza in the role of Bob Whitewood released?

Given Contexts:
Text 1: The Bad News Bears
"The Bad News Bears is a 1976 American sports comedy film directed by Michael Ritchie. It stars Walter Matthau and Tatum O'Neal... The film was followed by two sequels..."
Table 1: Filmography
/*
col : Year | Title | Role | Notes
row 0 : 1957 | A Dangerous Age | David | 
row 1 : 1976 | The Bad News Bears | Bob Whitewood | 
row 2 : 1979 | Nightwing | Roger Piggott | 
*/

Step-by-Step Reasoning:
1. The question asks for the release date of the movie where **Ben Piazza** played **Bob Whitewood**.
2. Looking at **Table 1**, specifically row 1, we see that **Ben Piazza** played the role of **Bob Whitewood** in the film titled "**The Bad News Bears**".
3. The same row in **Table 1** (and confirmed by **Text 1**) indicates that "**The Bad News Bears**" was released in **1976**.
ANSWER: 1976

--- Demonstration 2 ---
Example Question: Which film did Ben Piazza appear in first: "Nightwing" or the movie that shows half of a woman's face on the poster?

Given Contexts:
Table 1: Filmography
/*
col : Year | Title | Role
row 0 : 1970 | Tell Me That You Love Me, Junie Moon | Jesse
row 1 : 1976 | The Bad News Bears | Bob Whitewood
row 2 : 1979 | Nightwing | Roger Piggott
*/
Image 1: 1. **[Dense Caption]**: The image is a movie poster... On the right side, there is a stylized graphic of a man's face... To the left of the face, the title "Tell me that you love me, Junie Moon" is written... 2. **[Retrieval Entities]**: Tell me that you love me, Junie Moon, Otto Preminger...

Step-by-Step Reasoning:
1. The question requires comparing the release dates of two films: "**Nightwing**" and a movie identified by its poster description ("shows half of a woman's face").
2. Looking at **Image 1**, the description mentions a movie poster with the title "**Tell Me That You Love Me, Junie Moon**". Let's assume this matches the visual description implied.
3. Now we look up the dates in **Table 1**.
4. **Table 1**, row 2 shows that "**Nightwing**" was released in **1979**.
5. **Table 1**, row 0 shows that "**Tell Me That You Love Me, Junie Moon**" was released in **1970**.
6. Comparing 1979 and 1970, the film that appeared first is **1970**.
ANSWER: Tell Me That You Love Me, Junie Moon
"""

    # --- 3. Process Current Contexts ---
    formatted_contexts_list = []

    # 3.1 Text Contexts (Input: list of [title, text])
    if text_list:
        for i, item in enumerate(text_list):
            # 解包 title 和 text
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                title, txt = item[0], item[1]
                clean_title = str(title).strip()
                clean_txt = str(txt).strip()
                # 格式: Text i: Title \n Content
                formatted_contexts_list.append(f"Text {i+1}: {clean_title}\n{clean_txt}")
            else:
                # Fallback
                formatted_contexts_list.append(f"Text {i+1}: {str(item).strip()}")

    # 3.2 Table Contexts (Input: list of [title, df])
    if table_list:
        for i, item in enumerate(table_list):
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                title, tbl = item[0], item[1]
                clean_title = str(title).strip()
                formatted_tbl = _format_table(tbl)
                if formatted_tbl:
                    # 格式: Table i: Title \n Content
                    formatted_contexts_list.append(f"Table {i+1}: {clean_title}\n{formatted_tbl}")
            else:
                # Fallback
                formatted_tbl = _format_table(item)
                if formatted_tbl:
                    formatted_contexts_list.append(f"Table {i+1}:\n{formatted_tbl}")

    # 3.3 Image Contexts (Captions) (Input: list of strings)
    if image_list:
        for i, img_cap in enumerate(image_list):
            clean_cap = str(img_cap).strip()
            formatted_contexts_list.append(f"Image {i+1}: {clean_cap}")

    full_context_str = "\n\n".join(formatted_contexts_list)

    # --- 4. Assemble Final Prompt ---
    parts = []
    
    # System Instructions
    # parts.append("You are an expert multi-modal reasoning engine. Your goal is to answer complex questions by connecting facts from text, structured tables, and image descriptions.")
    # parts.append("\n### INSTRUCTIONS:")
    # parts.append("1. **Analyze the Question:** Identify the entities and the specific comparison or retrieval task.")
    # parts.append("2. **Locate Evidence:** Search through Text, Tables, and Images. Tables are formatted with 'col' and 'row' indices.")
    # parts.append("3. **Link Facts:** Connect information across different modalities (e.g., map a movie description in an Image to a date in a Table).")
    # parts.append("4. **Derive Answer:** Perform the necessary comparison or extraction to form the final answer.")
    
    # parts.append("\n### RESPONSE FORMAT:")
    # parts.append("- **Step-by-Step Reasoning:** Briefly explain the logical steps (e.g., 1. Identified movie X from Image... 2. Found year Y in Table... 3. Concluded Z).")
    # parts.append("- **ANSWER:** Output ONLY the final answer line starting with 'ANSWER: '.")

    parts.append("You are an expert multi-modal reasoning engine. Your goal is to answer complex questions by connecting facts from text, structured tables, and image descriptions.")
    parts.append("\n### INSTRUCTIONS:")
    parts.append("1. **Analyze the Question:** Identify the key entity and the specific relationship or comparison being queried.")
    parts.append("2. **Search & Bridge:** Locate the initial entity in Text, Tables, or Images. Identify the connecting link (bridge entity) that maps information across different modalities (e.g., use a visual description in an Image to find a Name, then link that Name to a row in a Table).")
    parts.append("3. **Filter Noise:** Ignore irrelevant context snippets, table rows, or image details that do not contribute to the logical chain.")
    parts.append("4. **Derive Answer:** Combine the connected facts to perform the necessary comparison or extraction to form the final answer.")
    
    parts.append("\n### RESPONSE FORMAT:")
    parts.append("- **Step-by-Step Reasoning:** Briefly explain the logical steps (e.g., 1. Found Entity X in Image... 2. Linked X to Y in Text... 3. Concluded Z).")
    parts.append("- **ANSWER:** Output ONLY the final answer line starting with 'ANSWER: '.")

    # Demonstrations
    parts.append("\n### DEMONSTRATIONS:")
    parts.append(fixed_demos_text)

    # Current Task
    parts.append("\n### CURRENT TASK:")
    parts.append(f"statement: {question}")  # 直接使用传入的 question
    
    parts.append("\n<input>")
    parts.append(full_context_str)
    parts.append("</input>")

    parts.append("\n### YOUR OUTPUT:")
    parts.append("<output>")
    parts.append("Step-by-Step Reasoning:")

    return "\n".join(parts)


import json

GENERATION_PROMPT = """
Your goal is to give the best full answer to question the user input according to the given context below.
Given Context: {context_data}

Give the best full answer to question: {question}.

Answer this question in as fewer number of words as possible. Don't output your thinking process!
JUST OUTPUT THE ANSWER BELOW WITH THE PREFIX "ANSWER", SUCH AS "ANSWER: <YOUR OUTPUT>".
"""

def row_to_prompt_general(row, current_contexts, question_col='question', context_col=None):
    """
    Revised function using the specified GENERATION_PROMPT template.
    
    1. Input signature matches the original function.
    2. Context generation logic fuses structured row data (paragraphs) with the fallback 'current_contexts'.
    3. The final prompt is constructed by formatting the 'GENERATION_PROMPT' string with the extracted context and question.
    """

    # --- 1. Extract Question ---
    if question_col in row.index:
        question = str(row[question_col])
    else:
        question = ''
        # Fallback: find any column containing 'question'
        for c in row.index:
            if 'question' in str(c).lower():
                question = str(row[c])
                break

    # --- 2. Prepare Context String ---
    # Strategy: Try to find structured data in the row first (e.g., 'paragraphs'), 
    # otherwise fall back to the passed 'current_contexts'.
    target_col = context_col if context_col else 'paragraphs'
    raw_data_list = row.get(target_col, [])
    
    formatted_contexts_list = []
    
    # Strategy A: Use raw structured data from the row if available (dict format)

    for content in current_contexts:

            formatted_contexts_list.append(f"Content: {content}")

    # Join all context items into a single string
    full_context_str = "\n\n".join(formatted_contexts_list)

    # --- 3. Construct Final Prompt ---
    # Use the global GENERATION_PROMPT template provided in the instructions
    final_prompt = GENERATION_PROMPT.format(
        context_data=full_context_str,
        question=question
    )

    return final_prompt