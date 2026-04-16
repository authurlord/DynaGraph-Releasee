"""
KGQA-Cache Module 2: Prompt Builder with CacheBlend Separators

Mirrors the logic of utils/retrieve.py:row_to_prompt_cot_fixed_demos()
but inserts BLEND_SPECIAL_STR between reusable chunks so LMCache can
identify and cache them independently.
"""
import json

# Must match LMCACHE_BLEND_SPECIAL_STR in configs/lmcache_env.sh
BLEND_SPECIAL_STR = "§§§"

# ---------- Fixed prompt components (shared across ALL queries) ----------

SYSTEM_INSTRUCTIONS = """You are an expert multi-hop reasoning engine. Your goal is to answer complex questions by connecting facts from multiple provided context snippets.

### INSTRUCTIONS:
1. **Analyze the Question:** Identify the key entity and what relationship is being queried.
2. **Search & Bridge:** Find the initial entity, identify the connecting link (bridge entity), and locate the final information.
3. **Filter Noise:** Ignore context snippets that do not contribute to the logical chain.
4. **Derive Answer:** Combine the facts to form the final answer.

### RESPONSE FORMAT:
- **Step-by-Step Reasoning:** Briefly explain the logical steps (e.g., 1. Found Entity X... 2. Linked to Y... 3. Concluded Z).
- **ANSWER:** Output ONLY the final answer line starting with 'ANSWER: '."""

FIXED_DEMOS = """
DEMONSTRATIONS:

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


def format_doc_chunk(title: str, content: str) -> str:
    """Format a single document chunk in the same way as the original pipeline."""
    quoted_content = json.dumps(content, ensure_ascii=False)
    if title:
        return f"Title: {title}\nContent: {quoted_content}\n"
    else:
        return f"Content: {quoted_content}\n"


def row_to_prompt_cot_fixed_demos_cached(row, current_contexts, question_col='question', context_col=None):
    """
    CacheBlend-aware prompt builder.

    Inserts BLEND_SPECIAL_STR between:
      [System+Instructions] SEP [Demos] SEP [Question+Contexts_header]
      SEP [Doc1] SEP [Doc2] ... SEP [DocK] SEP [Output_trigger]

    This allows LMCache to independently cache and reuse:
      - The system+demo prefix (shared by ALL queries)
      - Each document chunk (shared across queries that retrieve the same doc)
    """
    # --- 1. Get question ---
    if question_col in row.index:
        question = str(row[question_col])
    else:
        question = ''
        for c in row.index:
            if 'question' in str(c).lower():
                question = str(row[c])
                break

    # --- 2. Build prompt with separators ---
    # IMPORTANT: Use direct concatenation (not "\n".join") so separator tokens
    # remain BPE-stable and chunk boundaries are exact. Newlines go INSIDE
    # content parts, not between separator and content.
    parts = []

    # [A] System Instructions (fixed, cacheable)
    parts.append(SYSTEM_INSTRUCTIONS)
    parts.append(BLEND_SPECIAL_STR)

    # [B] Demonstrations (fixed, cacheable)
    parts.append(FIXED_DEMOS)
    parts.append(BLEND_SPECIAL_STR)

    # [C] Question + Context header (unique per query, not cached)
    parts.append(f"\nCURRENT TASK:\nQuestion: {question}\nGiven Contexts:")
    parts.append(BLEND_SPECIAL_STR)

    # [D] Each document as an independent cacheable chunk
    for i, (title, content) in enumerate(current_contexts):
        parts.append(format_doc_chunk(title, content))
        parts.append(BLEND_SPECIAL_STR)

    # [E] Output trigger
    parts.append("\n### YOUR OUTPUT:\nStep-by-Step Reasoning:")

    return "".join(parts)


def row_to_prompt_docs_first(row, current_contexts, question_col='question', context_col=None):
    """
    Docs-first prompt builder for maximum vLLM native prefix caching.

    Structure:
      [System+Instructions] [Demos] [Contexts_header]
      [Doc1] [Doc2] ... [DocK]  (sorted by title for canonical ordering)
      [Question] [Output_trigger]

    By placing documents BEFORE the question and sorting them by title,
    queries that share the same top-K documents will share a longer
    contiguous prefix → more prefix cache hits in vLLM.
    """
    # --- 1. Get question ---
    if question_col in row.index:
        question = str(row[question_col])
    else:
        question = ''
        for c in row.index:
            if 'question' in str(c).lower():
                question = str(row[c])
                break

    # --- 2. Build prompt: docs before question ---
    parts = []

    # [A] System Instructions (fixed prefix)
    parts.append(SYSTEM_INSTRUCTIONS)

    # [B] Demonstrations (fixed prefix)
    parts.append(FIXED_DEMOS)

    # [C] Context header + documents (sorted by title for canonical ordering)
    parts.append("\nGiven Contexts:\n")
    sorted_contexts = sorted(current_contexts, key=lambda x: x[0])
    for title, content in sorted_contexts:
        parts.append(format_doc_chunk(title, content))

    # [D] Question LAST (unique per query — prefix diverges here)
    parts.append(f"\nCURRENT TASK:\nQuestion: {question}")

    # [E] Output trigger
    parts.append("\n### YOUR OUTPUT:\nStep-by-Step Reasoning:")

    return "".join(parts)


def get_system_prefix_text() -> str:
    """Return the fixed system+demo prefix text for KV pre-computation."""
    return SYSTEM_INSTRUCTIONS + "\n" + FIXED_DEMOS


def build_online_prompt(question: str, doc_contexts: list) -> str:
    """Canonical prompt builder for all online benchmarks.

    Uses the same SYSTEM_INSTRUCTIONS + 3 FIXED_DEMOS as prompt_builder.py
    to ensure consistent prompt structure across all experiments.

    Args:
        question: The query string.
        doc_contexts: List of (title, content) tuples for retrieved documents.

    Returns:
        Complete prompt string.
    """
    ctx_parts = []
    for title, content in doc_contexts:
        quoted = json.dumps(content, ensure_ascii=False)
        if title:
            ctx_parts.append(f"Given Context: Title: {title}\nContent: {quoted}")
        else:
            ctx_parts.append(f"Given Context: Content: {quoted}")
    ctx_text = "\n".join(ctx_parts)

    prompt = f"""{SYSTEM_INSTRUCTIONS}
{FIXED_DEMOS}

CURRENT TASK:
Question: {question}
{ctx_text}

### YOUR OUTPUT:
Step-by-Step Reasoning:"""
    return prompt


def build_online_prompts_from_retrieval(corpus_df, data_df, retrieved_docs, n_queries):
    """Build prompts and answers for n_queries using canonical prompt format.

    Args:
        corpus_df: DataFrame with 'title' and 'text' columns.
        data_df: DataFrame with 'question' and 'answer' columns.
        retrieved_docs: Array of arrays of doc IDs per query.
        n_queries: Number of unique queries to build.

    Returns:
        (prompts, answers): Lists of prompt strings and answer strings.
    """
    prompts = []
    for qid in range(n_queries):
        question = str(data_df.iloc[qid].get('question', ''))
        doc_contexts = []
        for doc_id in retrieved_docs[qid]:
            title = corpus_df.iloc[doc_id]['title']
            content = corpus_df.iloc[doc_id]['text']
            doc_contexts.append((title, content))
        prompts.append(build_online_prompt(question, doc_contexts))
    answers = data_df['answer'].tolist()[:n_queries]
    return prompts, answers
