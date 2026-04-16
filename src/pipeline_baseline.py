"""
KGQA Pipeline Baseline - faithful reproduction of KGQA/pipeline.py
Runs from KGQA-Cache/src/ directory, using symlinked data/ and tmp_data/
"""
from utils.retrieve import faiss_cosine_search_cpu, row_to_prompt_cot_fixed_demos
from utils.async_llm import infer_prompts
import numpy as np
import pandas as pd
import time
from utils.triple_prompt import generate_rewrite_prompt, parse_rewrite_response
from utils.rerank import control_vllm_sleep, check_server_status, switch_vllm_mode
from utils.rerank import batch_fuzzy_match, process_triples_with_cleaning, set_intersetion, ndcg_at_k, batch_ndcg_at_k, density_based_reward, reranker_func_vllm, reranker_func_vllm_parallel
import argparse
from utils.retrieve import row_to_prompt_cot_fixed_demos_musique
from utils.triple_prompt import generate_triple_extraction_prompt, parse_triple_response
from openai import OpenAI
from loguru import logger
import requests
import sys
from tqdm import tqdm
import faiss
import torch
from json_repair import repair_json
from utils.evaluate import calculate_metric_scores
import semchunk
import tiktoken
from transformers import AutoTokenizer
import ast
import os
import json
import httpx


def preprocess_KG_list(input):
    _2wiki_kg_result_new = []
    for question_id in range(len(input)):
        assert isinstance(input[question_id], list)
        question_triples = []
        for element in input[question_id]:
            str_triple = [str(e) for e in element]
            question_triples.append(str_triple)
        _2wiki_kg_result_new.append(question_triples)
    return _2wiki_kg_result_new


# ================= 配置区域 =================
openai_api_key = "embed-key-m3"
openai_api_base = "http://localhost:8001/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
    http_client=httpx.Client(trust_env=False),
)

models = client.models.list()
model = models.data[0].id
logger.info(f"Using model: {model}")


def get_embed(text_list, dimensions: int = None):
    models = client.models.list()
    model_name = models.data[0].id
    request_kwargs = {
        "input": text_list,
        "model": model_name,
    }
    if dimensions is not None:
        request_kwargs["dimensions"] = dimensions
    try:
        responses = client.embeddings.create(**request_kwargs)
    except Exception as e:
        if "matryoshka" in str(e).lower() or "dimensions" in str(e).lower():
            logger.warning(f"Model does not support dimensions={dimensions}, falling back to full dimensions")
            request_kwargs.pop("dimensions", None)
            responses = client.embeddings.create(**request_kwargs)
        else:
            raise
    embeddings = np.zeros((len(text_list), len(responses.data[0].embedding)))
    for data in responses.data:
        embeddings[data.index] = np.array(data.embedding)
    return embeddings


# ================= 参数解析 =================
parser = argparse.ArgumentParser(description="KGQA-Cache Baseline Pipeline")
parser.add_argument("--dataset", type=str, choices=['2wikimultihopqa', 'musique', 'hotpotqa'], default='2wikimultihopqa', help="dataset_name")
parser.add_argument("--model_name", type=str, choices=['qwen3-4b', 'qwen3-30b', 'qwen3.5-4b', 'qwen3.5-35b'], default='qwen3-4b', help="llm model name")
parser.add_argument("--kg_match_type", type=str, choices=['fuzzy', 'exact'], default='exact', help="KG match method")
parser.add_argument("--construct_kg_index", action='store_true', help="whether to construct kg faiss index")
parser.add_argument("--chunk_size", type=int, default=128, help="semantic chunk size")
parser.add_argument("--exp_id", type=str, default="0", help="exp id for load/save")
parser.add_argument("--question_num", type=int, default=None, help="override question_num for fast testing (e.g. 50)")
parser.add_argument("--triple_size", type=str, default=None, help="override model_size for triple loading (e.g. 35B to use 35B triples with 4B LLM)")
args = parser.parse_args()

dataset_name = args.dataset
exp_id = args.exp_id

_MODEL_ROOT = os.environ.get("MODEL_PATH_ROOT", "./models")
CONFIG = {
    "qwen3-4b": os.path.join(_MODEL_ROOT, 'Qwen3-4B-Instruct-2507'),
    'qwen3-30b': os.path.join(_MODEL_ROOT, 'Qwen3-30B-A3B-Instruct-2507-FP8'),
    'qwen3.5-4b': os.path.join(_MODEL_ROOT, 'Qwen3.5-4B'),
    'qwen3.5-35b': os.path.join(_MODEL_ROOT, 'Qwen3.5-35B-A3B-FP8'),
}

MODEL_SIZE_MAP = {
    'qwen3-4b': '4B',
    'qwen3-30b': '30B',
    'qwen3.5-4b': '4B',
    'qwen3.5-35b': '35B',
}
SERVED_NAME_MAP = {
    'qwen3-4b': 'qwen3-4b',
    'qwen3-30b': 'qwen3-30b',
    'qwen3.5-4b': 'qwen3.5-4b',
    'qwen3.5-35b': 'qwen3.5-35b',
}
model_size = MODEL_SIZE_MAP.get(args.model_name, 'Unknown')
served_name = SERVED_NAME_MAP.get(args.model_name, 'qwen3-4b')
triple_size = args.triple_size or model_size  # decouple KG model from generation model

# ================= 日志与超参数配置 =================
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_dir = os.path.join(REPO_ROOT, "logs")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_filename = f"{log_dir}/{dataset_name}-{exp_id}-{model_size}-baseline.log"

logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add(log_filename, rotation="500 MB", level="INFO", mode="w")

HYPER_PARAMS_ALL = {}

HYPER_PARAMS_ALL['4B'] = {
    '2wikimultihopqa': {
        'top_k_hop_1': 5,
        'top_k_hop_2': 20,
        'top_k_candidate': 30,
        'reward_factor': 0.24,
        'KG_buff': 2,
        'top_k_final': 20,
        'chunk_buff': 0.85,
        'question_num': 1000,
    },
    'musique': {
        'top_k_hop_1': 5,
        'top_k_hop_2': 20,
        'top_k_candidate': 30,
        'reward_factor': 1.1,
        'KG_buff': 0.3,
        'top_k_final': 20,
        'chunk_buff': 0.3,
        'question_num': 1000,
    },
    'hotpotqa': {
        'top_k_hop_1': 5,
        'top_k_hop_2': 20,
        'top_k_candidate': 30,
        'reward_factor': 0.24,
        'KG_buff': 1.7,
        'top_k_final': 10,
        'chunk_buff': 0.85,
        'question_num': 1000,
    }
}

HYPER_PARAMS_ALL['30B'] = {
    '2wikimultihopqa': {
        'top_k_hop_1': 5,
        'top_k_hop_2': 20,
        'top_k_candidate': 30,
        'reward_factor': 0.24,
        'KG_buff': 2,
        'top_k_final': 20,
        'chunk_buff': 0.85,
        'question_num': 1000,
    },
    'musique': {
        'top_k_hop_1': 5,
        'top_k_hop_2': 20,
        'top_k_candidate': 30,
        'reward_factor': 1.1,
        'KG_buff': 0.3,
        'top_k_final': 20,
        'chunk_buff': 0.3,
        'question_num': 1000,
    },
    'hotpotqa': {
        'top_k_hop_1': 5,
        'top_k_hop_2': 20,
        'top_k_candidate': 30,
        'reward_factor': 0.24,
        'KG_buff': 0.7,
        'top_k_final': 15,
        'chunk_buff': 0.85,
        'question_num': 1000,
    }
}
# 35B uses same hyper params as 30B (same MoE architecture family)
HYPER_PARAMS_ALL['35B'] = HYPER_PARAMS_ALL['30B']
HYPER_PARAMS = HYPER_PARAMS_ALL[model_size]

current_config = {
    "Experiment Meta": {
        "Dataset": dataset_name,
        "Exp ID": exp_id,
        "Model Size": model_size,
        "Pipeline": "baseline",
        "Time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    },
    "Command Args": vars(args),
    "Hyper Parameters": HYPER_PARAMS[dataset_name],
    "Model Path": CONFIG[args.model_name]
}

logger.info("=" * 30 + " EXPERIMENT CONFIGURATION " + "=" * 30)
logger.info(f"\n{json.dumps(current_config, indent=4)}")
logger.info("=" * 80)

# ================= 赋值变量 =================
top_k_hop_1 = HYPER_PARAMS[dataset_name]['top_k_hop_1']
top_k_hop_2 = HYPER_PARAMS[dataset_name]['top_k_hop_2']
top_k_candidate = HYPER_PARAMS[dataset_name]['top_k_candidate']
reward_factor = HYPER_PARAMS[dataset_name]['reward_factor']
KG_buff = HYPER_PARAMS[dataset_name]['KG_buff']
top_k_final = HYPER_PARAMS[dataset_name]['top_k_final']
chunk_buff = HYPER_PARAMS[dataset_name]['chunk_buff']
question_num = args.question_num if args.question_num else HYPER_PARAMS[dataset_name]['question_num']

llm_path = CONFIG[args.model_name]
dataset = args.dataset
chunk_size = args.chunk_size

# ================= 数据加载 =================
_2wiki_corpus_path = f'{REPO_ROOT}/data/{dataset}_corpus.json'
_2wiki_data_path = f'{REPO_ROOT}/data/{dataset}.json'
_2wiki_corpus = pd.read_json(_2wiki_corpus_path)
_2wiki_data = pd.read_json(_2wiki_data_path)

if dataset == 'musique':
    fact_dict = {}
    for question_id in tqdm(range(question_num), desc="Loading Musique Facts"):
        fact_dict_single = []
        paragraph = _2wiki_data.iloc[question_id]['paragraphs']
        for item in paragraph:
            title = item['title']
            text = item['paragraph_text']
            select_pd = _2wiki_corpus[(_2wiki_corpus['title'] == title) & (_2wiki_corpus['text'] == text)]
            if item['is_supporting'] == True:
                fact_dict_single.append(list(select_pd.index)[0])
        fact_dict[question_id] = fact_dict_single
    exist_list_all = fact_dict
else:
    exist_list_all = []
    for question_id in range(question_num):
        supporting_facts = _2wiki_data.iloc[question_id]['supporting_facts']
        exist_list = []
        for i, j in supporting_facts:
            try:
                id = list(_2wiki_corpus[_2wiki_corpus['title'] == i].index)[0]
                if id not in exist_list:
                    exist_list.append(id)
            except IndexError:
                continue
        exist_list_all.append(exist_list)

### Phase 1: Offline Indexing KG
logger.info(f"Start to construct KG index for {dataset_name} dataset")

offline_index_time = 0
online_retrieval_time = 0
token_count = []
token_num = 0
time_0 = time.time()

### Chunk Text
all_text_chunk = []
chunker = semchunk.chunkerify(AutoTokenizer.from_pretrained(llm_path), chunk_size)
for i in tqdm(range(len(_2wiki_corpus)), desc="Chunking Text"):
    text = _2wiki_corpus.iloc[i, -1]
    title = _2wiki_corpus.iloc[i, 1]
    chunks = chunker(text, offsets=False, overlap=0.5)
    for chunk in chunks:
        all_text_chunk.append([title, chunk, i])

corpus_chunk_pd = pd.DataFrame(all_text_chunk, columns=['title', 'text', 'id'])
corpus_dict = {}
for index, row in corpus_chunk_pd.iterrows():
    corpus_dict[index] = row['id']

if args.construct_kg_index:
    switch_vllm_mode("generation")
    prompt_list = [generate_triple_extraction_prompt(title=_2wiki_corpus.iloc[i]['title'], text=_2wiki_corpus.iloc[i]['text']) for i in range(len(_2wiki_corpus))]
    qa_final = infer_prompts(prompt_list, temperature=0.7, top_p=0.8, sample_num=1, llm_path=llm_path, llm_name=served_name)

    token_count.append(qa_final[2])
    token_num += qa_final[2]['total_prompt_tokens']
    token_num += qa_final[2]['total_completion_tokens']

    return_str_list = {}
    for i in range(len(_2wiki_corpus)):
        try:
            return_str = eval(qa_final[0][i][0])
        except:
            try:
                return_str = ast.literal_eval(repair_json(qa_final[0][i][0]))
            except:
                return_str = parse_triple_response(qa_final[0][i][0])
        try:
            return_str_list[i] = return_str['triplets']
            assert isinstance(return_str, dict)
        except:
            return_str_list[i] = return_str
            assert isinstance(return_str, list)
    np.save(f'{REPO_ROOT}/tmp_data/{dataset}/triple_extraction_{model_size}.npy', return_str_list)
    _2wiki_kg_result = return_str_list
else:
    _2wiki_kg_result = np.load(f'{REPO_ROOT}/tmp_data/{dataset}/triple_extraction_{triple_size}.npy', allow_pickle=True).item()
    logger.info(f"Loaded triples from triple_extraction_{triple_size}.npy")

### Construct KG
try:
    df_result = process_triples_with_cleaning(_2wiki_kg_result, _2wiki_corpus)
except:
    _2wiki_kg_result = preprocess_KG_list(_2wiki_kg_result)
    df_result = process_triples_with_cleaning(_2wiki_kg_result, _2wiki_corpus)
triple_list_pd = df_result

time_1 = time.time()
offline_index_time = time_1 - time_0

### Phase 2: Init Online Retrieval
logger.info(f"Start Init Online Retrieval for {dataset_name} dataset")
switch_vllm_mode("retrieve")

questions = _2wiki_data['question'].to_list()
responses = client.embeddings.create(input=questions, model=model)
questions_embedding = np.zeros((len(questions), len(responses.data[0].embedding)))
for data in responses.data:
    questions_embedding[data.index] = np.array(data.embedding)

corpus = [f"Title: {row['title']}\n{row['text']}" for _, row in _2wiki_corpus.iterrows()]
responses = client.embeddings.create(input=corpus, model=model)
corpus_embedding = np.zeros((len(corpus), len(responses.data[0].embedding)))
for data in responses.data:
    corpus_embedding[data.index] = np.array(data.embedding)

indices, scores = faiss_cosine_search_cpu(questions_embedding, corpus_embedding, 6000)
questions_embedding_chunk = get_embed(questions, dimensions=512)
corpus_chunk = [f"Title: {row['title']}\n{row['text']}" for _, row in corpus_chunk_pd.iterrows()]
corpus_embedding_chunk = get_embed(corpus_chunk, dimensions=512)
indices_chunk, scores_chunk = faiss_cosine_search_cpu(questions_embedding_chunk, corpus_embedding_chunk, 6000)

### Phase 3 Query Rewrite
logger.info(f"Start Query Rewrite for {dataset_name} dataset")
switch_vllm_mode('generation')

prompt_list = []
for question_id in tqdm(range(question_num), desc="Query Rewrite"):
    question = _2wiki_data.iloc[question_id]['question']
    hop_1_ind = indices[question_id][:top_k_hop_1]
    retrieved_docs = [f'Title:{_2wiki_corpus.iloc[entity_id]["title"]}\nText:{_2wiki_corpus.iloc[entity_id]["text"]}' for entity_id in hop_1_ind]
    prompt = generate_rewrite_prompt(question, retrieved_docs)
    prompt_list.append(prompt)

qa_final_rewrite = infer_prompts(prompt_list, temperature=0.7, top_p=0.8, sample_num=1, llm_path=llm_path, llm_name=served_name)

token_count.append(qa_final_rewrite[2])
token_num += qa_final_rewrite[2]['total_prompt_tokens']
token_num += qa_final_rewrite[2]['total_completion_tokens']

logger.info(f"Current tokens used: {token_num}")

rewrite_query = {}
fail_count = 0
for i in range(question_num):
    try:
        return_str = eval(qa_final_rewrite[0][i][0])
    except:
        return_str = ast.literal_eval(repair_json(qa_final_rewrite[0][i][0]))
    try:
        rewrite_query[i] = return_str['expanded_query']
    except:
        rewrite_query[i] = _2wiki_data.iloc[i]['question']
        fail_count += 1
np.save(f'{REPO_ROOT}/tmp_data/{dataset_name}/rewrite_query_{model_size}.npy', rewrite_query)

### Phase 4: KG Match
logger.info(f"Start KG Match for {dataset_name} dataset")
count_id = []
hot_2_score_all = []
match_method = args.kg_match_type

for question_id in tqdm(range(question_num), desc="KG Match"):
    question = _2wiki_data.iloc[question_id]['question']
    hop_1_ind = indices[question_id][:top_k_hop_1]
    init_pd = triple_list_pd[triple_list_pd['sample_id'].isin(hop_1_ind)]
    tail_set = init_pd['tail'].unique().tolist()
    if match_method == 'fuzzy':
        result = batch_fuzzy_match(tail_set, list(triple_list_pd.fillna('').iloc[:, 0].unique()), threshold=65)
        head_set = []
        for item in result.keys():
            for value in result[item]:
                if value[0] not in head_set:
                    head_set.append(value[0])
    elif match_method == 'exact':
        head_set = tail_set
    hop_2_pd = triple_list_pd[triple_list_pd['head'].isin(head_set)]
    hop_2_pd_all = pd.concat([init_pd, hop_2_pd], ignore_index=True)
    selected_items = hop_2_pd_all['sample_id'].unique()
    count_id.append([int(s) for s in selected_items])

KG_list = count_id

count = 0
for question_id in range(question_num):
    cover = set_intersetion(exist_list_all[question_id], KG_list[question_id])
    if len(cover) == len(exist_list_all[question_id]):
        count += 1
logger.info(f'[KG Match] NDCG@{top_k_final}: {batch_ndcg_at_k(exist_list_all, KG_list, top_k_final)} with match count {count}')

### Phase 5: Corpus Rerank
logger.info(f"Start Corpus Rerank for {dataset_name} dataset")
switch_vllm_mode("retrieve")

count_id = []
hop_2_scores_all = []
question_add_list = []

for question_id in tqdm(range(question_num), desc="Corpus Rerank Prep"):
    question = rewrite_query[question_id]
    hop_1_ind = indices[question_id][:top_k_hop_1]
    hop_2_scores = np.zeros(len(_2wiki_corpus))
    for entity_id in hop_1_ind:
        context = _2wiki_corpus.iloc[entity_id]['text']
        question_add = f'Query:{question}\nBackground Information:\n{context}'
        question_add_list.append(question_add)

indices_2 = faiss_cosine_search_cpu(get_embed(question_add_list), corpus_embedding, top_k_hop_2)[0]

rerank_question = []
rerank_doc = []
for index in tqdm(range(len(question_add_list)), desc="Rerank Data Gen"):
    question_id = index // top_k_hop_1
    for text_id in range(top_k_hop_2):
        rerank_question.append(question_add_list[index])
        text_index = indices_2[index][text_id]
        rerank_doc.append(f'Title:{_2wiki_corpus.iloc[text_index]["title"]}\nText:{_2wiki_corpus.iloc[text_index]["text"]}')

scores_all = reranker_func_vllm_parallel(rerank_question, rerank_doc,
    urls=["http://localhost:8002/score", "http://localhost:8003/score"])

for question_id in tqdm(range(question_num), desc="Calculating Scores"):
    hop_2_scores = np.zeros(len(_2wiki_corpus))
    for entity_index in range(top_k_hop_1):
        flat_query_idx = question_id * top_k_hop_1 + entity_index
        current_retrieved_indices = indices_2[flat_query_idx]
        for rank_k, doc_index in enumerate(current_retrieved_indices):
            score_idx = flat_query_idx * top_k_hop_2 + rank_k
            score = scores_all[score_idx]
            hop_2_scores[doc_index] += score
    hop_2_scores_all.append(hop_2_scores)
    selected_items = np.argsort(-hop_2_scores)[:top_k_candidate].tolist()
    count_id.append(selected_items)

corpus_score = hop_2_scores_all
corpus_list = count_id
count = 0
for question_id in range(question_num):
    cover = set_intersetion(exist_list_all[question_id], corpus_list[question_id])
    if len(cover) == len(exist_list_all[question_id]):
        count += 1
logger.info(f'[Corpus Level] NDCG@{top_k_final}: {batch_ndcg_at_k(exist_list_all, corpus_list, top_k_final)} with match count {count}')

### Phase 6: Chunk Rerank
logger.info(f"Start Chunk Rerank for {dataset_name} dataset")

count_id = []
hop_2_scores_chunk = []
question_add_list = []

logger.info("Phase 6.1: Preparing Queries...")
for question_id in tqdm(range(question_num)):
    question = rewrite_query[question_id]
    hop_1_ind = indices_chunk[question_id][:top_k_hop_1]
    for entity_id in hop_1_ind:
        context = corpus_chunk_pd.iloc[entity_id]['text']
        question_add = f'Query:{question}\nBackground Information:\n{context}'
        question_add_list.append(question_add)

logger.info("Phase 6.2: Batch Embedding & Retrieval...")
query_embeddings = get_embed(question_add_list, dimensions=512)
indices_2 = faiss_cosine_search_cpu(query_embeddings, corpus_embedding_chunk, top_k_hop_2)[0]

logger.info("Phase 6.3: Preparing Rerank Pairs...")
rerank_question = []
rerank_doc = []
for index in tqdm(range(len(question_add_list))):
    for text_id in range(top_k_hop_2):
        chunk_index = indices_2[index][text_id]
        rerank_question.append(question_add_list[index])
        rerank_doc.append(f'Title:{corpus_chunk_pd.iloc[chunk_index]["title"]}\nText:{corpus_chunk_pd.iloc[chunk_index]["text"]}')

logger.info("Phase 6.4: Batch Reranking...")
scores_all = reranker_func_vllm_parallel(rerank_question, rerank_doc, batch_size=512,
    urls=["http://localhost:8002/score", "http://localhost:8003/score"])

logger.info("Phase 6.5: Aggregating Scores & Mapping Chunk to Doc...")
for question_id in tqdm(range(question_num)):
    hop_2_scores = np.zeros(len(corpus_chunk_pd))
    for hop1_k in range(top_k_hop_1):
        flat_query_idx = question_id * top_k_hop_1 + hop1_k
        current_retrieved_indices = indices_2[flat_query_idx]
        for rank_k, chunk_index in enumerate(current_retrieved_indices):
            score_idx = flat_query_idx * top_k_hop_2 + rank_k
            hop_2_scores[chunk_index] += scores_all[score_idx]

    hop_2_scores_chunk.append(hop_2_scores)
    selected_chunks = np.argsort(-hop_2_scores)[:top_k_candidate].tolist()
    selected_docs = [corpus_dict[i] for i in selected_chunks]
    selected_docs = list(dict.fromkeys(selected_docs))
    count_id.append(selected_docs)

chunk_list = count_id
chunk_score = hop_2_scores_chunk

count = 0
for question_id in range(question_num):
    cover = set_intersetion(exist_list_all[question_id], chunk_list[question_id])
    if len(cover) == len(exist_list_all[question_id]):
        count += 1
logger.info(f'[Chunk Level] NDCG@{top_k_final}: {batch_ndcg_at_k(exist_list_all, corpus_list, top_k_final)} with match count {count}')

retrieve_result = {}
retrieve_result['corpus_score'] = corpus_score
retrieve_result['corpus_list'] = corpus_list
retrieve_result['chunk_score'] = chunk_score
retrieve_result['kg_list'] = KG_list

np.save(f'{REPO_ROOT}/tmp_data/{dataset}/retrieve_result_{model_size}_{exp_id}.npy', retrieve_result)

## Phase 7: Aggregate all
logger.info(f"Start Aggregate all for {dataset_name} dataset")

results = {}
count_id_ = []
for question_id in range(question_num):
    score_all = np.zeros(len(_2wiki_corpus))
    corpus_score_single = corpus_score[question_id]
    chunk_score_single = chunk_score[question_id]

    for i in range(len(corpus_score_single)):
        score_all[i] += corpus_score_single[i]

    selected_items_ = np.argsort(-chunk_score_single)[:top_k_final].tolist()
    selected_items_ = [corpus_dict[i] for i in selected_items_]
    selected_items_ = list(dict.fromkeys(selected_items_))

    for i in selected_items_:
        score_all[i] += float(chunk_buff)

    for id in KG_list[question_id]:
        score_all[id] += float(KG_buff)

    score_all_density = density_based_reward(score_all, window_size=20, reward_factor=reward_factor, strategy='proportional')
    selected_items = np.argsort(-score_all_density)[:top_k_final]
    count_id_.append(selected_items)

ndcg_value = batch_ndcg_at_k(exist_list_all, count_id_, top_k_final)

count = 0
for question_id in range(question_num):
    cover = set_intersetion(exist_list_all[question_id], count_id_[question_id])
    if len(cover) == len(exist_list_all[question_id]):
        count += 1
logger.info(f'[Global Level] NDCG@{top_k_final}: {ndcg_value} with match count {count}')

retrieved_docs = count_id_
retrieve_result['retrieved_docs'] = retrieved_docs
np.save(f'{REPO_ROOT}/tmp_data/{dataset}/retrieve_result_{model_size}_{exp_id}.npy', retrieve_result)

# Phase 8: 生成答案
logger.info(f"Start generating answers for {dataset_name} dataset")

input_df = _2wiki_data

if dataset_name != 'musique':
    prompt_gt_with_all_text = [row_to_prompt_cot_fixed_demos(_2wiki_data.iloc[id], current_contexts=[(_2wiki_corpus.iloc[i]['title'], _2wiki_corpus.iloc[i]['text']) for i in retrieved_docs[id]]) for id in range(len(retrieved_docs))]
else:
    prompt_gt_with_all_text = [row_to_prompt_cot_fixed_demos_musique(_2wiki_data.iloc[id], current_contexts=[(_2wiki_corpus.iloc[i]['title'], _2wiki_corpus.iloc[i]['text']) for i in retrieved_docs[id]]) for id in range(len(retrieved_docs))]
answers = input_df['answer'].to_list()[:question_num]

switch_vllm_mode('generation')

qa_final = infer_prompts(prompt_gt_with_all_text, temperature=0, top_p=1, sample_num=1, llm_path=llm_path, llm_name=served_name)

token_count.append(qa_final[2])
token_num += qa_final[2]['total_prompt_tokens']
token_num += qa_final[2]['total_completion_tokens']

result = calculate_metric_scores(answers=answers, predictions=[q[0] for q in qa_final[0]])

logger.info("Final Metric Scores:")
logger.info(result['pooled_results'])

np.save(f'{REPO_ROOT}/tmp_data/{dataset_name}/result_{model_size}_{exp_id}.npy', result)

time_2 = time.time()
online_retrieval_time = time_2 - time_1

efficiency_measure = {
    'offline_index_time': offline_index_time,
    'online_retrieval_time': online_retrieval_time,
    'total_token_num': token_num}
efficiency_measure['time_stat'] = token_count

np.save(f'{REPO_ROOT}/tmp_data/{dataset_name}/efficiency_measure_{model_size}_{exp_id}.npy', efficiency_measure)

logger.info("Efficiency Measures:")
logger.info(efficiency_measure)
