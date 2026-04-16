import requests
import time
from typing import Literal, Optional

# --- 配置区域 ---
# 定义三个服务的连接信息
VLLM_SERVERS = {
    "generation": {
        "url": "http://localhost:8000",
        "key": "api-key-qwen3",
        "name": "Qwen3-Generation"
    },
    "embedding": {
        "url": "http://localhost:8001",
        "key": "embed-key-m3",
        "name": "Qwen3-Embedding"
    },
    "rerank": {
        "url": "http://localhost:8002",
        "key": "reranker-key-m3",
        "name": "BGE-Reranker-1 (GPU 1)"
    },
    "rerank_2": {
        "url": "http://localhost:8003",
        "key": "reranker-key-m3",
        "name": "BGE-Reranker-2 (GPU 0)"
    },
}

def control_vllm_sleep(
    target: Literal["generation", "embedding", "rerank", "rerank_2", "all"],
    action: Literal["sleep", "wake"],
    level: int = 1
):
    """
    控制 vLLM 服务的休眠与唤醒。
    
    Args:
        target: 目标服务名称 ('generation', 'embedding', 'rerank') 或 'all' (全部)
        action: 动作 ('sleep' 或 'wake')
        level: 休眠等级 (默认为 1，仅对 sleep 有效)
    """
    
    # 确定要操作的服务列表
    targets = list(VLLM_SERVERS.keys()) if target == "all" else [target]
    
    for srv_key in targets:
        server_info = VLLM_SERVERS.get(srv_key)
        if not server_info:
            print(f"❌ 未找到配置: {srv_key}")
            continue

        base_url = server_info['url']
        api_key = server_info['key']
        name = server_info['name']
        
        headers = {"Authorization": f"Bearer {api_key}"}
        
        try:
            if action == "sleep":
                # Sleep 接口: POST /sleep?level=1
                url = f"{base_url}/sleep"
                params = {"level": level}
                print(f"⏳ 正在让 {name} 进入休眠 (Level {level})...")
                resp = requests.post(url, headers=headers, params=params, timeout=5)
            
            elif action == "wake":
                # Wake 接口: POST /wake_up
                url = f"{base_url}/wake_up"
                print(f"⚡ 正在唤醒 {name} ...")
                resp = requests.post(url, headers=headers, timeout=5)
            
            # 处理响应
            if resp.status_code == 200:
                print(f"✅ {name}: 成功 - {resp.text}")
            elif resp.status_code == 404:
                print(f"⏭️ {name}: 不支持 sleep/wake (跳过)")
            else:
                print(f"⚠️ {name}: 失败 (Code {resp.status_code}) - {resp.text}")
                
        except requests.exceptions.ConnectionError:
            print(f"❌ {name}: 连接失败。请检查服务是否已启动 (端口 {base_url.split(':')[-1]})")
        except Exception as e:
            print(f"❌ {name}: 发生错误 - {str(e)}")

# --- 辅助函数：查看状态 ---
def check_server_status():
    """检查所有服务的存活与休眠状态"""
    print(f"{'服务名称':<20} {'端口':<8} {'连接状态':<10} {'休眠状态'}")
    print("-" * 60)
    
    for key, conf in VLLM_SERVERS.items():
        url = f"{conf['url']}/health" # 通常用 health 或 version 检查存活
        sleep_url = f"{conf['url']}/is_sleeping"
        headers = {"Authorization": f"Bearer {conf['key']}"}
        port = conf['url'].split(':')[-1]
        
        try:
            # 1. 检查是否休眠
            sleep_resp = requests.get(sleep_url, headers=headers, timeout=2)
            is_sleeping = "未知"
            if sleep_resp.status_code == 200:
                is_sleeping = "😴 休眠中" if sleep_resp.json().get('is_sleeping') else "🟢 运行中"
            
            print(f"{conf['name']:<20} {port:<8} {'在线':<10} {is_sleeping}")
            
        except requests.exceptions.ConnectionError:
            print(f"{conf['name']:<20} {port:<8} {'🔴 离线':<10} {'-'}")
def switch_vllm_mode(mode: Literal["retrieve", "generation"], timeout: int = 30):
    """
    切换系统模式：在 'retrieve' (检索) 和 'generation' (生成) 之间切换。
    
    机制：
    1. 发送休眠指令给不需要的服务。
    2. 循环监控这些服务的状态，直到它们真正进入休眠（释放显存）。
    3. 唤醒需要的服务。
    
    Args:
        mode: 目标模式 ('retrieve' 或 'generation')
        timeout: 等待休眠完成的最大秒数
    """
    print(f"\n🔄 正在切换系统模式至: [ {mode.upper()} ]")
    
    # 1. 定义分组
    gen_group = ["generation"]
    retrieve_group = ["embedding", "rerank", "rerank_2"]
    
    if mode == "retrieve":
        to_sleep = gen_group
        to_wake = retrieve_group
    elif mode == "generation":
        to_sleep = retrieve_group
        to_wake = gen_group
    else:
        print("❌ 模式错误：仅支持 'retrieve' 或 'generation'")
        return

    # 2. 执行休眠 (Sleep Phase)
    print(f"--- 阶段 1/3: 休眠非必要服务 {to_sleep} ---")
    for target in to_sleep:
        control_vllm_sleep(target, "sleep")

    # 3. 监控确认 (Monitor Phase)
    # 必须确保服务真正 sleep 了，显存释放了，才能继续，否则可能 OOM
    print(f"--- 阶段 2/3: 等待显存释放 (超时设置: {timeout}s) ---")
    start_time = time.time()
    
    for target in to_sleep:
        server_conf = VLLM_SERVERS[target]
        check_url = f"{server_conf['url']}/is_sleeping"
        headers = {"Authorization": f"Bearer {server_conf['key']}"}
        
        print(f"👀 监控 {server_conf['name']} 状态...", end="", flush=True)
        
        while True:
            try:
                # 检查是否超时
                if time.time() - start_time > timeout:
                    print(f"\n⚠️ 警告: 等待 {target} 休眠超时，强行继续唤醒流程...")
                    break
                
                # 请求状态
                resp = requests.get(check_url, headers=headers, timeout=2)
                if resp.status_code == 200:
                    data = resp.json()
                    is_sleeping = data.get('is_sleeping', False)
                    
                    if is_sleeping:
                        print(" ✅ 已休眠 (资源已释放)")
                        break
                    else:
                        # 还在运行中，等待
                        time.sleep(1)
                        print(".", end="", flush=True)
                elif resp.status_code == 404:
                    print(" ⏭️ 不支持 sleep (跳过)")
                    break
                else:
                    print(f" ❌ 状态检查失败 ({resp.status_code})")
                    time.sleep(1)
                    
            except requests.exceptions.RequestException:
                print(" ❌ 连接丢失 (可能服务正在重启)")
                time.sleep(1)

    # 4. 执行唤醒 (Wake Phase)
    print(f"--- 阶段 3/3: 唤醒目标服务 {to_wake} ---")
    for target in to_wake:
        control_vllm_sleep(target, "wake")
        
    print(f"\n✨ 模式切换 [ {mode.upper()} ] 完成\n")

from rapidfuzz import fuzz, process
def batch_fuzzy_match(query_list, target_list, threshold=78): ## KG模糊查询
    """
    使用 RapidFuzz 的 cdist 进行矩阵化计算。
    """
    # 1. 计算距离矩阵 (返回的是 uint8 或 float 矩阵)
    # workers=-1 表示使用所有 CPU 核心
    # score_cutoff 可以加速，但在 cdist 中部分版本可能仅用于截断
    matrix = process.cdist(
        query_list, 
        target_list, 
        scorer=fuzz.token_sort_ratio, 
        workers=-1
    )
    
    # matrix[i][j] 代表 query_list[i] 和 target_list[j] 的分数
    
    # 2. 使用 NumPy 快速筛选满足阈值的索引
    # np.where 返回的是 (行索引数组, 列索引数组)
    rows, cols = np.where(matrix >= threshold)
    
    # 3. 组装结果
    results = {}
    for r, c in zip(rows, cols):
        q_text = query_list[r]
        t_text = target_list[c]
        score = matrix[r, c]
        
        if q_text not in results:
            results[q_text] = []
        results[q_text].append((t_text, score))
        
    return results
import pandas as pd
import re

def process_triples_with_cleaning(_2wiki_kg_result, _2wiki_corpus): ## 处理三元组列表
    """
    处理三元组列表，包含以下扩展逻辑：
    1. 保留原始三元组。
    2. 添加 head 替换为页面 title 的三元组。
    3. (新增) 如果 head 包含括号 (e.g. "Name (Film)"), 添加去括号版本。
    4. (新增) 如果 tail 包含括号, 添加去括号版本。
    """
    
    triple_list = []
    
    # 预编译正则，提高循环内的匹配效率
    # r"\(.*?\)" 匹配括号及其中间的内容，.*? 为非贪婪匹配
    pattern = re.compile(r"\(.*?\)")

    def remove_parentheses(text):
        if not isinstance(text, str):
            return str(text)
        # 将括号内容替换为空，并去除首尾空格
        return re.sub(pattern, "", text).strip()

    for i in range(len(_2wiki_kg_result)):
        all_triples = _2wiki_kg_result[i]
        
        # 获取当前 sample 对应的 title
        # 注意：这里假设 _2wiki_corpus 的索引和 _2wiki_kg_result 是对齐的
        head_title = _2wiki_corpus.iloc[i]['title']
        
        for triple in all_triples:
            # 提取原始三元组元素 (避免引用修改，直接取值)
            # print(triple)
            if len(triple)>=3:
                raw_head = triple[0]
                relation = triple[1]
                raw_tail = triple[2]
                
                # --- 1. 添加原始三元组 ---
                # [head, relation, tail, sample_id]
                triple_list.append([raw_head, relation, raw_tail, i])
                
                # --- 2. 原始逻辑: 强制将 Head 替换为 Title ---
                # 此时 head 变成 head_title，tail 保持 raw_tail
                triple_list.append([head_title, relation, raw_tail, i])
                
                # --- 3. 新增逻辑: 检查 Head 是否有括号 ---
                clean_head = remove_parentheses(raw_head)
                # 如果清洗后的文本与原文不同（说明有括号被去掉了），且不为空
                if clean_head != raw_head and len(clean_head) > 0:
                    # 保持 tail 不变，只变 head
                    triple_list.append([clean_head, relation, raw_tail, i])
                
                # --- 4. 新增逻辑: 检查 Tail 是否有括号 ---
                clean_tail = remove_parentheses(raw_tail)
                # 如果清洗后的文本与原文不同
                if clean_tail != raw_tail and len(clean_tail) > 0:
                    # 保持 head 不变 (这里用 raw_head)，只变 tail
                    triple_list.append([raw_head, relation, clean_tail, i])

    # 转为 DataFrame 并去重
    triple_list_pd = pd.DataFrame(
        triple_list, 
        columns=['head', 'relation', 'tail', 'sample_id']
    ).drop_duplicates()
    
    # 去除 sample_id 为空的行 (虽然逻辑上不太可能产生空id，保留原有的健壮性)
    triple_list_pd = triple_list_pd[~pd.isnull(triple_list_pd['sample_id'])]
    
    return triple_list_pd
def set_intersetion(A,B): ## 求交集，保序
    exist_list = []
    # set_A = set(A)
    # set_B = set(B)
    for b in B:
        if b in A:
            exist_list.append(b)
    return exist_list
import math

def ndcg_at_k(ground_truth, predicted_list, k):
    """
    计算单个查询的 NDCG@k（归一化折损累积增益）
    
    参数：
        ground_truth (set/list): 真实目标集合/列表（如{5, 8}或[5]）
        predicted_list (list): 预测的有序列表（如[3, 5, 9, 8, 2]）
        k (int): 评估的前k个位置（k<=0时返回0）
    
    返回：
        float: NDCG@k 值（范围0~1，值越大表示检索质量越好）
    """
    # 边界条件：k<=0 或预测列表为空，直接返回0
    if k <= 0 or len(predicted_list) == 0:
        return 0.0
    
    gt_set = set(ground_truth) if isinstance(ground_truth, (list, set)) else {ground_truth}
    # 取前k个元素（若预测列表长度不足k，取全部）
    top_k = predicted_list[:k]
    
    # 1. 计算 DCG@k（折损累积增益）：命中元素的权重随位置递增而递减
    dcg = 0.0
    for idx, item in enumerate(top_k):
        if item in gt_set:
            # 排名从1开始，权重=1/log2(排名+1) = 1/log2(idx+2)（idx从0开始）
            dcg += 1.0 / math.log2(idx + 2)
    
    # 2. 计算 IDCG@k（理想DCG）：所有ground truth元素都出现在前k位的最优情况
    # 理想情况下，ground truth元素按最靠前的位置排列
    ideal_hit_count = min(len(gt_set), k)  # 最多命中min(真实目标数, k)个
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hit_count))
    
    # 3. 归一化：避免IDCG=0（无真实目标）时的除零错误
    return dcg / idcg if idcg > 0 else 0.0

def batch_ndcg_at_k(ground_truths, predicted_lists, k): ## Batch NDCG@k计算
    """
    批量计算多个查询的 NDCG@k 平均值
    
    参数：
        ground_truths (list): 每个元素是单个查询的ground truth（如[[5], {8,9}, [10]]）
        predicted_lists (list): 每个元素是对应查询的预测有序列表
        k (int): 评估的前k个位置
    
    返回：
        float: 所有查询的 NDCG@k 平均值
    """
    # 边界条件：输入列表为空或长度不匹配
    if not ground_truths or not predicted_lists:
        raise ValueError("ground_truths 和 predicted_lists 不能为空")
    if len(ground_truths) != len(predicted_lists):
        raise ValueError("ground_truths 和 predicted_lists 长度必须一致")
    
    total_ndcg = 0.0
    for gt, pred in zip(ground_truths, predicted_lists):
        total_ndcg += ndcg_at_k(gt, pred, k)
    
    # 返回平均值
    return total_ndcg / len(ground_truths)
import numpy as np
import matplotlib.pyplot as plt

def density_based_reward(score_list, window_size=20, reward_factor=0.5, strategy='max_only'):
    """
    输入:
        score_list: 原始分数列表 (list or np.array)
        window_size: 密度估计的窗口长度 (间隔长度)
        reward_factor: 奖励系数 (例如 0.5 代表增加 50%)
        strategy: 'max_only' (只奖励密度最高的区间) 或 'proportional' (按密度比例奖励)
    
    输出:
        new_scores: 奖励后的新分数列表
        density_profile: 每个位置对应的密度值
    """
    scores = np.array(score_list, dtype=float)
    length = len(scores)
    
    # 1. 标记非零元素 (二值化)
    # is_nonzero 是一个 0/1 数组，1 代表该位置有值
    is_nonzero = (scores != 0).astype(int)
    
    # 2. 计算密度 (使用卷积/滑动窗口)
    # np.convolve 会计算窗口内的求和。mode='same' 保证输出长度与原数组一致
    # 这里的 density_profile[i] 代表以 i 为中心的窗口内的非零元素个数
    window = np.ones(window_size)
    density_profile = np.convolve(is_nonzero, window, mode='same')
    
    # 初始化奖励乘数数组，默认为 1.0 (即不奖励)
    reward_multipliers = np.ones(length)
    
    if strategy == 'max_only':
        # --- 策略 A: 只奖励密度最大的那个窗口 ---
        
        # 找到密度最大的值
        max_density = np.max(density_profile)
        
        if max_density > 0:
            # 找到所有达到最大密度的位置索引
            # 注意：可能有多个区间的密度并列第一
            max_indices = np.where(density_profile == max_density)[0]
            
            # 对这些高密度中心点附近的元素应用奖励
            # 因为 convolution 是以中心对齐的，我们需要还原回窗口覆盖的范围
            half_window = window_size // 2
            
            for center_idx in max_indices:
                start = max(0, center_idx - half_window)
                end = min(length, center_idx + half_window)
                
                # 给这个区间内的非零元素增加奖励
                # 逻辑：如果是最大密度区间的非零元素，值 * (1 + reward_factor)
                reward_multipliers[start:end] = 1 + reward_factor
                
    elif strategy == 'proportional':
        # --- 策略 B: 密度越高，奖励越多 ---
        # 归一化密度，使其范围在 0~1 之间
        max_density = np.max(density_profile)
        if max_density > 0:
            normalized_density = density_profile / max_density
            # 奖励公式：基础 1 + (系数 * 密度强度)
            reward_multipliers = 1 + (reward_factor * normalized_density)

    # 3. 应用奖励
    # 只有非零元素才会被奖励 (保持0依然为0)
    new_scores = scores * reward_multipliers
    
    return new_scores
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


# def reranker_func_vllm(question,text_list,url = "http://localhost:8002/score",MODEL_NAME = "../models/bge-reranker-v2-m3"):
#     response = requests.post(url,
#                          json={
#                              "model": MODEL_NAME,
#                              "text_1": question,
#                              "text_2": text_list,
#                              "truncate_prompt_tokens": -1,
#                          }).json()
#     scores_all = []
#     # print(response)
#     for item in response['data']:
#         scores_all.append(item['score'])
#     return scores_all
def reranker_func_vllm(question, text_list, url="http://localhost:8002/score", MODEL_NAME="bge-reranker-v2-m3", batch_size=128):
    """
    Args:
        question: List[str] - 问题列表
        text_list: List[str] - 文档列表 (必须与 question 长度一致且一一对应)
        url: vLLM server url
        MODEL_NAME: model path or name
        batch_size: 每次请求发送的 pair 数量
    """
    scores_all = []
    total_len = len(question)
    
    # 确保输入长度一致
    assert len(question) == len(text_list), "Questions and Texts must have the same length"

    # 分批处理
    for i in tqdm(range(0, total_len, batch_size), desc="Reranking Batches", leave=False):
        # 切片获取当前 batch
        batch_question = question[i : i + batch_size]
        batch_text = text_list[i : i + batch_size]
        
        try:
            response = requests.post(url,
                                 json={
                                     "model": MODEL_NAME,
                                     "text_1": batch_question,
                                     "text_2": batch_text,
                                     "truncate_prompt_tokens": -1,
                                 }).json()
            
            # 收集分数
            if 'data' in response:
                for item in response['data']:
                    scores_all.append(item['score'])
            else:
                print(f"Warning: Batch {i} returned no data. Response: {response}")
                # 填充默认值防止索引错位 (可选，视具体需求定)
                scores_all.extend([-999.0] * len(batch_question))
                
        except Exception as e:
            print(f"Error processing batch {i}: {e}")
            scores_all.extend([-999.0] * len(batch_question))

    return scores_all
import concurrent.futures
import numpy as np
from tqdm import tqdm
import requests

# 复用之前的单路处理逻辑，稍作修改以适配线程调用
def _worker_process(args):
    """
    内部工作函数，用于单个线程处理分片数据
    """
    questions, texts, url, model_name, batch_size, worker_id = args
    
    # 这里直接调用之前的逻辑，或者把之前的逻辑内联进来
    # 为了避免多线程tqdm显示错乱，建议在内部关闭tqdm，或者使用 position 参数（较麻烦）
    # 这里简单处理：仅打印简单的进度信息
    
    scores = []
    total_len = len(questions)
    
    # 简单的 batch 循环
    for i in range(0, total_len, batch_size):
        batch_q = questions[i : i + batch_size]
        batch_t = texts[i : i + batch_size]
        
        try:
            response = requests.post(url, json={
                "model": model_name,
                "text_1": batch_q,
                "text_2": batch_t,
                "truncate_prompt_tokens": -1,
            }).json()
            
            if 'data' in response:
                scores.extend([item['score'] for item in response['data']])
            else:
                scores.extend([-999.0] * len(batch_q))
        except Exception as e:
            print(f"Worker {worker_id} Error: {e}")
            scores.extend([-999.0] * len(batch_q))
            
    return scores

def reranker_func_vllm_parallel(question, text_list,
                                urls=["http://localhost:8002/score"],
                                MODEL_NAME="bge-reranker-v2-m3",
                                batch_size=128):
    """
    并行 Rerank 函数
    """
    # 1. 基础校验
    assert len(question) == len(text_list), "Lengths must match"
    num_workers = len(urls)
    total_items = len(question)
    
    if total_items == 0:
        return []

    print(f"Parallel Reranking: {total_items} items across {num_workers} servers...")

    # 2. 数据切分 (Split)
    # 使用 numpy.array_split 可以比较均匀地把 list 切分成 n 份
    # 注意：array_split 返回的是 numpy array，如果你需要 list，需要转回 list
    # 为了通用性，这里手动切片，保证类型安全
    chunk_size = int(np.ceil(total_items / num_workers))
    
    futures = []
    results_map = {} # 用于存储 {worker_index: result_list}

    # 3. 提交任务 (Submit)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        for i in range(num_workers):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, total_items)
            
            if start_idx >= total_items:
                break
            
            # 切片
            sub_q = question[start_idx:end_idx]
            sub_t = text_list[start_idx:end_idx]
            current_url = urls[i]
            
            # 打包参数
            args = (sub_q, sub_t, current_url, MODEL_NAME, batch_size, i)
            
            # 提交并记录顺序
            future = executor.submit(_worker_process, args)
            futures.append((i, future))

        # 4. 等待结果 (Wait & Gather)
        # 使用 tqdm 显示总体线程完成进度
        for i, future in tqdm(futures, desc="Waiting for servers"):
            try:
                # 获取结果
                res = future.result()
                results_map[i] = res
            except Exception as e:
                print(f"Main Thread Error collecting result from worker {i}: {e}")
                results_map[i] = [] # 异常时填充空，或者根据需要填充默认值

    # 5. 合并结果 (Merge) - 严格按照切分时的顺序 i
    final_scores = []
    for i in range(len(results_map)):
        final_scores.extend(results_map[i])

    # 简单校验长度
    if len(final_scores) != total_items:
        print(f"Warning: Result length mismatch! Expected {total_items}, got {len(final_scores)}")

    return final_scores