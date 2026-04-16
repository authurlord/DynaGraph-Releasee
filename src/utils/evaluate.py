# ...existing code...
from hashlib import md5
import tiktoken
import html
import re
import string
from typing import List, Dict, Tuple, Optional, Union, Callable, Any
import numpy as np
from collections import Counter
from loguru import logger
import os
import json
import re
def encode_string_by_tiktoken(content: str, model_name: str = "cl100k_base"):
    ENCODER = tiktoken.get_encoding(model_name)
    tokens = ENCODER.encode(content)
    return tokens

def decode_string_by_tiktoken(tokens: list[int], model_name: str = "cl100k_base"):
    ENCODER = tiktoken.get_encoding(model_name)
    string = ENCODER.decode(tokens)
    return string

def mdhash_id(content, prefix: str = ""):
    return prefix + md5(content.encode()).hexdigest()

def clean_str(input) -> str:
    """Clean an input string by removing HTML escapes, control characters, and other unwanted characters."""
    # If we get non-string input, just give it back
    if not isinstance(input, str):
        return input

    result = html.unescape(input.strip())
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
    result = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)

    # Remove non-alphanumeric characters and convert to lowercase
    return re.sub('[^A-Za-z0-9 ]', ' ', result.lower()).strip()

def normalize_name(name):
    suffixes = ["'s", "s'", "'"]
    for suffix in suffixes:
        if name.endswith(suffix):
            return name[:-len(suffix)].strip()
    return name

def normalize_answer(answer: str) -> str:
    """
    Normalize a given string by applying the following transformations:
    1. Convert the string to lowercase.
    2. Remove punctuation characters.
    3. Remove the articles "a", "an", and "the".
    4. Normalize whitespace by collapsing multiple spaces into one.

    Args:
        answer (str): The input string to be normalized.

    Returns:
        str: The normalized string.
    """
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(answer))))
def calculate_metric_scores(answers: List[Union[str, List[str]]], predictions: List[str], aggregation_fn: Callable = np.max) -> Dict[str, Any]:
    """
    独立运行的评分函数。
    输入:
      - answers: 每个样本的 gold answer 或者 gold answers 列表（List[str] 或 str）
      - predictions: 模型输出字符串列表（可能包含 "ANSWER: ..." 前缀）
      - aggregation_fn: 多个 gold answers 时的聚合函数（默认 np.max）
    输出:
      - dict 包含 "pooled_results" 和 "specific_results"
    """
    if len(answers) != len(predictions):
        raise ValueError("Length of answers and predictions must be the same.")

    def extract_pred(pred_raw: str) -> str:
        try:
            matches = list(re.finditer(r'ANSWER:\s*(.*)', pred_raw))
            if matches:
                return matches[-1].group(1).strip()
        except Exception:
            pass
        return pred_raw.strip()

    def compute_single(gold: str, predicted: str) -> Dict[str, float]:
        normalize_gold = normalize_answer(gold)
        normalize_pred = normalize_answer(predicted)
        gold_tokens = normalize_gold.split()
        predicted_tokens = normalize_pred.split()
        common = Counter(predicted_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            return {"F1": 0.0, "ACC": 0.0}

        precision = 1.0 * num_same / (len(predicted_tokens) if predicted_tokens else 1)
        recall = 1.0 * num_same / (len(gold_tokens) if gold_tokens else 1)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        acc = 1.0 if normalize_gold in normalize_pred else 0.0
        return {"F1": f1, "ACC": acc}

    specific_results: List[Dict[str, Any]] = []
    total_f1 = 0.0
    total_acc = 0.0

    for idx, (gold_item, pred_raw) in enumerate(zip(answers, predictions)):
        # 标准化 gold list
        if isinstance(gold_item, str):
            gold_list = [gold_item]
        else:
            gold_list = list(gold_item)

        pred = extract_pred(pred_raw)

        scores = [compute_single(g, pred) for g in gold_list]
        f1s = [s["F1"] for s in scores]
        accs = [s["ACC"] for s in scores]

        aggregated_f1 = float(aggregation_fn(f1s)) if f1s else 0.0
        aggregated_acc = float(aggregation_fn(accs)) if accs else 0.0

        specific_results.append({
            "id": idx,
            "answer": gold_list,
            "generation": pred,
            "F1": aggregated_f1,
            "ACC": aggregated_acc
        })

        total_f1 += aggregated_f1
        total_acc += aggregated_acc

    n = len(answers) if answers else 1
    avg_f1 = total_f1 / n
    avg_acc = total_acc / n

    pooled = {"F1": float(avg_f1), "ACC": float(avg_acc)}

    return {"pooled_results": pooled, "specific_results": specific_results}
# ...existing code...

import json
import argparse
import re
import string
import numpy as np
from collections import Counter
from typing import List, Set, Tuple, Union
from scipy.optimize import linear_sum_assignment
from word2number.w2n import word_to_num


# from oscar.utils.common_utils import *


# From here through _match_numbers_if_present was originally copied from the evaluation code of DROP dataset:
# https://github.com/allenai/allennlp-reading-comprehension/blob/master/allennlp_rc/eval/drop_eval.py

def _remove_articles(text: str) -> str:
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    return re.sub(regex, " ", text)


def _white_space_fix(text: str) -> str:
    return " ".join(text.split())


EXCLUDE = set(string.punctuation)


def _remove_punc(text: str) -> str:
    if not _is_number(text):
        return "".join(ch for ch in text if ch not in EXCLUDE)
    else:
        return text


def _lower(text: str) -> str:
    return text.lower()


def _tokenize(text: str) -> List[str]:
    return re.split(" |-", text)


def _normalize_answer(text: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    parts = [
        _white_space_fix(_remove_articles(_normalize_number(_remove_punc(_lower(token)))))
        for token in _tokenize(text)
    ]
    parts = [part for part in parts if part.strip()]
    normalized = " ".join(parts).strip()
    return normalized


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def _is_word_number(text: str) -> bool:
    try:
        word_to_num(text)
        return True
    except ValueError:
        return False


def _normalize_number(text: str) -> str:
    if _is_number(text):
        return str(float(text))
    # TODO: this is not included in the original drop evaluation script, we need to have our own in the end anyways.
    elif _is_word_number(text):
        return str(float(word_to_num(text)))
    else:
        return text


def _answer_to_bags(
        answer: Union[str, List[str], Tuple[str, ...]]
) -> Tuple[List[str], List[Set[str]]]:
    if isinstance(answer, (list, tuple)):
        raw_spans = answer
    else:
        raw_spans = [answer]
    normalized_spans: List[str] = []
    token_bags = []
    for raw_span in raw_spans:
        normalized_span = _normalize_answer(raw_span)
        normalized_spans.append(normalized_span)
        token_bags.append(set(normalized_span.split()))
    return normalized_spans, token_bags


def _align_bags(predicted: List[Set[str]], gold: List[Set[str]]) -> List[float]:
    """
    Takes gold and predicted answer sets and first finds the optimal 1-1 alignment
    between them and gets maximum metric values over all the answers.
    """
    scores = np.zeros([len(gold), len(predicted)])
    for gold_index, gold_item in enumerate(gold):
        for pred_index, pred_item in enumerate(predicted):
            if _match_numbers_if_present(gold_item, pred_item):
                scores[gold_index, pred_index] = _compute_f1(pred_item, gold_item)
    row_ind, col_ind = linear_sum_assignment(-scores)

    max_scores = np.zeros([max(len(gold), len(predicted))])
    for row, column in zip(row_ind, col_ind):
        max_scores[row] = max(max_scores[row], scores[row, column])
    return max_scores


def _compute_f1(predicted_bag: Set[str], gold_bag: Set[str]) -> float:
    intersection = len(gold_bag.intersection(predicted_bag))
    if not predicted_bag:
        precision = 1.0
    else:
        precision = intersection / float(len(predicted_bag))
    if not gold_bag:
        recall = 1.0
    else:
        recall = intersection / float(len(gold_bag))
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if not (precision == 0.0 and recall == 0.0)
        else 0.0
    )
    return f1


def _match_numbers_if_present(gold_bag: Set[str], predicted_bag: Set[str]) -> bool:
    gold_numbers = set()
    predicted_numbers = set()
    for word in gold_bag:
        if _is_number(word):
            gold_numbers.add(word)
    for word in predicted_bag:
        if _is_number(word):
            predicted_numbers.add(word)
    if (not gold_numbers) or gold_numbers.intersection(predicted_numbers):
        return True
    return False


def list_em(predicted, gold):
    predicted_bags = _answer_to_bags(predicted)
    gold_bags = _answer_to_bags(gold)
    if set(predicted_bags[0]) == set(gold_bags[0]) and len(predicted_bags[0]) == len(gold_bags[0]):
        return 1.0
    else:
        return 0.0


def list_f1(predicted, gold):
    predicted_bags = _answer_to_bags(predicted)
    gold_bags = _answer_to_bags(gold)
    f1_per_bag = _align_bags(predicted_bags[1], gold_bags[1])
    f1 = np.mean(f1_per_bag)
    f1 = round(f1, 2)
    return f1


def metric_max_over_ground_truths(metric_fn, prediction, gold_answers):
    scores_for_ground_truths = []
    for gold_answer in gold_answers:
        score = metric_fn(prediction, gold_answer)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate_predictions(predictions, gold_answers, example_types=None):
    """To support multiple gold annotations, `gold_answers` should be a list,
    with each item (either a string or a list) corresponding to one valid reference answer."""
    instance_eval_results = {}
    instance_eval_results_by_types = {}
    eval_funcs = {
        "list_em": list_em,
        "list_f1": list_f1
    }
    for qas_id in gold_answers:
        ref_answers = gold_answers[qas_id]
        if qas_id not in predictions:
            print(f"Missing prediction for question {qas_id}, and all scores for this question are set to zero")
            instance_eval_results[qas_id] = {
                metric: 0.0 for metric in eval_funcs.keys()
            }
        else:
            pred_answer = predictions[qas_id]
            instance_eval_results[qas_id] = {
                metric: metric_max_over_ground_truths(
                    func, pred_answer, ref_answers
                ) for metric, func in eval_funcs.items()
            }
        if example_types is not None:
            example_type = example_types[qas_id]
            if example_type not in instance_eval_results_by_types:
                instance_eval_results_by_types[example_type] = {}
            instance_eval_results_by_types[example_type][qas_id] = instance_eval_results[qas_id]

    eval_scores = {metric: np.mean([result[metric] for result in instance_eval_results.values()]) * 100
                   for metric in eval_funcs.keys()}

    if example_types is not None:
        eval_scores_by_types = {}
        for example_type, type_instance_eval_results in instance_eval_results_by_types.items():
            eval_scores_by_types[example_type] = {
                metric: np.mean([result[metric] for result in type_instance_eval_results.values()]) * 100 for metric in
                eval_funcs.keys()
            }
        return eval_scores, instance_eval_results, eval_scores_by_types
    else:
        return eval_scores, instance_eval_results


def evaluate_predictions_example(prediction, gold_answer, example_types=None):
    eval_funcs = {
        "list_em": list_em,
        "list_f1": list_f1
    }
    instance_eval_results = {
        metric: metric_max_over_ground_truths(
            func, prediction, gold_answer
        ) for metric, func in eval_funcs.items()
    }

    return instance_eval_results