# -*- coding: utf-8 -*-
"""
Evaluator for computing metrics on emotion analysis results
"""
from typing import Dict, List, Any
from collections import defaultdict


EMOTIONS = ["angry", "fear", "happy", "neutral", "sad", "surprise"]
MBTI16 = {
    "INTJ", "INTP", "ENTJ", "ENTP",
    "INFJ", "INFP", "ENFJ", "ENFP",
    "ISTJ", "ISFJ", "ESTJ", "ESFJ",
    "ISTP", "ISFP", "ESTP", "ESFP",
}


def compute_mae(y_true: List[float], y_pred: List[float]) -> float:
    """Compute Mean Absolute Error."""
    if not y_true:
        return 0.0
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_true)


def compute_mse(y_true: List[float], y_pred: List[float]) -> float:
    """Compute Mean Squared Error."""
    if not y_true:
        return 0.0
    return sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true)


def compute_accuracy(y_true: List[str], y_pred: List[str]) -> float:
    """Compute classification accuracy."""
    if not y_true:
        return 0.0
    hits = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return hits / len(y_true)


def compute_macro_f1(y_true: List[str], y_pred: List[str], labels: List[str]) -> float:
    """Compute macro F1 score."""
    if not y_true:
        return 0.0

    f1_scores = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)

    return sum(f1_scores) / len(f1_scores)


def compute_confusion_matrix(y_true: List[str], y_pred: List[str], labels: List[str]) -> List[List[int]]:
    """Compute confusion matrix."""
    idx = {label: i for i, label in enumerate(labels)}
    n = len(labels)
    matrix = [[0] * n for _ in range(n)]

    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            matrix[idx[t]][idx[p]] += 1

    return matrix


def evaluate_predictions(
    predictions: List[Dict[str, Any]],
    gold_labels: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Evaluate predictions against gold labels.

    Args:
        predictions: List of predicted results
        gold_labels: List of gold standard labels

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Emotion regression metrics
    reg_true = defaultdict(list)
    reg_pred = defaultdict(list)

    # Classification metrics
    primary_true = []
    primary_pred = []
    mbti_true = []
    mbti_pred = []

    # Format metrics
    json_ok = 0
    cot_complete = 0

    for pred, gold in zip(predictions, gold_labels):
        # Emotion scores
        for e in EMOTIONS:
            gold_val = gold.get("emotion_analysis", {}).get(e, 0.0)
            pred_val = pred.get("scores", {}).get(e, 0.0)
            reg_true[e].append(gold_val)
            reg_pred[e].append(pred_val)

        # Primary emotion
        if "primary_emotion" in gold:
            primary_true.append(gold["primary_emotion"])
            primary_pred.append(pred.get("primary_emotion", ""))

        # MBTI
        if "mbti_type" in gold:
            mbti_true.append(gold["mbti_type"])
            mbti_pred.append(pred.get("mbti_type", ""))

        # Format checks
        if pred.get("json_parse_ok"):
            json_ok += 1
        if pred.get("cot_complete"):
            cot_complete += 1

    n = len(predictions)
    metrics["json_parse_rate"] = json_ok / n if n > 0 else 0.0
    metrics["cot7_complete_rate"] = cot_complete / n if n > 0 else 0.0

    # Emotion MAE/MSE per dimension
    per_dim_mae = {}
    per_dim_mse = {}
    maes = []
    mses = []

    for e in EMOTIONS:
        if reg_true[e]:
            mae = compute_mae(reg_true[e], reg_pred[e])
            mse = compute_mse(reg_true[e], reg_pred[e])
            per_dim_mae[e] = round(mae, 6)
            per_dim_mse[e] = round(mse, 6)
            maes.append(mae)
            mses.append(mse)

    if maes:
        metrics["emotion_macro_mae"] = round(sum(maes) / len(maes), 6)
        metrics["emotion_macro_mse"] = round(sum(mses) / len(mses), 6)
        metrics["emotion_per_dim_mae"] = per_dim_mae
        metrics["emotion_per_dim_mse"] = per_dim_mse

    # Primary emotion classification
    if primary_true:
        metrics["primary_cls_accuracy"] = round(compute_accuracy(primary_true, primary_pred), 6)
        metrics["primary_cls_macro_f1"] = round(compute_macro_f1(primary_true, primary_pred, EMOTIONS), 6)
        metrics["primary_cls_confusion_matrix"] = {
            "labels": EMOTIONS,
            "matrix": compute_confusion_matrix(primary_true, primary_pred, EMOTIONS)
        }

    # MBTI prediction
    if mbti_true:
        metrics["mbti_accuracy"] = round(compute_accuracy(mbti_true, mbti_pred), 6)
        metrics["mbti_macro_f1"] = round(
            compute_macro_f1(mbti_true, mbti_pred, sorted(MBTI16)), 6
        )

    return metrics


def compute_sample_metrics(pred: Dict[str, Any]) -> Dict[str, Any]:
    """Compute metrics for a single prediction sample."""
    scores = pred.get("scores", {})

    # Find top 3 emotions
    sorted_emotions = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top3 = [{"emotion": e, "score": s} for e, s in sorted_emotions[:3]]

    return {
        "primary_emotion": pred.get("primary_emotion", ""),
        "confidence": pred.get("confidence", 0.0),
        "mbti_type": pred.get("mbti_type", ""),
        "top_emotions": top3,
        "json_parse_ok": pred.get("json_parse_ok", False),
        "cot_complete": pred.get("cot_complete", False),
        "latency_ms": pred.get("latency_ms", 0.0)
    }
