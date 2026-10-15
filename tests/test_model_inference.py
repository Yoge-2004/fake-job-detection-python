"""
JobGuard ML Inference Model Evaluation & Benchmark Test Suite
============================================================
Evaluates the multi-modal neural architecture (DistilBERT + Autoencoder
+ Isolation Forest + Calibrated Meta-Ensemble + LIME + Semantic Engine)
against 24 gold-standard test job descriptions (12 Fake, 12 Real).

Usage:
  .venv/bin/python tests/test_model_inference.py
  .venv/bin/pytest tests/test_model_inference.py
"""

import json
import os
import sys
import time
from typing import List, Dict, Any

# Ensure parent directory is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app import app, detector


def load_benchmark_dataset() -> List[Dict[str, Any]]:
    json_path = os.path.join(CURRENT_DIR, "test_job_descriptions.json")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_benchmark(verbose: bool = True) -> Dict[str, Any]:
    dataset = load_benchmark_dataset()
    client = app.test_client()

    y_true = []
    y_pred = []
    results_detail = []

    total_inference_time = 0.0

    print("=" * 80)
    print("JOBGUARD SOTA FRAUD DETECTION SYSTEM - BENCHMARK EVALUATION")
    print("=" * 80)
    print(f"Total Test Samples: {len(dataset)} (12 Fake / 12 Real)")
    print("-" * 80)

    for item in dataset:
        sample_id = item["id"]
        category = item["category"]
        expected = item["expected_label"]
        title = item["title"]
        text = item["description"]

        t0 = time.time()
        res = client.post("/predict", json={"text": text})
        elapsed = time.time() - t0
        total_inference_time += elapsed

        assert res.status_code == 200, f"Inference failed for {sample_id}: {res.data.decode()}"
        data = res.get_json()

        prob = data["fraud_probability"]
        # Classification threshold: >= 50% = Fake (1), < 50% = Real (0)
        pred_label = 1 if prob >= 50.0 else 0
        is_correct = (pred_label == expected)

        y_true.append(expected)
        y_pred.append(pred_label)

        verdict_str = "FAKE / SCAM" if pred_label == 1 else "REAL / CLEAN"
        status_icon = "✅ PASS" if is_correct else "❌ FAIL"

        results_detail.append({
            "id": sample_id,
            "title": title,
            "category": category,
            "prob": prob,
            "verdict": verdict_str,
            "status": status_icon,
            "elapsed": elapsed
        })

        if verbose:
            print(f"[{status_icon}] [{sample_id}] {title[:38]:<38} | Prob: {prob:>5.1f}% | Time: {elapsed:.2f}s")

    # Metrics computation
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    total = len(y_true)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    avg_latency = total_inference_time / total if total > 0 else 0.0

    print("=" * 80)
    print("BENCHMARK PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"Accuracy:          {accuracy * 100:.2f}% ({tp + tn}/{total})")
    print(f"Precision:         {precision * 100:.2f}%")
    print(f"Recall:            {recall * 100:.2f}% ({tp}/{tp + fn})")
    print(f"F1 Score:          {f1 * 100:.2f}%")
    print(f"Confusion Matrix:  TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"Avg Latency/Job:   {avg_latency:.2f} seconds")
    print("=" * 80)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "avg_latency": avg_latency,
        "details": results_detail
    }


def test_ml_benchmark_accuracy():
    """PyTest runner hook: asserts that model achieves >= 90% benchmark accuracy."""
    metrics = evaluate_benchmark(verbose=False)
    assert metrics["accuracy"] >= 0.90, f"Model accuracy {metrics['accuracy']*100:.1f}% below 90% target"
    assert metrics["recall"] >= 0.90, f"Model scam recall {metrics['recall']*100:.1f}% below 90% target"


if __name__ == "__main__":
    evaluate_benchmark(verbose=True)
