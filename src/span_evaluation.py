# src/span_evaluation.py
"""
Span-level evaluation metrics for italic detection

This module evaluates model performance at the phrase/span level rather than
token level, which is more meaningful for the end-user experience.
"""
import json
from typing import List, Dict, Tuple, Set
from pathlib import Path
import config
from utils import setup_logging

logger = setup_logging(__name__)


class SpanEvaluator:
    """Evaluate model performance at span level"""

    def __init__(self):
        self.label2id = config.LABEL2ID
        self.id2label = config.ID2LABEL

    def tokens_to_spans(
        self, tokens: List[str], labels: List[str]
    ) -> List[Tuple[int, int, str]]:
        """
        Convert BIO-tagged tokens to spans

        Args:
            tokens: List of tokens
            labels: List of BIO labels (B, I, O)

        Returns:
            List of (start_idx, end_idx, phrase) tuples
        """
        spans = []
        current_span_start = None
        current_span_tokens = []

        for idx, (token, label) in enumerate(zip(tokens, labels)):
            if label == "B":
                # Save previous span if exists
                if current_span_start is not None:
                    spans.append(
                        (
                            current_span_start,
                            idx - 1,
                            " ".join(current_span_tokens),
                        )
                    )

                # Start new span
                current_span_start = idx
                current_span_tokens = [token]

            elif label == "I" and current_span_start is not None:
                # Continue current span
                current_span_tokens.append(token)

            else:  # label == "O"
                # End current span if exists
                if current_span_start is not None:
                    spans.append(
                        (
                            current_span_start,
                            idx - 1,
                            " ".join(current_span_tokens),
                        )
                    )
                    current_span_start = None
                    current_span_tokens = []

        # Don't forget last span
        if current_span_start is not None:
            spans.append(
                (
                    current_span_start,
                    len(tokens) - 1,
                    " ".join(current_span_tokens),
                )
            )

        return spans

    def compute_span_metrics(
        self,
        true_spans: List[Tuple[int, int, str]],
        pred_spans: List[Tuple[int, int, str]],
        match_type: str = "exact",
    ) -> Dict[str, float]:
        """
        Compute precision, recall, F1 at span level

        Args:
            true_spans: Ground truth spans [(start, end, phrase), ...]
            pred_spans: Predicted spans [(start, end, phrase), ...]
            match_type: "exact" (exact boundary match) or "overlap" (any overlap)

        Returns:
            Dictionary with precision, recall, f1, true_positives, false_positives, false_negatives
        """
        # Convert to sets of (start, end) for matching
        true_set = {(start, end) for start, end, _ in true_spans}
        pred_set = {(start, end) for start, end, _ in pred_spans}

        if match_type == "exact":
            # Exact boundary match
            true_positives = len(true_set & pred_set)
            false_positives = len(pred_set - true_set)
            false_negatives = len(true_set - pred_set)

        elif match_type == "overlap":
            # Any overlap counts as match
            true_positives = 0
            matched_true = set()
            matched_pred = set()

            for pred_start, pred_end in pred_set:
                for true_start, true_end in true_set:
                    # Check if spans overlap
                    if not (pred_end < true_start or pred_start > true_end):
                        true_positives += 1
                        matched_true.add((true_start, true_end))
                        matched_pred.add((pred_start, pred_end))
                        break

            false_positives = len(pred_set) - len(matched_pred)
            false_negatives = len(true_set) - len(matched_true)

        else:
            raise ValueError(f"Unknown match_type: {match_type}")

        # Compute metrics
        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0.0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "num_true_spans": len(true_set),
            "num_pred_spans": len(pred_set),
        }

    def evaluate_dataset(
        self,
        dataset: List[Dict],
        predictions: List[List[int]],
        match_types: List[str] = ["exact", "overlap"],
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate entire dataset at span level

        Args:
            dataset: List of samples with 'tokens' and 'labels'
            predictions: List of predicted label IDs for each sample
            match_types: Types of matching to compute

        Returns:
            Dictionary of metrics for each match type
        """
        all_metrics = {}

        for match_type in match_types:
            total_tp = 0
            total_fp = 0
            total_fn = 0
            total_true_spans = 0
            total_pred_spans = 0

            for sample, pred_ids in zip(dataset, predictions):
                tokens = sample["tokens"]
                true_labels = [config.ID2LABEL[lid] for lid in sample["labels"]]
                pred_labels = [config.ID2LABEL[pid] for pid in pred_ids]

                # Convert to spans
                true_spans = self.tokens_to_spans(tokens, true_labels)
                pred_spans = self.tokens_to_spans(tokens, pred_labels)

                # Compute metrics
                metrics = self.compute_span_metrics(
                    true_spans, pred_spans, match_type
                )

                total_tp += metrics["true_positives"]
                total_fp += metrics["false_positives"]
                total_fn += metrics["false_negatives"]
                total_true_spans += metrics["num_true_spans"]
                total_pred_spans += metrics["num_pred_spans"]

            # Aggregate metrics
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            all_metrics[match_type] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "true_positives": total_tp,
                "false_positives": total_fp,
                "false_negatives": total_fn,
                "num_true_spans": total_true_spans,
                "num_pred_spans": total_pred_spans,
            }

            logger.info(f"\n{match_type.upper()} MATCH:")
            logger.info(f"  Precision: {precision:.4f}")
            logger.info(f"  Recall:    {recall:.4f}")
            logger.info(f"  F1 Score:  {f1:.4f}")
            logger.info(f"  True Positives:  {total_tp}")
            logger.info(f"  False Positives: {total_fp}")
            logger.info(f"  False Negatives: {total_fn}")

        return all_metrics

    def find_mismatched_spans(
        self,
        true_spans: List[Tuple[int, int, str]],
        pred_spans: List[Tuple[int, int, str]],
    ) -> Dict[str, List[Tuple[int, int, str]]]:
        """
        Find false positives and false negatives

        Returns:
            Dictionary with 'false_positives' and 'false_negatives' lists
        """
        true_set = {(start, end) for start, end, _ in true_spans}
        pred_set = {(start, end) for start, end, _ in pred_spans}

        # False positives: predicted but not in ground truth
        fp_positions = pred_set - true_set
        false_positives = [
            span for span in pred_spans if (span[0], span[1]) in fp_positions
        ]

        # False negatives: in ground truth but not predicted
        fn_positions = true_set - pred_set
        false_negatives = [
            span for span in true_spans if (span[0], span[1]) in fn_positions
        ]

        return {
            "false_positives": false_positives,
            "false_negatives": false_negatives,
        }


def main():
    """Example usage of span evaluation"""
    logger.info("="*60)
    logger.info("Span-Level Evaluation")
    logger.info("="*60)

    # Load test dataset
    logger.info("\nLoading test dataset...")
    with open(config.DATA_DIR / "test.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # Convert to expected format
    dataset = []
    for item in test_data:
        dataset.append({
            "tokens": item["tokens"],
            "labels": [config.LABEL2ID[l] for l in item["labels"]],
        })

    logger.info(f"Loaded {len(dataset)} test samples")

    # For demonstration, use ground truth as predictions
    # In real usage, load model predictions
    predictions = [sample["labels"] for sample in dataset]

    # Evaluate
    evaluator = SpanEvaluator()
    metrics = evaluator.evaluate_dataset(dataset, predictions)

    # Save results
    results_path = config.MODEL_DIR / "span_evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
