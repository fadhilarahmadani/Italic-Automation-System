# src/error_analysis.py
"""
Analisis kesalahan dan dokumentasi kasus kegagalan untuk deteksi italic

Modul ini membantu mengidentifikasi pola kesalahan model untuk memandu perbaikan.
"""
import json
from typing import List, Dict, Tuple
from pathlib import Path
from collections import Counter, defaultdict
import config
from span_evaluation import SpanEvaluator
from utils import setup_logging

logger = setup_logging(__name__)


class ErrorAnalyzer:
    """Analyze model errors and document failure cases"""

    def __init__(self):
        self.span_evaluator = SpanEvaluator()
        self.errors = {
            "false_positives": [],
            "false_negatives": [],
            "boundary_errors": [],
        }

    def analyze_sample(
        self,
        sample_id: str,
        tokens: List[str],
        true_labels: List[str],
        pred_labels: List[str],
        confidences: List[float] = None,
    ) -> Dict:
        """
        Analyze errors in a single sample

        Args:
            sample_id: Sample identifier
            tokens: List of tokens
            true_labels: Ground truth labels
            pred_labels: Predicted labels
            confidences: Prediction confidences (optional)

        Returns:
            Dictionary with error analysis
        """
        # Convert to spans
        true_spans = self.span_evaluator.tokens_to_spans(tokens, true_labels)
        pred_spans = self.span_evaluator.tokens_to_spans(tokens, pred_labels)

        # Find mismatches
        mismatches = self.span_evaluator.find_mismatched_spans(
            true_spans, pred_spans
        )

        # Analyze each error type
        analysis = {
            "sample_id": sample_id,
            "text": " ".join(tokens),
            "errors": {
                "false_positives": [],
                "false_negatives": [],
                "boundary_errors": [],
            },
        }

        # False Positives: Model predicted italic but shouldn't be
        for start, end, phrase in mismatches["false_positives"]:
            error_info = {
                "phrase": phrase,
                "position": (start, end),
                "context": self._get_context(tokens, start, end),
                "error_type": self._classify_fp_error(phrase, tokens, start, end),
            }

            if confidences:
                error_info["avg_confidence"] = sum(
                    confidences[start : end + 1]
                ) / len(confidences[start : end + 1])

            analysis["errors"]["false_positives"].append(error_info)

        # False Negatives: Model missed italic that should be detected
        for start, end, phrase in mismatches["false_negatives"]:
            error_info = {
                "phrase": phrase,
                "position": (start, end),
                "context": self._get_context(tokens, start, end),
                "error_type": self._classify_fn_error(phrase, tokens, start, end),
            }

            if confidences:
                error_info["avg_confidence"] = sum(
                    confidences[start : end + 1]
                ) / len(confidences[start : end + 1])

            analysis["errors"]["false_negatives"].append(error_info)

        # Boundary errors: Partial overlap (detected but wrong boundaries)
        boundary_errors = self._find_boundary_errors(
            true_spans, pred_spans, tokens
        )
        analysis["errors"]["boundary_errors"] = boundary_errors

        return analysis

    def _get_context(
        self, tokens: List[str], start: int, end: int, window: int = 3
    ) -> str:
        """Get surrounding context for error"""
        left_context = " ".join(tokens[max(0, start - window) : start])
        phrase = " ".join(tokens[start : end + 1])
        right_context = " ".join(
            tokens[end + 1 : min(len(tokens), end + 1 + window)]
        )

        return f"...{left_context} [{phrase}] {right_context}..."

    def _classify_fp_error(
        self, phrase: str, tokens: List[str], start: int, end: int
    ) -> str:
        """Classify type of false positive error"""
        phrase_lower = phrase.lower()

        # Common patterns
        if any(char.isdigit() for char in phrase):
            return "contains_numbers"
        elif len(phrase) <= 2:
            return "very_short_phrase"
        elif phrase.isupper():
            return "all_uppercase"
        elif "-" in phrase or "_" in phrase:
            return "contains_separator"
        elif any(c in phrase for c in "()[]{}"):
            return "contains_brackets"
        else:
            return "other"

    def _classify_fn_error(
        self, phrase: str, tokens: List[str], start: int, end: int
    ) -> str:
        """Classify type of false negative error"""
        phrase_lower = phrase.lower()

        # Common patterns
        if len(phrase.split()) > 3:
            return "long_phrase"
        elif any(char.isdigit() for char in phrase):
            return "mixed_alphanumeric"
        elif phrase[0].isupper() and start == 0:
            return "sentence_start"
        elif phrase.istitle():
            return "title_case"
        else:
            return "other"

    def _find_boundary_errors(
        self,
        true_spans: List[Tuple[int, int, str]],
        pred_spans: List[Tuple[int, int, str]],
        tokens: List[str],
    ) -> List[Dict]:
        """Find spans with partial overlap (boundary errors)"""
        boundary_errors = []

        for pred_start, pred_end, pred_phrase in pred_spans:
            for true_start, true_end, true_phrase in true_spans:
                # Check for overlap but not exact match
                if (pred_start, pred_end) != (true_start, true_end):
                    overlap = not (pred_end < true_start or pred_start > true_end)

                    if overlap:
                        error = {
                            "predicted": {
                                "phrase": pred_phrase,
                                "position": (pred_start, pred_end),
                            },
                            "ground_truth": {
                                "phrase": true_phrase,
                                "position": (true_start, true_end),
                            },
                            "context": self._get_context(
                                tokens,
                                min(pred_start, true_start),
                                max(pred_end, true_end),
                            ),
                        }
                        boundary_errors.append(error)

        return boundary_errors

    def analyze_dataset(
        self,
        dataset: List[Dict],
        predictions: List[List[int]],
        confidences: List[List[float]] = None,
    ) -> Dict:
        """
        Analyze errors across entire dataset

        Args:
            dataset: List of samples
            predictions: Model predictions
            confidences: Prediction confidences (optional)

        Returns:
            Comprehensive error analysis
        """
        logger.info("Analyzing errors across dataset...")

        all_errors = []
        error_type_counts = Counter()
        phrase_error_freq = defaultdict(int)

        for idx, (sample, pred_ids) in enumerate(zip(dataset, predictions)):
            sample_id = sample.get("id", f"sample_{idx}")
            tokens = sample["tokens"]
            true_labels = [config.ID2LABEL[lid] for lid in sample["labels"]]
            pred_labels = [config.ID2LABEL[pid] for pid in pred_ids]

            conf = confidences[idx] if confidences else None

            # Analyze this sample
            analysis = self.analyze_sample(
                sample_id, tokens, true_labels, pred_labels, conf
            )

            # Collect errors
            if any(
                len(errors) > 0
                for errors in analysis["errors"].values()
            ):
                all_errors.append(analysis)

                # Count error types
                for fp in analysis["errors"]["false_positives"]:
                    error_type_counts[f"FP_{fp['error_type']}"] += 1
                    phrase_error_freq[fp["phrase"].lower()] += 1

                for fn in analysis["errors"]["false_negatives"]:
                    error_type_counts[f"FN_{fn['error_type']}"] += 1
                    phrase_error_freq[fn["phrase"].lower()] += 1

                error_type_counts["boundary_errors"] += len(
                    analysis["errors"]["boundary_errors"]
                )

        # Generate summary statistics
        summary = {
            "total_samples": len(dataset),
            "samples_with_errors": len(all_errors),
            "error_rate": len(all_errors) / len(dataset),
            "error_type_distribution": dict(error_type_counts),
            "most_common_error_phrases": dict(
                Counter(phrase_error_freq).most_common(20)
            ),
        }

        logger.info(f"\nError Analysis Summary:")
        logger.info(f"  Total samples: {summary['total_samples']}")
        logger.info(f"  Samples with errors: {summary['samples_with_errors']}")
        logger.info(f"  Error rate: {summary['error_rate']:.2%}")

        logger.info(f"\nError Type Distribution:")
        for error_type, count in error_type_counts.most_common():
            logger.info(f"  {error_type}: {count}")

        return {
            "summary": summary,
            "detailed_errors": all_errors[:100],  # Save top 100 for review
        }

    def generate_error_report(
        self, error_analysis: Dict, output_path: Path
    ) -> None:
        """
        Generate human-readable error report

        Args:
            error_analysis: Output from analyze_dataset()
            output_path: Path to save report
        """
        logger.info(f"\nGenerating error report...")

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("ERROR ANALYSIS REPORT")
        report_lines.append("Italic Detection Model")
        report_lines.append("=" * 80)
        report_lines.append("")

        # Summary section
        summary = error_analysis["summary"]
        report_lines.append("SUMMARY")
        report_lines.append("-" * 80)
        report_lines.append(f"Total Samples: {summary['total_samples']}")
        report_lines.append(
            f"Samples with Errors: {summary['samples_with_errors']}"
        )
        report_lines.append(f"Error Rate: {summary['error_rate']:.2%}")
        report_lines.append("")

        # Error type distribution
        report_lines.append("ERROR TYPE DISTRIBUTION")
        report_lines.append("-" * 80)
        for error_type, count in summary["error_type_distribution"].items():
            report_lines.append(f"  {error_type:30s}: {count:5d}")
        report_lines.append("")

        # Most common error phrases
        report_lines.append("MOST COMMON ERROR PHRASES")
        report_lines.append("-" * 80)
        for phrase, count in list(summary["most_common_error_phrases"].items())[
            :20
        ]:
            report_lines.append(f"  {phrase:40s}: {count:3d} times")
        report_lines.append("")

        # Detailed examples
        report_lines.append("DETAILED ERROR EXAMPLES (Top 20)")
        report_lines.append("-" * 80)

        for idx, error_sample in enumerate(
            error_analysis["detailed_errors"][:20], 1
        ):
            report_lines.append(f"\n{idx}. Sample ID: {error_sample['sample_id']}")
            report_lines.append(f"   Text: {error_sample['text'][:100]}...")

            # False Positives
            if error_sample["errors"]["false_positives"]:
                report_lines.append("   False Positives:")
                for fp in error_sample["errors"]["false_positives"][:3]:
                    report_lines.append(
                        f"     - '{fp['phrase']}' ({fp['error_type']})"
                    )
                    report_lines.append(f"       Context: {fp['context']}")

            # False Negatives
            if error_sample["errors"]["false_negatives"]:
                report_lines.append("   False Negatives:")
                for fn in error_sample["errors"]["false_negatives"][:3]:
                    report_lines.append(
                        f"     - '{fn['phrase']}' ({fn['error_type']})"
                    )
                    report_lines.append(f"       Context: {fn['context']}")

            # Boundary Errors
            if error_sample["errors"]["boundary_errors"]:
                report_lines.append("   Boundary Errors:")
                for be in error_sample["errors"]["boundary_errors"][:2]:
                    report_lines.append(
                        f"     - Predicted: '{be['predicted']['phrase']}'"
                    )
                    report_lines.append(
                        f"       True: '{be['ground_truth']['phrase']}'"
                    )

        # Save report
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

        logger.info(f"Error report saved to {output_path}")


def main():
    """Run error analysis on test set"""
    logger.info("="*60)
    logger.info("Error Analysis")
    logger.info("="*60)

    # Load test dataset
    logger.info("\nLoading test dataset...")
    with open(config.DATA_DIR / "test.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)

    dataset = []
    for item in test_data:
        dataset.append({
            "id": item["id"],
            "tokens": item["tokens"],
            "labels": [config.LABEL2ID[l] for l in item["labels"]],
        })

    logger.info(f"Loaded {len(dataset)} test samples")

    # For demonstration, use ground truth as predictions
    # In real usage, load model predictions
    predictions = [sample["labels"] for sample in dataset]

    # Analyze
    analyzer = ErrorAnalyzer()
    error_analysis = analyzer.analyze_dataset(dataset, predictions)

    # Save results
    results_path = config.MODEL_DIR / "error_analysis.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(error_analysis, f, indent=2, ensure_ascii=False)

    logger.info(f"\nError analysis saved to {results_path}")

    # Generate human-readable report
    report_path = config.MODEL_DIR / "error_analysis_report.txt"
    analyzer.generate_error_report(error_analysis, report_path)


if __name__ == "__main__":
    main()
