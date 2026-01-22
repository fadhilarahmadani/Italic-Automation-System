# src/evaluate.py
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    DataCollatorForTokenClassification,
    TrainingArguments
)
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import config
from utils import setup_logging, tokenize_and_align_labels

# Setup logging
logger = setup_logging(__name__)

def load_test_dataset():
    """Load test dataset"""
    with open(config.DATA_DIR / "test.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    for item in data:
        records.append({
            "id": item["id"],
            "tokens": item["tokens"],
            "labels": [config.LABEL2ID[l] for l in item["labels"]],
        })

    return Dataset.from_list(records)

def compute_detailed_metrics(predictions, labels):
    """Compute detailed classification report"""
    # Remove ignored indices
    true_predictions = []
    true_labels = []

    for prediction, label in zip(predictions, labels):
        for pred_id, label_id in zip(prediction, label):
            if label_id == -100:
                continue
            true_predictions.append(pred_id)
            true_labels.append(label_id)

    # Classification report
    label_names = [config.ID2LABEL[i] for i in range(len(config.LABEL_LIST))]
    report = classification_report(
        true_labels,
        true_predictions,
        target_names=label_names,
        digits=4
    )

    # Confusion matrix
    cm = confusion_matrix(true_labels, true_predictions)

    return report, cm


def main():
    logger.info("="*60)
    logger.info("Evaluating Model on Test Set")
    logger.info("="*60)

    # Load model and tokenizer
    model_path = config.MODEL_DIR / "indobert-italic" / "final"
    logger.info(f"Loading model from {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    logger.info("Model loaded successfully")

    # Load test dataset
    logger.info("Loading test dataset...")
    test_dataset = load_test_dataset()
    logger.info(f"Loaded {len(test_dataset)} test samples")

    # Tokenize
    logger.info("Tokenizing test data...")
    tokenized_test = test_dataset.map(
        lambda examples: tokenize_and_align_labels(examples, tokenizer),
        batched=True,
        remove_columns=test_dataset.column_names
    )
    logger.info("Tokenization complete")

    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # Create trainer for evaluation
    training_args = TrainingArguments(
        output_dir=str(config.MODEL_DIR / "temp"),
        per_device_eval_batch_size=config.EVAL_BATCH_SIZE
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator
    )

    # Get predictions
    logger.info("Running predictions on test set...")
    predictions_output = trainer.predict(tokenized_test)
    predictions = np.argmax(predictions_output.predictions, axis=2)
    labels = predictions_output.label_ids

    # Compute metrics
    logger.info("Computing detailed metrics...")
    report, cm = compute_detailed_metrics(predictions, labels)

    # Print results
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(report)

    print("\n" + "="*60)
    print("CONFUSION MATRIX")
    print("="*60)
    print("       ", "  ".join(f"{config.ID2LABEL[i]:^5}" for i in range(len(config.LABEL_LIST))))
    for i, row in enumerate(cm):
        print(f"{config.ID2LABEL[i]:^5}", "  ".join(f"{val:^5}" for val in row))

    # Save best model results
    results_path = config.MODEL_DIR / "indobert-italic" / "test_results.json"
    results = {
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "test_samples": len(test_dataset)
    }

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {results_path}")
    logger.info("="*60)

if __name__ == "__main__":
    main()
