# src/train.py
import json
import torch
import numpy as np
from pathlib import Path
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import config
from utils import setup_logging, tokenize_and_align_labels

# Setup logging
logger = setup_logging(__name__)

# Set seed for reproducibility
torch.manual_seed(config.SEED)
np.random.seed(config.SEED)


def load_json_dataset(path: str) -> Dataset:
    """Load dataset from JSON file"""
    logger.info(f"Loading dataset from {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Convert labels from string to integer
    records = []
    for item in data:
        records.append({
            "id": item["id"],
            "tokens": item["tokens"],
            "labels": [config.LABEL2ID[l] for l in item["labels"]],
        })

    dataset = Dataset.from_list(records)
    logger.info(f"Loaded {len(dataset)} samples")
    return dataset


def compute_metrics(pred):
    """
    Calculate metrics: accuracy, precision, recall, F1
    """
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = []
    true_labels = []
    
    for prediction, label in zip(predictions, labels):
        for pred_id, label_id in zip(prediction, label):
            if label_id == -100:
                continue
            true_predictions.append(pred_id)
            true_labels.append(label_id)

    # Calculate metrics
    accuracy = accuracy_score(true_labels, true_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, 
        true_predictions, 
        average='macro',
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def main():
    logger.info("="*60)
    logger.info("Starting IndoBERT Training for Italic Detection")
    logger.info("="*60)

    # 1. Load datasets
    logger.info("Loading datasets...")
    train_dataset = load_json_dataset(config.DATA_DIR / "train.json")
    val_dataset = load_json_dataset(config.DATA_DIR / "validation.json")

    datasets = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset
    })

    # 2. Load tokenizer and model
    logger.info(f"Loading model: {config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    model = AutoModelForTokenClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=len(config.LABEL_LIST),
        id2label=config.ID2LABEL,
        label2id=config.LABEL2ID,
    )
    logger.info("Model loaded successfully")

    # 3. Tokenization
    logger.info("Tokenizing datasets...")
    tokenized_datasets = datasets.map(
        lambda examples: tokenize_and_align_labels(examples, tokenizer),
        batched=True,
        remove_columns=datasets["train"].column_names,
        desc="Tokenizing"
    )
    logger.info("Tokenization complete")

    # 4. Data collator (for dynamic padding)
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True
    )

    # 5. Training arguments
    output_dir = config.MODEL_DIR / "indobert-italic"
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config.LEARNING_RATE,
        per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
        num_train_epochs=config.NUM_EPOCHS,
        weight_decay=config.WEIGHT_DECAY,
        warmup_ratio=config.WARMUP_RATIO,
        logging_dir=str(config.LOG_DIR),
        logging_steps=config.LOGGING_STEPS,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=config.METRIC_FOR_BEST_MODEL,
        greater_is_better=True,
        seed=config.SEED,
        report_to="tensorboard"
    )

    # 6. Initialize trainer
    logger.info("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=config.EARLY_STOPPING_PATIENCE
        )]
    )

    # 7. Train!
    logger.info("="*60)
    logger.info("Starting Training...")
    logger.info("="*60)

    train_result = trainer.train()

    # 8. Save final model
    logger.info("Saving final model...")
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))

    # 9. Evaluate on validation set
    logger.info("Final Evaluation on Validation Set:")
    metrics = trainer.evaluate()

    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"{key}: {value:.4f}")
        else:
            logger.info(f"{key}: {value}")

    # 10. Save training metrics
    metrics_path = output_dir / "training_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info("="*60)
    logger.info("Training Complete!")
    logger.info(f"Model saved to: {output_dir / 'final'}")
    logger.info(f"Metrics saved to: {metrics_path}")
    logger.info(f"View logs: tensorboard --logdir {config.LOG_DIR}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
