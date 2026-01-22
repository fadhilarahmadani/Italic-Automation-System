# src/utils.py
"""
Utility functions shared across the project
"""
import logging
import sys
from pathlib import Path
import config


def setup_logging(name: str, log_level: str = "INFO") -> logging.Logger:
    """
    Setup logging configuration

    Args:
        name: Logger name (usually __name__)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Avoid duplicate handlers
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

    return logger


def tokenize_and_align_labels(examples, tokenizer):
    """
    Tokenize and align labels with subword tokens

    IndoBERT will split some words into subwords.
    We need to align labels so each subword has the correct label.

    Args:
        examples: Dictionary containing 'tokens' and 'labels' keys
        tokenizer: HuggingFace tokenizer instance

    Returns:
        Tokenized inputs with aligned labels
    """
    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=config.MAX_LENGTH,
        padding=False  # Padding will be done by data collator
    )

    all_labels = examples["labels"]
    new_labels = []

    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_id = None

        for word_id in word_ids:
            # Special tokens (CLS, SEP, PAD) are given label -100 (ignored in loss)
            if word_id is None:
                label_ids.append(-100)
            # First subword of a word
            elif word_id != previous_word_id:
                label_ids.append(labels[word_id])
            # Second+ subword of the same word
            else:
                # Give same label or -100 (choose one)
                # Here we use -100 so only the first subword is counted
                label_ids.append(-100)

            previous_word_id = word_id

        new_labels.append(label_ids)

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs
