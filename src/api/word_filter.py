"""
Rule-based filtering for foreign word detection

This module provides additional filtering rules to remove obvious false positives
that pass through ML model and KBBI verification but shouldn't be italicized.
"""

import re
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class ForeignWordFilter:
    """Filter obvious false positives using rule-based approach"""

    def __init__(self):
        # Common abbreviations/acronyms that shouldn't be italicized
        self.common_abbreviations = {
            # Technical
            'api', 'cpu', 'gpu', 'json', 'xml', 'html', 'css', 'sql',
            'http', 'https', 'url', 'uri', 'pdf', 'csv',

            # ML/AI
            'nlp', 'ml', 'ai', 'nn', 'cnn', 'rnn', 'lstm', 'bert',
            'f1', 'mae', 'mse', 'rmse', 'auc', 'roc',

            # Units
            'kb', 'mb', 'gb', 'tb', 'ms', 'fps', 'dpi',

            # BIO tags
            'b', 'i', 'o', 'bio',

            # Common
            'id', 'no', 'vs', 'etc', 'dll', 'dsb',
        }

    def should_filter_word(self, word: str) -> tuple[bool, str]:
        """
        Check if word should be filtered out

        Args:
            word: Word to check

        Returns:
            Tuple of (should_filter, reason)
            - should_filter: True if word should be removed
            - reason: Explanation why it was filtered
        """
        if not word or not word.strip():
            return True, "empty_word"

        word_clean = word.strip()
        word_lower = word_clean.lower()

        # Rule 1: Single character (except 'a' which might be valid in some contexts)
        if len(word_clean) == 1:
            return True, "single_character"

        # Rule 2: Contains numbers
        if any(char.isdigit() for char in word_clean):
            return True, "contains_numbers"

        # Rule 3: Contains special characters/punctuation (except hyphen and apostrophe)
        # Allow: hyphen (-), apostrophe ('), for words like "e-mail", "D'Artagnan"
        special_chars = r'[^\w\s\-\']'
        if re.search(special_chars, word_clean):
            return True, "contains_special_chars"

        # Rule 4: Contains brackets or parentheses
        if any(char in word_clean for char in '()[]{}'):
            return True, "contains_brackets"

        # Rule 5: All uppercase (likely abbreviation/acronym)
        if word_clean.isupper() and len(word_clean) > 1:
            return True, "all_uppercase_abbreviation"

        # Rule 6: Known abbreviation
        if word_lower in self.common_abbreviations:
            return True, "known_abbreviation"

        # Rule 7: Contains whitespace (word fragments)
        if ' ' in word_clean or '\t' in word_clean or '\n' in word_clean:
            return True, "contains_whitespace"

        # Rule 8: Starts or ends with punctuation
        if word_clean[0] in '.,;:!?' or word_clean[-1] in '.,;:!?':
            return True, "starts_ends_punctuation"

        # Rule 9: Mixed case with numbers (like "2e", "3d", "4k")
        if re.match(r'^\d+[a-zA-Z]+$|^[a-zA-Z]+\d+$', word_clean):
            return True, "mixed_alphanumeric"

        # Rule 10: Too short (less than 2 characters)
        if len(word_clean) < 2:
            return True, "too_short"

        return False, "passed_all_rules"

    def batch_filter_words(self, words: List[str]) -> Dict:
        """
        Filter list of words using rules

        Args:
            words: List of words to filter

        Returns:
            Dictionary with:
                - 'valid': List of valid foreign words
                - 'filtered': Dictionary of filtered words by reason
                - 'valid_count': Count of valid words
                - 'filtered_count': Count of filtered words
        """
        valid_words = []
        filtered_by_reason = {}

        for word in words:
            should_filter, reason = self.should_filter_word(word)

            if should_filter:
                if reason not in filtered_by_reason:
                    filtered_by_reason[reason] = []
                filtered_by_reason[reason].append(word)
            else:
                valid_words.append(word)

        total_filtered = sum(len(words) for words in filtered_by_reason.values())

        result = {
            'valid': valid_words,
            'filtered': filtered_by_reason,
            'valid_count': len(valid_words),
            'filtered_count': total_filtered
        }

        # Log filtering summary
        if total_filtered > 0:
            logger.debug(
                f"Rule-based filtering: {len(words)} input → "
                f"{len(valid_words)} valid, {total_filtered} filtered"
            )
            for reason, filtered_words in filtered_by_reason.items():
                logger.debug(f"  - {reason}: {len(filtered_words)} words")

        return result


# Singleton instance
_filter_instance = None


def get_foreign_word_filter() -> ForeignWordFilter:
    """Get or create filter singleton"""
    global _filter_instance
    if _filter_instance is None:
        _filter_instance = ForeignWordFilter()
    return _filter_instance
