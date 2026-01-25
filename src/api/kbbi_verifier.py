"""
KBBI (Kamus Besar Bahasa Indonesia) Word Verification

This module provides verification of Indonesian words using a local KBBI database
containing 115,978 Indonesian words. It's used to filter false positives from the
ML model by checking if detected foreign words actually exist in Indonesian dictionary.

Database source: https://github.com/dyazincahya/KBBI-SQL-database
"""

import json
import logging
from pathlib import Path
from typing import Set, Dict, List

logger = logging.getLogger(__name__)


class KBBIVerifier:
    """KBBI word verification using local JSON database (115k+ words)"""

    def __init__(self):
        self.kbbi_words: Set[str] = set()
        self._load_kbbi_database()

    def _load_kbbi_database(self):
        """Load KBBI database from JSON file into memory"""
        # Path to KBBI JSON database
        kbbi_file = Path(__file__).parent.parent / "data" / "dictionary_JSON.json"

        if not kbbi_file.exists():
            error_msg = (
                f"KBBI database not found at: {kbbi_file}\n"
                f"Please download from: https://github.com/dyazincahya/KBBI-SQL-database\n"
                f"Place 'dictionary_JSON.json' in: src/data/"
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            logger.info(f"Loading KBBI database from: {kbbi_file}")
            with open(kbbi_file, 'r', encoding='utf-8') as f:
                kbbi_data = json.load(f)

            # Extract only the words (we don't need definitions for verification)
            # Handle different JSON formats
            words_list = []

            if isinstance(kbbi_data, dict) and 'dictionary' in kbbi_data:
                # Format: {"dictionary": [{"word": "...", "arti": "...", ...}, ...]}
                words_list = kbbi_data['dictionary']
            elif isinstance(kbbi_data, list):
                # Format: [{"word": "...", ...}, ...] or ["word1", "word2", ...]
                words_list = kbbi_data

            for entry in words_list:
                if isinstance(entry, dict):
                    # Format: {"word": "kata", "arti": "...", ...}
                    word = entry.get('word', '').strip().lower()
                elif isinstance(entry, str):
                    # Format: "kata"
                    word = entry.strip().lower()
                else:
                    continue

                if word:
                    self.kbbi_words.add(word)

            logger.info(f"✅ Loaded {len(self.kbbi_words):,} KBBI words into memory")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse KBBI JSON: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load KBBI database: {e}")
            raise

    def is_indonesian_word(self, word: str) -> bool:
        """
        Check if word exists in KBBI dictionary

        Args:
            word: Word to check

        Returns:
            True if word IS Indonesian (found in KBBI)
            False if word is NOT Indonesian (foreign word)
        """
        if not word:
            return False

        clean_word = word.lower().strip()
        return clean_word in self.kbbi_words

    def batch_filter_foreign_words(self, words: List[str]) -> Dict:
        """
        Filter list of words to only keep foreign (non-KBBI) words

        Args:
            words: List of words to filter

        Returns:
            Dictionary with:
                - 'foreign': List of foreign words (NOT in KBBI)
                - 'indonesian': List of Indonesian words (found in KBBI)
                - 'foreign_count': Count of foreign words
                - 'indonesian_count': Count of Indonesian words
        """
        foreign = []
        indonesian = []

        for word in words:
            if not word:
                continue

            if self.is_indonesian_word(word):
                indonesian.append(word)
            else:
                foreign.append(word)

        result = {
            'foreign': foreign,
            'indonesian': indonesian,
            'foreign_count': len(foreign),
            'indonesian_count': len(indonesian)
        }

        logger.debug(
            f"KBBI filtering: {len(words)} total → "
            f"{result['foreign_count']} foreign, "
            f"{result['indonesian_count']} Indonesian"
        )

        return result


# Singleton instance (load once at startup)
_kbbi_verifier_instance = None


def get_kbbi_verifier() -> KBBIVerifier:
    """
    Get or create KBBI verifier singleton

    Returns:
        KBBIVerifier instance
    """
    global _kbbi_verifier_instance
    if _kbbi_verifier_instance is None:
        _kbbi_verifier_instance = KBBIVerifier()
    return _kbbi_verifier_instance
