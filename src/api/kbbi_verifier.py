# src/api/kbbi_verifier.py
import json
from pathlib import Path
from typing import Set

class KBBIVerifier:
    """KBBI word verification using local database (115k+ words)"""
    
    def __init__(self):
        self.kbbi_words: Set[str] = set()
        self._load_kbbi_database()
    
    def _load_kbbi_database(self):
        """Load KBBI database from JSON file"""
        kbbi_file = Path(__file__).parent.parent / "data" / "dictionary_JSON.json"
        
        if not kbbi_file.exists():
            raise FileNotFoundError(f"KBBI database not found: {kbbi_file}")
        
        with open(kbbi_file, 'r', encoding='utf-8') as f:
            kbbi_data = json.load(f)
        
        # Extract only the words (we don't need definitions for verification)
        for entry in kbbi_data:
            word = entry.get('word', '').strip().lower()
            if word:
                self.kbbi_words.add(word)
        
        print(f"✅ Loaded {len(self.kbbi_words)} KBBI words into memory")
    
    def is_indonesian_word(self, word: str) -> bool:
        """
        Check if word exists in KBBI dictionary
        Returns True if word IS Indonesian (found in KBBI)
        Returns False if word is NOT Indonesian (foreign word)
        """
        return word.lower().strip() in self.kbbi_words
    
    def batch_filter_foreign_words(self, words: list[str]) -> dict:
        """
        Filter list of words to only keep foreign (non-KBBI) words
        Returns dict with 'foreign' and 'indonesian' word lists
        """
        foreign = []
        indonesian = []
        
        for word in words:
            if self.is_indonesian_word(word):
                indonesian.append(word)
            else:
                foreign.append(word)
        
        return {
            'foreign': foreign,
            'indonesian': indonesian,
            'foreign_count': len(foreign),
            'indonesian_count': len(indonesian)
        }

# Singleton instance (load once at startup)
kbbi_verifier = None

def get_kbbi_verifier() -> KBBIVerifier:
    """Get or create KBBI verifier singleton"""
    global kbbi_verifier
    if kbbi_verifier is None:
        kbbi_verifier = KBBIVerifier()
    return kbbi_verifier
