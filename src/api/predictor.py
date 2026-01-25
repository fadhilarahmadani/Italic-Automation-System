import torch
import time
from transformers import AutoTokenizer, AutoModelForTokenClassification
from typing import List, Dict, Tuple
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
import config
from utils import setup_logging

# Setup logging
logger = setup_logging(__name__)


class ItalicDetectionService:
    """
    Service for italic detection using IndoBERT
    Output based on character offset (safe for Microsoft Word)
    """

    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = config.MODEL_DIR / "indobert-italic" / "final"

        self.model_path = Path(model_path)
        self.tokenizer = None
        self.model = None
        self.device = None
        self.is_loaded = False

        self._load_model()

    def _load_model(self):
        try:
            logger.info(f"Loading model from {self.model_path}")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForTokenClassification.from_pretrained(self.model_path)
            self.model.eval()

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)

            self.is_loaded = True
            logger.info(f"Model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def predict_tokens(
        self,
        text: str
    ) -> Tuple[List[Dict], float]:
        """
        Token-level prediction with offset mapping tokenizer

        Returns:
            predictions: list of tokens with label, confidence, char offset
            processing_time: time taken for prediction
        """
        start_time = time.time()

        if not text or not text.strip():
            return [], 0.0

        # Tokenize with offset mapping (CRITICAL FIX)
        inputs = self.tokenizer(
            text,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=True,
            max_length=config.MAX_LENGTH
        )

        offset_mapping = inputs.pop("offset_mapping")[0]  # (seq_len, 2)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(probs, dim=-1)

        results = []

        for idx, (start_char, end_char) in enumerate(offset_mapping):
            if start_char == end_char:
                continue  # special tokens

            label_id = predictions[idx].item()
            label = config.ID2LABEL[label_id]
            confidence = probs[idx][label_id].item()

            results.append({
                "label": label,
                "confidence": confidence,
                "start_pos": int(start_char),
                "end_pos": int(end_char)
            })

        processing_time = time.time() - start_time
        return results, processing_time

    def _expand_to_word_boundary(self, text: str, start: int, end: int) -> Tuple[int, int]:
        """
        Expand span ke batas kata yang utuh (hindari kata terpotong seperti 'prom' dari 'prompt')
        """
        # Expand ke kiri sampai awal kata
        while start > 0 and text[start - 1].isalnum():
            start -= 1
        
        # Expand ke kanan sampai akhir kata
        while end < len(text) and text[end].isalnum():
            end += 1
        
        return start, end

    def _clean_word(self, word: str) -> Tuple[str, int, int]:
        """
        Bersihkan kata dari tanda baca di awal dan akhir.
        Juga hapus singkatan dalam kurung di akhir.
        Returns: (cleaned_word, left_offset, right_offset)
        """
        import string
        import re
        punctuation = string.punctuation + "()[]{}\"'""''«»‹›"
        
        original_len = len(word)
        left_offset = 0
        right_offset = 0
        
        # Hapus singkatan dalam kurung di akhir, contoh: "Natural Language Processing (NLP" -> "Natural Language Processing"
        # Pattern: spasi + kurung buka + huruf kapital + akhir string
        abbrev_pattern = r'\s*\([A-Z]+$'
        match = re.search(abbrev_pattern, word)
        if match:
            word = word[:match.start()]
            right_offset += len(match.group())
        
        # Strip dari kiri
        while word and word[0] in punctuation:
            word = word[1:]
            left_offset += 1
        
        # Strip dari kanan
        while word and word[-1] in punctuation:
            word = word[:-1]
            right_offset += 1
        
        return word, left_offset, right_offset

    def _is_numeric_only(self, word: str) -> bool:
        """
        Cek apakah kata hanya berisi angka (dan tanda baca angka seperti . , -)
        Contoh: "2024", "100.000", "1,234" -> True
        Contoh: "COVID-19", "Web3" -> False (ada huruf)
        """
        import re
        # Hapus spasi dan cek apakah hanya angka dan tanda baca numerik
        cleaned = word.strip()
        if not cleaned:
            return True
        
        # Pattern: hanya angka, titik, koma, spasi, tanda minus
        numeric_pattern = r'^[\d.,\-\s%]+$'
        return bool(re.match(numeric_pattern, cleaned))

    def extract_italic_phrases(
        self,
        text: str,
        confidence_threshold: float = 0.8
    ) -> Tuple[List[Dict], float]:
        """
        Combine B/I tokens into italic spans
        Confidence is applied at SPAN LEVEL (not token level)
        """

        token_preds, processing_time = self.predict_tokens(text)

        italic_phrases = []
        current_span = None

        for token in token_preds:
            label = token["label"]

            if label == "B":
                if current_span:
                    italic_phrases.append(current_span)

                current_span = {
                    "start_pos": token["start_pos"],
                    "end_pos": token["end_pos"],
                    "confidences": [token["confidence"]]
                }

            elif label == "I" and current_span:
                current_span["end_pos"] = token["end_pos"]
                current_span["confidences"].append(token["confidence"])

            else:
                if current_span:
                    italic_phrases.append(current_span)
                    current_span = None

        if current_span:
            italic_phrases.append(current_span)

        # Apply confidence threshold at span level
        final_results = []
        for span in italic_phrases:
            avg_conf = sum(span["confidences"]) / len(span["confidences"])

            if avg_conf >= confidence_threshold:
                # Expand ke batas kata yang utuh (fix kata terpotong)
                expanded_start, expanded_end = self._expand_to_word_boundary(
                    text, span["start_pos"], span["end_pos"]
                )
                
                raw_word = text[expanded_start:expanded_end]
                
                # Bersihkan tanda baca dari kata
                clean_word, left_off, right_off = self._clean_word(raw_word)
                
                # Adjust posisi berdasarkan cleaning
                final_start = expanded_start + left_off
                final_end = expanded_end - right_off
                
                # Skip jika kata kosong setelah cleaning
                if not clean_word.strip():
                    continue
                
                # Skip jika kata hanya berisi angka
                if self._is_numeric_only(clean_word):
                    continue
                
                final_results.append({
                    "start_pos": final_start,
                    "end_pos": final_end,
                    "confidence": avg_conf,
                    "label": "ITALIC",
                    "word": clean_word
                })

        return final_results, processing_time

    def get_model_info(self) -> Dict:
        return {
            "model_name": config.MODEL_NAME,
            "model_path": str(self.model_path),
            "device": str(self.device),
            "is_loaded": self.is_loaded,
            "labels": config.LABEL_LIST
        }


# Singleton instance
_service_instance = None


def get_predictor_service() -> ItalicDetectionService:
    global _service_instance
    if _service_instance is None:
        _service_instance = ItalicDetectionService()
    return _service_instance
