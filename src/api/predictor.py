# src/api/predictor.py
import torch
import time
from transformers import AutoTokenizer, AutoModelForTokenClassification
from typing import List, Dict, Tuple
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
import config

class ItalicDetectionService:
    """Service untuk deteksi italic menggunakan IndoBERT"""
    
    def __init__(self, model_path: str = None):
        """
        Initialize service dengan model
        
        Args:
            model_path: Path ke model. Jika None, gunakan default dari config
        """
        if model_path is None:
            model_path = config.MODEL_DIR / "indobert-italic" / "final"
        
        self.model_path = Path(model_path)
        self.tokenizer = None
        self.model = None
        self.device = None
        self.is_loaded = False
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load model dan tokenizer"""
        try:
            print(f"📂 Loading model from {self.model_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForTokenClassification.from_pretrained(self.model_path)
            self.model.eval()
            
            # Setup device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            
            self.is_loaded = True
            print(f"   ✅ Model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"   ❌ Failed to load model: {e}")
            raise
    
    def predict_tokens(
        self, 
        text: str, 
        confidence_threshold: float = 0.8
    ) -> Tuple[List[Dict], float]:
        """
        Predict dengan detail token-level
        
        Args:
            text: Input text
            confidence_threshold: Minimum confidence
            
        Returns:
            Tuple of (predictions list, processing time)
        """
        start_time = time.time()
        
        # Split text by space (simple tokenization)
        tokens = text.split()
        
        if not tokens:
            return [], 0.0
        
        # Tokenize
        inputs = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=config.MAX_LENGTH
        )
        
        # Get word IDs before moving to device
        word_ids = inputs.word_ids(batch_index=0)
        
        # Move to device
        inputs_on_device = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs_on_device)
            predictions = torch.argmax(outputs.logits, dim=-1)[0].cpu()
            probabilities = torch.softmax(outputs.logits, dim=-1)[0].cpu()
        
        # Extract results
        results = []
        prev_word_id = None
        char_position = 0
        
        for idx, word_id in enumerate(word_ids):
            if word_id is None:  # Special token
                continue
            
            # Only process first subword of each word
            if word_id != prev_word_id:
                token = tokens[word_id]
                pred_label = config.ID2LABEL[predictions[idx].item()]
                confidence = probabilities[idx][predictions[idx]].item()
                
                # Calculate character positions
                start_pos = char_position
                end_pos = char_position + len(token)
                
                # Only add if confidence meets threshold
                if confidence >= confidence_threshold:
                    results.append({
                        "token": token,
                        "label": pred_label,
                        "confidence": confidence,
                        "start_pos": start_pos,
                        "end_pos": end_pos
                    })
                
                char_position = end_pos + 1  # +1 for space
            
            prev_word_id = word_id
        
        processing_time = time.time() - start_time
        return results, processing_time
    
    def extract_italic_phrases(
        self,
        text: str,
        confidence_threshold: float = 0.8
    ) -> Tuple[List[Dict], float]:
        """
        Extract phrases yang perlu di-italic (menggabungkan B dan I)
        
        Returns:
            Tuple of (italic phrases list, processing time)
        """
        predictions, processing_time = self.predict_tokens(text, confidence_threshold)
        
        italic_phrases = []
        current_phrase = {
            "words": [],
            "start_pos": None,
            "end_pos": None,
            "confidences": []
        }
        
        for pred in predictions:
            if pred["label"] == "B":
                # Save previous phrase if exists
                if current_phrase["words"]:
                    italic_phrases.append({
                        "word": " ".join(current_phrase["words"]),
                        "start_pos": current_phrase["start_pos"],
                        "end_pos": current_phrase["end_pos"],
                        "confidence": sum(current_phrase["confidences"]) / len(current_phrase["confidences"]),
                        "label": "B" if len(current_phrase["words"]) == 1 else "B-I"
                    })
                
                # Start new phrase
                current_phrase = {
                    "words": [pred["token"]],
                    "start_pos": pred["start_pos"],
                    "end_pos": pred["end_pos"],
                    "confidences": [pred["confidence"]]
                }
            
            elif pred["label"] == "I" and current_phrase["words"]:
                # Continue current phrase
                current_phrase["words"].append(pred["token"])
                current_phrase["end_pos"] = pred["end_pos"]
                current_phrase["confidences"].append(pred["confidence"])
        
        # Don't forget last phrase
        if current_phrase["words"]:
            italic_phrases.append({
                "word": " ".join(current_phrase["words"]),
                "start_pos": current_phrase["start_pos"],
                "end_pos": current_phrase["end_pos"],
                "confidence": sum(current_phrase["confidences"]) / len(current_phrase["confidences"]),
                "label": "B" if len(current_phrase["words"]) == 1 else "B-I"
            })
        
        return italic_phrases, processing_time
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            "model_name": config.MODEL_NAME,
            "model_path": str(self.model_path),
            "device": str(self.device),
            "is_loaded": self.is_loaded,
            "labels": config.LABEL_LIST
        }


# Global service instance (singleton)
_service_instance = None

def get_predictor_service() -> ItalicDetectionService:
    """Get or create predictor service instance"""
    global _service_instance
    if _service_instance is None:
        _service_instance = ItalicDetectionService()
    return _service_instance
