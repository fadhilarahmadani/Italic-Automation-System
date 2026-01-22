# src/predict.py
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import config
from utils import setup_logging

# Setup logging
logger = setup_logging(__name__)


class ItalicPredictor:
    def __init__(self, model_path):
        """Load trained model"""
        logger.info(f"Loading model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info(f"Model loaded on {self.device}")
    
    def predict(self, text):
        """
        Predict italic words in text

        Args:
            text: Sentence string (will be split by space)

        Returns:
            List of tuples (token, label, confidence)
        """
        # Split text into tokens (simple)
        tokens = text.split()

        # Tokenize
        inputs = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=config.MAX_LENGTH
        )

        # IMPORTANT: Save word_ids BEFORE moving to device
        word_ids = inputs.word_ids(batch_index=0)

        # Move to device
        inputs_on_device = {k: v.to(self.device) for k, v in inputs.items()}

        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs_on_device)
            predictions = torch.argmax(outputs.logits, dim=-1)[0]
            probabilities = torch.softmax(outputs.logits, dim=-1)[0]

        # Move predictions back to CPU for processing
        predictions = predictions.cpu()
        probabilities = probabilities.cpu()

        results = []
        prev_word_id = None

        for idx, word_id in enumerate(word_ids):
            if word_id is None:  # Special token
                continue

            # Only take first subword of each word
            if word_id != prev_word_id:
                token = tokens[word_id]
                pred_label = config.ID2LABEL[predictions[idx].item()]
                confidence = probabilities[idx][predictions[idx]].item()

                results.append({
                    "token": token,
                    "label": pred_label,
                    "confidence": confidence
                })

            prev_word_id = word_id

        return results

    def get_italic_words(self, text):
        """Extract only words that need italic (label B or I)"""
        predictions = self.predict(text)
        italic_words = []
        current_phrase = []

        for pred in predictions:
            if pred["label"] == "B":
                # Save previous phrase if exists
                if current_phrase:
                    italic_words.append(" ".join(current_phrase))
                # Start new phrase
                current_phrase = [pred["token"]]
            elif pred["label"] == "I" and current_phrase:
                # Continue phrase
                current_phrase.append(pred["token"])
            else:
                # Label O, save phrase if exists
                if current_phrase:
                    italic_words.append(" ".join(current_phrase))
                    current_phrase = []

        # Don't forget last phrase
        if current_phrase:
            italic_words.append(" ".join(current_phrase))

        return italic_words


def main():
    # Load model
    model_path = config.MODEL_DIR / "indobert-italic" / "final"
    predictor = ItalicPredictor(model_path)

    # Test sentences
    test_sentences = [
        "Untuk memastikan aksesibilitas, kami menjalankan back-end sementara dokumentasi mengikuti responsive design secara berkala",
        "Di sisi klien, user experience berjalan berdampingan dengan front-end guna menekan waktu muat",
        "Meskipun content management system telah digunakan, tim tetap melakukan responsive design agar konsistensi tata letak terjaga"
    ]

    logger.info("="*60)
    logger.info("Testing Italic Detection")
    logger.info("="*60)

    for i, sentence in enumerate(test_sentences, 1):
        print(f"\nTest {i}:")
        print(f"Sentence: {sentence}\n")

        # Detailed predictions
        predictions = predictor.predict(sentence)
        print("Token-level predictions:")
        for pred in predictions:
            icon = "🔴" if pred["label"] != "O" else "⚪"
            print(f"   {icon} {pred['token']:20s} → {pred['label']} (conf: {pred['confidence']:.3f})")

        # Italic words
        italic_words = predictor.get_italic_words(sentence)
        print(f"\nWords to italicize: {italic_words}")
        print("-" * 60)


if __name__ == "__main__":
    main()
