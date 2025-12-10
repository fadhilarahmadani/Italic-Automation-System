# src/api/models.py
from pydantic import BaseModel, Field, validator
from typing import List, Optional

class ItalicWord(BaseModel):
    """Model untuk kata yang perlu di-italic"""
    word: str
    start_pos: int
    end_pos: int
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0-1")
    label: str = Field(description="BIO label: B or I")
    
    class Config:
        json_schema_extra = {
            "example": {
                "word": "machine learning",
                "start_pos": 20,
                "end_pos": 36,
                "confidence": 0.95,
                "label": "B-I"
            }
        }


class DetectRequest(BaseModel):
    """Request untuk deteksi italic"""
    text: str = Field(
        ..., 
        min_length=1, 
        max_length=10000,
        description="Teks yang akan dianalisis"
    )
    confidence_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum confidence untuk deteksi (0-1)"
    )
    
    @validator('text')
    def text_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Text tidak boleh kosong')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Penelitian ini menggunakan machine learning dan deep learning untuk analisis data",
                "confidence_threshold": 0.8
            }
        }


class DetectResponse(BaseModel):
    """Response hasil deteksi"""
    success: bool
    text: str
    italic_words: List[ItalicWord]
    total_detected: int
    processing_time: float = Field(description="Waktu processing dalam detik")
    model_info: Optional[dict] = None
    
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "text": "Penelitian ini menggunakan machine learning...",
                "italic_words": [
                    {
                        "word": "machine learning",
                        "start_pos": 24,
                        "end_pos": 40,
                        "confidence": 0.95,
                        "label": "B-I"
                    }
                ],
                "total_detected": 1,
                "processing_time": 0.123
            }
        }


class BatchDetectRequest(BaseModel):
    """Request untuk batch detection (multiple paragraphs)"""
    paragraphs: List[str] = Field(
        ...,
        min_items=1,
        max_items=100,
        description="List paragraf untuk dianalisis"
    )
    confidence_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "paragraphs": [
                    "Paragraph pertama dengan machine learning",
                    "Paragraph kedua dengan deep learning"
                ],
                "confidence_threshold": 0.8
            }
        }


class ParagraphResult(BaseModel):
    """Hasil deteksi untuk satu paragraph"""
    paragraph_index: int
    text: str
    italic_words: List[ItalicWord]
    word_count: int


class BatchDetectResponse(BaseModel):
    """Response untuk batch detection"""
    success: bool
    results: List[ParagraphResult]
    total_paragraphs: int
    total_words_detected: int
    processing_time: float


class HealthResponse(BaseModel):
    """Response untuk health check"""
    status: str
    model_loaded: bool
    model_name: str
    device: str
    version: str



class ErrorResponse(BaseModel):
    """Response untuk error"""
    success: bool = False
    error: str
    detail: Optional[str] = None
