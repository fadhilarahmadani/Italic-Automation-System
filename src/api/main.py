# src/api/main.py
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from api.models import (
    DetectRequest,
    DetectResponse,
    BatchDetectRequest,
    BatchDetectResponse,
    ParagraphResult,
    ItalicWord,
    HealthResponse,
    ErrorResponse
)
from api.predictor import get_predictor_service
from utils import setup_logging

# Setup logging
logger = setup_logging(__name__, os.getenv("LOG_LEVEL", "INFO"))

# Load predictor on startup/shutdown
predictor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    global predictor
    # Startup
    logger.info("="*60)
    logger.info("Starting Italic Automation API")
    logger.info("="*60)
    try:
        predictor = get_predictor_service()
        logger.info("API ready to serve requests")
    except Exception as e:
        logger.error(f"Failed to start API: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down API")

# Initialize FastAPI app
app = FastAPI(
    title="Italic Automation API",
    description="IndoBERT-based automatic italic detection for Indonesian text (PUEBI compliant)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Get CORS origins from environment variable
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,https://localhost:3000").split(",")

# CORS middleware - Allow Word Add-in to access API
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Italic Automation API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint
    
    Returns model status and system information
    """
    model_info = predictor.get_model_info()
    
    return HealthResponse(
        status="healthy" if predictor.is_loaded else "unhealthy",
        model_loaded=predictor.is_loaded,
        model_name=model_info["model_name"],
        device=model_info["device"],
        version="1.0.0"
    )


@app.post(
    "/api/detect",
    response_model=DetectResponse,
    tags=["Detection"],
    summary="Detect italic words in text",
    description="Analyze text and detect foreign words that should be italicized according to PUEBI"
)
async def detect_italic(request: DetectRequest):
    """
    Detect words that need italic formatting
    
    - **text**: Text to analyze (Indonesian)
    - **confidence_threshold**: Minimum confidence (0.0-1.0)
    
    Returns list of detected words with positions and confidence scores
    """
    try:
        # Get predictions
        italic_phrases, processing_time = predictor.extract_italic_phrases(
            text=request.text,
            confidence_threshold=request.confidence_threshold
        )
        
        # Convert to response model
        italic_words = [
            ItalicWord(
                word=phrase["word"],
                start_pos=phrase["start_pos"],
                end_pos=phrase["end_pos"],
                confidence=phrase["confidence"],
                label=phrase["label"]
            )
            for phrase in italic_phrases
        ]
        
        return DetectResponse(
            success=True,
            text=request.text,
            italic_words=italic_words,
            total_detected=len(italic_words),
            processing_time=processing_time,
            model_info=predictor.get_model_info()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during detection: {str(e)}"
        )


@app.post(
    "/api/batch-detect",
    response_model=BatchDetectResponse,
    tags=["Detection"],
    summary="Batch detect italic words in multiple paragraphs"
)
async def batch_detect(request: BatchDetectRequest):
    """
    Detect italic words in multiple paragraphs
    
    Useful for processing entire documents
    """
    try:
        start_time = time.time()
        results = []
        total_words = 0
        
        for idx, paragraph in enumerate(request.paragraphs):
            if not paragraph.strip():
                continue
            
            # Detect in this paragraph
            italic_phrases, _ = predictor.extract_italic_phrases(
                text=paragraph,
                confidence_threshold=request.confidence_threshold
            )
            
            # Convert to response model
            italic_words = [
                ItalicWord(
                    word=phrase["word"],
                    start_pos=phrase["start_pos"],
                    end_pos=phrase["end_pos"],
                    confidence=phrase["confidence"],
                    label=phrase["label"]
                )
                for phrase in italic_phrases
            ]
            
            results.append(ParagraphResult(
                paragraph_index=idx,
                text=paragraph,
                italic_words=italic_words,
                word_count=len(italic_words)
            ))
            
            total_words += len(italic_words)
        
        processing_time = time.time() - start_time
        
        return BatchDetectResponse(
            success=True,
            results=results,
            total_paragraphs=len(request.paragraphs),
            total_words_detected=total_words,
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during batch detection: {str(e)}"
        )


@app.get("/api/model-info", tags=["Info"])
async def get_model_info():
    """Get model information"""
    return predictor.get_model_info()


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc)
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=os.getenv("API_RELOAD", "true").lower() == "true",
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )
