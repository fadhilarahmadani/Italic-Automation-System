# tests/test_api.py
import requests
import json

API_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("🔍 Testing /health endpoint...")
    response = requests.get(f"{API_URL}/health")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_detect():
    """Test detect endpoint"""
    print("🔍 Testing /api/detect endpoint...")
    
    data = {
        "text": "Penelitian ini menggunakan machine learning dan deep learning untuk analisis data",
        "confidence_threshold": 0.8
    }
    
    response = requests.post(f"{API_URL}/api/detect", json=data)
    print(f"   Status: {response.status_code}")
    result = response.json()
    print(f"   Success: {result['success']}")
    print(f"   Total detected: {result['total_detected']}")
    print(f"   Processing time: {result['processing_time']:.3f}s")
    print(f"   Italic words:")
    for word in result['italic_words']:
        print(f"      - {word['word']} (confidence: {word['confidence']:.3f})")
    print()

def test_batch_detect():
    """Test batch detect endpoint"""
    print("🔍 Testing /api/batch-detect endpoint...")
    
    data = {
        "paragraphs": [
            "Untuk memastikan aksesibilitas, kami menjalankan back-end sementara dokumentasi mengikuti responsive design secara berkala",
            "Di sisi klien, user experience berjalan berdampingan dengan front-end guna menekan waktu muat"
        ],
        "confidence_threshold": 0.8
    }
    
    response = requests.post(f"{API_URL}/api/batch-detect", json=data)
    print(f"   Status: {response.status_code}")
    result = response.json()
    print(f"   Success: {result['success']}")
    print(f"   Total paragraphs: {result['total_paragraphs']}")
    print(f"   Total words detected: {result['total_words_detected']}")
    print(f"   Processing time: {result['processing_time']:.3f}s")
    print()

if __name__ == "__main__":
    print("="*60)
    print("🧪 Testing Italic Automation API")
    print("="*60)
    print()
    
    test_health()
    test_detect()
    test_batch_detect()
    
    print("="*60)
    print("✅ All tests completed!")
    print("="*60)
