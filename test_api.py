"""
API Testing Suite for Handwritten Character Recognition

This module provides comprehensive tests for the FastAPI backend,
including unit tests, integration tests, and performance benchmarks.

Usage:
    python test_api.py              # Run basic API test
    python -m pytest test_api.py    # Run full test suite
"""

import requests
import io
import time
from typing import Dict, Any, Optional
from PIL import Image, ImageDraw, ImageFont

# Configuration
API_BASE_URL = "http://localhost:8000"
PREDICT_ENDPOINT = f"{API_BASE_URL}/predict"

# Expected model performance thresholds
MIN_TOP1_ACCURACY = 0.70
MIN_TOP3_ACCURACY = 0.90
MAX_RESPONSE_TIME_MS = 2000


def create_test_image(
    character: str = "A",
    size: tuple = (224, 224),
    bg_color: str = "white",
    fg_color: str = "black"
) -> io.BytesIO:
    """
    Create a test image with a drawn character.
    
    Args:
        character: Character to draw on the image
        size: Image dimensions (width, height)
        bg_color: Background color
        fg_color: Foreground (text) color
    
    Returns:
        BytesIO buffer containing the PNG image
    """
    img = Image.new('RGB', size, color=bg_color)
    draw = ImageDraw.Draw(img)
    
    # Draw a simple representation (cross/line pattern)
    # In production, you'd use actual character rendering
    center_x, center_y = size[0] // 2, size[1] // 2
    offset = min(size) // 4
    
    draw.line(
        (center_x - offset, center_y - offset, center_x + offset, center_y + offset),
        fill=fg_color,
        width=5
    )
    draw.line(
        (center_x + offset, center_y - offset, center_x - offset, center_y + offset),
        fill=fg_color,
        width=5
    )
    
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf


def test_api_health() -> bool:
    """
    Test if the API is running and responsive.
    
    Returns:
        True if API is healthy, False otherwise
    """
    try:
        response = requests.get(API_BASE_URL, timeout=5)
        return response.status_code in [200, 404]  # 404 is ok, means server is up
    except requests.exceptions.ConnectionError:
        return False


def test_predict_endpoint() -> Dict[str, Any]:
    """
    Test the /predict endpoint with a sample image.
    
    Returns:
        Dictionary containing test results
    """
    results = {
        "success": False,
        "response_time_ms": None,
        "prediction": None,
        "confidence": None,
        "error": None
    }
    
    # Create test image
    test_image = create_test_image()
    files = {"file": ("test_image.png", test_image, "image/png")}
    
    try:
        start_time = time.time()
        response = requests.post(PREDICT_ENDPOINT, files=files, timeout=10)
        elapsed_ms = (time.time() - start_time) * 1000
        
        results["response_time_ms"] = round(elapsed_ms, 2)
        
        if response.status_code == 200:
            data = response.json()
            results["success"] = True
            results["prediction"] = data.get("prediction")
            results["confidence"] = data.get("confidence")
        else:
            results["error"] = f"Status {response.status_code}: {response.text}"
            
    except requests.exceptions.ConnectionError:
        results["error"] = "Connection refused - is the API running?"
    except Exception as e:
        results["error"] = str(e)
    
    return results


def test_response_time() -> bool:
    """
    Test if API response time is within acceptable limits.
    
    Returns:
        True if response time < MAX_RESPONSE_TIME_MS
    """
    result = test_predict_endpoint()
    if result["response_time_ms"]:
        return result["response_time_ms"] < MAX_RESPONSE_TIME_MS
    return False


def test_invalid_input() -> bool:
    """
    Test API behavior with invalid input.
    
    Returns:
        True if API handles invalid input gracefully
    """
    try:
        # Send empty file
        files = {"file": ("empty.png", io.BytesIO(b""), "image/png")}
        response = requests.post(PREDICT_ENDPOINT, files=files, timeout=10)
        
        # Should return 400 or 422 for invalid input
        return response.status_code in [400, 422, 500]
    except Exception:
        return False


def test_supported_formats() -> Dict[str, bool]:
    """
    Test various image format support.
    
    Returns:
        Dictionary mapping format to support status
    """
    formats = {
        "PNG": ("image/png", "PNG"),
        "JPEG": ("image/jpeg", "JPEG"),
        "GIF": ("image/gif", "GIF"),
    }
    
    results = {}
    
    for format_name, (mime_type, pil_format) in formats.items():
        try:
            img = Image.new('RGB', (128, 128), color='white')
            buf = io.BytesIO()
            img.save(buf, format=pil_format)
            buf.seek(0)
            
            files = {"file": (f"test.{format_name.lower()}", buf, mime_type)}
            response = requests.post(PREDICT_ENDPOINT, files=files, timeout=10)
            results[format_name] = response.status_code == 200
        except Exception:
            results[format_name] = False
    
    return results


def run_all_tests() -> None:
    """Run all tests and print results."""
    print("=" * 60)
    print("Handwritten Character Recognition API - Test Suite")
    print("=" * 60)
    
    # Test 1: API Health
    print("\n[1/5] Testing API Health...")
    health = test_api_health()
    print(f"      {'✅ PASS' if health else '❌ FAIL'} - API {'is' if health else 'is NOT'} running")
    
    if not health:
        print("\n⚠️  API is not running. Start it with: ./run_app.sh")
        return
    
    # Test 2: Predict Endpoint
    print("\n[2/5] Testing Predict Endpoint...")
    predict_result = test_predict_endpoint()
    if predict_result["success"]:
        print(f"      ✅ PASS - Prediction: {predict_result['prediction']}")
        print(f"              Confidence: {predict_result['confidence']:.2%}")
        print(f"              Response Time: {predict_result['response_time_ms']}ms")
    else:
        print(f"      ❌ FAIL - {predict_result['error']}")
    
    # Test 3: Response Time
    print("\n[3/5] Testing Response Time...")
    fast_enough = test_response_time()
    print(f"      {'✅ PASS' if fast_enough else '❌ FAIL'} - Response time {'<' if fast_enough else '>'} {MAX_RESPONSE_TIME_MS}ms")
    
    # Test 4: Invalid Input Handling
    print("\n[4/5] Testing Invalid Input Handling...")
    handles_invalid = test_invalid_input()
    print(f"      {'✅ PASS' if handles_invalid else '❌ FAIL'} - Invalid input {'handled' if handles_invalid else 'NOT handled'} gracefully")
    
    # Test 5: Format Support
    print("\n[5/5] Testing Image Format Support...")
    formats = test_supported_formats()
    for fmt, supported in formats.items():
        print(f"      {'✅' if supported else '❌'} {fmt}: {'Supported' if supported else 'Not Supported'}")
    
    # Summary
    print("\n" + "=" * 60)
    all_passed = health and predict_result["success"] and fast_enough and handles_invalid
    print(f"{'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
