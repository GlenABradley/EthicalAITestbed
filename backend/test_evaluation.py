import requests
import json
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class TestRequest(BaseModel):
    text: str
    context: Optional[Dict[str, Any]] = None
    tau_slider: Optional[float] = None

def test_evaluation(text: str, base_url: str = "http://localhost:8001"):
    """Test the evaluation endpoint with the given text"""
    url = f"{base_url}/api/evaluate"
    payload = {
        "text": text,
        "context": {},
        "tau_slider": None
    }
    
    try:
        # Validate the request model first
        request = TestRequest(**payload)
        
        # Send the request
        response = requests.post(url, json=request.dict())
        response.raise_for_status()
        
        # Try to parse the response
        try:
            result = response.json()
            print(f"✅ Success! Response: {json.dumps(result, indent=2)}")
            return True, result
        except json.JSONDecodeError:
            print(f"❌ Invalid JSON response. Status: {response.status_code}, Content: {response.text}")
            return False, response.text
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {str(e)}")
        return False, str(e)
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False, str(e)

if __name__ == "__main__":
    # Test with the problematic text
    test_texts = [
        "OK, but this is an example again of you doing exactly the opposite of what the fuck I just asked you...",
        "Well, things are looking up and like Nebuchadnezzar at the end of his time this might actually work..."
    ]
    
    for text in test_texts:
        print(f"\n🔍 Testing text: {text[:50]}...")
        success, result = test_evaluation(text)
        
        if not success:
            print("Test failed. Please check the error message above.")
            break
        
        print("Test passed!")
    else:
        print("\n🎉 All tests completed successfully!")
