"""
Simple test script for the Literature Review API
"""
import requests
import json

def test_health():
    """Test health check endpoint"""
    print("Testing /health endpoint...")
    response = requests.get("http://localhost:3000/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_literature_review():
    """Test literature review endpoint"""
    print("Testing /literature_review endpoint...")
    
    url = "http://localhost:3000/literature_review"
    payload = {
        "query": "What are the latest advances in transformer models?"
    }
    
    print(f"Request: {json.dumps(payload, indent=2)}")
    print("Streaming response:")
    print("-" * 80)
    
    response = requests.post(url, json=payload, stream=True)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return
    
    # Process streaming response
    content = ""
    for line in response.iter_lines():
        if line:
            line_str = line.decode('utf-8')
            if line_str.startswith('data: '):
                data_str = line_str[6:]  # Remove 'data: ' prefix
                if data_str == '[DONE]':
                    print("\n[DONE]")
                    break
                try:
                    data = json.loads(data_str)
                    if 'choices' in data and len(data['choices']) > 0:
                        delta = data['choices'][0].get('delta', {})
                        if 'content' in delta:
                            chunk = delta['content']
                            print(chunk, end='', flush=True)
                            content += chunk
                    elif 'error' in data:
                        # Error message
                        error_msg = data.get('error', {}).get('message', 'Unknown error')
                        print(f"\n[ERROR] {error_msg}")
                except json.JSONDecodeError:
                    pass
    
    print("\n" + "-" * 80)
    print(f"Total content length: {len(content)} characters")
    print()

if __name__ == "__main__":
    print("=" * 80)
    print("Literature Review API Test")
    print("=" * 80)
    print()
    
    try:
        test_health()
        test_literature_review()
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API. Make sure the server is running on http://localhost:3000")
    except Exception as e:
        print(f"Error: {e}")

