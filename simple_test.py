#!/usr/bin/env python3
"""
Test the simplest possible response creation.
"""

import llm
import httpx

def test_minimal_response():
    """Test creating the most minimal response possible."""

    try:
        api_key = llm.get_key(alias='zai', env='ZAI_API_KEY')
    except Exception as e:
        print(f"❌ No API key: {e}")
        return

    # Load plugin
    llm.load_plugins()
    from llm_zai import ZaiChat

    model = ZaiChat('zai-glm-4.6')
    print(f"✅ Model loaded: {model}")

    # Make API call directly
    url = "https://api.z.ai/api/coding/paas/v4/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    data = {
        "model": "GLM-4.6",
        "messages": [{"role": "user", "content": "Just say hello"}],
        "max_tokens": 10
    }

    print("🔧 Making API call...")
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.post(url, headers=headers, json=data)

            if response.status_code == 200:
                result = response.json()
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                print(f"✅ API success! Content: '{content}'")
                return content
            else:
                print(f"❌ API failed: {response.status_code} - {response.text}")
                return None
    except Exception as e:
        print(f"❌ API exception: {e}")
        return None

def test_response_creation_ways(content):
    """Test different ways to create responses."""

    llm.load_plugins()
    from llm_zai import ZaiChat

    model = ZaiChat('zai-glm-4.6')

    # Create a proper prompt
    prompt = llm.Prompt("test", model=model)
    print(f"✅ Prompt created: {prompt}")

    # Test different response creation approaches
    approaches = [
        ("Direct Response with content", lambda: create_simple_response(prompt, model, content)),
        ("Response with text property", lambda: create_text_response(prompt, model, content)),
        ("Minimal response", lambda: create_minimal_response(prompt, model)),
    ]

    for name, method in approaches:
        print(f"\n🔧 Testing: {name}")
        try:
            response = method()
            print(f"   ✅ Response created: {type(response)}")

            # Try to get text from it
            try:
                text = str(response)
                print(f"   ✅ String conversion: '{text[:50]}...'")
            except Exception as e:
                print(f"   ❌ String conversion failed: {e}")

        except Exception as e:
            print(f"   ❌ Creation failed: {e}")
            import traceback
            traceback.print_exc()

def create_simple_response(prompt, model, content):
    """Create a simple response."""
    response = llm.Response(prompt=prompt, model=model, stream=False)
    response._content = content
    return response

def create_text_response(prompt, model, content):
    """Create response with text property."""
    response = llm.Response(prompt=prompt, model=model, stream=False)
    response.text = content
    response._content = content
    return response

def create_minimal_response(prompt, model):
    """Create minimal response."""
    return llm.Response(prompt=prompt, model=model, stream=False)

if __name__ == "__main__":
    print("🔍 Testing Minimal Response Creation")
    print("=" * 50)

    content = test_minimal_response()

    if content:
        test_response_creation_ways(content)
    else:
        print("❌ Could not get API content to test with")