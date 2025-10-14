#!/usr/bin/env python3
"""
Manual API testing script for Z.ai API.
This helps test the API directly without going through LLM.
"""

import os
import json
import sys
import httpx

def test_api_directly(api_key=None):
    """Test Z.ai API directly with provided key."""

    # Get API key from various sources
    if not api_key:
        api_key = os.environ.get('ZAI_API_KEY')

    if not api_key:
        try:
            import llm
            api_key = llm.get_key(alias='zai', env='ZAI_API_KEY')
        except:
            pass

    if not api_key:
        print("❌ No API key found.")
        print("Please either:")
        print("1. Set ZAI_API_KEY environment variable")
        print("2. Pass key as argument: python3 test_api_manually.py YOUR_KEY")
        print("3. Set with: llm keys set zai")
        return False

    print(f"🔑 Testing with API key (length: {len(api_key)})")

    # Prepare request
    url = "https://api.z.ai/api/paas/v4/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "glm-4.6",
        "messages": [
            {
                "role": "user",
                "content": "Please respond with just 'API test successful'"
            }
        ],
        "temperature": 0.1,
        "max_tokens": 50
    }

    print(f"📡 Making request to: {url}")
    print(f"📤 Request data: {json.dumps(data, indent=2)}")

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(url, headers=headers, json=data)

            print(f"📥 Response status: {response.status_code}")
            print(f"📥 Response headers: {dict(response.headers)}")

            if response.status_code == 200:
                result = response.json()
                print("✅ API call successful!")
                print(f"📥 Response: {json.dumps(result, indent=2)}")
                return True
            else:
                print(f"❌ API call failed with status {response.status_code}")
                print(f"📥 Response body: {response.text}")

                # Specific error analysis
                if response.status_code == 401:
                    print("💡 This suggests: Invalid or expired API key")
                elif response.status_code == 429:
                    print("💡 This suggests: Rate limit exceeded")
                elif response.status_code == 404:
                    print("💡 This suggests: Model not found or endpoint changed")
                elif response.status_code >= 500:
                    print("💡 This suggests: Z.ai server error")

                return False

    except httpx.RequestError as e:
        print(f"❌ Network error: {e}")
        print("💡 This suggests: Connectivity issues or DNS problems")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_different_models(api_key):
    """Test different model names to see what works."""
    models_to_test = ["glm-4.6", "glm-4.5", "glm-4.5-air", "glm-4.5v", "glm-4-32b"]

    print(f"\n🔍 Testing different model names...")

    for model in models_to_test:
        print(f"\nTesting model: {model}")

        url = "https://api.z.ai/api/paas/v4/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": "test"
                }
            ],
            "max_tokens": 10
        }

        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.post(url, headers=headers, json=data)

                if response.status_code == 200:
                    print(f"✅ {model} - WORKS")
                elif response.status_code == 404:
                    print(f"❌ {model} - Not found")
                elif response.status_code == 401:
                    print(f"❌ {model} - Auth error (same for all models)")
                    break  # No point testing more if auth fails
                else:
                    print(f"⚠️  {model} - Error {response.status_code}: {response.text[:50]}...")

        except Exception as e:
            print(f"❌ {model} - Exception: {e}")

if __name__ == "__main__":
    api_key = None
    if len(sys.argv) > 1:
        api_key = sys.argv[1]
        print("🔑 Using API key from command line argument")

    success = test_api_directly(api_key)

    if success:
        print("\n🎉 Basic API test passed! Testing different models...")
        # Get the key again for model testing
        if not api_key:
            try:
                import llm
                api_key = llm.get_key(alias='zai', env='ZAI_API_KEY')
            except:
                api_key = os.environ.get('ZAI_API_KEY')

        if api_key:
            test_different_models(api_key)

    print("\n📋 Summary:")
    print("- If you get 401 errors: Check your API key")
    print("- If you get 404 errors: Model names may have changed")
    print("- If you get network errors: Check internet connection")
    print("- If everything works: The issue might be in the LLM plugin integration")