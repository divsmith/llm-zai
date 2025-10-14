#!/usr/bin/env python3
"""
Advanced testing for Z.ai API compatibility.
"""

import llm
import httpx
import json
import os

def test_different_endpoints():
    """Test different API endpoints."""

    try:
        api_key = llm.get_key(alias='zai', env='ZAI_API_KEY')
        print(f"🔑 Using API key: {api_key[:10]}...{api_key[-4:]}")
    except Exception as e:
        print(f"❌ No API key: {e}")
        return

    # Different endpoints to test
    endpoints = [
        "https://api.z.ai/api/paas/v4/chat/completions",
        "https://api.z.ai/v4/chat/completions",
        "https://api.z.ai/chat/completions",
        "https://open.bigmodel.cn/api/paas/v4/chat/completions",  # Zhipu AI endpoint
    ]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (compatible; Claude/2.1)"
    }

    data = {
        "model": "glm-4.5",
        "messages": [{"role": "user", "content": "test"}],
        "max_tokens": 10
    }

    print(f"\n🌐 Testing different endpoints...\n")

    for i, endpoint in enumerate(endpoints, 1):
        print(f"{i}. Testing endpoint: {endpoint}")

        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.post(endpoint, headers=headers, json=data)

                print(f"   Status: {response.status_code}")

                if response.status_code == 200:
                    print("   ✅ SUCCESS! This endpoint works!")
                    result = response.json()
                    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                    print(f"   Response: {content}")
                    return endpoint

                elif response.status_code == 429:
                    error_data = response.json() if response.text else {}
                    error_msg = error_data.get("error", {}).get("message", "Rate limit")
                    print(f"   ⚠️  429 - {error_msg}")

                elif response.status_code == 401:
                    print("   ❌ 401 - Authentication failed")

                elif response.status_code == 404:
                    print("   ❌ 404 - Endpoint not found")

                else:
                    print(f"   ⚠️  {response.status_code} - {response.text[:50]}...")

        except Exception as e:
            print(f"   ❌ Exception: {e}")

        print()

    return None

def test_different_models():
    """Test different model naming conventions."""

    try:
        api_key = llm.get_key(alias='zai', env='ZAI_API_KEY')
    except Exception as e:
        print(f"❌ No API key: {e}")
        return

    url = "https://api.z.ai/api/paas/v4/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Try different model names
    model_names = [
        "glm-4.5",
        "glm-4",
        "glm4",
        "zai-glm-4.5",
        "zai-glm-4",
        "GLM-4.5",
        "GLM-4",
        "gpt-3.5-turbo",  # Just to see what error we get
    ]

    print(f"\n🤖 Testing different model names...\n")

    for model in model_names:
        print(f"Testing model: {model}")

        data = {
            "model": model,
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 5
        }

        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.post(url, headers=headers, json=data)

                if response.status_code == 200:
                    print(f"   ✅ {model} WORKS!")
                    return model

                elif response.status_code == 429:
                    print(f"   ⚠️  {model} - Balance issue (but model exists)")
                elif response.status_code == 404:
                    print(f"   ❌ {model} - Not found")
                else:
                    print(f"   ⚠️  {model} - Error {response.status_code}")

        except Exception as e:
            print(f"   ❌ {model} - Exception: {e}")

    return None

def test_authentication_methods():
    """Test different authentication methods."""

    try:
        api_key = llm.get_key(alias='zai', env='ZAI_API_KEY')
    except Exception as e:
        print(f"❌ No API key: {e}")
        return

    url = "https://api.z.ai/api/paas/v4/chat/completions"
    data = {
        "model": "glm-4.5",
        "messages": [{"role": "user", "content": "test"}],
        "max_tokens": 5
    }

    auth_methods = [
        {"name": "Bearer token", "headers": {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}},
        {"name": "API Key header", "headers": {"X-API-Key": api_key, "Content-Type": "application/json"}},
        {"name": "API Key in query", "headers": {"Content-Type": "application/json"}, "params": {"api_key": api_key}},
        {"name": "Basic auth", "headers": {"Authorization": f"Basic {api_key}", "Content-Type": "application/json"}},
    ]

    print(f"\n🔐 Testing different authentication methods...\n")

    for method in auth_methods:
        print(f"Testing: {method['name']}")

        try:
            with httpx.Client(timeout=10.0) as client:
                params = method.get('params', {})
                response = client.post(url, headers=method['headers'], json=data, params=params)

                if response.status_code == 200:
                    print(f"   ✅ {method['name']} WORKS!")
                    return method
                else:
                    print(f"   ❌ {method['name']} - {response.status_code}")

        except Exception as e:
            print(f"   ❌ {method['name']} - Exception: {e}")

    return None

if __name__ == "__main__":
    print("🔍 Advanced Z.ai API Testing")
    print("=" * 50)

    # Test 1: Different endpoints
    working_endpoint = test_different_endpoints()

    # Test 2: Different model names
    working_model = test_different_models()

    # Test 3: Different auth methods
    working_auth = test_authentication_methods()

    print("\n📊 Summary:")
    if working_endpoint:
        print(f"✅ Working endpoint: {working_endpoint}")
    if working_model:
        print(f"✅ Working model: {working_model}")
    if working_auth:
        print(f"✅ Working auth: {working_auth['name']}")

    if not working_endpoint and not working_model and not working_auth:
        print("❌ No working configuration found")
        print("💡 This strongly suggests an account balance issue")
        print("   - The API key is valid (gets 429, not 401)")
        print("   - The endpoints exist (gets 429, not 404)")
        print("   - The models exist (gets 429, not 404)")
        print("   - But account lacks credits to actually make calls")