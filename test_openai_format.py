#!/usr/bin/env python3
"""
Test OpenAI protocol compliance and different request formats.
"""

import llm
import httpx
import json

def test_openai_format_variations():
    """Test different OpenAI-compatible request formats."""

    try:
        api_key = llm.get_key(alias='zai', env='ZAI_API_KEY')
        print(f"🔑 Using API key: {api_key[:10]}...{api_key[-4:]}")
    except Exception as e:
        print(f"❌ No API key: {e}")
        return

    url = "https://api.z.ai/api/paas/v4/chat/completions"

    # Test different OpenAI-compatible request formats
    test_formats = [
        {
            "name": "Strict OpenAI format",
            "headers": {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "OpenAI/Python"
            },
            "data": {
                "model": "glm-4.5",
                "messages": [{"role": "user", "content": "Say 'test successful'"}],
                "temperature": 1.0,
                "max_tokens": 50,
                "stream": False
            }
        },
        {
            "name": "Minimal OpenAI format",
            "headers": {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            "data": {
                "model": "glm-4.5",
                "messages": [{"role": "user", "content": "test"}]
            }
        },
        {
            "name": "With OpenAI-specific fields",
            "headers": {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "OpenAI/Python v1"
            },
            "data": {
                "model": "glm-4.5",
                "messages": [{"role": "user", "content": "test"}],
                "temperature": 1.0,
                "max_tokens": 50,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "stream": False
            }
        },
        {
            "name": "Different model naming",
            "headers": {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            "data": {
                "model": "gpt-3.5-turbo",  # OpenAI model name to see how API responds
                "messages": [{"role": "user", "content": "test"}]
            }
        },
        {
            "name": "Claude-style format",
            "headers": {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "Claude/2.1",
                "Accept": "application/json"
            },
            "data": {
                "model": "glm-4.5",
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 4096
            }
        }
    ]

    print(f"\n🧪 Testing {len(test_formats)} different request formats...\n")

    for i, format_test in enumerate(test_formats, 1):
        print(f"{i}. Testing: {format_test['name']}")
        print(f"   Headers: {format_test['headers']}")
        print(f"   Data: {json.dumps(format_test['data'], indent=6)}")

        try:
            with httpx.Client(timeout=15.0) as client:
                response = client.post(url, headers=format_test['headers'], json=format_test['data'])

                print(f"   Status: {response.status_code}")

                if response.status_code == 200:
                    print("   ✅ SUCCESS! This format works!")
                    result = response.json()
                    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                    print(f"   Response: {content[:100]}...")
                    return format_test

                elif response.status_code == 429:
                    try:
                        error_data = response.json()
                        error_msg = error_data.get("error", {}).get("message", "Rate limit")
                        print(f"   ⚠️  429 - {error_msg}")
                        if "balance" in error_msg.lower() or "recharge" in error_msg.lower():
                            print("   💡 This is a balance issue, not a format issue!")
                    except:
                        print("   ⚠️  429 - Rate limit or balance issue")

                elif response.status_code == 401:
                    print("   ❌ 401 - Authentication failed")

                elif response.status_code == 400:
                    print("   ❌ 400 - Bad request. This format might be incompatible.")
                    try:
                        error_data = response.json()
                        print(f"   Error: {error_data}")
                    except:
                        print(f"   Error text: {response.text[:100]}")

                else:
                    print(f"   ⚠️  {response.status_code} - {response.text[:100]}...")

        except Exception as e:
            print(f"   ❌ Exception: {e}")

        print()

    return None

def test_response_format():
    """Test if we need to handle response format differently."""

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

    data = {
        "model": "glm-4.5",
        "messages": [{"role": "user", "content": "test"}],
        "max_tokens": 10
    }

    print(f"\n📥 Testing response format...\n")

    try:
        with httpx.Client(timeout=15.0) as client:
            response = client.post(url, headers=headers, json=data)

            print(f"Status: {response.status_code}")
            print(f"Headers: {dict(response.headers)}")

            if response.text:
                print(f"Raw response: {response.text}")

                try:
                    response_json = response.json()
                    print(f"Parsed JSON: {json.dumps(response_json, indent=2)}")
                except:
                    print("Response is not valid JSON")
            else:
                print("No response body")

    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    print("🔍 OpenAI Protocol Compliance Testing")
    print("=" * 50)

    # Test different formats
    working_format = test_openai_format_variations()

    # Test response format
    test_response_format()

    print("\n📊 Analysis:")
    if working_format:
        print(f"✅ Found working format: {working_format['name']}")
        print("💡 We should update the plugin to use this format")
    else:
        print("❌ No format worked completely")
        print("💡 If we get 429 errors with balance messages, the format might be correct")
        print("   but the account needs to be recharged")

    print("\n🔧 Recommended next steps:")
    print("1. If balance issues: Recharge Z.ai account")
    print("2. If format issues: Update plugin to use working format")
    print("3. Test again with recharged account")