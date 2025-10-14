#!/usr/bin/env python3
"""
Test different header combinations to match working tools.
"""

import llm
import httpx
import json

def test_different_headers():
    """Test various header combinations."""

    # Get API key
    try:
        api_key = llm.get_key(alias='zai', env='ZAI_API_KEY')
        print(f"🔑 Using API key: {api_key[:10]}...{api_key[-4:]}")
    except Exception as e:
        print(f"❌ No API key: {e}")
        return

    url = "https://api.z.ai/api/paas/v4/chat/completions"

    # Base request data
    data = {
        "model": "glm-4.5",
        "messages": [
            {
                "role": "user",
                "content": "Please respond with just 'test successful'"
            }
        ],
        "temperature": 0.1,
        "max_tokens": 50
    }

    # Different header combinations to test
    header_combinations = [
        {
            "name": "Current plugin headers",
            "headers": {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        },
        {
            "name": "With User-Agent (browser-like)",
            "headers": {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            }
        },
        {
            "name": "With User-Agent (curl-like)",
            "headers": {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "curl/8.1.2"
            }
        },
        {
            "name": "With User-Agent (Python httpx)",
            "headers": {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "python-httpx/0.28.1"
            }
        },
        {
            "name": "With additional headers",
            "headers": {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "python-httpx/0.28.1",
                "Accept": "application/json",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive"
            }
        },
        {
            "name": "Claude-like headers",
            "headers": {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0 (compatible; Claude/2.1)",
                "Accept": "application/json",
                "Accept-Encoding": "gzip, deflate, br",
                "Accept-Language": "en-US,en;q=0.9",
                "Connection": "keep-alive",
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "cross-site"
            }
        },
        {
            "name": "With Origin header",
            "headers": {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                "Accept": "application/json",
                "Origin": "https://z.ai",
                "Referer": "https://z.ai/"
            }
        }
    ]

    print(f"\n🧪 Testing {len(header_combinations)} different header combinations...\n")

    for i, combo in enumerate(header_combinations, 1):
        print(f"{i}. Testing: {combo['name']}")
        print(f"   Headers: {list(combo['headers'].keys())}")

        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.post(url, headers=combo['headers'], json=data)

                print(f"   Status: {response.status_code}")

                if response.status_code == 200:
                    print("   ✅ SUCCESS! This header combination works!")
                    result = response.json()
                    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                    print(f"   Response: {content[:100]}...")
                    return combo['headers']  # Return working headers

                elif response.status_code == 429:
                    error_data = response.json()
                    error_msg = error_data.get("error", {}).get("message", "Unknown")
                    print(f"   ⚠️  429 - {error_msg}")

                elif response.status_code == 401:
                    print("   ❌ 401 - Authentication failed")

                elif response.status_code == 404:
                    print("   ❌ 404 - Not found")

                else:
                    print(f"   ⚠️  {response.status_code} - {response.text[:50]}...")

        except Exception as e:
            print(f"   ❌ Exception: {e}")

        print()

    print("🔍 No header combination worked. The issue might be:")
    print("   - Account balance/recharge required")
    print("   - Different request payload format")
    print("   - Different API endpoint required")
    print("   - Additional authentication needed")

    return None

def test_different_payloads(api_key, working_headers):
    """Test different payload formats."""

    url = "https://api.z.ai/api/paas/v4/chat/completions"

    payload_variations = [
        {
            "name": "Current payload",
            "data": {
                "model": "glm-4.5",
                "messages": [{"role": "user", "content": "test"}],
                "temperature": 0.1,
                "max_tokens": 50
            }
        },
        {
            "name": "Without temperature",
            "data": {
                "model": "glm-4.5",
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 50
            }
        },
        {
            "name": "Minimal payload",
            "data": {
                "model": "glm-4.5",
                "messages": [{"role": "user", "content": "test"}]
            }
        },
        {
            "name": "With stream=false",
            "data": {
                "model": "glm-4.5",
                "messages": [{"role": "user", "content": "test"}],
                "stream": False
            }
        },
        {
            "name": "Different model format",
            "data": {
                "model": "zai-glm-4.5",  # With zai- prefix
                "messages": [{"role": "user", "content": "test"}]
            }
        }
    ]

    print(f"\n🎯 Testing different payload formats with working headers...\n")

    for i, payload in enumerate(payload_variations, 1):
        print(f"{i}. Testing: {payload['name']}")
        print(f"   Data: {payload['data']}")

        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.post(url, headers=working_headers, json=payload['data'])

                print(f"   Status: {response.status_code}")

                if response.status_code == 200:
                    print("   ✅ SUCCESS! This payload format works!")
                    return payload['data']

                else:
                    error_text = response.text[:100] if response.text else "No response body"
                    print(f"   ❌ Failed: {error_text}")

        except Exception as e:
            print(f"   ❌ Exception: {e}")

        print()

    return None

if __name__ == "__main__":
    working_headers = test_different_headers()

    if working_headers:
        print("\n🎯 Found working headers! Testing payload variations...")
        import llm
        api_key = llm.get_key(alias='zai', env='ZAI_API_KEY')
        working_payload = test_different_payloads(api_key, working_headers)

        if working_payload:
            print("\n🎉 SUCCESS! Found working configuration:")
            print(f"Headers: {working_headers}")
            print(f"Payload: {working_payload}")
        else:
            print("\n⚠️  Headers work but need payload adjustments")
    else:
        print("\n❌ No header combination worked")