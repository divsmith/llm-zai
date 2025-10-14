#!/usr/bin/env python3
"""
Test the corrected API endpoint and model naming.
"""

import llm
import httpx
import json

def test_corrected_format():
    """Test the corrected endpoint and model naming."""

    try:
        api_key = llm.get_key(alias='zai', env='ZAI_API_KEY')
        print(f"🔑 Using API key: {api_key[:10]}...{api_key[-4:]}")
    except Exception as e:
        print(f"❌ No API key: {e}")
        return

    # Test the corrected endpoint
    url = "https://api.z.ai/api/coding/paas/v4/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Test different model name formats
    test_models = [
        "GLM-4.6",
        "GLM-4.5",
        "GLM-4.5-Air",
        "GLM-4.5V",
        "GLM-4-32B"
    ]

    print(f"\n🎯 Testing corrected API endpoint: {url}")
    print(f"📋 Testing {len(test_models)} model name formats...\n")

    for model in test_models:
        print(f"Testing model: {model}")

        data = {
            "model": model,
            "messages": [{"role": "user", "content": "Please respond with just 'API test successful'"}],
            "max_tokens": 50
        }

        try:
            with httpx.Client(timeout=15.0) as client:
                response = client.post(url, headers=headers, json=data)

                print(f"   Status: {response.status_code}")

                if response.status_code == 200:
                    print("   ✅ SUCCESS! This model works!")
                    result = response.json()
                    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                    print(f"   Response: {content[:100]}...")
                    return model

                elif response.status_code == 429:
                    try:
                        error_data = response.json()
                        error_msg = error_data.get("error", {}).get("message", "Rate limit")
                        print(f"   ⚠️  429 - {error_msg}")
                    except:
                        print("   ⚠️  429 - Rate limit")

                elif response.status_code == 401:
                    print("   ❌ 401 - Authentication failed")

                elif response.status_code == 404:
                    print("   ❌ 404 - Model not found")

                elif response.status_code == 400:
                    try:
                        error_data = response.json()
                        error_msg = error_data.get("error", {}).get("message", "Bad request")
                        print(f"   ❌ 400 - {error_msg}")
                    except:
                        print("   ❌ 400 - Bad request")

                else:
                    print(f"   ⚠️  {response.status_code} - {response.text[:50]}...")

        except Exception as e:
            print(f"   ❌ Exception: {e}")

        print()

    return None

def test_plugin_with_corrected_endpoint():
    """Test the plugin with the corrected endpoint."""

    try:
        # Clear module cache
        import sys
        modules_to_clear = [k for k in sys.modules.keys() if 'llm_zai' in k]
        for mod in modules_to_clear:
            del sys.modules[mod]

        # Reload plugin
        import llm
        llm.load_plugins()

        from llm_zai import ZaiChat

        model = ZaiChat('zai-glm-4.6')
        print(f"🔧 Plugin model: {model}")
        print(f"🔧 Plugin endpoint: {model.api_base}")

        # Test model name conversion
        converted_name = model.model_id.replace("zai-", "").upper().replace("-AIR", "-Air")
        print(f"🔧 Converted model name: {converted_name}")

        return True

    except Exception as e:
        print(f"❌ Plugin test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Testing Corrected Z.ai API Implementation")
    print("=" * 60)

    # Test 1: Direct API calls with corrected format
    working_model = test_corrected_format()

    # Test 2: Plugin with corrected endpoint
    plugin_works = test_plugin_with_corrected_endpoint()

    print("\n📊 Results:")
    if working_model:
        print(f"✅ Working model found: {working_model}")
        print("✅ Corrected endpoint and model naming work!")
    else:
        print("❌ No working model found with corrected format")

    if plugin_works:
        print("✅ Plugin updated successfully")
    else:
        print("❌ Plugin update failed")

    print("\n🎯 Next steps:")
    if working_model and plugin_works:
        print("✅ Ready to test with LLM command:")
        print("   ~/.local/bin/llm -m glm-4.6 'Hello, test message'")
    else:
        print("❌ Need further investigation")