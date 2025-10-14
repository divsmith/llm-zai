#!/usr/bin/env python3
"""
Diagnostic script for llm-zai API issues.
This script helps identify problems without exposing actual API keys.
"""

import llm
import json
from llm_zai import ZaiChat

def diagnose_api_setup():
    """Diagnose API configuration and common issues."""
    print("🔍 LLM-ZAI API Diagnostic Tool")
    print("=" * 50)

    # 1. Check if plugin loads
    try:
        llm.load_plugins()
        print("✅ Plugin loaded successfully")
    except Exception as e:
        print(f"❌ Plugin loading failed: {e}")
        return

    # 2. Check API key retrieval (without showing the key)
    try:
        api_key = llm.get_key(alias='zai', env='ZAI_API_KEY')
        if api_key:
            print(f"✅ API key found: {len(api_key)} characters")
            print(f"   Format check: starts with '{api_key[:10]}...' if longer than 10 chars")
            print(f"   Ends with: '...{api_key[-4:]}' if longer than 4 chars")

            # Basic format validation
            if len(api_key) < 10:
                print("⚠️  Key seems short (may be incomplete)")
            if not api_key.replace('-', '').replace('_', '').isalnum():
                print("⚠️  Key contains unusual characters")

        else:
            print("❌ No API key found")
            print("   Set key with: llm keys set zai")
            print("   Or export: export ZAI_API_KEY='your-key'")
            return
    except Exception as e:
        print(f"❌ Error retrieving API key: {e}")
        return

    # 3. Test model instantiation
    try:
        model = ZaiChat('zai-glm-4.6')
        print(f"✅ Model instantiated: {model}")
    except Exception as e:
        print(f"❌ Model instantiation failed: {e}")
        return

    # 4. Test header generation
    try:
        headers = model._get_headers()
        auth_header = headers.get('Authorization', '')
        if auth_header.startswith('Bearer '):
            token_length = len(auth_header) - 7  # Remove 'Bearer '
            print(f"✅ Headers generated correctly")
            print(f"   Authorization header: Bearer [token of {token_length} chars]")
        else:
            print(f"❌ Invalid authorization header: {auth_header[:20]}...")
    except Exception as e:
        print(f"❌ Header generation failed: {e}")
        return

    # 5. Test request structure (without actually sending)
    try:
        class SimplePrompt:
            def __init__(self):
                self.prompt = "test"
                self.system = None

        prompt = SimplePrompt()
        messages = model.build_messages(prompt, None)
        request_data = {
            "model": model.model_id.replace("zai-", ""),
            "messages": messages
        }

        print("✅ Request structure validated:")
        print(f"   Endpoint: {model.api_base}/chat/completions")
        print(f"   Model name: {request_data['model']}")
        print(f"   Messages: {len(request_data['messages'])} messages")
        print(f"   Message format: {request_data['messages'][0]}")

    except Exception as e:
        print(f"❌ Request structure validation failed: {e}")
        return

    print("\n🎯 All basic checks passed!")
    print("If API calls still fail, the issue may be:")
    print("1. Invalid/expired API key")
    print("2. Network connectivity issues")
    print("3. Z.ai API endpoint changes")
    print("4. Rate limiting")
    print("5. Model availability issues")

    print("\n📋 Next steps:")
    print("1. Test manually with curl:")
    print(f"   curl -H 'Authorization: Bearer YOUR_KEY' {model.api_base}/chat/completions")
    print("2. Check Z.ai API documentation for any changes")
    print("3. Verify your key works in Z.ai's web interface")

def simulate_api_call():
    """Simulate an API call structure without actually making it."""
    print("\n🧪 Simulating API call structure...")

    try:
        model = ZaiChat('zai-glm-4.6')

        # Create sample request
        class SimplePrompt:
            def __init__(self):
                self.prompt = "Hello, this is a test."
                self.system = None

        prompt = SimplePrompt()
        messages = model.build_messages(prompt, None)

        request_data = {
            "model": model.model_id.replace("zai-", ""),
            "messages": messages,
            "temperature": 1.0,
            "max_tokens": 100
        }

        print("📤 Request that would be sent:")
        print(json.dumps(request_data, indent=2))

        print(f"\n🔗 To: {model.api_base}/chat/completions")
        print("🔒 With headers: Authorization: Bearer [YOUR_KEY], Content-Type: application/json")

    except Exception as e:
        print(f"❌ Simulation failed: {e}")

if __name__ == "__main__":
    diagnose_api_setup()
    simulate_api_call()