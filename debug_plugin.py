#!/usr/bin/env python3
"""
Debug the plugin hanging issue.
"""

import llm
import httpx
import json
import time

def debug_plugin_execution():
    """Debug what happens during plugin execution."""

    try:
        api_key = llm.get_key(alias='zai', env='ZAI_API_KEY')
        print(f"🔑 API key: {api_key[:10]}...")
    except Exception as e:
        print(f"❌ No API key: {e}")
        return

    # Load the plugin
    llm.load_plugins()
    from llm_zai import ZaiChat

    print("🔧 Creating model instance...")
    model = ZaiChat('zai-glm-4.6')
    print(f"   Model: {model}")
    print(f"   Endpoint: {model.api_base}")

    print("🔧 Building messages...")

    # Create a simple prompt
    class SimplePrompt:
        def __init__(self, text):
            self.prompt = text
            self.system = None

    prompt = SimplePrompt("Just say 'hello'")
    messages = model.build_messages(prompt, None)
    print(f"   Messages: {messages}")

    print("🔧 Making API request...")

    try:
        start_time = time.time()
        response_data = model._make_request(messages, {}, None)
        end_time = time.time()

        print(f"   ✅ Request completed in {end_time - start_time:.2f}s")
        print(f"   Response type: {type(response_data)}")
        print(f"   Response keys: {list(response_data.keys()) if isinstance(response_data, dict) else 'Not a dict'}")

        if isinstance(response_data, dict):
            choices = response_data.get("choices", [])
            print(f"   Choices: {choices}")
            if choices:
                content = choices[0].get("message", {}).get("content", "")
                print(f"   Content: '{content}'")

    except Exception as e:
        print(f"   ❌ API request failed: {e}")
        return

    print("🔧 Testing response creation...")

    try:
        # Test the current response creation method
        print("   Current method...")
        response = llm.Response(
            prompt=prompt,
            model=model,
            stream=False,
            conversation=None
        )
        response._content = "test content"
        print(f"   ✅ Response created: {response}")

        # Try to access content
        try:
            print(f"   Response content: {response._content}")
        except Exception as e:
            print(f"   ❌ Content access failed: {e}")

        try:
            print(f"   Response str: {str(response)}")
        except Exception as e:
            print(f"   ❌ String conversion failed: {e}")

    except Exception as e:
        print(f"   ❌ Response creation failed: {e}")
        import traceback
        traceback.print_exc()

def test_different_response_methods():
    """Test different ways to create LLM responses."""

    try:
        api_key = llm.get_key(alias='zai', env='ZAI_API_KEY')
    except:
        print("❌ No API key")
        return

    llm.load_plugins()
    from llm_zai import ZaiChat

    model = ZaiChat('zai-glm-4.6')

    class SimplePrompt:
        def __init__(self, text):
            self.prompt = text
            self.system = None

    prompt = SimplePrompt("test")

    # Make a real API call first
    messages = model.build_messages(prompt, None)
    response_data = model._make_request(messages, {}, None)
    content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")

    print(f"📥 API returned content: '{content}'")

    # Test different response creation methods
    methods = [
        ("Current method", lambda: llm.Response(prompt=prompt, model=model, stream=False, conversation=None)),
        ("Minimal response", lambda: llm.Response(prompt=prompt, model=model, stream=False)),
        ("None conversation", lambda: llm.Response(prompt=prompt, model=model, stream=False, conversation=None)),
    ]

    for name, method in methods:
        print(f"\n🔧 Testing: {name}")
        try:
            response = method()
            print(f"   ✅ Response object created: {type(response)}")

            # Try to set content
            try:
                response._content = content
                print(f"   ✅ Content set successfully")
            except Exception as e:
                print(f"   ❌ Content setting failed: {e}")

            # Try to convert to string
            try:
                response_str = str(response)
                print(f"   ✅ String conversion: '{response_str[:50]}...'")
            except Exception as e:
                print(f"   ❌ String conversion failed: {e}")

        except Exception as e:
            print(f"   ❌ Method failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    print("🔍 Debug Plugin Hanging Issue")
    print("=" * 50)

    debug_plugin_execution()

    print("\n" + "="*50)
    test_different_response_methods()