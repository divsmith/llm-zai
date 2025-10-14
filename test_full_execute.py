#!/usr/bin/env python3
"""
Test the full execute method.
"""

import llm
import time

def test_full_execute():
    """Test the full execute method end-to-end."""

    try:
        llm.load_plugins()
        from llm_zai import ZaiChat

        print("🔧 Creating model...")
        model = ZaiChat('zai-glm-4.6')

        print("🔧 Creating prompt...")
        prompt = llm.Prompt("What is 2+2? Just give the number.", model=model)
        print(f"   Prompt: {prompt}")

        print("🔧 Starting execute...")
        start_time = time.time()

        try:
            response = model.execute(prompt)
            end_time = time.time()

            print(f"✅ Execute completed in {end_time - start_time:.2f}s")
            print(f"✅ Response type: {type(response)}")
            print(f"✅ Response: \"{response}\"")

            # Test string conversion
            response_str = str(response)
            print(f"✅ String conversion: \"{response_str}\"")

            return True

        except Exception as e:
            end_time = time.time()
            print(f"❌ Execute failed after {end_time - start_time:.2f}s: {e}")
            import traceback
            traceback.print_exc()
            return False

    except Exception as e:
        print(f"❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_manual_execute():
    """Manually test what execute does step by step."""

    try:
        llm.load_plugins()
        from llm_zai import ZaiChat, ZaiOptions

        model = ZaiChat('zai-glm-4.6')
        prompt = llm.Prompt("What is 2+2?", model=model)

        print("🔧 Step 1: Building messages...")
        messages = model.build_messages(prompt, None)
        print(f"   Messages: {messages}")

        print("🔧 Step 2: Creating options...")
        options = ZaiOptions().dict(exclude_unset=True)
        print(f"   Options: {options}")

        print("🔧 Step 3: Making API request...")
        response_data = model._make_request(messages, options, None)
        print(f"   Response received: {type(response_data)}")

        print("🔧 Step 4: Extracting content...")
        choice = response_data.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content", "")
        if not content:
            content = message.get("reasoning_content", "")
        print(f"   Content: \"{content[:50]}...\"")

        print("🔧 Step 5: Creating response...")
        from llm_zai import ZaiResponse
        response = ZaiResponse(
            content=content,
            prompt=prompt,
            model=model,
            stream=False
        )
        print(f"   Response created: {type(response)}")

        print("🔧 Step 6: Testing response...")
        response_str = str(response)
        print(f"   Final result: \"{response_str}\"")

        return True

    except Exception as e:
        print(f"❌ Manual execute failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔍 Testing Full Execute Method")
    print("=" * 50)

    print("\n1. Testing manual step-by-step execute:")
    manual_success = test_manual_execute()

    print("\n2. Testing full execute method:")
    full_success = test_full_execute()

    print(f"\n📊 Results:")
    print(f"   Manual execute: {'✅' if manual_success else '❌'}")
    print(f"   Full execute: {'✅' if full_success else '❌'}")

    if manual_success and not full_success:
        print("💡 The individual steps work, but execute() has issues")
    elif manual_success and full_success:
        print("🎉 Everything works!")
    else:
        print("❌ There are fundamental issues")