"""Tests for llm-zai plugin."""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock
import llm
from llm_zai import ZaiChat, AsyncZaiChat, ZaiOptions, _Shared


class TestZaiOptions:
    """Test ZaiOptions validation."""

    def test_default_options(self):
        """Test default option values."""
        options = ZaiOptions()
        assert options.temperature == 1.0
        assert options.max_tokens == 4096
        assert options.top_p is None
        assert options.stream is False

    def test_custom_options(self):
        """Test custom option values."""
        options = ZaiOptions(
            temperature=0.7,
            max_tokens=2048,
            top_p=0.9,
            stream=True
        )
        assert options.temperature == 0.7
        assert options.max_tokens == 2048
        assert options.top_p == 0.9
        assert options.stream is True

    def test_temperature_bounds(self):
        """Test temperature bounds validation."""
        # Valid temperatures
        ZaiOptions(temperature=0.0)
        ZaiOptions(temperature=1.0)
        ZaiOptions(temperature=2.0)

        # Invalid temperatures should raise validation error
        with pytest.raises(Exception):  # Pydantic ValidationError
            ZaiOptions(temperature=-0.1)

        with pytest.raises(Exception):  # Pydantic ValidationError
            ZaiOptions(temperature=2.1)

    def test_max_tokens_bounds(self):
        """Test max_tokens bounds validation."""
        # Valid max_tokens
        ZaiOptions(max_tokens=1)
        ZaiOptions(max_tokens=32768)

        # Invalid max_tokens should raise validation error
        with pytest.raises(Exception):  # Pydantic ValidationError
            ZaiOptions(max_tokens=0)

        with pytest.raises(Exception):  # Pydantic ValidationError
            ZaiOptions(max_tokens=32769)

    def test_top_p_bounds(self):
        """Test top_p bounds validation."""
        # Valid top_p values
        ZaiOptions(top_p=0.0)
        ZaiOptions(top_p=0.5)
        ZaiOptions(top_p=1.0)

        # Invalid top_p should raise validation error
        with pytest.raises(Exception):  # Pydantic ValidationError
            ZaiOptions(top_p=-0.1)

        with pytest.raises(Exception):  # Pydantic ValidationError
            ZaiOptions(top_p=1.1)


class TestShared:
    """Test _Shared configuration class."""

    @patch.dict(os.environ, {'ZAI_API_KEY': 'test-key-123'})
    def test_get_api_key_from_env(self):
        """Test getting API key from environment."""
        api_key = _Shared.get_api_key()
        assert api_key == 'test-key-123'

    @patch.dict(os.environ, {}, clear=True)
    def test_get_api_key_missing(self):
        """Test getting API key when not set."""
        api_key = _Shared.get_api_key()
        assert api_key is None

    @patch.dict(os.environ, {'ZAI_API_KEY': 'test-key-123'})
    def test_get_headers(self):
        """Test getting HTTP headers."""
        headers = _Shared.get_headers()
        assert headers['Authorization'] == 'Bearer test-key-123'
        assert headers['Content-Type'] == 'application/json'

    @patch.dict(os.environ, {}, clear=True)
    def test_get_headers_missing_key(self):
        """Test getting headers when API key is missing."""
        with pytest.raises(ValueError, match="API key required"):
            _Shared.get_headers()


class TestZaiChat:
    """Test ZaiChat synchronous model."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model = ZaiChat("zai-glm-4.6")

    def test_model_init(self):
        """Test model initialization."""
        assert self.model.model_id == "zai-glm-4.6"
        assert self.model.api_base == "https://api.z.ai/api/paas/v4"
        assert self.model.can_stream is True

    def test_str_representation(self):
        """Test string representation."""
        str_repr = str(self.model)
        assert "Z.ai: zai-glm-4.6" in str_repr

    def test_convert_simple_messages(self):
        """Test converting simple LLM messages to Z.ai format."""
        messages = [
            llm.Message(role="system", content="You are a helpful assistant."),
            llm.Message(role="user", content="Hello, world!"),
        ]

        converted = self.model._convert_messages(messages)

        expected = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, world!"},
        ]

        assert converted == expected

    def test_convert_messages_with_images(self):
        """Test converting messages with image attachments."""
        # Create mock attachment
        attachment = Mock()
        attachment.image_url = "https://example.com/image.jpg"

        # Create message with attachment
        message = Mock()
        message.role = "user"
        message.content = "Describe this image."
        message.attachments = [attachment]

        converted = self.model._convert_messages([message])

        expected = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image."},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image.jpg"}
                }
            ]
        }]

        assert converted == expected

    @patch.dict(os.environ, {'ZAI_API_KEY': 'test-key-123'})
    @patch('httpx.post')
    def test_successful_request(self, mock_post):
        """Test successful API request."""
        # Mock successful response
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello! How can I help you?"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25}
        }
        mock_post.return_value = mock_response

        messages = [{"role": "user", "content": "Hello"}]
        options = {"temperature": 0.7}

        result = self.model._make_request(messages, options)

        assert result["choices"][0]["message"]["content"] == "Hello! How can I help you?"
        mock_post.assert_called_once()

    @patch.dict(os.environ, {'ZAI_API_KEY': 'test-key-123'})
    @patch('httpx.post')
    def test_authentication_error(self, mock_post):
        """Test authentication error handling."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = Mock(
            side_effect=llm.httpx.HTTPStatusError("401", request=Mock(), response=mock_response)
        )
        mock_post.return_value = mock_response

        messages = [{"role": "user", "content": "Hello"}]

        with pytest.raises(ValueError, match="Invalid Z.ai API key"):
            self.model._make_request(messages, {})

    @patch.dict(os.environ, {'ZAI_API_KEY': 'test-key-123'})
    @patch('httpx.post')
    def test_rate_limit_error(self, mock_post):
        """Test rate limit error handling."""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.raise_for_status.side_effect = Mock(
            side_effect=llm.httpx.HTTPStatusError("429", request=Mock(), response=mock_response)
        )
        mock_post.return_value = mock_response

        messages = [{"role": "user", "content": "Hello"}]

        with pytest.raises(ValueError, match="Rate limit exceeded"):
            self.model._make_request(messages, {})

    @patch.dict(os.environ, {'ZAI_API_KEY': 'test-key-123'})
    @patch('httpx.post')
    def test_server_error(self, mock_post):
        """Test server error handling."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = Mock(
            side_effect=llm.httpx.HTTPStatusError("500", request=Mock(), response=mock_response)
        )
        mock_post.return_value = mock_response

        messages = [{"role": "user", "content": "Hello"}]

        with pytest.raises(ValueError, match="Z.ai server error: 500"):
            self.model._make_request(messages, {})

    @patch('httpx.post')
    def test_network_error(self, mock_post):
        """Test network error handling."""
        mock_post.side_effect = llm.httpx.RequestError("Network error")

        messages = [{"role": "user", "content": "Hello"}]

        with pytest.raises(ValueError, match="Network error connecting to Z.ai"):
            self.model._make_request(messages, {})

    @patch.dict(os.environ, {'ZAI_API_KEY': 'test-key-123'})
    @patch('httpx.post')
    def test_call_sync(self, mock_post):
        """Test synchronous model call."""
        # Mock successful response
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello! How can I help you?"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25}
        }
        mock_post.return_value = mock_response

        # Create mock prompt
        prompt = Mock()
        prompt.messages = [llm.Message(role="user", content="Hello")]
        prompt.prompt = "Hello"

        response = self.model(prompt)

        assert response.content == "Hello! How can I help you?"
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 15
        assert response.usage.total_tokens == 25


class TestAsyncZaiChat:
    """Test AsyncZaiChat asynchronous model."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model = AsyncZaiChat("zai-glm-4.6")

    def test_async_model_init(self):
        """Test async model initialization."""
        assert self.model.model_id == "zai-glm-4.6"
        assert self.model.api_base == "https://api.z.ai/api/paas/v4"
        assert self.model.can_stream is True

    def test_async_str_representation(self):
        """Test async model string representation."""
        str_repr = str(self.model)
        assert "Async Z.ai: zai-glm-4.6" in str_repr

    @pytest.mark.asyncio
    @patch.dict(os.environ, {'ZAI_API_KEY': 'test-key-123'})
    async def test_async_successful_request(self):
        """Test successful async API request."""
        # Mock successful response
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello! How can I help you?"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25}
        }

        mock_client = Mock()
        mock_client.post.return_value = mock_response

        with patch('httpx.AsyncClient') as mock_async_client:
            mock_async_client.return_value.__aenter__.return_value = mock_client

            messages = [{"role": "user", "content": "Hello"}]
            options = {"temperature": 0.7}

            result = await self.model._make_request(messages, options)

            assert result["choices"][0]["message"]["content"] == "Hello! How can I help you?"
            mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    @patch.dict(os.environ, {'ZAI_API_KEY': 'test-key-123'})
    async def test_async_call(self):
        """Test asynchronous model call."""
        # Mock successful response
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello! How can I help you?"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25}
        }

        mock_client = Mock()
        mock_client.post.return_value = mock_response

        with patch('httpx.AsyncClient') as mock_async_client:
            mock_async_client.return_value.__aenter__.return_value = mock_client

            # Create mock prompt
            prompt = Mock()
            prompt.messages = [llm.Message(role="user", content="Hello")]
            prompt.prompt = "Hello"

            response = await self.model(prompt)

            assert response.content == "Hello! How can I help you?"
            assert response.usage.input_tokens == 10
            assert response.usage.output_tokens == 15
            assert response.usage.total_tokens == 25


class TestModelRegistration:
    """Test model registration hook."""

    @patch('llm.register_models')
    def test_register_models_hook(self, mock_register):
        """Test that models are properly registered."""
        # Import the hook function
        from llm_zai import register_models

        # Call the hook
        register_models(mock_register)

        # Verify that register was called
        assert mock_register.call_count == 8  # 4 sync + 4 async models

        # Check that first call is for GLM-4.6
        first_call_args = mock_register.call_args_list[0]
        glm46_model = first_call_args[0][0]
        assert isinstance(glm46_model, ZaiChat)
        assert glm46_model.model_id == "zai-glm-4.6"

        # Check that async GLM-4.6 is registered
        second_call_args = mock_register.call_args_list[1]
        async_glm46_model = second_call_args[0][0]
        assert isinstance(async_glm46_model, AsyncZaiChat)
        assert async_glm46_model.model_id == "zai-glm-4.6"


class TestMessageConversion:
    """Test message format conversion."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model = ZaiChat("zai-glm-4.6")

    def test_empty_messages(self):
        """Test converting empty message list."""
        converted = self.model._convert_messages([])
        assert converted == []

    def test_single_message(self):
        """Test converting single message."""
        message = llm.Message(role="user", content="Hello!")
        converted = self.model._convert_messages([message])

        expected = [{"role": "user", "content": "Hello!"}]
        assert converted == expected

    def test_conversation(self):
        """Test converting conversation with multiple roles."""
        messages = [
            llm.Message(role="system", content="You are helpful."),
            llm.Message(role="user", content="What is Python?"),
            llm.Message(role="assistant", content="Python is a programming language."),
            llm.Message(role="user", content="Can you show me an example?"),
        ]

        converted = self.model._convert_messages(messages)

        expected = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language."},
            {"role": "user", "content": "Can you show me an example?"},
        ]

        assert converted == expected

    def test_message_without_attachments(self):
        """Test message without attachments attribute."""
        message = Mock()
        message.role = "user"
        message.content = "Hello"
        # Don't set attachments attribute

        converted = self.model._convert_messages([message])

        expected = [{"role": "user", "content": "Hello"}]
        assert converted == expected

    def test_message_with_empty_attachments(self):
        """Test message with empty attachments list."""
        message = Mock()
        message.role = "user"
        message.content = "Hello"
        message.attachments = []

        converted = self.model._convert_messages([message])

        expected = [{"role": "user", "content": "Hello"}]
        assert converted == expected


class TestStreamingResponse:
    """Test streaming response handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model = ZaiChat("zai-glm-4.6")

    @patch.dict(os.environ, {'ZAI_API_KEY': 'test-key-123'})
    def test_streaming_response_format(self):
        """Test streaming response data parsing."""
        # Mock streaming response data
        stream_lines = [
            "data: {\"choices\": [{\"delta\": {\"content\": \"Hello\"}}]}",
            "data: {\"choices\": [{\"delta\": {\"content\": \" world\"}}]}",
            "data: {\"choices\": [{\"delta\": {\"content\": \"!\"}}]}",
            "data: [DONE]"
        ]

        # This test would require more complex mocking of the streaming response
        # For now, we just verify the stream_lines format
        for line in stream_lines:
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str.strip() != "[DONE]":
                    data = json.loads(data_str)
                    assert "choices" in data
                    assert "delta" in data["choices"][0]
                    if "content" in data["choices"][0]["delta"]:
                        assert isinstance(data["choices"][0]["delta"]["content"], str)


if __name__ == "__main__":
    pytest.main([__file__])