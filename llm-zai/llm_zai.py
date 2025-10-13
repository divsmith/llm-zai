"""LLM plugin for Z.ai's GLM models."""

import os
import json
import llm
import httpx
from typing import Optional, Dict, Any, List, AsyncGenerator, Iterator
from pydantic import Field, BaseModel


class ZaiOptions(llm.Options):
    """Options for Z.ai models."""

    temperature: Optional[float] = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Controls randomness in the output (0.0 to 2.0)"
    )

    max_tokens: Optional[int] = Field(
        default=4096,
        ge=1,
        le=32768,
        description="Maximum number of tokens to generate"
    )

    top_p: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter (0.0 to 1.0)"
    )

    stream: Optional[bool] = Field(
        default=False,
        description="Whether to stream the response"
    )


class _Shared:
    """Shared configuration for Z.ai models."""

    needs_key = "zai"
    key_env_var = "ZAI_API_KEY"

    @classmethod
    def get_api_key(cls):
        """Get the API key from environment."""
        return os.environ.get(cls.key_env_var)

    @classmethod
    def get_headers(cls):
        """Get HTTP headers for API requests."""
        api_key = cls.get_api_key()
        if not api_key:
            raise ValueError(f"API key required. Set {cls.key_env_var} environment variable.")

        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }


class ZaiChat(llm.Model):
    """Synchronous Z.ai chat model."""

    model_id = "zai-glm-4.6"
    can_stream = True

    def __init__(self, model_id: str, api_base: str = "https://api.z.ai/api/paas/v4"):
        self.model_id = model_id
        self.api_base = api_base
        self._shared = _Shared()

    def __str__(self):
        return f"Z.ai: {self.model_id}"

    def _convert_messages(self, messages: List[llm.Message]) -> List[Dict[str, Any]]:
        """Convert LLM message format to Z.ai API format."""
        converted = []

        for message in messages:
            msg_dict = {
                "role": message.role,
                "content": message.content
            }

            # Handle attachments/images for vision models
            if hasattr(message, 'attachments') and message.attachments:
                content = []
                content.append({"type": "text", "text": message.content})

                for attachment in message.attachments:
                    if hasattr(attachment, 'image_url'):
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": attachment.image_url
                            }
                        })

                msg_dict["content"] = content

            converted.append(msg_dict)

        return converted

    def _make_request(self, messages: List[Dict[str, Any]], options: Dict[str, Any]) -> Dict[str, Any]:
        """Make API request to Z.ai."""
        request_data = {
            "model": self.model_id.replace("zai-", ""),  # Remove "zai-" prefix
            "messages": messages,
            **options
        }

        try:
            response = httpx.post(
                f"{self.api_base}/chat/completions",
                headers=self._shared.get_headers(),
                json=request_data,
                timeout=60.0
            )
            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise ValueError("Invalid Z.ai API key")
            elif e.response.status_code == 429:
                raise ValueError("Rate limit exceeded. Please try again later.")
            elif e.response.status_code >= 500:
                raise ValueError(f"Z.ai server error: {e.response.status_code}")
            else:
                raise ValueError(f"Z.ai API error: {e.response.text}")

        except httpx.RequestError as e:
            raise ValueError(f"Network error connecting to Z.ai: {str(e)}")

    def __call__(self, prompt: llm.Prompt, stream: bool = False, **kwargs) -> llm.Response:
        """Generate a response from the model."""
        messages = list(prompt.messages)
        options = ZaiOptions(**kwargs).dict(exclude_unset=True)

        # Set stream option
        if stream:
            options["stream"] = True

        # Handle streaming
        if options.get("stream", False):
            return self._stream_response(messages, options)

        # Make non-streaming request
        request_options = options.copy()
        request_options.pop("stream", None)

        response_data = self._make_request(messages, request_options)

        # Extract response text and usage
        content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = response_data.get("usage", {})

        return llm.Response(
            model=self,
            content=content,
            prompt=prompt.prompt,
            usage=llm.Usage(
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0)
            )
        )

    def _stream_response(self, messages: List[Dict[str, Any]], options: Dict[str, Any]) -> llm.Response:
        """Handle streaming response."""
        request_data = {
            "model": self.model_id.replace("zai-", ""),
            "messages": messages,
            **options
        }

        try:
            with httpx.stream(
                "POST",
                f"{self.api_base}/chat/completions",
                headers=self._shared.get_headers(),
                json=request_data,
                timeout=60.0
            ) as response:
                response.raise_for_status()

                def generate_content():
                    for line in response.iter_lines():
                        if line.strip():
                            if line.startswith("data: "):
                                data_str = line[6:]  # Remove "data: " prefix
                                if data_str.strip() == "[DONE]":
                                    break

                                try:
                                    data = json.loads(data_str)
                                    delta = data.get("choices", [{}])[0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        yield content
                                except json.JSONDecodeError:
                                    continue

                # Create response with streaming content
                content_stream = generate_content()
                full_content = "".join(content_stream)

                return llm.Response(
                    model=self,
                    content=full_content,
                    prompt=messages[-1].get("content", ""),
                    usage=llm.Usage()  # Usage info not available in streaming mode
                )

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise ValueError("Invalid Z.ai API key")
            elif e.response.status_code == 429:
                raise ValueError("Rate limit exceeded. Please try again later.")
            else:
                raise ValueError(f"Z.ai API error: {e.response.text}")

        except httpx.RequestError as e:
            raise ValueError(f"Network error connecting to Z.ai: {str(e)}")


class AsyncZaiChat(llm.AsyncModel):
    """Asynchronous Z.ai chat model."""

    model_id = "zai-glm-4.6"
    can_stream = True

    def __init__(self, model_id: str, api_base: str = "https://api.z.ai/api/paas/v4"):
        self.model_id = model_id
        self.api_base = api_base
        self._shared = _Shared()

    def __str__(self):
        return f"Async Z.ai: {self.model_id}"

    def _convert_messages(self, messages: List[llm.Message]) -> List[Dict[str, Any]]:
        """Convert LLM message format to Z.ai API format."""
        converted = []

        for message in messages:
            msg_dict = {
                "role": message.role,
                "content": message.content
            }

            # Handle attachments/images for vision models
            if hasattr(message, 'attachments') and message.attachments:
                content = []
                content.append({"type": "text", "text": message.content})

                for attachment in message.attachments:
                    if hasattr(attachment, 'image_url'):
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": attachment.image_url
                            }
                        })

                msg_dict["content"] = content

            converted.append(msg_dict)

        return converted

    async def _make_request(self, messages: List[Dict[str, Any]], options: Dict[str, Any]) -> Dict[str, Any]:
        """Make async API request to Z.ai."""
        request_data = {
            "model": self.model_id.replace("zai-", ""),  # Remove "zai-" prefix
            "messages": messages,
            **options
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.api_base}/chat/completions",
                    headers=self._shared.get_headers(),
                    json=request_data
                )
                response.raise_for_status()
                return response.json()

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise ValueError("Invalid Z.ai API key")
            elif e.response.status_code == 429:
                raise ValueError("Rate limit exceeded. Please try again later.")
            elif e.response.status_code >= 500:
                raise ValueError(f"Z.ai server error: {e.response.status_code}")
            else:
                raise ValueError(f"Z.ai API error: {e.response.text}")

        except httpx.RequestError as e:
            raise ValueError(f"Network error connecting to Z.ai: {str(e)}")

    async def __call__(self, prompt: llm.Prompt, stream: bool = False, **kwargs) -> llm.Response:
        """Generate an async response from the model."""
        messages = list(prompt.messages)
        options = ZaiOptions(**kwargs).dict(exclude_unset=True)

        # Set stream option
        if stream:
            options["stream"] = True

        # Handle streaming
        if options.get("stream", False):
            return await self._stream_response(messages, options)

        # Make non-streaming request
        request_options = options.copy()
        request_options.pop("stream", None)

        response_data = await self._make_request(messages, request_options)

        # Extract response text and usage
        content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = response_data.get("usage", {})

        return llm.Response(
            model=self,
            content=content,
            prompt=prompt.prompt,
            usage=llm.Usage(
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0)
            )
        )

    async def _stream_response(self, messages: List[Dict[str, Any]], options: Dict[str, Any]) -> llm.Response:
        """Handle async streaming response."""
        request_data = {
            "model": self.model_id.replace("zai-", ""),
            "messages": messages,
            **options
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    "POST",
                    f"{self.api_base}/chat/completions",
                    headers=self._shared.get_headers(),
                    json=request_data
                ) as response:
                    response.raise_for_status()

                    async def generate_content():
                        async for line in response.aiter_lines():
                            if line.strip():
                                if line.startswith("data: "):
                                    data_str = line[6:]  # Remove "data: " prefix
                                    if data_str.strip() == "[DONE]":
                                        break

                                    try:
                                        data = json.loads(data_str)
                                        delta = data.get("choices", [{}])[0].get("delta", {})
                                        content = delta.get("content", "")
                                        if content:
                                            yield content
                                    except json.JSONDecodeError:
                                        continue

                    # Create response with streaming content
                    content_stream = generate_content()
                    full_content = "".join([c async for c in content_stream])

                    return llm.Response(
                        model=self,
                        content=full_content,
                        prompt=messages[-1].get("content", ""),
                        usage=llm.Usage()  # Usage info not available in streaming mode
                    )

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise ValueError("Invalid Z.ai API key")
            elif e.response.status_code == 429:
                raise ValueError("Rate limit exceeded. Please try again later.")
            else:
                raise ValueError(f"Z.ai API error: {e.response.text}")

        except httpx.RequestError as e:
            raise ValueError(f"Network error connecting to Z.ai: {str(e)}")


@llm.hookimpl
def register_models(register):
    """Register Z.ai models with the LLM tool."""

    # GLM-4.6 - Latest text model
    register(
        ZaiChat("zai-glm-4.6"),
        AsyncZaiChat("zai-glm-4.6"),
        aliases=["glm-4.6"],
        can_stream=True,
        supports_tools=False,
        supports_images=False,
        supports_pdf=False,
        default_max_tokens=4096,
    )

    # GLM-4.5V - Vision model
    register(
        ZaiChat("zai-glm-4.5v"),
        AsyncZaiChat("zai-glm-4.5v"),
        aliases=["glm-4.5v"],
        can_stream=True,
        supports_tools=False,
        supports_images=True,
        supports_pdf=False,
        default_max_tokens=4096,
    )

    # GLM-4-32b-0414-128K - Large context model
    register(
        ZaiChat("zai-glm-4-32b"),
        AsyncZaiChat("zai-glm-4-32b"),
        aliases=["glm-4-32b", "glm-4-32b-0414-128k"],
        can_stream=True,
        supports_tools=False,
        supports_images=False,
        supports_pdf=False,
        default_max_tokens=8192,
    )

    # Z.ai Coder model (assuming it's available)
    register(
        ZaiChat("zai-coder"),
        AsyncZaiChat("zai-coder"),
        aliases=["coder", "zai-coder-llm"],
        can_stream=True,
        supports_tools=False,
        supports_images=False,
        supports_pdf=False,
        default_max_tokens=4096,
    )