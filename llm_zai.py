"""LLM plugin for Z.ai's GLM models."""

import os
import json
import llm
import httpx
from typing import Optional, Dict, Any, List
from pydantic import Field


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

    @classmethod
    def get_api_key(cls, key=None):
        """Get the API key using LLM's native secrets management."""
        return llm.get_key(key, alias="zai", env="ZAI_API_KEY")

    @classmethod
    def get_headers(cls, key=None):
        """Get HTTP headers for API requests."""
        api_key = cls.get_api_key(key)
        if not api_key:
            raise ValueError("API key required. Use 'llm keys set zai' or set ZAI_API_KEY environment variable.")

        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }


class ZaiChat(llm.KeyModel):
    """Synchronous Z.ai chat model."""

    model_id = "zai-glm-4.6"
    can_stream = False  # Disabled for now

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.api_base = "https://api.z.ai/api/paas/v4"
        super().__init__()

    def __str__(self):
        return f"Z.ai: {self.model_id}"

    def build_messages(self, prompt: llm.Prompt, conversation: llm.Conversation) -> List[Dict[str, Any]]:
        """Build messages for API request."""
        messages = []

        # Add system message if provided
        if prompt.system:
            messages.append({"role": "system", "content": prompt.system})

        # Add conversation history
        if conversation:
            for response in conversation.responses:
                if response.prompt:
                    messages.append({"role": "user", "content": response.prompt.prompt})
                if response.response:
                    messages.append({"role": "assistant", "content": response.response})

        # Add current prompt
        messages.append({"role": "user", "content": prompt.prompt})

        return messages

    def _make_request(self, messages: List[Dict[str, Any]], options: Dict[str, Any], key: str = None) -> Dict[str, Any]:
        """Make API request to Z.ai."""
        request_data = {
            "model": self.model_id.replace("zai-", ""),  # Remove "zai-" prefix
            "messages": messages,
            **options
        }

        try:
            response = httpx.post(
                f"{self.api_base}/chat/completions",
                headers=self.get_headers(key),
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

    def execute(self, prompt: llm.Prompt, stream: bool = False, conversation: llm.Conversation = None, key: str = None, **kwargs) -> llm.Response:
        """Generate a response from the model."""
        messages = self.build_messages(prompt, conversation or llm.Conversation())
        options = ZaiOptions(**kwargs).dict(exclude_unset=True)

        # Remove stream option for now
        request_options = {k: v for k, v in options.items() if k != "stream"}

        response_data = self._make_request(messages, request_options, key)

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


class AsyncZaiChat(llm.AsyncKeyModel):
    """Asynchronous Z.ai chat model."""

    model_id = "zai-glm-4.6"
    can_stream = False  # Disabled for now

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.api_base = "https://api.z.ai/api/paas/v4"
        super().__init__()

    def __str__(self):
        return f"Async Z.ai: {self.model_id}"

    def build_messages(self, prompt: llm.Prompt, conversation: llm.AsyncConversation) -> List[Dict[str, Any]]:
        """Build messages for API request."""
        messages = []

        # Add system message if provided
        if prompt.system:
            messages.append({"role": "system", "content": prompt.system})

        # Add current prompt
        messages.append({"role": "user", "content": prompt.prompt})

        return messages

    async def _make_request(self, messages: List[Dict[str, Any]], options: Dict[str, Any], key: str = None) -> Dict[str, Any]:
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
                    headers=self.get_headers(key),
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

    async def execute(self, prompt: llm.Prompt, stream: bool = False, conversation: llm.AsyncConversation = None, key: str = None, **kwargs) -> llm.Response:
        """Generate an async response from the model."""
        messages = self.build_messages(prompt, conversation or llm.AsyncConversation())
        options = ZaiOptions(**kwargs).dict(exclude_unset=True)

        # Remove stream option for now
        request_options = {k: v for k, v in options.items() if k != "stream"}

        response_data = await self._make_request(messages, request_options, key)

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


@llm.hookimpl
def register_models(register):
    """Register Z.ai models with the LLM tool."""

    # GLM-4.6 - Latest text model
    register(
        ZaiChat("zai-glm-4.6"),
        AsyncZaiChat("zai-glm-4.6"),
        aliases=["glm-4.6"],
        can_stream=False,  # Disabled for now
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
        can_stream=False,  # Disabled for now
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
        can_stream=False,  # Disabled for now
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
        can_stream=False,  # Disabled for now
        supports_tools=False,
        supports_images=False,
        supports_pdf=False,
        default_max_tokens=4096,
    )