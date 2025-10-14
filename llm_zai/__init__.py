"""LLM plugin for Z.ai's GLM models."""

import llm
from llm import hookimpl
from llm import KeyModel, AsyncKeyModel
import httpx
from typing import Optional, Dict, Any, List
from pydantic import Field


class ZaiOptions(llm.Options):
    """Options for Z.ai models."""

    temperature: Optional[float] = Field(
        description="What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.",
        ge=0,
        le=2,
        default=None,
    )

    max_tokens: Optional[int] = Field(
        description="Maximum number of tokens to generate.",
        default=None,
    )

    top_p: Optional[float] = Field(
        description="An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass.",
        ge=0,
        le=1,
        default=None,
    )


class _ZaiShared:
    """Shared functionality for Z.ai models."""

    api_base = "https://api.z.ai/api/coding/paas/v4"
    needs_key = "zai"
    key_env_var = "ZAI_API_KEY"

    def __init__(self, model_id: str):
        self.model_id = model_id
        super().__init__()

    def __str__(self):
        return "Z.ai: {}".format(self.model_id)

    def _get_api_key(self, key: Optional[str] = None) -> str:
        """Get the API key using LLM's native secrets management."""
        api_key = llm.get_key(key, alias=self.needs_key, env=self.key_env_var)
        if not api_key:
            raise ValueError(f"API key required. Use 'llm keys set {self.needs_key}' or set {self.key_env_var} environment variable.")
        return api_key

    def _get_headers(self, key: Optional[str] = None) -> Dict[str, str]:
        """Get HTTP headers for API requests."""
        api_key = self._get_api_key(key)
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _build_request_data(self, messages: List[Dict[str, Any]], options: Dict[str, Any]) -> Dict[str, Any]:
        """Build the request data for Z.ai API."""
        return {
            "model": self.model_id.replace("zai-", "").upper().replace("-AIR", "-Air"),
            "messages": messages,
            **options
        }

    def _extract_content(self, response_data: Dict[str, Any]) -> str:
        """Extract content from Z.ai response, handling reasoning_content field."""
        choice = response_data.get("choices", [{}])[0]
        message = choice.get("message", {})

        # Try content field first, then reasoning_content
        content = message.get("content", "")
        if not content:
            content = message.get("reasoning_content", "")

        return content

    def _set_usage(self, response, usage: Dict[str, Any]):
        """Set usage information on response."""
        if usage:
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            response.set_usage(input=input_tokens, output=output_tokens, details={})


class ZaiChat(_ZaiShared, KeyModel):
    """Synchronous Z.ai chat model."""

    model_id = "zai-glm-4.6"
    can_stream = False

    def build_messages(self, prompt, conversation):
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

    def execute(self, prompt, stream, response, conversation=None, key=None, **kwargs):
        """Generate a response from the model."""
        messages = self.build_messages(prompt, conversation or llm.Conversation(model=self))
        options = ZaiOptions(**kwargs).model_dump(exclude_unset=True)

        # Remove unsupported options
        request_options = {k: v for k, v in options.items() if k != "stream"}

        request_data = self._build_request_data(messages, request_options)

        try:
            httpx_response = httpx.post(
                f"{self.api_base}/chat/completions",
                headers=self._get_headers(key),
                json=request_data,
                timeout=30.0
            )
            httpx_response.raise_for_status()
            response_data = httpx_response.json()
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

        # Store response data and set usage
        response.response_json = response_data
        self._set_usage(response, response_data.get("usage", {}))

        # Extract and yield content
        content = self._extract_content(response_data)
        if content is not None:
            yield content


class AsyncZaiChat(_ZaiShared, AsyncKeyModel):
    """Asynchronous Z.ai chat model."""

    model_id = "zai-glm-4.6"
    can_stream = False

    def build_messages(self, prompt, conversation):
        """Build messages for API request."""
        messages = []

        # Add system message if provided
        if prompt.system:
            messages.append({"role": "system", "content": prompt.system})

        # Add current prompt
        messages.append({"role": "user", "content": prompt.prompt})

        return messages

    async def execute(self, prompt, stream, response, conversation=None, key=None, **kwargs):
        """Generate an async response from the model."""
        messages = self.build_messages(prompt, conversation or llm.AsyncConversation(model=self))
        options = ZaiOptions(**kwargs).model_dump(exclude_unset=True)

        # Remove unsupported options
        request_options = {k: v for k, v in options.items() if k != "stream"}

        request_data = self._build_request_data(messages, request_options)

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                httpx_response = await client.post(
                    f"{self.api_base}/chat/completions",
                    headers=self._get_headers(key),
                    json=request_data
                )
                httpx_response.raise_for_status()
                response_data = httpx_response.json()
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

        # Store response data and set usage
        response.response_json = response_data
        self._set_usage(response, response_data.get("usage", {}))

        # Extract and yield content
        content = self._extract_content(response_data)
        if content is not None:
            yield content


class ZaiPlugin:
    @hookimpl
    def register_models(self, register):
        """Register Z.ai models with the LLM tool."""

        # GLM-4.6 - Latest text model
        register(
            ZaiChat("zai-glm-4.6"),
            AsyncZaiChat("zai-glm-4.6"),
            aliases=["glm-4.6"],
        )

        # GLM-4.5V - Vision model
        register(
            ZaiChat("zai-glm-4.5v"),
            AsyncZaiChat("zai-glm-4.5v"),
            aliases=["glm-4.5v"],
        )

        # GLM-4.5 - Standard text model
        register(
            ZaiChat("zai-glm-4.5"),
            AsyncZaiChat("zai-glm-4.5"),
            aliases=["glm-4.5"],
        )

        # GLM-4.5-Air - Lightweight text model
        register(
            ZaiChat("zai-glm-4.5-air"),
            AsyncZaiChat("zai-glm-4.5-air"),
            aliases=["glm-4.5-air"],
        )

        # GLM-4-32b-0414-128K - Large context model
        register(
            ZaiChat("zai-glm-4-32b"),
            AsyncZaiChat("zai-glm-4-32b"),
            aliases=["glm-4-32b", "glm-4-32b-0414-128k"],
        )

# Create plugin instance with __name__ attribute for compatibility
plugin = ZaiPlugin()
plugin.__name__ = "llm_zai"

# For backward compatibility, keep the function available
def register_models(register):
    """Register Z.ai models with the LLM tool."""
    return plugin.register_models(register)


__all__ = ["ZaiChat", "AsyncZaiChat", "ZaiOptions", "register_models", "plugin"]