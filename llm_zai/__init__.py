"""LLM plugin for Z.ai's GLM models."""

import llm
from llm import hookimpl
import httpx
from typing import Optional, Dict, Any, List
from pydantic import Field


class ZaiOptions(llm.Options):
    """Options for Z.ai models."""

    class Config:
        extra = "ignore"  # Allow extra fields to prevent validation errors

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


class ZaiChat(llm.KeyModel):
    """Synchronous Z.ai chat model."""

    model_id = "zai-glm-4.6"
    can_stream = False  # Disabled for now

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.api_base = "https://api.z.ai/api/coding/paas/v4"
        super().__init__()

    def __str__(self):
        return f"Z.ai: {self.model_id}"

    def _get_api_key(self, key: Optional[str] = None) -> str:
        """Get the API key using LLM's native secrets management."""
        api_key = llm.get_key(key, alias="zai", env="ZAI_API_KEY")
        if not api_key:
            raise ValueError("API key required. Use 'llm keys set zai' or set ZAI_API_KEY environment variable.")
        return api_key

    def _get_headers(self, key: Optional[str] = None) -> Dict[str, str]:
        """Get HTTP headers for API requests."""
        api_key = self._get_api_key(key)
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "llm-zai/0.1.0 (python-httpx)",
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }

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
            "model": self.model_id.replace("zai-", "").upper().replace("-AIR", "-Air"),  # Convert to GLM-4.6 format
            "messages": messages,
            **options
        }

        try:
            response = httpx.post(
                f"{self.api_base}/chat/completions",
                headers=self._get_headers(key),
                json=request_data,
                timeout=60.0
            )
            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise ValueError("Invalid Z.ai API key")
            elif e.response.status_code == 429:
                # Try to extract more detailed error message
                try:
                    error_data = e.response.json()
                    message = error_data.get("error", {}).get("message", "Rate limit or account balance issue")
                    if "balance" in message.lower() or "recharge" in message.lower() or "充值" in message:
                        raise ValueError(f"Account balance issue: {message}. Please check your Z.ai account balance and recharge if needed.")
                    else:
                        raise ValueError(f"Rate limit exceeded: {message}")
                except:
                    raise ValueError("Rate limit exceeded or account balance issue. Please check your Z.ai account.")
            elif e.response.status_code >= 500:
                raise ValueError(f"Z.ai server error: {e.response.status_code}")
            else:
                raise ValueError(f"Z.ai API error: {e.response.text}")

        except httpx.RequestError as e:
            raise ValueError(f"Network error connecting to Z.ai: {str(e)}")

    def execute(self, prompt, stream, response, conversation=None, key=None, **kwargs):
        """Generate a response from the model."""
        messages = self.build_messages(prompt, conversation or llm.Conversation(model=self))
        options = ZaiOptions(**kwargs).model_dump(exclude_unset=True)

        # Remove stream option for now
        request_options = {k: v for k, v in options.items() if k != "stream"}

        response_data = self._make_request(messages, request_options, key)

        # Extract response text and usage
        choice = response_data.get("choices", [{}])[0]
        message = choice.get("message", {})

        # Try content field first, then reasoning_content
        content = message.get("content", "")
        if not content:
            content = message.get("reasoning_content", "")

        usage = response_data.get("usage", {})

        # Store response data
        response.response_json = response_data

        # Set usage if available
        if usage:
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            response.set_usage(input=input_tokens, output=output_tokens, details={})

        # Yield the content (this is what LLM expects)
        if content is not None:
            yield content


class AsyncZaiChat(llm.AsyncKeyModel):
    """Asynchronous Z.ai chat model."""

    model_id = "zai-glm-4.6"
    can_stream = False  # Disabled for now

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.api_base = "https://api.z.ai/api/coding/paas/v4"
        super().__init__()

    def __str__(self):
        return f"Async Z.ai: {self.model_id}"

    def _get_api_key(self, key: Optional[str] = None) -> str:
        """Get the API key using LLM's native secrets management."""
        api_key = llm.get_key(key, alias="zai", env="ZAI_API_KEY")
        if not api_key:
            raise ValueError("API key required. Use 'llm keys set zai' or set ZAI_API_KEY environment variable.")
        return api_key

    def _get_headers(self, key: Optional[str] = None) -> Dict[str, str]:
        """Get HTTP headers for API requests."""
        api_key = self._get_api_key(key)
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "llm-zai/0.1.0 (python-httpx)",
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }

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
            "model": self.model_id.replace("zai-", "").upper().replace("-AIR", "-Air"),  # Convert to GLM-4.6 format
            "messages": messages,
            **options
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.api_base}/chat/completions",
                    headers=self._get_headers(key),
                    json=request_data
                )
                response.raise_for_status()
                return response.json()

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise ValueError("Invalid Z.ai API key")
            elif e.response.status_code == 429:
                # Try to extract more detailed error message
                try:
                    error_data = e.response.json()
                    message = error_data.get("error", {}).get("message", "Rate limit or account balance issue")
                    if "balance" in message.lower() or "recharge" in message.lower() or "充值" in message:
                        raise ValueError(f"Account balance issue: {message}. Please check your Z.ai account balance and recharge if needed.")
                    else:
                        raise ValueError(f"Rate limit exceeded: {message}")
                except:
                    raise ValueError("Rate limit exceeded or account balance issue. Please check your Z.ai account.")
            elif e.response.status_code >= 500:
                raise ValueError(f"Z.ai server error: {e.response.status_code}")
            else:
                raise ValueError(f"Z.ai API error: {e.response.text}")

        except httpx.RequestError as e:
            raise ValueError(f"Network error connecting to Z.ai: {str(e)}")

    async def execute(self, prompt, stream, response, conversation=None, key=None, **kwargs):
        """Generate an async response from the model."""
        messages = self.build_messages(prompt, conversation or llm.AsyncConversation(model=self))
        options = ZaiOptions(**kwargs).model_dump(exclude_unset=True)

        # Remove stream option for now
        request_options = {k: v for k, v in options.items() if k != "stream"}

        response_data = await self._make_request(messages, request_options, key)

        # Extract response text and usage
        choice = response_data.get("choices", [{}])[0]
        message = choice.get("message", {})

        # Try content field first, then reasoning_content
        content = message.get("content", "")
        if not content:
            content = message.get("reasoning_content", "")

        usage = response_data.get("usage", {})

        # Store response data
        response.response_json = response_data

        # Set usage if available
        if usage:
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            response.set_usage(input=input_tokens, output=output_tokens, details={})

        # Yield the content (this is what LLM expects)
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