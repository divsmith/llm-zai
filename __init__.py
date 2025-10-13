"""LLM plugin for Z.ai's GLM models."""

__version__ = "0.1.0"
__author__ = "LLM Z.ai Plugin Contributors"
__description__ = "LLM plugin for Z.ai's GLM models"

from .llm_zai import ZaiChat, AsyncZaiChat, ZaiOptions

__all__ = ["ZaiChat", "AsyncZaiChat", "ZaiOptions"]