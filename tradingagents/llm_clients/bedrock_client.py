import os
from typing import Any, Optional

from botocore.exceptions import ClientError
from langchain_aws import ChatBedrockConverse
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .base_client import BaseLLMClient, normalize_content

_MODEL_ALIASES = {
    "claude-haiku": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    "claude-sonnet": "us.anthropic.claude-sonnet-4-20250514-v1:0",
    "claude-opus": "us.anthropic.claude-opus-4-20250514-v1:0",
    "claude-haiku-4-5": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    "claude-sonnet-4-5": "us.anthropic.claude-sonnet-4-20250514-v1:0",
    "claude-sonnet-4-6": "us.anthropic.claude-sonnet-4-20250514-v1:0",
    "claude-opus-4-5": "us.anthropic.claude-opus-4-20250514-v1:0",
    "claude-opus-4-6": "us.anthropic.claude-opus-4-20250514-v1:0",
}

_PASSTHROUGH_KWARGS = ("max_tokens", "temperature", "callbacks")


def resolve_model_id(model: str) -> str:
    return _MODEL_ALIASES.get(model, model)


class NormalizedChatBedrockConverse(ChatBedrockConverse):
    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(ClientError),
    )
    def invoke(self, input, config=None, **kwargs):
        return normalize_content(super().invoke(input, config, **kwargs))


class BedrockClient(BaseLLMClient):
    def __init__(self, model: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(model, base_url, **kwargs)

    def get_llm(self) -> Any:
        region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
        llm_kwargs = {
            "model_id": resolve_model_id(self.model),
            "region_name": region,
        }
        for key in _PASSTHROUGH_KWARGS:
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]
        return NormalizedChatBedrockConverse(**llm_kwargs)

    def validate_model(self) -> bool:
        return True
