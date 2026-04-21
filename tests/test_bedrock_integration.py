"""Integration tests for AWS Bedrock provider.

Run with: pytest -m bedrock
Requires valid AWS credentials and Bedrock model access.
"""

import pytest

from tradingagents.llm_clients.bedrock_client import BedrockClient

pytestmark = pytest.mark.bedrock


@pytest.fixture
def client():
    return BedrockClient("claude-haiku", max_tokens=256, temperature=0.0)


def test_invoke_returns_string(client):
    """Verify a real Bedrock call returns a non-empty string response."""
    llm = client.get_llm()
    result = llm.invoke("Reply with exactly: hello")
    assert isinstance(result.content, str)
    assert len(result.content) > 0


def test_validate_model(client):
    assert client.validate_model() is True
