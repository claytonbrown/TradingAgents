import os
import unittest
from unittest.mock import MagicMock, patch

from botocore.exceptions import ClientError

from tradingagents.llm_clients.bedrock_client import (
    BedrockClient,
    NormalizedChatBedrockConverse,
    resolve_model_id,
)
from tradingagents.llm_clients.factory import create_llm_client


class TestResolveModelId(unittest.TestCase):
    def test_alias_resolution(self):
        self.assertEqual(
            resolve_model_id("claude-haiku"),
            "us.anthropic.claude-3-5-haiku-20241022-v1:0",
        )
        self.assertEqual(
            resolve_model_id("claude-sonnet"),
            "us.anthropic.claude-sonnet-4-20250514-v1:0",
        )

    def test_passthrough_unknown(self):
        self.assertEqual(resolve_model_id("some-custom-model"), "some-custom-model")


@patch("tradingagents.llm_clients.bedrock_client.ChatBedrockConverse.__init__", return_value=None)
class TestBedrockClientGetLlm(unittest.TestCase):
    @patch.dict(os.environ, {"AWS_REGION": "us-west-2"}, clear=False)
    def test_uses_aws_region(self, mock_init):
        client = BedrockClient("claude-haiku")
        llm = client.get_llm()
        self.assertIsInstance(llm, NormalizedChatBedrockConverse)
        mock_init.assert_called_once_with(
            model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0",
            region_name="us-west-2",
        )

    @patch.dict(os.environ, {"AWS_DEFAULT_REGION": "eu-west-1"}, clear=False)
    def test_falls_back_to_default_region(self, mock_init):
        env = os.environ.copy()
        env.pop("AWS_REGION", None)
        with patch.dict(os.environ, env, clear=True):
            os.environ["AWS_DEFAULT_REGION"] = "eu-west-1"
            client = BedrockClient("claude-sonnet")
            client.get_llm()
            mock_init.assert_called_once_with(
                model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
                region_name="eu-west-1",
            )

    @patch.dict(os.environ, {}, clear=True)
    def test_defaults_to_us_east_1(self, mock_init):
        client = BedrockClient("claude-opus")
        client.get_llm()
        mock_init.assert_called_once_with(
            model_id="us.anthropic.claude-opus-4-20250514-v1:0",
            region_name="us-east-1",
        )

    @patch.dict(os.environ, {"AWS_REGION": "us-east-1"}, clear=False)
    def test_passthrough_kwargs(self, mock_init):
        client = BedrockClient("claude-haiku", max_tokens=1024, temperature=0.5)
        client.get_llm()
        mock_init.assert_called_once_with(
            model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0",
            region_name="us-east-1",
            max_tokens=1024,
            temperature=0.5,
        )


class TestNormalizedInvokeRetry(unittest.TestCase):
    @patch("tradingagents.llm_clients.bedrock_client.ChatBedrockConverse.__init__", return_value=None)
    def test_retries_on_client_error(self, _):
        llm = NormalizedChatBedrockConverse(model_id="x", region_name="us-east-1")
        error_response = {"Error": {"Code": "ThrottlingException", "Message": "slow down"}}
        success = MagicMock()
        success.content = "ok"

        with patch(
            "langchain_aws.ChatBedrockConverse.invoke",
            side_effect=[ClientError(error_response, "Invoke"), success],
        ):
            result = llm.invoke("hello")
            self.assertEqual(result.content, "ok")


class TestNormalizedInvokeNormalize(unittest.TestCase):
    @patch("tradingagents.llm_clients.bedrock_client.ChatBedrockConverse.__init__", return_value=None)
    def test_normalizes_list_content(self, _):
        llm = NormalizedChatBedrockConverse(model_id="x", region_name="us-east-1")
        response = MagicMock()
        response.content = [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}]

        with patch("langchain_aws.ChatBedrockConverse.invoke", return_value=response):
            result = llm.invoke("test")
            self.assertEqual(result.content, "hello\nworld")


class TestFactory(unittest.TestCase):
    def test_creates_bedrock_client(self):
        client = create_llm_client("bedrock", "claude-sonnet")
        self.assertIsInstance(client, BedrockClient)


class TestValidateModel(unittest.TestCase):
    def test_always_true(self):
        client = BedrockClient("anything")
        self.assertTrue(client.validate_model())


if __name__ == "__main__":
    unittest.main()
