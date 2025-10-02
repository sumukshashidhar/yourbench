import asyncio
from unittest.mock import AsyncMock, patch

from yourbench.utils.inference.inference_core import InferenceCall, Model, _get_response


class _DummyResponse:
    def __init__(self, content: str):
        message = type("Message", (), {"content": content})()
        choice = type("Choice", (), {"message": message})()
        self.choices = [choice]


def test_get_response_merges_extra_parameters():
    created_clients = []

    class _DummyClient:
        def __init__(self, *_, **__):
            self.latest_kwargs = None
            self.chat_completion = AsyncMock(side_effect=self._chat_completion)
            created_clients.append(self)

        async def _chat_completion(self, **kwargs):
            self.latest_kwargs = kwargs
            return _DummyResponse("ok")

    model = Model(
        model_name="openrouter/test",
        provider=None,
        base_url="https://example.com/v1",
        api_key="token",
        bill_to=None,
        max_concurrent_requests=2,
        encoding_name="cl100k_base",
        extra_parameters={"reasoning": {"effort": "medium"}},
    )
    call = InferenceCall(
        messages=[{"role": "user", "content": "Hello"}],
        temperature=None,
        tags=["unit"],
        extra_parameters={"metadata": {"trace": True}},
    )

    async def _run():
        with patch("yourbench.utils.inference.inference_core.AsyncInferenceClient", _DummyClient):
            return await _get_response(model, call)

    response_text, metrics = asyncio.run(_run())

    assert response_text == "ok"
    assert metrics.success is True
    assert metrics.model_name == "openrouter/test"

    assert created_clients, "Expected AsyncInferenceClient to be instantiated"
    sent_kwargs = created_clients[0].latest_kwargs
    assert sent_kwargs["extra_body"] == {
        "reasoning": {"effort": "medium"},
        "metadata": {"trace": True},
    }
    assert sent_kwargs["messages"] == call.messages
