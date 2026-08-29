# SPDX-License-Identifier: Apache-2.0
"""Test that ``chat_template_kwargs`` extra keys reach ``apply_chat_template``.

Pins issue #2474: ``chat_template_kwargs["reasoning_effort"]`` was silently
dropped before the template render, so Qwen3.8 was permanently pinned to its
``xhigh`` template default. The fix threads the client-supplied dict through
route → engine → ``shared_apply_chat_template``, merging unknown keys into
``template_kwargs`` without overriding server-resolved values.
"""

from __future__ import annotations

from vllm_mlx.engine.batched import BatchedEngine
from vllm_mlx.utils.chat_template import apply_chat_template


class FakeTokenizer:
    """Minimal tokenizer that records the kwargs it receives."""

    def __init__(self):
        self.received_kwargs: dict | None = None

    def apply_chat_template(self, messages, **kwargs):
        self.received_kwargs = kwargs
        return "rendered"

    def encode(self, _text):
        return [1, 2]


def _make_applicator():
    tok = FakeTokenizer()
    # Guard against the sanitiser rejecting the fake applicator.
    tok.chat_template = "fake"
    return tok


class TestChatTemplateKwargsPassthrough:
    """``chat_template_kwargs`` extra keys are merged into template_kwargs."""

    def test_reasoning_effort_reaches_the_template(self):
        tok = _make_applicator()
        apply_chat_template(
            tok,
            [{"role": "user", "content": "hi"}],
            chat_template_kwargs={"reasoning_effort": "low"},
        )
        assert tok.received_kwargs is not None
        assert tok.received_kwargs["reasoning_effort"] == "low"

    def test_server_resolved_keys_are_not_overridden(self):
        tok = _make_applicator()
        apply_chat_template(
            tok,
            [{"role": "user", "content": "hi"}],
            enable_thinking=False,
            chat_template_kwargs={"enable_thinking": True},
        )
        assert tok.received_kwargs["enable_thinking"] is False

    def test_tokenize_and_add_generation_prompt_not_overridden(self):
        tok = _make_applicator()
        apply_chat_template(
            tok,
            [{"role": "user", "content": "hi"}],
            add_generation_prompt=False,
            chat_template_kwargs={
                "tokenize": True,
                "add_generation_prompt": True,
            },
        )
        assert tok.received_kwargs["tokenize"] is False
        assert tok.received_kwargs["add_generation_prompt"] is False

    def test_none_chat_template_kwargs_is_a_noop(self):
        tok = _make_applicator()
        apply_chat_template(tok, [{"role": "user", "content": "hi"}])
        assert tok.received_kwargs is not None
        assert "reasoning_effort" not in tok.received_kwargs

    def test_tools_key_not_overridden_by_client(self):
        tok = _make_applicator()
        server_tools = [{"type": "function", "function": {"name": "f"}}]
        apply_chat_template(
            tok,
            [{"role": "user", "content": "hi"}],
            tools=server_tools,
            chat_template_kwargs={
                "tools": [{"type": "function", "function": {"name": "evil"}}]
            },
        )
        assert tok.received_kwargs["tools"] == server_tools


class TestBatchedEngineChatTemplateKwargs:
    """All BatchedEngine chat entry points preserve template kwargs."""

    def _engine(self) -> BatchedEngine:
        received: dict = {}
        engine = BatchedEngine("test-model")
        engine._loaded = True
        engine._prepare_cache_stable_messages = lambda messages: (messages, None)

        def apply_chat_template(*_args, **kwargs):
            received.update(kwargs.get("chat_template_kwargs") or {})
            return "prompt"

        engine._apply_chat_template = apply_chat_template
        engine._prepare_harmony_no_thinking_prompt = lambda prompt, **_kwargs: (
            prompt,
            None,
        )
        return engine, received

    async def test_chat_forwards_chat_template_kwargs(self):
        engine, received = self._engine()
        seen_kwargs = {}

        async def generate(**kwargs):
            seen_kwargs.update(kwargs)
            return "output"

        engine.generate = generate

        await engine.chat(
            [{"role": "user", "content": "hi"}],
            chat_template_kwargs={"reasoning_effort": "low"},
        )

        assert received == {"reasoning_effort": "low"}
        assert seen_kwargs.get("chat_template_kwargs") is None

    async def test_stream_chat_forwards_chat_template_kwargs(self):
        engine, received = self._engine()
        engine._create_output_router = lambda: None
        engine.stream_generate = lambda **kwargs: kwargs

        async def route_stream(stream, _router):
            stream["consumed"] = True
            yield "output"

        engine._stream_with_output_router = route_stream

        outputs = [
            output
            async for output in engine.stream_chat(
                [{"role": "user", "content": "hi"}],
                chat_template_kwargs={"reasoning_effort": "medium"},
            )
        ]

        assert outputs == ["output"]
        assert received == {"reasoning_effort": "medium"}

    async def test_generate_with_schema_forwards_chat_template_kwargs(
        self, monkeypatch
    ):
        class GuidedEngine(BatchedEngine):
            @property
            def supports_guided_generation(self):
                return True

            @property
            def tokenizer(self):
                return FakeTokenizer()

        engine = GuidedEngine("test-model")
        engine._loaded = True
        engine._run_guided_generation = lambda **_kwargs: '{"ok": true}'

        received = {}

        def fake_shared_apply(tokenizer, messages, **kwargs):
            received.update(kwargs.get("chat_template_kwargs") or {})
            return kwargs.get("chat_template_kwargs")

        monkeypatch.setattr(
            "vllm_mlx.engine.batched.shared_apply_chat_template",
            fake_shared_apply,
        )

        output = await engine.generate_with_schema(
            messages=[{"role": "user", "content": "hi"}],
            json_schema={"type": "object"},
            chat_template_kwargs={"reasoning_effort": "low"},
        )

        assert output.text == '{"ok": true}'
        assert received == {"reasoning_effort": "low"}
