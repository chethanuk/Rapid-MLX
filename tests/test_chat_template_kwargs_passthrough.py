# SPDX-License-Identifier: Apache-2.0
"""Test that ``chat_template_kwargs`` extra keys reach ``apply_chat_template``.

Pins issue #2474: ``chat_template_kwargs["reasoning_effort"]`` was silently
dropped before the template render, so Qwen3.8 was permanently pinned to its
``xhigh`` template default. The fix threads the client-supplied dict through
route → engine → ``shared_apply_chat_template``, merging unknown keys into
``template_kwargs`` without overriding server-resolved values.
"""

from __future__ import annotations

from vllm_mlx.utils.chat_template import apply_chat_template


class FakeTokenizer:
    """Minimal tokenizer that records the kwargs it receives."""

    def __init__(self):
        self.received_kwargs: dict | None = None

    def apply_chat_template(self, messages, **kwargs):
        self.received_kwargs = kwargs
        return "rendered"


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
            chat_template_kwargs={"tools": [{"type": "function", "function": {"name": "evil"}}]},
        )
        assert tok.received_kwargs["tools"] == server_tools
