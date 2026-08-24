# SPDX-License-Identifier: Apache-2.0
"""Behavioral contracts for the Cohere Command typed-channel detector."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vllm_mlx.reasoning import get_parser
from vllm_mlx.reasoning.cohere_command_parser import (
    ACTION_START,
    TEXT_END,
    TEXT_START,
    THINK_END,
    THINK_START,
    CohereCommand4ReasoningParser,
)


def _stream(
    text: str,
    chunks: list[str],
    *,
    json_mode: bool = False,
) -> tuple[str | None, str | None]:
    parser = CohereCommand4ReasoningParser()
    parser.configure_request(json_mode=json_mode)
    previous = ""
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    for chunk in chunks:
        current = previous + chunk
        delta = parser.extract_reasoning_streaming(previous, current, chunk)
        previous = current
        if delta and delta.reasoning:
            reasoning_parts.append(delta.reasoning)
        if delta and delta.content:
            content_parts.append(delta.content)
    final = parser.finish_stream()
    if final and final.reasoning:
        reasoning_parts.append(final.reasoning)
    if final and final.content:
        content_parts.append(final.content)
    return "".join(reasoning_parts) or None, "".join(content_parts) or None


def _all_two_part_splits(text: str) -> list[list[str]]:
    return [[text], *[[text[:i], text[i:]] for i in range(1, len(text))], list(text)]


class TestRegistration:
    def test_protocol_and_legacy_names_resolve_to_same_parser(self):
        assert get_parser("cohere_command4") is CohereCommand4ReasoningParser
        assert get_parser("north") is CohereCommand4ReasoningParser

    def test_aliases_use_protocol_name(self):
        aliases = json.loads(
            (Path(__file__).parent.parent / "vllm_mlx" / "aliases.json").read_text()
        )
        for alias in ("north-mini-code-4bit", "north-mini-code-bf16"):
            assert aliases[alias]["reasoning_parser"] == "cohere_command4"

    def test_raw_checkpoint_path_uses_protocol_name(self):
        from vllm_mlx.model_auto_config import detect_model_config

        config = detect_model_config("mlx-community/North-Mini-Code-1.0-4bit")
        assert config is not None
        assert config.reasoning_parser == "cohere_command4"
        assert config.tool_call_parser is None


@pytest.mark.parametrize(
    ("wire", "expected"),
    [
        (
            f"plan{THINK_END}{TEXT_START}answer{TEXT_END}",
            ("plan", "answer"),
        ),
        (
            f"{THINK_START}plan{THINK_END}{TEXT_START}answer{TEXT_END}",
            ("plan", "answer"),
        ),
        (f"{TEXT_START}direct{TEXT_END}", (None, "direct")),
        (
            f"plan{THINK_END}{ACTION_START}[{{\"tool_name\":\"f\"}}]<|END_ACTION|>",
            ("plan", f'{ACTION_START}[{{"tool_name":"f"}}]<|END_ACTION|>'),
        ),
        ("unfinished thought", ("unfinished thought", None)),
        (f"plan{THINK_END}{TEXT_START}partial", ("plan", "partial")),
    ],
)
def test_full_parse_protocol_shapes(wire, expected):
    assert CohereCommand4ReasoningParser().extract_reasoning(wire) == expected


@pytest.mark.parametrize(
    "wire",
    [
        f"plan{THINK_END}{TEXT_START}answer{TEXT_END}",
        f"{THINK_START}plan{THINK_END}{TEXT_START}answer{TEXT_END}",
        f"{TEXT_START}direct{TEXT_END}",
        f"plan{THINK_END}{ACTION_START}[{{\"tool_name\":\"f\"}}]<|END_ACTION|>",
        "unfinished thought",
        f"plan{THINK_END}{TEXT_START}partial",
    ],
)
def test_streaming_is_invariant_at_every_boundary(wire):
    expected = CohereCommand4ReasoningParser().extract_reasoning(wire)
    for chunks in _all_two_part_splits(wire):
        assert _stream(wire, chunks) == expected, chunks


@pytest.mark.parametrize("document", ['{"answer":4}', "\n[1,2]", '"ok"', "42", "true", "null"])
def test_json_mode_routes_bare_json_to_content(document):
    parser = CohereCommand4ReasoningParser()
    assert parser.extract_reasoning(document, json_mode=True) == (None, document)
    for chunks in _all_two_part_splits(document):
        assert _stream(document, chunks, json_mode=True) == (None, document)


def test_json_looking_thought_stays_private_without_json_request():
    document = '{"draft":1} — reconsider'
    assert CohereCommand4ReasoningParser().extract_reasoning(document) == (
        document,
        None,
    )


def test_nonstream_orchestrator_passes_json_request_contract():
    from vllm_mlx.service.helpers import _finalize_content_and_reasoning

    document = '{"answer":4}'
    content, reasoning = _finalize_content_and_reasoning(
        raw_text=document,
        cleaned_text=document,
        tool_calls=[],
        reasoning_parser=CohereCommand4ReasoningParser(),
        json_mode=True,
    )
    assert content == document
    assert reasoning is None


def test_action_marker_is_preserved_for_downstream_tool_parser():
    action = f'{ACTION_START}[{{"tool_name":"weather","parameters":{{}}}}]<|END_ACTION|>'
    wire = f"check forecast{THINK_END}{action}"
    reasoning, content = _stream(wire, list(wire))
    assert reasoning == "check forecast"
    assert content == action


def test_finish_releases_partial_marker_in_reasoning_phase():
    wire = "thinking<|END_THI"
    assert _stream(wire, ["thinking", "<|END_THI"]) == (wire, None)


def test_finish_releases_partial_text_end_as_answer_bytes():
    wire = f"plan{THINK_END}{TEXT_START}answer<|END_TE"
    assert _stream(wire, ["plan", THINK_END, TEXT_START, "answer<|END_TE"]) == (
        "plan",
        "answer<|END_TE",
    )


def test_finish_is_idempotent():
    parser = CohereCommand4ReasoningParser()
    parser.extract_reasoning_streaming("", "abc<|END_THI", "abc<|END_THI")
    assert parser.finish_stream() is not None
    assert parser.finish_stream() is None


def test_configure_request_resets_incremental_state():
    parser = CohereCommand4ReasoningParser()
    parser.extract_reasoning_streaming("", THINK_END, THINK_END)
    parser.configure_request(json_mode=True)
    message = parser.extract_reasoning_streaming("", '{"ok":', '{"ok":')
    assert message is None
    final = parser.finish_stream()
    assert final is not None
    assert final.reasoning is None
    assert final.content == '{"ok":'


def test_forced_reasoning_end_keeps_later_model_close_structural():
    parser = CohereCommand4ReasoningParser()
    first = parser.extract_reasoning_streaming("", "abcdefgh", "abcdefgh")
    assert first is not None and first.reasoning == "abcdefgh"

    parser.prepare_forced_reasoning_end()
    assert parser.extract_reasoning_streaming("", THINK_END, THINK_END) is None
    tail = f"ijkl{THINK_END}{TEXT_START}done{TEXT_END}"
    message = parser.extract_reasoning_streaming("", tail, tail)
    assert message is not None
    assert message.reasoning is None
    assert message.content == "ijkldone"


def test_prompt_priming_detects_command_markers_and_mixed_templates():
    from vllm_mlx.service.helpers import _should_start_in_thinking

    template = (
        "{# historical <think></think> markers #}"
        "{% if add_generation_prompt %}<|START_THINKING|>{% endif %}"
    )
    assert _should_start_in_thinking(template, None, unconditional=True) is True


def test_request_flags_keep_parser_active_for_implicit_protocol():
    parser = CohereCommand4ReasoningParser()
    assert parser.sanitize_when_thinking_disabled is True
    assert parser.implicit_reasoning_until_close is True
    assert parser.reasoning_end_str == THINK_END


class TestChatRouteStreaming:
    @staticmethod
    def _read_channels(response_text: str) -> tuple[str, str]:
        reasoning_parts: list[str] = []
        content_parts: list[str] = []
        for event in response_text.split("\n\n"):
            for line in event.splitlines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                try:
                    chunk = json.loads(line.removeprefix("data: "))
                except json.JSONDecodeError:
                    continue
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta", {})
                    if delta.get("reasoning_content"):
                        reasoning_parts.append(delta["reasoning_content"])
                    if delta.get("content"):
                        content_parts.append(delta["content"])
        return "".join(reasoning_parts), "".join(content_parts)

    @pytest.mark.parametrize(
        ("deltas", "finish_reason", "expected_reasoning", "expected_content"),
        [
            (
                ["Provide answer: 4.", THINK_END, TEXT_START, "4", TEXT_END],
                "stop",
                "Provide answer: 4.",
                "4",
            ),
            (
                ["deliberating", " about it<|END_THI"],
                "length",
                "deliberating about it<|END_THI",
                None,
            ),
            (
                ["plan", THINK_END, TEXT_START, "answer<|END_TE"],
                "length",
                "plan",
                "answer<|END_TE",
            ),
        ],
    )
    def test_server_sse_protocol_and_eof_drain(
        self,
        deltas,
        finish_reason,
        expected_reasoning,
        expected_content,
    ):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from vllm_mlx.config import reset_config
        from vllm_mlx.engine.base import GenerationOutput
        from vllm_mlx.routes.chat import router as chat_router

        class Engine:
            preserve_native_tool_format = False
            is_mllm = False
            supports_guided_generation = False
            tokenizer = None

            def build_prompt(self, messages, tools=None, enable_thinking=None):
                return "PROMPT"

            async def stream_chat(self, messages, **kwargs):
                accumulated = ""
                for index, delta in enumerate(deltas):
                    accumulated += delta
                    final = index == len(deltas) - 1
                    yield GenerationOutput(
                        text=accumulated,
                        new_text=delta,
                        prompt_tokens=4,
                        completion_tokens=index + 1,
                        finished=final,
                        finish_reason=finish_reason if final else None,
                    )

        config = reset_config()
        try:
            config.engine = Engine()
            config.model_name = "cohere-command-test"
            config.model_registry = None
            config.reasoning_parser = CohereCommand4ReasoningParser()
            config.reasoning_parser_name = "cohere_command4"
            config.tool_parser = None
            config.no_thinking = False

            app = FastAPI()
            app.include_router(chat_router)
            response = TestClient(app).post(
                "/v1/chat/completions",
                json={
                    "model": "cohere-command-test",
                    "messages": [{"role": "user", "content": "2+2?"}],
                    "stream": True,
                    "max_tokens": 100,
                },
            )
            assert response.status_code == 200
            reasoning, content = self._read_channels(response.text)
            assert reasoning == expected_reasoning
            if expected_content is not None:
                assert content == expected_content
        finally:
            reset_config()
