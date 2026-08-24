"""Malformed Hermes calls stay in the tool loop without being executed."""

import json

from tests.test_postprocessor import _make_cfg, _make_output
from vllm_mlx.service.postprocessor import StreamingPostProcessor
from vllm_mlx.tool_parsers.hermes_tool_parser import HermesToolParser

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "browse",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    }
]
REQUEST = {"tools": TOOLS}
MALFORMED = (
    "<tool_call>\n"
    "<function=browse>\n"
    "<parameter=url>\n"
    "https://example.com/guide\n"
    "</tool_call>"
)
MALFORMED_JSON = (
    '<tool_call>{"name":"browse","arguments":{"url":'
    '"https://example.com/guide"</tool_call>'
)


def test_parser_surfaces_declared_malformed_call_with_unparseable_arguments():
    result = HermesToolParser(None).extract_tool_calls(MALFORMED, request=REQUEST)

    assert result.tools_called is True
    assert result.content is None
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call["name"] == "browse"
    # The orchestration layer must reject this before dispatch and return its
    # normal parse-error tool result to the model. Do not silently repair and
    # execute a request whose structure the model failed to close.
    try:
        json.loads(call["arguments"])
    except json.JSONDecodeError:
        pass
    else:
        raise AssertionError("malformed arguments unexpectedly became executable JSON")


def test_parser_does_not_promote_undeclared_malformed_function():
    result = HermesToolParser(None).extract_tool_calls(
        MALFORMED.replace("function=browse", "function=delete_everything"),
        request=REQUEST,
    )
    assert result.tools_called is False
    assert result.tool_calls == []
    assert result.content == MALFORMED.replace(
        "function=browse", "function=delete_everything"
    )


def test_parser_surfaces_declared_malformed_json_call_for_executor_feedback():
    result = HermesToolParser(None).extract_tool_calls(MALFORMED_JSON, request=REQUEST)

    assert result.tools_called is True
    assert result.content is None
    assert result.tool_calls[0]["name"] == "browse"
    try:
        json.loads(result.tool_calls[0]["arguments"])
    except json.JSONDecodeError:
        pass
    else:
        raise AssertionError("malformed JSON unexpectedly became executable")


def test_stream_finalize_emits_standard_tool_call_not_raw_xml():
    cfg = _make_cfg(
        enable_auto_tool_choice=True, tool_parser_instance=HermesToolParser(None)
    )
    processor = StreamingPostProcessor(cfg, tools_requested=True, request=REQUEST)
    processor.reset()

    streamed = processor.process_chunk(_make_output(MALFORMED))
    finalized = processor.finalize()
    events = [*streamed, *finalized]

    tool_events = [event for event in events if event.type == "tool_call"]
    assert len(tool_events) == 1
    call = tool_events[0].tool_calls[0]
    assert call["function"]["name"] == "browse"
    assert MALFORMED not in "".join(
        event.content or "" for event in events if event.type in {"content", "finish"}
    )
