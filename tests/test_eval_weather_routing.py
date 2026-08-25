# SPDX-License-Identifier: Apache-2.0
"""Offline tests for the issue #2222 weather-routing eval coverage.

We cannot start a model here (CI has no resident model), so these tests exercise
the pure, decision-relevant helpers of ``evals/run_eval.py`` — the tool-subset
resolver and the forbidden-tool rejection logic — and the scenario's own
configuration. They prove the NEGATIVE paths that guard the #2222 contract
(weather selected, never web_search):
  * a response that calls weather AND web_search must be rejected,
  * a final completion that calls web_search must be rejected even when its text
    carries a result marker,
  * a weather-only response must pass,
  * a malformed tools config must fail fast.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_EVAL = _REPO_ROOT / "evals" / "run_eval.py"
_PROMPTS = _REPO_ROOT / "evals" / "prompts" / "tool_calling.json"


def _load():
    spec = importlib.util.spec_from_file_location("eval_run_eval", _EVAL)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def re():
    return _load()


def _tc():
    data = json.loads(_PROMPTS.read_text())
    return next(c for c in data if c["id"] == "tc31-weather-explicit")


def _call(name):
    return {"id": "c", "type": "function", "function": {"name": name, "arguments": "{}"}}


class TestForbiddenToolRejection:
    def test_weather_plus_web_search_first_turn_rejected(self, re):
        # issue #2222: "use weather, not web search" — calling web_search at all,
        # even after a correct weather call, violates the contract.
        sc = _tc()
        assert re._forbidden_tool_names(
            [_call("weather"), _call("web_search")], sc["forbid_tools"]
        ) == ["web_search"]

    def test_weather_only_ok(self, re):
        sc = _tc()
        assert re._forbidden_tool_names([_call("weather")], sc["forbid_tools"]) == []

    def test_unrelated_extra_tool_not_forbidden(self, re):
        sc = _tc()
        # forbid_tools only bans web_search; an unrelated extra is a separate concern.
        assert re._forbidden_tool_names(
            [_call("weather"), _call("exec")], sc["forbid_tools"]
        ) == []

    def test_empty_or_none_calls_no_forbidden(self, re):
        sc = _tc()
        assert re._forbidden_tool_names([], sc["forbid_tools"]) == []
        assert re._forbidden_tool_names(None, sc["forbid_tools"]) == []

    def test_final_completion_calling_forbidden_tool_rejected(self, re):
        # The exact round-5 shape: final text carries a result marker AND the final
        # completion calls web_search. Must be rejected.
        sc = _tc()
        final_calls = [_call("weather"), _call("web_search")]
        assert re._forbidden_tool_names(
            final_calls, sc["forbid_tools"]
        ) == ["web_search"]

    def test_scenario_forbids_web_search(self, re):
        sc = _tc()
        assert sc["forbid_tools"] == ["web_search"]
        assert sc["verify_final_text"] is True
        assert sc["first_call_stream"] is False


class TestResolveTools:
    def test_tc31_resolves_both_desktop_schemas(self, re):
        sc = _tc()
        names = [t["function"]["name"] for t in re._resolve_tools(sc)]
        assert names == ["weather", "web_search"]
        web = next(
            t for t in re._resolve_tools(sc) if t["function"]["name"] == "web_search"
        )
        # The Desktop-authentic web_search carries the weather cross-reference.
        assert "Do not use it for current weather" in web["function"]["description"]

    def test_absent_tools_defaults_to_shared_list(self, re):
        tools = re._resolve_tools({})
        assert len(tools) == len(re.TOOLS)
        assert all(t["function"]["name"] != "weather" for t in tools)

    def test_unknown_tool_name_fails_fast(self, re):
        with pytest.raises(ValueError, match="unknown tool name"):
            re._resolve_tools({"id": "x", "tools": ["weather", "web_seach"]})

    @pytest.mark.parametrize("bad", [[], {"name": "weather"}, 123, "weather"])
    def test_malformed_tools_fails_fast(self, re, bad):
        with pytest.raises(ValueError, match="malformed tools"):
            re._resolve_tools({"id": "x", "tools": bad})
