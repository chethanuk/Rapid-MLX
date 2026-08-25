# Issue #2222 — Weather routing + strict-format: reproduction + root-cause classification (2026-08-25)

Owner: ds0732 (temporary human-owner exception; RC2 path untouched). Worktree:
`fix/issue-2222-weather-routing` from `origin/main` `bf2ff335`. No code changed yet —
this documents baseline reproduction + classification before disposition.

## Reported symptoms (issue #2222, filed at base `4bbca765`, build 0.12.18)
1. Explicit Weather prompt routed to `web_search`; the answer claimed no Weather tool
   was available while still listing `weather` in its own advertised inventory.
2. "Write exactly two sentences" prompt produced one long sentence.

## Baseline reproduction (current origin/main, model `ornith-1.5-9b-bf16`)

Request shape replicated the Desktop contract: `tools` = [weather, web_search] with
the exact shipped `WeatherTool` + `WebSearchTool` schemas (including their
"not web_search" / "not for current weather" cross-references), `tool_choice: "auto"`
(free-typed prompt → no `forcedTool`; the #2244 native router), temperature 0.7.

Weather prompt (`What is the current weather in Tokyo? Use the Weather tool, not web
search, and report the tool result.`):
- 5/5 samples → tool call `weather` with valid args
  (`{"location":"Tokyo","country":"Japan","units":"metric"}` variant).
- No web_search, no "Weather unavailable" messaging, no permission ask.

Strict-format prompt (`Explain why local AI can be useful. Write exactly two sentences,
no heading or bullets.`):
- 5/5 samples → exactly two sentences (correct terminal punctuation).

Control model `qwen3.5-9b-4bit` (same arch family, reference):
- Weather 4/4 → `weather`; strict format 4/4 → two sentences.

Effective system/tool-choice inputs (captured from server): the request carried both
tool schemas + `tool_choice:auto`; the rendered prompt registered ~601 tokens (tools +
system content). First-round request carries no tool-result grounding preamble — it
matches the Desktop's free-typed turn exactly.

## Root-cause classification

### Symptom 1 — Weather routed to web_search: Rapid product-contract bug, ALREADY FIXED
The old Desktop routing (present at the issue's base) forced `web_search` for any
free-typed prompt whose text contained a live-evidence phrase, including
`"current weather"` / `"当前"` (see removed `forcedToolForUserTurn` /
`promptRequiresFreshWebEvidence` in `ChatViewModel`). Because the issue's prompt
contains "current weather", the app force-dispatched `web_search` even though the
`weather` tool was enabled — the observed misroute.

PR **#2244** ("fix(mac): preserve Qwen context and weather routing") removed that
heuristic and adopted **native schema-driven tool routing** (send both schemas with
`tool_choice:auto`; the model picks; the schemas cross-reference `weather`↔`web_search`).
Verified at current `origin/main`: the exact prompt routes 5/5 to `weather` (both
Ornith 1.5 9B and the qwen3.5-9b-4bit control). No product code change required.

Note: when the Weather tool is genuinely absent from the supplied list (old
single-tool contract, `weather_only_web_search`), the model truthfully reports no
Weather tool is available and offers web search — that is correct, not a contradiction.

### Symptom 2 — strict sentence count: model capability, NOT a product defect
At current head both the reported model and the control satisfy the constraint. There
is no product code path that can enforce a strict sentence count without a prompt
rewrite/trick (explicitly a non-goal). This is an instruction-following / eval-coverage
observation, not a routing or contract defect.

## Proposed disposition (per the anti-scope-creep rule)
- The two symptoms do NOT share a product-contract defect.
- Weather routing: already fixed by #2244 on current main → verify + close/dispose as
  resolved, no product change.
- Strict format: split/disposition as a separate model-eval format evidence item
  (add a model-agnostic eval case), no product fix.

## Candidates for in-scope, model-agnostic regression coverage (pending Atlas go/no-go)
The shipped `evals/prompts/tool_calling.json` suite cannot currently exercise
weather-vs-web_search disambiguation: its global tool list has no `weather` and
scenarios cannot override the tool subset. A model-agnostic eval case that sends BOTH
schemas with `tool_choice:auto` and asserts `weather` for an explicit current-weather
request would lock the corrected contract. This is within allowed scope
("focused model-agnostic regression/eval coverage"), touches no product routing, and
is not a full-ci/RC change. Any such change awaits Atlas approval (scope rule).

## Acceptance status
1. Baseline reproduction + root-cause classification documented: DONE (this doc).
2. Fix model-agnostic, uses existing tool contracts: Weather already uses native
   schema routing (#2244); no new product code needed.
3. Explicit Weather request → `weather`, never contradictory messaging: verified 5/5.
4. Disabled/missing Weather, auto choice, unrelated tools not regressed: verified
   (`weather_only_web_search` truthful fallback; auto + both tools healthy).
5. Strict-format: satisfied by the model family; disposed/split as separate eval item.
6. Local Release build + real-model dogfood + focused tests + review + PR validation:
   repro done via local server + real model; remaining PR/review only if Atlas approves
   the eval-coverage change.
7. No full-ci until Atlas clears RC2 lanes: respected.

## Repro commands (reproducible)
```
python3.12 -m vllm_mlx.cli serve ornith-1.5-9b-bf16 --port 8899 \
  --tool-call-parser hermes --reasoning-parser qwen3 --log-level DEBUG
python3.12 /tmp/issue2222-evidence/repro.py     # both prompts + diagnostics
python3.12 /tmp/issue2222-evidence/multisample.py 5
# control
python3.12 -m vllm_mlx.cli serve qwen3.5-9b-4bit --port 8898 \
  --tool-call-parser hermes --reasoning-parser qwen3
```
Raw evidence: `/tmp/issue2222-evidence/{results.json,control_results.txt,multisample output}`.

## Verification of the eval-coverage change (2026-08-25)
The approved change (WEATHER_TOOL + WEB_SEARCH_TOOL + `_resolve_tools` in
`evals/run_eval.py`, scenario `tc31-weather-explicit`) was run once against the cached
`ornith-1.5-9b-bf16` model, then updated for Codex round-1 findings:
- tc31 advertises the two **Desktop-authentic** schemas inline (WeatherTool +
  WebSearchTool, including the "do not use web_search for current weather" guard) so it
  models the real Desktop two-schema contract, not a synthetic setup.
- `verify_final_text` (opt-in) runs one more non-streaming completion after the tool
  result and requires non-empty final content; tc31 sets it. Verified end-to-end:
  the first call routes to **`weather`**, and after feeding the weather result the final
  completion yields a clean, non-contradictory narrative report (no "web_search
  unavailable" claim).
- `_resolve_tools` now supports name refs OR inline schema dicts and **fails fast** on an
  unknown tool name / malformed entry (rather than silently dropping, which could let a
  routing case pass by omitting web_search). Scenarios without `tools` keep the shared
  `TOOLS` list unchanged.
- Unit checks: JSON valid; `run_eval.py` parses; resolver resolves `[weather,
  web_search]`, keeps default behavior, raises on `weather`+`web_seach` typo.
- Caveat: the eval suite's `stream_chat` auto-grades only structured
  `delta.tool_calls` in SSE. This model family emits the tool call as streamed content
  text, so the streaming auto-grade reports "no tool call" for every tool-detection
  scenario (tc01–tc17, tc21–tc31) — a PRE-EXISTING harness × model limitation, present
  on the unmodified harness, not a regression from this change. Models that emit
  structured streaming tool_calls auto-grade normally. The scenario encodes the
  corrected contract (weather over web_search) and is model-agnostic by construction.

