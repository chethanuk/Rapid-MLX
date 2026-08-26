#!/usr/bin/env python3
"""Run the fixed Qwen3.8 day-0 correctness panel against a local server."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path


def _long_document() -> str:
    needles = {
        50: "The Harbor recovery code is HARBOR-7319.",
        200: "The maintenance window begins at 18:40.",
        400: "The cold-room reading is -17°C.",
        700: "The final audit sequence is 6042.",
    }
    return "\n".join(
        f"Record {index:03d}: {needles.get(index, 'Routine status is nominal.')}"
        for index in range(1, 801)
    )


def _tool(name: str, properties: dict, required: list[str]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"Call {name} with the exact requested arguments.",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            },
        }
    ]


def _cases() -> list[dict]:
    cases = [
        (
            "R1",
            "reasoning",
            "A bat and ball cost $1.10 total. The bat costs $1 more than the ball. What does the ball cost? Justify briefly.",
        ),
        (
            "R2",
            "reasoning",
            "Three boxes labeled Apples, Oranges, and Mixed are all labeled incorrectly. You may draw one fruit from one box. Explain how to relabel every box correctly.",
        ),
        (
            "R3",
            "reasoning",
            "A car travels 120 km at 60 km/h and returns 120 km at 40 km/h. Give the average speed and show total distance divided by total time.",
        ),
        (
            "R4",
            "reasoning",
            "Find the smallest positive integer that leaves remainder 1 when divided by each of 2, 3, 4, 5, and 6, and is divisible by 7. Verify it.",
        ),
        (
            "C1",
            "code",
            "Write Python merge_intervals(intervals) for closed integer intervals. Do not mutate the input. Return exactly one complete fenced Python block.",
        ),
        (
            "C2",
            "code",
            "Write Python first_unique(text) returning the first Unicode character occurring once, or None. Cover empty input. Return exactly one complete fenced Python block.",
        ),
        (
            "C3",
            "code",
            "Write Python topo_sort(graph) returning the lexicographically smallest deterministic topological order and raising ValueError on cycles. Return exactly one complete fenced Python block.",
        ),
        (
            "C4",
            "code",
            "Write Python json_pointer_get(document, pointer) implementing RFC 6901 object/array lookup, including empty pointer, ~0 and ~1. Return exactly one complete fenced Python block.",
        ),
        (
            "Z1",
            "chinese",
            "请用不超过四句中文解释为什么三门问题中换门获胜概率是三分之二。",
        ),
        (
            "Z2",
            "chinese",
            "只翻译成中文，不要解释：The cache must preserve explicit operator overrides while choosing a safe automatic default.",
        ),
        (
            "Z3",
            "chinese",
            "一个价格先上涨20%，再下降20%。请简洁说明最终相对原价的变化。",
        ),
        (
            "Z4",
            "chinese",
            "列出恰好三项可执行的启动检查。必须编号，每项不超过15个汉字。",
        ),
    ]
    result = [{"id": i, "category": c, "prompt": p} for i, c, p in cases]
    result.extend(
        [
            {
                "id": "T1",
                "category": "tool_json",
                "prompt": "Get the weather in Tokyo in celsius.",
                "tools": _tool(
                    "get_weather",
                    {
                        "city": {"type": "string"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    ["city", "unit"],
                ),
            },
            {
                "id": "T2",
                "category": "tool_json",
                "prompt": "Get the current stock price for AAPL.",
                "tools": _tool(
                    "get_stock_price", {"symbol": {"type": "string"}}, ["symbol"]
                ),
            },
            {
                "id": "T3",
                "category": "tool_json",
                "prompt": "Schedule 'Architecture review' at 2026-08-27T17:00:00Z for 30 minutes with alice@example.com and bob@example.com.",
                "tools": _tool(
                    "schedule_meeting",
                    {
                        "title": {"type": "string"},
                        "start": {"type": "string"},
                        "duration_minutes": {"type": "integer"},
                        "participants": {"type": "array", "items": {"type": "string"}},
                    },
                    ["title", "start", "duration_minutes", "participants"],
                ),
            },
            {
                "id": "T4",
                "category": "tool_json",
                "prompt": "Search documentation for hybrid prefix cache admission and return the top 3.",
                "tools": _tool(
                    "search_docs",
                    {"query": {"type": "string"}, "top_k": {"type": "integer"}},
                    ["query", "top_k"],
                ),
            },
        ]
    )
    document = _long_document()
    for case_id, question in (
        ("L1", "What is the Harbor recovery code? Return only the code."),
        ("L2", "When does the maintenance window begin? Return only the time."),
        ("L3", "What is the cold-room reading? Return only the temperature."),
        ("L4", "What is the final audit sequence? Return only the number."),
    ):
        result.append(
            {
                "id": case_id,
                "category": "long_recall",
                "prompt": f"Use this record set to answer the question.\n\n{document}\n\nQuestion: {question}",
            }
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    headers = {"Content-Type": "application/json"}
    if token := os.environ.get("API_KEY"):
        headers["Authorization"] = f"Bearer {token}"

    with args.output.open("w", encoding="utf-8") as output:
        for case in _cases():
            payload = {
                "model": args.model,
                "messages": [{"role": "user", "content": case["prompt"]}],
                "temperature": 0,
                "max_tokens": 1024,
            }
            if tools := case.get("tools"):
                payload.update(tools=tools, tool_choice="required")
            request = urllib.request.Request(
                f"{args.base_url.rstrip('/')}/chat/completions",
                data=json.dumps(payload).encode(),
                headers=headers,
                method="POST",
            )
            started = time.perf_counter()
            with urllib.request.urlopen(request, timeout=900) as response:
                body = json.load(response)
            record = {
                "id": case["id"],
                "category": case["category"],
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "response": body,
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            print(f"{case['id']}: {record['elapsed_seconds']:.3f}s")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
