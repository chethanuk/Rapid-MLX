#!/usr/bin/env python3
"""Observe first-token timing without changing CLI benchmark semantics."""

import sys
import time

from vllm_mlx.engine_core import AsyncEngineCore

_add_request = AsyncEngineCore.add_request
_stream_outputs = AsyncEngineCore.stream_outputs
_request_starts: dict[str, float] = {}


async def _observed_add_request(self, *args, **kwargs):
    request_id = await _add_request(self, *args, **kwargs)
    _request_starts[request_id] = time.perf_counter()
    return request_id


async def _observed_stream_outputs(self, request_id, *args, **kwargs):
    observed_first_token = False
    async for output in _stream_outputs(self, request_id, *args, **kwargs):
        if not observed_first_token and (output.new_token_ids or output.new_text):
            observed_first_token = True
            elapsed = time.perf_counter() - _request_starts[request_id]
            print(
                f"PERF_TTFT request_id={request_id} seconds={elapsed:.9f}",
                file=sys.stderr,
                flush=True,
            )
        yield output


AsyncEngineCore.add_request = _observed_add_request
AsyncEngineCore.stream_outputs = _observed_stream_outputs

from vllm_mlx.cli import cli_entrypoint  # noqa: E402

cli_entrypoint()
