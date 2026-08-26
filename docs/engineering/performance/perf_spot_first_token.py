#!/usr/bin/env python3
"""Observe first-token timing without changing CLI benchmark semantics."""

import sys
import time
from collections.abc import Callable


class FirstRequestTTFT:
    """Measure the first non-empty token for only the first admitted request."""

    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        self._request_id: str | None = None
        self._started_at: float | None = None
        self._emitted = False

    def admitted(self, request_id: str) -> None:
        if self._request_id is None:
            self._request_id = request_id
            self._started_at = self._clock()

    def observe(self, request_id: str, *, has_token: bool) -> float | None:
        if (
            request_id != self._request_id
            or not has_token
            or self._emitted
            or self._started_at is None
        ):
            return None
        self._emitted = True
        return self._clock() - self._started_at


def main() -> None:
    from vllm_mlx.engine_core import AsyncEngineCore

    add_request = AsyncEngineCore.add_request
    stream_outputs = AsyncEngineCore.stream_outputs
    observer = FirstRequestTTFT()

    async def observed_add_request(self, *args, **kwargs):
        request_id = await add_request(self, *args, **kwargs)
        observer.admitted(request_id)
        return request_id

    async def observed_stream_outputs(self, request_id, *args, **kwargs):
        async for output in stream_outputs(self, request_id, *args, **kwargs):
            elapsed = observer.observe(
                request_id,
                has_token=bool(output.new_token_ids or output.new_text),
            )
            if elapsed is not None:
                print(
                    f"PERF_TTFT request_id={request_id} seconds={elapsed:.9f}",
                    file=sys.stderr,
                    flush=True,
                )
            yield output

    AsyncEngineCore.add_request = observed_add_request
    AsyncEngineCore.stream_outputs = observed_stream_outputs

    from vllm_mlx.cli import cli_entrypoint

    cli_entrypoint()


if __name__ == "__main__":
    main()
