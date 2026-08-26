#!/usr/bin/env python3
"""Run one real image request through a freshly built Desktop sidecar."""

from __future__ import annotations

import argparse
import base64
import json
import os
import signal
import socket
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path


def _free_local_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _completion_is_sensible(text: object) -> bool:
    """Reject empty/error output and require recognition of the known fixture."""
    if not isinstance(text, str):
        return False
    words = {word.strip(".,!?;:()[]{}\"'").lower() for word in text.split()}
    fixture_labels = {"cheetah", "leopard", "cat", "feline"}
    uncertainty = {
        "cannot",
        "can't",
        "unable",
        "unclear",
        "unknown",
        "unsure",
        "not",
        "no",
    }
    return len(words) >= 3 and bool(words & fixture_labels) and not words & uncertainty


def _request_json(url: str, payload: dict | None, timeout: float) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"{url} returned HTTP {response.status}")
        return json.load(response)


def _wait_until_ready(base_url: str, process: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"sidecar server exited early with {process.returncode}")
        try:
            _request_json(f"{base_url}/v1/models", None, 2)
            return
        except (OSError, ValueError, RuntimeError) as exc:
            last_error = exc
            time.sleep(0.5)
    raise RuntimeError(f"sidecar server was not ready after {timeout}s: {last_error}")


def _resolve_model(model: str, revision: str | None) -> Path:
    local_path = Path(model)
    if local_path.exists():
        return local_path
    if not revision:
        raise SystemExit(
            "vision smoke: a repository model requires --revision so the "
            "release proof is content-addressed"
        )
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=model,
            revision=revision,
            local_files_only=True,
        )
    )


def _stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sidecar-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--startup-timeout", type=float, default=240)
    parser.add_argument("--request-timeout", type=float, default=240)
    args = parser.parse_args()

    executable = args.sidecar_root / "bin" / "rapid-mlx"
    for path, label in (
        (executable, "sidecar executable"),
        (args.image, "image"),
    ):
        if not path.exists():
            raise SystemExit(f"vision smoke: {label} not found: {path}")
    model = _resolve_model(args.model, args.revision)

    port = _free_local_port()
    base_url = f"http://127.0.0.1:{port}"
    with tempfile.NamedTemporaryFile(
        prefix="rapid-sidecar-vision-", suffix=".log", delete=False
    ) as log:
        log_path = Path(log.name)
        process = subprocess.Popen(
            [str(executable), "serve", str(model), "--mllm", "--port", str(port)],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, "PYTHONNOUSERSITE": "1"},
        )

    succeeded = False
    try:
        _wait_until_ready(base_url, process, args.startup_timeout)
        image = base64.b64encode(args.image.read_bytes()).decode("ascii")
        payload = {
            "model": str(model),
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe the main animal in this image in one short sentence.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64," + image},
                        },
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 64,
            "stream": False,
        }
        body = _request_json(
            f"{base_url}/v1/chat/completions", payload, args.request_timeout
        )
        content = body.get("choices", [{}])[0].get("message", {}).get("content")
        if not _completion_is_sensible(content):
            raise RuntimeError(
                f"vision smoke returned an implausible description: {content!r}"
            )
        print(f"vision smoke: HTTP 200; description={content!r}")
        succeeded = True
        return 0
    except Exception:
        print(f"vision smoke server log: {log_path}")
        print(log_path.read_text(errors="replace"))
        raise
    finally:
        _stop_process(process)
        if succeeded:
            log_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
