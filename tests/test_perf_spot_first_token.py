import importlib.util
from pathlib import Path

SCRIPT = (
    Path(__file__).parents[1]
    / "docs"
    / "engineering"
    / "performance"
    / "perf_spot_first_token.py"
)
SPEC = importlib.util.spec_from_file_location("perf_spot_first_token", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_observer_emits_once_for_only_first_admitted_request() -> None:
    times = iter((10.0, 12.5))
    observer = MODULE.FirstRequestTTFT(clock=lambda: next(times))

    observer.admitted("first")
    observer.admitted("second")

    assert observer.observe("second", has_token=True) is None
    assert observer.observe("first", has_token=False) is None
    assert observer.observe("first", has_token=True) == 2.5
    assert observer.observe("first", has_token=True) is None
