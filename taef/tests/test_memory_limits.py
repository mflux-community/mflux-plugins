"""The plugin leaves MLX's process-wide memory limits to mflux and to whoever runs the command."""

from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
# mflux sets the cache limit from --low-ram and --mlx-cache-limit-gb; mx.metal.* is the deprecated spelling of the same
# calls.
FORBIDDEN = ("set_wired_limit", "set_memory_limit", "set_cache_limit", "mx.metal.")


@pytest.mark.parametrize("name", FORBIDDEN)
def test_the_plugin_never_sets_an_mlx_memory_limit(name: str) -> None:
    """Bug: the plugin calls mx.set_cache_limit (or the wired or memory limit, or mx.metal.*) and silently overrides
    the limit mflux set from --low-ram or --mlx-cache-limit-gb, for the whole process."""
    sources = sorted(SRC.rglob("*.py"))
    assert sources, f"no Python files under {SRC}"
    found = [path.relative_to(SRC).as_posix() for path in sources if name in path.read_text(encoding="utf-8")]
    assert found == [], f"{name} appears in {found}"
