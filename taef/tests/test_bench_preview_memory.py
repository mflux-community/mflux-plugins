"""The memory reproducer in scripts/bench_preview_memory.py: its unit matrix, dry run, resume and stop rules, record
fields and checks, summary math, footprint reader, watchdog and process-group cleanup. No test loads a model: the
child runs the fake commands below."""

import ctypes
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "bench_preview_memory.py"
TESTS = Path(__file__).resolve().parent
GiB = 1024**3
MiB = 1024**2
# struct rusage_info_v4 in the macOS SDK's sys/resource.h: a 16-byte uuid, then 28 uint64 fields before
# ri_lifetime_max_phys_footprint, 35 uint64 fields in all.
LIFETIME_PEAK_OFFSET = 240
RUSAGE_V4_SIZE = 296


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("bench_preview_memory", SCRIPT)
    assert spec is not None and spec.loader is not None, f"cannot load {SCRIPT}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bench = _load_script()
# sha256 of the bytes the fakes write as their image.
FAKE_IMAGE_SHA256 = "5d2fe555eea829ac791af899610755bc8c5dc4f350bee9abbac1d2144d005fc8"


@pytest.fixture
def script() -> ModuleType:
    return bench


def _decoder_file(home: Path) -> Path:
    """Where mlx-taef keeps the converted Z-Image decoder for a user whose home is `home` (download.get_or_convert)."""
    from mlx_taef.kernels import get_kernel

    key = get_kernel("zimage").source.cache_key(role="decoder")
    return home / ".cache" / "mlx-taef" / "converted" / f"{key}.mlx.safetensors"


# The fake commands the child runs in place of the real entry points (the child imports this module by name, so the
# test passes this folder on PYTHONPATH). Each reads sys.argv the way a real command would, and writes its files next
# to its --output, the unit's folder.


def _output_path() -> Path:
    return Path(sys.argv[sys.argv.index("--output") + 1])


def _write_preview() -> Path:
    """Where mflux-taef writes its default preview: next to the output, named <output stem>.preview.png."""
    output = _output_path()
    preview = output.with_name(output.stem + ".preview.png")
    preview.write_bytes(b"fake preview")
    return preview


def fake_command() -> None:
    """A run that frees a 16 MiB buffer into MLX's cache, holds it there for a few watchdog polls, and writes its
    image; the -taef commands also write their preview, as mflux-taef does."""
    import mlx.core as mx

    freed = mx.ones((4 * MiB,))  # 16 MiB of float32
    mx.eval(freed)
    del freed
    time.sleep(0.3)
    _output_path().write_bytes(b"fake image")
    if sys.argv[0].endswith("-taef"):
        _write_preview()


def fake_command_image_only() -> None:
    _output_path().write_bytes(b"fake image")


def fake_command_no_image() -> None:
    pass


def fake_command_preview_failed() -> None:
    """mlx-taef's on_error="disable" path: an early preview exists, then a decode fails, the log says so, and the run
    still exits 0."""
    _output_path().write_bytes(b"fake image")
    _write_preview()
    print("live preview failed at step 3 (fake); disabling previews", file=sys.stderr, flush=True)


def fake_command_stale_preview() -> None:
    _output_path().write_bytes(b"fake image")
    preview = _write_preview()
    an_hour_ago = time.time() - 3600
    os.utime(preview, (an_hour_ago, an_hour_ago))


def fake_command_over_ceiling() -> None:
    import mlx.core as mx

    freed = mx.ones((4 * MiB,))  # 16 MiB of float32, far above the test's 1 MiB ceiling
    mx.eval(freed)
    del freed  # MLX keeps the buffer in its cache, so only active + cache crosses the ceiling
    time.sleep(30)


def fake_command_hangs() -> None:
    time.sleep(30)


def fake_command_fails() -> None:
    raise SystemExit(1)


def fake_command_exits_3() -> None:
    raise SystemExit(3)


def fake_command_sigkills_itself() -> None:
    os.kill(os.getpid(), signal.SIGKILL)


def fake_command_with_grandchild() -> None:
    sleeper = subprocess.Popen(["sleep", "60"])
    folder = _output_path().parent
    (folder / "child.pid").write_text(str(os.getpid()))
    (folder / "grandchild.pid").write_text(str(sleeper.pid))
    time.sleep(60)


def fake_command_reports_its_group() -> None:
    (_output_path().parent / "group.txt").write_text(str(os.getpgrp()))
    _output_path().write_bytes(b"fake image")


def _entries(plain: str, taef: str | None = None) -> dict[tuple[str, str], str]:
    taef = taef or plain
    return {
        (model, variant): f"test_bench_preview_memory:{name}"
        for model in ("turbo", "base")
        for variant, name in (("plain", plain), ("taef", taef))
    }


@pytest.fixture
def child_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The child finds the fake commands; HOME points at a folder whose mlx-taef cache holds a (fake) decoder."""
    monkeypatch.setenv("PYTHONPATH", str(TESTS))
    home = tmp_path / "home"
    decoder = _decoder_file(home)
    decoder.parent.mkdir(parents=True)
    decoder.write_bytes(b"fake decoder")
    monkeypatch.setenv("HOME", str(home))


def _gone(pid: int, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


def _kill_quietly(pid_file: Path) -> None:
    if pid_file.exists():
        try:
            os.kill(int(pid_file.read_text()), signal.SIGKILL)
        except ProcessLookupError:
            pass


def _wait_for(path: Path, timeout_s: float = 20.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert path.exists(), f"{path} never appeared"


# Helpers a test runs in a subprocess of its own (python -c "import test_bench_preview_memory as t; t.<name>()"), so
# a signal or a group kill there can never reach pytest.


def _subprocess_main() -> None:
    """Runs main() with the fake entry named in the environment, as a parent the test can signal."""
    argv = json.loads(os.environ["BENCH_TEST_ARGV"])
    sys.exit(bench.main(argv, entries=_entries(os.environ["BENCH_TEST_ENTRY"])))


def _subprocess_keep_group_unit() -> None:
    """Runs one --keep-group unit past its parent timeout; prints the outcome when this process survives the
    cleanup."""
    unit = bench.build_units(models=["turbo"], settings=["default"], reps=1)[0]
    kind = bench.run_unit(
        unit,
        Path(os.environ["BENCH_TEST_OUT"]),
        entry="test_bench_preview_memory:fake_command_with_grandchild",
        ceiling_gib=23.0,
        wall_backstop_s=60.0,
        parent_timeout_s=3.0,
        keep_group=True,
    )
    print(f"survived: {kind}", flush=True)


def _helper(name: str, env: dict[str, str], cwd: Path) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-c", f"import test_bench_preview_memory as t; t.{name}()"],
        cwd=cwd,
        env={**os.environ, "PYTHONPATH": str(TESTS), **env},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )


# Unit matrix and order


def test_the_full_matrix_has_36_units_with_every_arm_of_rep_1_before_rep_2(script: ModuleType) -> None:
    """Bug: the reps loop sits inside the arms loop, so all three reps of one arm run back to back and a slow drift
    of the machine (heat, cache) lands on one arm."""
    units = script.build_units(models=["turbo", "base"], settings=["default", "low-ram", "cache-1"], reps=3)
    names = [unit.name for unit in units]
    assert len(names) == 36
    assert names[:12] == [
        "turbo-default-plain-r1",
        "turbo-default-taef-r1",
        "turbo-low-ram-plain-r1",
        "turbo-low-ram-taef-r1",
        "turbo-cache-1-plain-r1",
        "turbo-cache-1-taef-r1",
        "base-default-plain-r1",
        "base-default-taef-r1",
        "base-low-ram-plain-r1",
        "base-low-ram-taef-r1",
        "base-cache-1-plain-r1",
        "base-cache-1-taef-r1",
    ]
    assert [unit.rep for unit in units] == [1] * 12 + [2] * 12 + [3] * 12


def test_subset_flags_select_only_the_named_models_settings_and_reps(script: ModuleType, tmp_path: Path) -> None:
    """Bug: --settings is ignored (or --reps is off by one), so a night meant for one setting runs all three."""
    ns = script.parse_args(["--out", str(tmp_path), "--models", "base", "--settings", "cache-1", "--reps", "2"])
    names = [unit.name for unit in script.units_from_args(ns)]
    assert names == ["base-cache-1-plain-r1", "base-cache-1-taef-r1", "base-cache-1-plain-r2", "base-cache-1-taef-r2"]


def test_a_relative_out_dir_puts_the_image_in_the_unit_folder(
    script: ModuleType, tmp_path: Path, child_path: None
) -> None:
    """Bug: a relative --out reaches the child unchanged, and the child runs inside the run folder, so the image goes
    to run/run/<unit>/image.png instead of the unit folder."""
    # The conftest runs each test in tmp_path, so "run" is tmp_path / "run".
    argv = ["--out", "run", "--models", "turbo", "--settings", "default", "--only", "turbo-default-plain-r1"]
    assert script.main(argv, entries=_entries("fake_command_image_only")) == 0
    assert (tmp_path / "run" / "turbo-default-plain-r1" / "image.png").read_bytes() == b"fake image"


def test_an_unknown_setting_is_refused(script: ModuleType, tmp_path: Path) -> None:
    """Bug: an unknown --settings name is dropped silently, so a typo runs nothing and the night reports success."""
    with pytest.raises(SystemExit) as raised:
        script.parse_args(["--out", str(tmp_path), "--settings", "cache-2"])
    assert raised.value.code == 2


def test_a_base_taef_unit_runs_the_taef_command_with_the_base_recipe_and_its_setting(
    script: ModuleType, tmp_path: Path
) -> None:
    """Bug: the Base unit drops --guidance 4.0 or --base-model z-image, or the setting's flag never reaches mflux, so
    the run measures a different recipe from the one the README states."""
    (unit,) = [u for u in script.build_units(models=["base"], settings=["cache-1"], reps=1) if u.variant == "taef"]
    assert script.command_for(unit) == "mflux-generate-z-image-taef"
    assert script.ENTRIES[("base", "taef")] == "mflux.extras.taef.z_image:main"
    assert script.mflux_args(unit, tmp_path / "image.png") == [
        "--model",
        "mflux-community/z-image-base-mflux-q4",
        "--base-model",
        "z-image",
        "--guidance",
        "4.0",
        "--prompt",
        "a red apple on a wooden table",
        "--steps",
        "8",
        "--seed",
        "42",
        "--width",
        "1024",
        "--height",
        "1024",
        "--output",
        str(tmp_path / "image.png"),
        "--mlx-cache-limit-gb",
        "1",
    ]


def test_a_turbo_plain_unit_runs_the_plain_mflux_command_with_low_ram(script: ModuleType, tmp_path: Path) -> None:
    """Bug: the plain arm runs the TAEF entry point (or loses --low-ram), so TAEF minus plain measures nothing."""
    (unit,) = [u for u in script.build_units(models=["turbo"], settings=["low-ram"], reps=1) if u.variant == "plain"]
    assert script.command_for(unit) == "mflux-generate-z-image-turbo"
    assert script.ENTRIES[("turbo", "plain")] == "mflux.models.z_image.cli.z_image_turbo_generate:main"
    args = script.mflux_args(unit, tmp_path / "image.png")
    assert args[:2] == ["--model", "mflux-community/z-image-turbo-mflux-q4"]
    assert "--guidance" not in args
    assert args[-1] == "--low-ram"


def test_the_child_environment_is_offline_even_when_the_caller_is_not(script: ModuleType) -> None:
    """Bug: the child inherits HF_HUB_OFFLINE=0 from the caller, so a model missing from the cache is downloaded in
    the middle of a measurement instead of failing the unit."""
    env = script.child_env({"HF_HUB_OFFLINE": "0", "PATH": "/usr/bin"})
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["PATH"] == "/usr/bin"


@pytest.mark.parametrize(
    ("model", "override", "expected"),
    [
        pytest.param("turbo", None, (600.0, 720.0), id="turbo-default"),
        pytest.param("base", None, (900.0, 1020.0), id="base-default"),
        pytest.param("base", 30.0, (30.0, 150.0), id="override"),
    ],
)
def test_each_model_has_its_wall_backstop_and_the_parent_waits_two_minutes_longer(
    script: ModuleType, model: str, override: float | None, expected: tuple[float, float]
) -> None:
    """Bug: Base gets Turbo's 600 s backstop (a slow Base run is aborted as hung), or the parent's timeout equals the
    backstop, so the parent kills a unit before its own watchdog can write the abort record."""
    assert script.unit_timeouts(model, override) == expected


# Dry run


def test_dry_run_prints_every_unit_its_command_and_the_eta(script: ModuleType, tmp_path: Path, capsys: Any) -> None:
    """Bug: the ETA counts one rep (or one model's minutes for both), so a 79-minute night is announced as 26."""
    out = tmp_path / "run"
    code = script.main(["--out", str(out), "--dry-run"])
    printed = capsys.readouterr().out
    assert code == 0
    assert printed.count("mflux-generate-z-image-turbo-taef") == 9
    assert "turbo-default-plain-r1" in printed and "base-cache-1-taef-r3" in printed
    # 18 Turbo units at 1.5 min and 18 Base units at 2.9 min.
    assert "ETA 79.2 min for 36 units" in printed
    assert not out.exists(), "a dry run created the out dir"


def test_dry_run_imports_no_mlx_mflux_or_hub_module(tmp_path: Path) -> None:
    """Bug: the script imports mlx or mflux at module level, so the dry run (and the orchestrator that holds the
    children) pays for MLX and could load what a child should load."""
    result = subprocess.run(
        [sys.executable, "-X", "importtime", str(SCRIPT), "--out", str(tmp_path / "run"), "--dry-run"],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "HF_HUB_OFFLINE": "1"},
    )
    assert result.returncode == 0, result.stderr
    imported = [line.rsplit("|", 1)[-1].strip() for line in result.stderr.splitlines() if "|" in line]
    heavy = [name for name in imported if name.split(".")[0] in {"mlx", "mflux", "mlx_taef", "huggingface_hub"}]
    assert imported, "-X importtime printed nothing; the check would pass vacuously"
    assert heavy == []


# Resume and stop


class FakeLaunch:
    """Stands in for the child launcher: returns the outcome kind it is told to for each unit and records the order."""

    def __init__(self, kinds: dict[str, str] | None = None) -> None:
        self.kinds = kinds or {}
        self.launched: list[str] = []

    def __call__(self, unit: Any) -> str:
        self.launched.append(unit.name)
        return self.kinds.get(unit.name, "ok")


def _record(
    model: str,
    setting: str,
    variant: str,
    rep: int,
    *,
    footprint: float | None = 9.0,
    peak: float = 7.0,
    wall: float = 80.0,
    cache_peak: float = 4.5,
    commit: str | None = "0123456789abcdef0123456789abcdef01234567",
    dirty: list[str] | None = None,
    image_sha256: str = FAKE_IMAGE_SHA256,
) -> dict[str, Any]:
    return {
        "image_sha256": image_sha256,
        "unit": f"{model}-{setting}-{variant}-r{rep}",
        "arm": f"{model}-{setting}-{variant}",
        "model": model,
        "setting": setting,
        "variant": variant,
        "rep": rep,
        "status": "ok",
        "started_utc": "2026-11-16T05:00:00+00:00",
        "git_commit": commit,
        "git_dirty_files": [] if dirty is None and commit is not None else dirty,
        "footprint_lifetime_peak_bytes": None if footprint is None else int(footprint * GiB),
        "mlx_peak_since_last_reset_bytes": int(peak * GiB),
        "mlx_cache_peak_bytes": int(cache_peak * GiB),
        "wall_s": wall,
        "hardware": {"chip": "Apple M1 Max", "memory_bytes": 32 * GiB, "macos": "27.2"},
        "versions": {"mflux": "0.24.0", "mlx": "0.32.3", "mlx-taef": "0.9.0", "mflux-taef": "0.1.0"},
    }


def _write_records(folder: Path, *records: dict[str, Any]) -> None:
    (folder / "records").mkdir(parents=True, exist_ok=True)
    for rec in records:
        (folder / "records" / f"{rec['unit']}.json").write_text(json.dumps(rec))


def test_resume_skips_a_unit_whose_record_exists(script: ModuleType, tmp_path: Path) -> None:
    """Bug: resume tests the unit's folder instead of its record, so a unit killed half way is skipped and never
    measured, or a finished unit runs again."""
    units = script.build_units(models=["turbo"], settings=["default"], reps=2)
    _write_records(tmp_path, _record("turbo", "default", "plain", 1))
    (tmp_path / "turbo-default-taef-r1").mkdir()  # a unit that was killed before its record was written
    launch = FakeLaunch()
    assert script.run_matrix(units, tmp_path, launch=launch, say=lambda _: None) == 0
    assert launch.launched == ["turbo-default-taef-r1", "turbo-default-plain-r2", "turbo-default-taef-r2"]


def test_resume_refuses_a_record_that_fails_its_check_and_names_the_file(script: ModuleType, tmp_path: Path) -> None:
    """Bug: resume skips a unit whenever records/<unit>.json exists, so an empty or copied-in file marks a unit as
    measured and the published median lacks a rep nobody noticed."""
    units = script.build_units(models=["turbo"], settings=["default"], reps=1)
    (tmp_path / "records").mkdir()
    (tmp_path / "records" / "turbo-default-plain-r1.json").write_text("{}")
    launch = FakeLaunch()
    with pytest.raises(script.BenchError, match=r"turbo-default-plain-r1\.json"):
        script.run_matrix(units, tmp_path, launch=launch, say=lambda _: None)
    assert launch.launched == []


def test_the_run_stops_at_the_first_watchdog_abort(script: ModuleType, tmp_path: Path) -> None:
    """Bug: an abort is treated like a failed unit, so the next unit starts on a machine that just hit the memory
    ceiling."""
    units = script.build_units(models=["turbo"], settings=["default", "low-ram"], reps=1)
    launch = FakeLaunch({"turbo-default-taef-r1": "aborted"})
    assert script.run_matrix(units, tmp_path, launch=launch, say=lambda _: None) == 3
    assert launch.launched == ["turbo-default-plain-r1", "turbo-default-taef-r1"]


def test_the_run_stops_at_the_first_killed_unit(script: ModuleType, tmp_path: Path) -> None:
    """Bug: a unit killed by a signal (a memory kill) counts as an ordinary failure and the run goes on."""
    units = script.build_units(models=["turbo"], settings=["default"], reps=2)
    launch = FakeLaunch({"turbo-default-plain-r1": "killed"})
    assert script.run_matrix(units, tmp_path, launch=launch, say=lambda _: None) == 4
    assert launch.launched == ["turbo-default-plain-r1"]


@pytest.mark.parametrize("kind", ["failed", "preview-off"])
def test_a_failed_or_preview_off_unit_lets_the_run_go_on_and_ends_with_exit_1(
    script: ModuleType, tmp_path: Path, kind: str
) -> None:
    """Bug: a failed unit (a model missing from the cache) or one whose preview switched off ends the run with exit
    0, so the summary silently lacks that arm's rep."""
    units = script.build_units(models=["turbo"], settings=["default"], reps=1)
    launch = FakeLaunch({"turbo-default-plain-r1": kind})
    assert script.run_matrix(units, tmp_path, launch=launch, say=lambda _: None) == 1
    assert launch.launched == ["turbo-default-plain-r1", "turbo-default-taef-r1"]


def test_an_earlier_unit_folder_is_renamed_aside_not_reused(script: ModuleType, tmp_path: Path) -> None:
    """Bug: the earlier attempt's folder is reused (its stale image or preview passes the new unit's checks) or
    deleted (its log, the only trace of why it failed, is lost)."""
    folder = tmp_path / "turbo-default-taef-r1"
    folder.mkdir()
    (folder / "log.txt").write_text("the earlier attempt")
    script._prepare_unit_dir(folder)
    assert list(folder.iterdir()) == []
    (aside,) = tmp_path.glob("turbo-default-taef-r1.prev-*")
    assert (aside / "log.txt").read_text() == "the earlier attempt"


def test_a_unit_folder_that_is_a_symlink_is_refused(script: ModuleType, tmp_path: Path) -> None:
    """Bug: a symlink planted at a unit's folder name passes exists() and iterdir(), so the unit writes its log and
    image into whatever folder the link points at."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (tmp_path / "turbo-default-plain-r1").symlink_to(elsewhere)
    with pytest.raises(script.BenchError, match="symbolic link"):
        script._prepare_unit_dir(tmp_path / "turbo-default-plain-r1")
    assert list(elsewhere.iterdir()) == []


def test_the_unit_log_is_never_opened_through_a_symlink(script: ModuleType, tmp_path: Path) -> None:
    """Bug: the log is opened with mode "wb", which follows a link planted at log.txt and truncates the file it
    points at."""
    victim = tmp_path / "victim.txt"
    victim.write_text("keep me")
    (tmp_path / "log.txt").symlink_to(victim)
    with pytest.raises(OSError):
        script._open_log(tmp_path / "log.txt")
    assert victim.read_text() == "keep me"


def test_write_json_never_writes_through_a_planted_temporary_name(script: ModuleType, tmp_path: Path) -> None:
    """Bug: write_json writes to the fixed name <record>.json.tmp with mode "w", which follows a link planted there
    and overwrites the file it points at."""
    victim = tmp_path / "victim.txt"
    victim.write_text("keep me")
    (tmp_path / "rec.json.tmp").symlink_to(victim)
    script.write_json(tmp_path / "rec.json", {"a": 1})
    assert victim.read_text() == "keep me"
    assert json.loads((tmp_path / "rec.json").read_text()) == {"a": 1}
    assert sorted(path.name for path in tmp_path.iterdir()) == ["rec.json", "rec.json.tmp", "victim.txt"]


# Record checks on load and resume


@pytest.mark.parametrize(
    ("change", "message"),
    [
        pytest.param({"status": "failed"}, "status", id="status-not-ok"),
        pytest.param({"unit": "turbo-default-plain-r2"}, "unit", id="unit-not-the-file-name"),
        pytest.param({"arm": "turbo-default-taef"}, "arm", id="arm-not-the-units"),
        pytest.param({"wall_s": "80"}, "wall_s", id="wall-a-string"),
        pytest.param({"mlx_peak_since_last_reset_bytes": True}, "mlx_peak_since_last_reset_bytes", id="peak-a-bool"),
        pytest.param({"footprint_lifetime_peak_bytes": False}, "footprint_lifetime_peak_bytes", id="footprint-a-bool"),
        pytest.param({"mlx_cache_peak_bytes": None}, "mlx_cache_peak_bytes", id="cache-peak-missing"),
        pytest.param({"image_sha256": "abc"}, "image_sha256", id="digest-not-sha256"),
    ],
)
def test_load_records_refuses_a_record_that_fails_its_check(
    script: ModuleType, tmp_path: Path, change: dict[str, Any], message: str
) -> None:
    """Bug: --summarize takes medians of whatever records/ holds, so a failed run, a file copied in under another
    unit's name, a record of the other arm or a non-number moves a published median."""
    rec = _record("turbo", "default", "plain", 1)
    (tmp_path / "records").mkdir()
    path = tmp_path / "records" / "turbo-default-plain-r1.json"
    path.write_text(json.dumps({**rec, **change}))
    with pytest.raises(script.BenchError, match=rf"turbo-default-plain-r1\.json.*{message}"):
        script.load_records(tmp_path)


def test_load_records_keeps_a_record_without_a_footprint(script: ModuleType, tmp_path: Path) -> None:
    """Bug: the check refuses a null footprint, which the script writes when macOS cannot report one, so a run on
    such a system cannot be summarized at all."""
    rec = _record("turbo", "default", "plain", 1, footprint=None)
    _write_records(tmp_path, rec)
    assert script.load_records(tmp_path) == [rec]


@pytest.mark.parametrize("make", ["missing-folder", "empty-records"])
def test_summarize_command_refuses_a_folder_without_records(
    script: ModuleType, tmp_path: Path, capsys: Any, make: str
) -> None:
    """Bug: --summarize on a mistyped path (or a run that recorded nothing) prints an empty summary and exits 0, so an
    outer runner reads success."""
    folder = tmp_path / "run"
    if make == "empty-records":
        (folder / "records").mkdir(parents=True)
    assert script.main(["--summarize", str(folder)]) == 2
    assert f"no records in {folder / 'records'}" in capsys.readouterr().err


def test_summarize_command_refuses_a_bad_record_with_exit_2(script: ModuleType, tmp_path: Path, capsys: Any) -> None:
    """Bug: a bad record ends --summarize in a KeyError traceback (or is skipped silently) instead of one error line
    that names the file."""
    (tmp_path / "records").mkdir()
    (tmp_path / "records" / "turbo-default-plain-r1.json").write_text("{}")
    assert script.main(["--summarize", str(tmp_path)]) == 2
    assert "turbo-default-plain-r1.json" in capsys.readouterr().err


# Summary math


def test_the_summary_takes_the_median_of_the_reps_and_taef_minus_plain(script: ModuleType) -> None:
    """Bug: the summary takes the mean (or the last rep), or subtracts TAEF from plain, so the README states the
    wrong excess."""
    records = [
        _record("turbo", "default", "plain", 1, footprint=8.5, peak=7.0, wall=80.0, cache_peak=4.0),
        _record("turbo", "default", "plain", 2, footprint=9.5, peak=7.5, wall=90.0, cache_peak=4.5),
        _record("turbo", "default", "plain", 3, footprint=9.0, peak=7.25, wall=81.0, cache_peak=5.0),
        _record("turbo", "default", "taef", 1, footprint=11.5, peak=8.0, wall=85.0, cache_peak=7.0),
        _record("turbo", "default", "taef", 2, footprint=11.75, peak=8.25, wall=83.0, cache_peak=7.5),
        _record("turbo", "default", "taef", 3, footprint=14.0, peak=8.5, wall=84.0, cache_peak=9.0),
    ]
    arms = script.summarize(records)
    assert arms["turbo-default-plain"] == {
        "n": 3,
        "n_footprint": 3,
        "footprint": 9.0 * GiB,
        "mlx_peak_since_reset": 7.25 * GiB,
        "mlx_cache_peak": 4.5 * GiB,
        "wall_s": 81.0,
    }
    assert arms["turbo-default-taef"] == {
        "n": 3,
        "n_footprint": 3,
        "footprint": 11.75 * GiB,
        "mlx_peak_since_reset": 8.25 * GiB,
        "mlx_cache_peak": 7.5 * GiB,
        "wall_s": 84.0,
    }
    assert script.taef_minus_plain(arms) == {
        ("turbo", "default"): {
            "footprint": 2.75 * GiB,
            "mlx_peak_since_reset": 1.0 * GiB,
            "mlx_cache_peak": 3.0 * GiB,
            "wall_s": 3.0,
        }
    }


def test_a_rep_without_a_footprint_is_left_out_of_that_median_only(script: ModuleType) -> None:
    """Bug: one null footprint makes the whole arm's footprint null (or counts as zero), the arm's MLX peak and wall
    vanish with it, or the count of reps with a footprint still says 2."""
    records = [
        _record("base", "cache-1", "plain", 1, footprint=None, peak=5.0, wall=160.0, cache_peak=1.0),
        _record("base", "cache-1", "plain", 2, footprint=6.0, peak=5.5, wall=162.0, cache_peak=1.0),
    ]
    arm = script.summarize(records)["base-cache-1-plain"]
    assert arm == {
        "n": 2,
        "n_footprint": 1,
        "footprint": 6.0 * GiB,
        "mlx_peak_since_reset": 5.25 * GiB,
        "mlx_cache_peak": 1.0 * GiB,
        "wall_s": 161.0,
    }
    assert script.taef_minus_plain(script.summarize(records)) == {}


def test_summarize_command_prints_the_header_and_each_delta_with_both_arms_counts(
    script: ModuleType, tmp_path: Path, capsys: Any
) -> None:
    """Bug: --summarize reads the abort records too (or none at all), so an aborted rep moves the median; or a delta
    line hides that its arms have uneven reps (3 plain, 1 TAEF after a partial resume) or fewer footprints."""
    _write_records(
        tmp_path,
        _record("turbo", "default", "plain", 1, footprint=9.0, peak=7.0, wall=80.0, cache_peak=4.0),
        _record("turbo", "default", "plain", 2, footprint=None, peak=7.0, wall=80.0, cache_peak=4.0),
        _record("turbo", "default", "plain", 3, footprint=9.0, peak=7.0, wall=80.0, cache_peak=4.0),
        _record("turbo", "default", "taef", 1, footprint=11.0, peak=8.0, wall=82.0, cache_peak=7.0),
    )
    (tmp_path / "problems").mkdir()
    aborted = _record("turbo", "default", "taef", 2, footprint=23.0, peak=23.0, wall=10.0)
    (tmp_path / "problems" / "turbo-default-taef-r2.aborted-1.json").write_text(json.dumps(aborted))
    assert script.main(["--summarize", str(tmp_path)]) == 0
    printed = capsys.readouterr().out
    assert "Apple M1 Max" in printed and "mlx 0.32.3" in printed
    assert "Commit: 0123456789abcdef0123456789abcdef01234567" in printed
    assert "Dirty files: none" in printed
    assert "MLX peak since last reset" in printed
    assert "  turbo-default-plain: n=3 (2 with a footprint), footprint 9.00 GiB" in printed
    assert (
        "  turbo default (n plain/TAEF 3/1, with a footprint 2/1): footprint +2.00 GiB, "
        "MLX peak since last reset +1.00 GiB, MLX cache peak +3.00 GiB, wall +2.0 s"
    ) in printed.splitlines()
    assert "Warning" not in printed


def test_summarize_warns_when_a_record_has_no_git_state(script: ModuleType, tmp_path: Path, capsys: Any) -> None:
    """Bug: records made outside a checkout (or without git) carry null commit fields and the summary prints numbers
    as if their source were known."""
    _write_records(
        tmp_path,
        _record("turbo", "default", "plain", 1),
        _record("turbo", "default", "taef", 1, commit=None, dirty=None),
    )
    assert script.main(["--summarize", str(tmp_path)]) == 0
    printed = capsys.readouterr().out
    assert (
        "Warning: git_commit and git_dirty_files are null in 1 of 2 records "
        "(git unavailable, or the script was not in a checkout)."
    ) in printed.splitlines()


def test_summarize_lists_the_dirty_files_of_a_dirty_tree(script: ModuleType, tmp_path: Path, capsys: Any) -> None:
    """Bug: the summary prints only the commit, so numbers measured on uncommitted changes look like that commit's."""
    _write_records(tmp_path, _record("turbo", "default", "plain", 1, dirty=[" M taef/scripts/bench_preview_memory.py"]))
    assert script.main(["--summarize", str(tmp_path)]) == 0
    assert "Dirty files:  M taef/scripts/bench_preview_memory.py" in capsys.readouterr().out


def test_summarize_says_whether_plain_and_taef_images_share_one_digest(
    script: ModuleType, tmp_path: Path, capsys: Any
) -> None:
    """Bug: the image check compares only within an arm (or counts records instead of digests), so a TAEF run whose
    final image differs from the plain run's (the preview changed the generation) reads as identical."""
    other = "6e39086d3f9591de3f3c253612c4a997e4d7fcd2e632c615d8c481573c627a19"
    _write_records(
        tmp_path,
        _record("turbo", "default", "plain", 1),
        _record("turbo", "default", "plain", 2),
        _record("turbo", "default", "taef", 1),
        _record("base", "low-ram", "plain", 1),
        _record("base", "low-ram", "taef", 1, image_sha256=other),
    )
    assert script.main(["--summarize", str(tmp_path)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert "  turbo default: one image digest across 2 plain and 1 TAEF images" in lines
    assert "  base low-ram: 2 image digests across 1 plain and 1 TAEF images" in lines


# Git state and start time


def _fake_git(tmp_path: Path, *, commit: str, status: str, fail: bool = False) -> Path:
    """A stand-in for the git executable: answers rev-parse HEAD and status --porcelain with fixed text."""
    fake = tmp_path / "git"
    rc = 128 if fail else 0
    fake.write_text(
        "#!/bin/sh\n"
        f'[ {rc} -ne 0 ] && {{ echo "fatal: not a git repository" >&2; exit {rc}; }}\n'
        f'case "$3" in rev-parse) printf "%s\\n" "{commit}";; status) printf "{status}";; esac\n'
    )
    fake.chmod(0o755)
    return fake


def test_git_state_reads_the_commit_and_every_porcelain_line(script: ModuleType, tmp_path: Path) -> None:
    """Bug: the dirty list drops untracked files (or strips the status columns), so a record from a tree with an
    uncommitted script edit reads as clean."""
    git = _fake_git(tmp_path, commit="89abcdef" * 5, status=" M taef/x.py\\n?? new file.txt\\n")
    assert script.git_state(tmp_path, git=str(git)) == {
        "git_commit": "89abcdef" * 5,
        "git_dirty_files": [" M taef/x.py", "?? new file.txt"],
    }


def test_git_state_of_a_clean_tree_is_an_empty_list(script: ModuleType, tmp_path: Path) -> None:
    """Bug: a clean tree records null (or [""]) for its dirty files, so the summary cannot tell clean from unknown."""
    git = _fake_git(tmp_path, commit="89abcdef" * 5, status="")
    assert script.git_state(tmp_path, git=str(git))["git_dirty_files"] == []


def test_git_state_is_null_outside_a_checkout_or_without_git(script: ModuleType, tmp_path: Path) -> None:
    """Bug: a failed git call records its error text as the commit, or crashes the unit, instead of null fields."""
    failing = _fake_git(tmp_path, commit="89abcdef" * 5, status="", fail=True)
    assert script.git_state(tmp_path, git=str(failing)) == {"git_commit": None, "git_dirty_files": None}
    assert script.git_state(tmp_path, git=str(tmp_path / "no-such-git")) == {
        "git_commit": None,
        "git_dirty_files": None,
    }


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_git_state_of_a_folder_outside_any_checkout_is_null(script: ModuleType, tmp_path: Path) -> None:
    """Bug: git_state runs git in the caller's folder instead of the one it is given, so a run started from inside
    some checkout records that checkout's commit."""
    assert script.git_state(tmp_path) == {"git_commit": None, "git_dirty_files": None}


def test_main_records_the_start_time_commit_dirty_files_and_cache_peak(
    script: ModuleType, tmp_path: Path, child_path: None
) -> None:
    """Bug: a record lacks its UTC start time, its commit, its dirty files or the peak MLX cache the watchdog saw, so
    a published number cannot be tied to the code and the date it came from."""
    out = tmp_path / "run"
    argv = ["--out", str(out), "--models", "turbo", "--settings", "default", "--reps", "1"]
    before = datetime.now(UTC).replace(microsecond=0)
    assert script.main(argv, entries=_entries("fake_command")) == 0
    after = datetime.now(UTC)
    rec = json.loads((out / "records" / "turbo-default-taef-r1.json").read_text())
    started = datetime.fromisoformat(rec["started_utc"])
    assert started.utcoffset() is not None and started.utcoffset().total_seconds() == 0
    assert before <= started <= after
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    if head.returncode == 0:
        assert rec["git_commit"] == head.stdout.strip()
        assert isinstance(rec["git_dirty_files"], list)
    else:
        assert rec["git_commit"] is None and rec["git_dirty_files"] is None
    # The fake frees 16 MiB into MLX's cache and keeps it there for 0.3 s, several 0.05 s polls.
    assert rec["mlx_cache_peak_bytes"] >= 16 * MiB
    assert rec["mlx_peak_since_last_reset_bytes"] >= 16 * MiB
    assert "mlx_peak_bytes" not in rec
    assert rec["image_sha256"] == FAKE_IMAGE_SHA256


# The decoder prerequisite


def test_main_refuses_before_any_unit_when_the_taef_decoder_is_missing(
    script: ModuleType, tmp_path: Path, child_path: None, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    """Bug: the TAEF decoder is not checked before the run, so every TAEF unit fails offline after its model load and
    a night of plain-only numbers looks like a finished run."""
    monkeypatch.setenv("HOME", str(tmp_path / "empty-home"))
    out = tmp_path / "run"
    argv = ["--out", str(out), "--models", "turbo", "--settings", "default", "--reps", "1"]
    assert script.main(argv, entries=_entries("fake_command")) == 5
    err = capsys.readouterr().err
    assert str(_decoder_file(tmp_path / "empty-home")) in err
    assert "run any -taef command once online first" in err
    assert not out.exists() or list(out.iterdir()) == [], "a unit ran before the refusal"


def test_a_plain_only_selection_runs_without_the_decoder(
    script: ModuleType, tmp_path: Path, child_path: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bug: the decoder check runs for every selection, so an outer runner driving plain units one by one is refused
    for a decoder those units never use."""
    monkeypatch.setenv("HOME", str(tmp_path / "empty-home"))
    argv = ["--out", str(tmp_path / "run"), "--models", "turbo", "--only", "turbo-default-plain-r1"]
    assert script.main(argv, entries=_entries("fake_command_image_only")) == 0


def test_the_decoder_check_passes_when_mlx_taef_has_the_decoder(
    script: ModuleType, tmp_path: Path, child_path: None
) -> None:
    """Bug: the check looks for a different file name or folder than mlx-taef's cache, so it refuses a run whose
    decoder is in place."""
    assert script.decoder_problem() is None


# Preview and outcome rules


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        pytest.param({}, ("ok", None), id="ok"),
        pytest.param({"timed_out": True}, ("killed", "parent timeout"), id="parent-timeout"),
        pytest.param({"rc": -9, "child_status": None}, ("killed", "signal SIGKILL"), id="signal"),
        pytest.param({"rc": 3, "child_status": "aborted"}, ("aborted", None), id="watchdog"),
        pytest.param({"rc": 3, "child_status": "failed"}, ("failed", None), id="command-exited-3"),
        pytest.param({"rc": 3, "child_status": None}, ("failed", None), id="exit-3-without-a-record"),
        pytest.param({"rc": 1, "child_status": "failed"}, ("failed", None), id="failed"),
        pytest.param({"image_exists": False}, ("failed", "no output image"), id="no-image"),
        pytest.param(
            {"preview": {"ok": False, "reason": "no preview file"}}, ("preview-off", "no preview file"), id="preview"
        ),
        pytest.param({"preview": {"ok": True, "reason": None}}, ("ok", None), id="preview-ran"),
    ],
)
def test_unit_outcome(script: ModuleType, given: dict[str, Any], expected: tuple[str, str | None]) -> None:
    """Bug: any exit code 3 counts as a watchdog abort (a command that exits 3 stops the whole matrix), an ok run with
    no image or with its preview switched off is counted, or a parent timeout is read as an abort."""
    facts = {"rc": 0, "timed_out": False, "child_status": "ok", "image_exists": True, "preview": None, **given}
    assert script.unit_outcome(**facts) == expected


def _taef_unit(script: ModuleType) -> Any:
    return [u for u in script.build_units(models=["turbo"], settings=["default"], reps=1) if u.variant == "taef"][0]


def _run_taef_unit(script: ModuleType, tmp_path: Path, fake: str) -> str:
    return script.run_unit(
        _taef_unit(script),
        tmp_path,
        entry=f"test_bench_preview_memory:{fake}",
        ceiling_gib=23.0,
        wall_backstop_s=60.0,
        parent_timeout_s=60.0,
    )


@pytest.mark.parametrize(
    ("fake", "reason"),
    [
        pytest.param("fake_command_image_only", "no preview file", id="no-preview"),
        pytest.param("fake_command_stale_preview", "preview file older than the unit", id="stale-preview"),
        pytest.param("fake_command_preview_failed", "log says live preview failed", id="log-says-failed"),
    ],
)
def test_a_taef_unit_whose_preview_switched_off_is_not_counted(
    script: ModuleType, tmp_path: Path, child_path: None, fake: str, reason: str
) -> None:
    """Bug: a TAEF run whose preview turned itself off (mlx-taef's on_error="disable" logs and goes on, exit 0) is
    recorded "ok", so its numbers, taken without a preview running, shrink the TAEF minus plain median."""
    assert _run_taef_unit(script, tmp_path, fake) == "preview-off"
    assert not (tmp_path / "records").exists()
    (problem,) = (tmp_path / "problems").glob("turbo-default-taef-r1.preview-off-*.json")
    rec = json.loads(problem.read_text())
    assert rec["status"] == "preview-off"
    assert rec["reason"] == reason
    assert rec["preview_check"]["ok"] is False


def test_a_taef_unit_with_a_fresh_preview_and_a_clean_log_is_ok(
    script: ModuleType, tmp_path: Path, child_path: None
) -> None:
    """Bug: the preview check looks for the wrong file name (or compares the mtime the wrong way), so every TAEF unit
    is refused and the night measures nothing."""
    assert _run_taef_unit(script, tmp_path, "fake_command") == "ok"
    rec = json.loads((tmp_path / "records" / "turbo-default-taef-r1.json").read_text())
    assert rec["preview_check"] == {
        "file": "turbo-default-taef-r1/image.preview.png",
        "exists": True,
        "written_after_start": True,
        "log_says_failed": False,
        "ok": True,
        "reason": None,
    }


def test_a_unit_without_its_output_image_is_failed(script: ModuleType, tmp_path: Path, child_path: None) -> None:
    """Bug: a run that exits 0 without saving its image is recorded "ok" and its numbers are counted."""
    unit = _one_unit(script)
    kind = script.run_unit(
        unit,
        tmp_path,
        entry="test_bench_preview_memory:fake_command_no_image",
        ceiling_gib=23.0,
        wall_backstop_s=60.0,
        parent_timeout_s=60.0,
    )
    assert kind == "failed"
    (problem,) = (tmp_path / "problems").glob(f"{unit.name}.failed-*.json")
    assert json.loads(problem.read_text())["reason"] == "no output image"


def test_a_command_that_exits_3_is_failed_and_the_run_goes_on(
    script: ModuleType, tmp_path: Path, child_path: None
) -> None:
    """Bug: "aborted" is decided by exit code 3 alone, so a command that exits 3 itself is reported as the watchdog
    firing and the whole matrix stops."""
    out = tmp_path / "run"
    argv = ["--out", str(out), "--models", "turbo", "--settings", "default", "--reps", "1"]
    assert script.main(argv, entries=_entries("fake_command_exits_3", "fake_command")) == 1
    (problem,) = (out / "problems").glob("turbo-default-plain-r1.failed-*.json")
    assert json.loads(problem.read_text())["exit_code"] == 3
    assert (out / "records" / "turbo-default-taef-r1.json").exists()


# The footprint reader


class FakeLibc:
    """Stands in for libproc's proc_pid_rusage: writes the given lifetime peak at the field's offset in the
    caller's buffer, or fails."""

    def __init__(self, *, peak: int | None, result: int = 0) -> None:
        self.peak, self.result = peak, result
        self.calls: list[tuple[int, int]] = []

    def proc_pid_rusage(self, pid: int, flavor: int, buffer: Any) -> int:
        self.calls.append((pid, flavor))
        if self.peak is not None:
            ctypes.memmove(ctypes.addressof(buffer) + LIFETIME_PEAK_OFFSET, self.peak.to_bytes(8, sys.byteorder), 8)
        return self.result


def test_the_footprint_reader_takes_the_lifetime_peak_field_of_rusage_v4(script: ModuleType) -> None:
    """Bug: the struct skips a field (or asks for flavor 3), so the reader returns the current footprint or an
    instructions count instead of the lifetime peak."""
    libc = FakeLibc(peak=12_345 * 1024 * 1024)
    assert script.lifetime_peak_footprint(4242, libc=libc) == 12_345 * 1024 * 1024
    assert libc.calls == [(4242, 4)]
    assert ctypes.sizeof(script.RusageInfoV4) == RUSAGE_V4_SIZE


def test_the_footprint_reader_returns_none_when_the_call_fails(script: ModuleType) -> None:
    """Bug: a failed proc_pid_rusage returns the zeroed buffer's 0, and the record claims a 0-byte peak."""
    assert script.lifetime_peak_footprint(4242, libc=FakeLibc(peak=7, result=-1)) is None


def test_the_footprint_reader_returns_none_without_proc_pid_rusage(script: ModuleType) -> None:
    """Bug: a system without proc_pid_rusage crashes the child after the run instead of recording null."""
    assert script.lifetime_peak_footprint(4242, libc=object()) is None


@pytest.mark.skipif(sys.platform != "darwin", reason="proc_pid_rusage is a macOS call")
def test_the_footprint_reader_reads_this_process_on_macos(script: ModuleType) -> None:
    """Bug: the real libproc lookup fails (wrong library or argument types) and every record carries null."""
    peak = script.lifetime_peak_footprint(os.getpid())
    assert peak is not None
    assert 1024 * 1024 < peak < 64 * GiB


# The script's own memory rule


@pytest.mark.parametrize(
    "name",
    ("set_wired_limit", "set_memory_limit", "set_cache_limit", "mx.metal.", "reset_peak_memory", "clear_cache"),
)
def test_no_script_sets_an_mlx_memory_limit_or_resets_mlx_state(name: str) -> None:
    """Bug: the reproducer sets an MLX cache, wired or memory limit itself, clears the cache or resets the peak, so
    what it measures no longer comes only from mflux's --low-ram or --mlx-cache-limit-gb."""
    sources = sorted((ROOT / "scripts").rglob("*.py"))
    assert sources, "no Python files under scripts/"
    found = [path.name for path in sources if name in path.read_text(encoding="utf-8")]
    assert found == [], f"{name} appears in {found}"


# The watchdog


class FakeReads:
    """Stands in for MLX's (active, cache) reads: returns the given samples in turn."""

    def __init__(self, samples: list[tuple[int, int]]) -> None:
        self.samples = iter(samples)

    def __call__(self) -> tuple[int, int]:
        return next(self.samples)


def _watchdog(script: ModuleType, samples: list[tuple[int, int]], trips: list[dict[str, Any]], ceiling: int = 100):
    return script.Watchdog(
        ceiling_bytes=ceiling, wall_backstop_s=60.0, read=FakeReads(samples), on_trip=trips.append, clock=lambda: 0.0
    )


def test_the_watchdog_keeps_the_highest_cache_it_saw(script: ModuleType) -> None:
    """Bug: the cache peak keeps the last sample (or the active memory), so a cache that rose and fell between the
    start and the end of a run is recorded as small."""
    trips: list[dict[str, Any]] = []
    watchdog = _watchdog(script, [(10, 5), (10, 9), (10, 2)], trips)
    for _ in range(3):
        assert watchdog.check_once() is False
    assert watchdog.cache_peak == 9
    assert trips == []


@pytest.mark.parametrize(
    ("sample", "tripped"),
    [pytest.param((60, 39), False, id="one-under"), pytest.param((60, 40), True, id="at-the-ceiling")],
)
def test_the_watchdog_trips_when_active_plus_cache_reaches_the_ceiling(
    script: ModuleType, sample: tuple[int, int], tripped: bool
) -> None:
    """Bug: the watchdog compares active memory alone (or uses > instead of >=), so a cache full of freed buffers
    takes the machine past the ceiling unseen."""
    trips: list[dict[str, Any]] = []
    assert _watchdog(script, [sample], trips).check_once() is tripped
    assert [t["reason"] for t in trips] == (["memory ceiling"] if tripped else [])


@pytest.mark.parametrize(
    ("ceiling_gib", "memory_size", "expected"),
    [
        pytest.param(None, 32 * GiB, 30_064_771_072, id="32-gib-machine"),
        pytest.param(None, 16 * GiB, 12_884_901_888, id="16-gib-machine"),
        pytest.param(10.0, 32 * GiB, 10_737_418_240, id="override-wins"),
    ],
)
def test_the_default_ceiling_is_memory_size_minus_4_gib(
    script: ModuleType, ceiling_gib: float | None, memory_size: int, expected: int
) -> None:
    """Bug: the ceiling stays a fixed 23 GiB, so on a 16 GB Mac the watchdog cannot fire before the system pages or
    kills the run; or --ceiling-gib is ignored when given."""
    assert script.resolve_ceiling_bytes(ceiling_gib, {"memory_size": memory_size, "device_name": "fake"}) == expected


def test_a_unit_without_ceiling_gib_records_the_ceiling_from_this_machine(
    script: ModuleType, tmp_path: Path, child_path: None
) -> None:
    """Bug: the child never resolves the default (a null or zero ceiling, so the watchdog trips at once or never),
    or reads it from a different machine property than MLX's memory_size."""
    import mlx.core as mx

    kind = script.run_unit(
        _one_unit(script),
        tmp_path,
        entry="test_bench_preview_memory:fake_command_image_only",
        ceiling_gib=None,
        wall_backstop_s=60.0,
        parent_timeout_s=60.0,
    )
    assert kind == "ok"
    rec = json.loads((tmp_path / "records" / "turbo-default-plain-r1.json").read_text())
    assert rec["ceiling_bytes"] == int(mx.device_info()["memory_size"]) - 4 * GiB


# Process-group cleanup and signals


def _one_unit(script: ModuleType) -> Any:
    return script.build_units(models=["turbo"], settings=["default"], reps=1)[0]


def test_a_unit_past_its_parent_timeout_is_killed_with_its_whole_process_group(
    script: ModuleType, tmp_path: Path, child_path: None
) -> None:
    """Bug: the parent kills only the child's pid, so a grandchild it started keeps running (and holding memory)
    after the unit is recorded as killed."""
    unit = _one_unit(script)
    grandchild = tmp_path / unit.name / "grandchild.pid"
    try:
        kind = script.run_unit(
            unit,
            tmp_path,
            entry="test_bench_preview_memory:fake_command_with_grandchild",
            ceiling_gib=23.0,
            wall_backstop_s=60.0,
            parent_timeout_s=3.0,
        )
        assert kind == "killed"
        assert _gone(int(grandchild.read_text())), "the grandchild outlived its unit"
        (problem,) = (tmp_path / "problems").glob(f"{unit.name}.killed-*.json")
        assert json.loads(problem.read_text())["reason"] == "parent timeout"
    finally:
        _kill_quietly(grandchild)


def test_an_interrupted_parent_kills_the_unit_process_group(
    script: ModuleType, tmp_path: Path, child_path: None
) -> None:
    """Bug: Ctrl-C in the parent leaves the child, in its own session, running a model load with no one to read its
    result."""
    unit = _one_unit(script)
    pid_file = tmp_path / unit.name / "grandchild.pid"

    def interrupted_wait(proc: subprocess.Popen[bytes], timeout_s: float) -> int:
        _wait_for(pid_file, 10)
        raise KeyboardInterrupt

    try:
        with pytest.raises(KeyboardInterrupt):
            script.run_unit(
                unit,
                tmp_path,
                entry="test_bench_preview_memory:fake_command_with_grandchild",
                ceiling_gib=23.0,
                wall_backstop_s=60.0,
                parent_timeout_s=60.0,
                wait=interrupted_wait,
            )
        assert _gone(int(pid_file.read_text()))
    finally:
        _kill_quietly(pid_file)


def test_sigterm_to_the_parent_ends_it_with_143_and_kills_the_unit(tmp_path: Path) -> None:
    """Bug: the SIGTERM handler is never installed (or never restored into an unwind), so SIGTERM ends the parent at
    once and the child, in its own session, keeps loading a model with no one to read its result."""
    out = tmp_path / "run"
    unit_dir = out / "turbo-default-plain-r1"
    argv = ["--out", str(out), "--models", "turbo", "--settings", "default", "--only", "turbo-default-plain-r1"]
    parent = _helper(
        "_subprocess_main",
        {"BENCH_TEST_ARGV": json.dumps(argv), "BENCH_TEST_ENTRY": "fake_command_with_grandchild"},
        tmp_path,
    )
    try:
        _wait_for(unit_dir / "grandchild.pid")
        child = int((unit_dir / "child.pid").read_text())
        parent.send_signal(signal.SIGTERM)
        assert parent.wait(timeout=20) == 143, parent.stdout.read() if parent.stdout else ""
        assert _gone(child), "the unit's child outlived the parent's SIGTERM"
        assert _gone(int((unit_dir / "grandchild.pid").read_text()))
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()
        if parent.stdout:
            parent.stdout.close()
        _kill_quietly(unit_dir / "child.pid")
        _kill_quietly(unit_dir / "grandchild.pid")


def test_a_child_killed_by_a_signal_is_recorded_killed_and_main_returns_4(
    script: ModuleType, tmp_path: Path, child_path: None
) -> None:
    """Bug: a negative return code (a memory kill by the system) is read as an ordinary failure, so the record hides
    the signal and the run goes on to the next unit."""
    out = tmp_path / "run"
    argv = ["--out", str(out), "--models", "turbo", "--settings", "default", "--reps", "1"]
    assert script.main(argv, entries=_entries("fake_command_sigkills_itself", "fake_command")) == 4
    (problem,) = (out / "problems").glob("turbo-default-plain-r1.killed-*.json")
    rec = json.loads(problem.read_text())
    assert rec["status"] == "killed" and rec["reason"] == "signal SIGKILL" and rec["exit_code"] == -9
    assert not (out / "turbo-default-taef-r1").exists(), "the run went on after a killed unit"


@pytest.mark.skipif(not hasattr(os, "waitid"), reason="os.waitid with WNOWAIT exists on macOS from Python 3.13")
def test_the_default_wait_leaves_the_child_unreaped_until_the_group_is_killed(script: ModuleType) -> None:
    """Bug: the wait reaps the child before the group kill, so the group id is free when killpg runs and a new group
    with the same id could receive the SIGKILL."""
    proc = subprocess.Popen(["sleep", "0.2"], start_new_session=True)
    try:
        script._default_wait(proc, 10.0)
        assert proc.returncode is None, "the wait reaped the child"
        exited = os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        assert exited is not None and exited.si_pid == proc.pid
        script._kill_group(proc)
        assert proc.returncode == 0
    finally:
        if proc.returncode is None:
            proc.kill()
            proc.wait()


def test_the_default_wait_times_out(script: ModuleType) -> None:
    """Bug: the wait ignores its timeout, so a hung unit holds the parent forever instead of being killed."""
    proc = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        started = time.monotonic()
        with pytest.raises(subprocess.TimeoutExpired):
            script._default_wait(proc, 0.3)
        assert time.monotonic() - started < 5
    finally:
        script._kill_group(proc)


# main() end to end with a fake child


def test_main_runs_each_unit_in_a_child_and_writes_its_record(
    script: ModuleType, tmp_path: Path, child_path: None
) -> None:
    """Bug: the child's record misses the MLX peak or the footprint, the image lands outside the unit's folder, or the
    record carries the run folder's absolute path (the operator's home directory) into published evidence."""
    out = tmp_path / "run"
    argv = ["--out", str(out), "--models", "turbo", "--settings", "cache-1", "--reps", "1", "--ceiling-gib", "20"]
    assert script.main(argv, entries=_entries("fake_command")) == 0
    text = (out / "records" / "turbo-cache-1-taef-r1.json").read_text()
    rec = json.loads(text)
    assert rec["status"] == "ok" and rec["exit_code"] == 0
    assert rec["command"] == "mflux-generate-z-image-turbo-taef"
    assert rec["argv"][-2:] == ["--mlx-cache-limit-gb", "1"]
    assert rec["argv"][rec["argv"].index("--output") + 1] == "turbo-cache-1-taef-r1/image.png"
    assert rec["log"] == "turbo-cache-1-taef-r1/log.txt"
    assert str(tmp_path) not in text
    assert rec["ceiling_bytes"] == 21_474_836_480  # 20 GiB
    assert isinstance(rec["mlx_cache_at_exit_bytes"], int)
    assert rec["footprint_lifetime_peak_bytes"] > 0
    assert rec["wall_s"] >= 0
    assert rec["versions"]["mlx"] and rec["hardware"]["chip"]
    assert (out / "turbo-cache-1-taef-r1" / "image.png").read_bytes() == b"fake image"
    plain = json.loads((out / "records" / "turbo-cache-1-plain-r1.json").read_text())
    assert plain["preview_check"] is None
    # Run again: both records exist, so nothing launches and nothing changes.
    before = (out / "records" / "turbo-cache-1-taef-r1.json").read_text()
    assert script.main(argv, entries=_entries("fake_command_fails")) == 0
    assert (out / "records" / "turbo-cache-1-taef-r1.json").read_text() == before


def test_main_stops_with_exit_3_and_leaves_the_abort_record_when_a_child_crosses_the_memory_ceiling(
    script: ModuleType, tmp_path: Path, child_path: None
) -> None:
    """Bug: the child's watchdog polls only active memory (or never starts), so a run past the ceiling goes on until
    the system kills it; or it exits before its record is written, and the run stops with no trace of why."""
    out = tmp_path / "run"
    argv = ["--out", str(out), "--models", "turbo", "--settings", "default", "--reps", "1", "--ceiling-gib", "0.001"]
    assert script.main(argv, entries=_entries("fake_command_image_only", "fake_command_over_ceiling")) == 3
    (problem,) = (out / "problems").glob("turbo-default-taef-r1.aborted-*.json")
    rec = json.loads(problem.read_text())
    assert rec["status"] == "aborted" and rec["exit_code"] == 3
    assert rec["ceiling_bytes"] == 1_073_741  # int(0.001 GiB)
    assert rec["abort"]["reason"] == "memory ceiling"
    assert rec["abort"]["active_bytes"] + rec["abort"]["cache_bytes"] >= rec["abort"]["ceiling_bytes"]
    assert rec["mlx_cache_peak_bytes"] >= rec["abort"]["cache_bytes"]  # the sample that tripped counts
    assert json.loads((out / "turbo-default-taef-r1" / "child-record.json").read_text())["status"] == "aborted"
    assert not (out / "records" / "turbo-default-taef-r1.json").exists()


def test_main_stops_with_exit_3_when_a_child_passes_its_wall_backstop(
    script: ModuleType, tmp_path: Path, child_path: None
) -> None:
    """Bug: the wall backstop is never checked, so a child stuck on a sleeping lid holds the night until the parent
    timeout, or forever."""
    out = tmp_path / "run"
    argv = ["--out", str(out), "--models", "turbo", "--settings", "default", "--reps", "1", "--wall-backstop-s", "1"]
    assert script.main(argv, entries=_entries("fake_command_hangs")) == 3
    (problem,) = (out / "problems").glob("turbo-default-plain-r1.aborted-*.json")
    assert json.loads(problem.read_text())["abort"]["reason"] == "wall backstop"


def test_main_records_a_failed_unit_and_retries_it_on_the_next_run(
    script: ModuleType, tmp_path: Path, child_path: None
) -> None:
    """Bug: a failed unit gets a record in records/, so resume skips it and the summary counts a run that never
    produced an image."""
    out = tmp_path / "run"
    argv = ["--out", str(out), "--models", "turbo", "--settings", "default", "--reps", "1"]
    assert script.main(argv, entries=_entries("fake_command_fails", "fake_command")) == 1
    assert list((out / "problems").glob("turbo-default-plain-r1.failed-*.json"))
    assert not (out / "records" / "turbo-default-plain-r1.json").exists()
    assert script.main(argv, entries=_entries("fake_command")) == 0
    assert (out / "records" / "turbo-default-plain-r1.json").exists()


# --only and --keep-group


def test_only_runs_exactly_the_named_units_in_matrix_order(script: ModuleType, tmp_path: Path) -> None:
    """Bug: --only keeps the order it was typed in (or adds the rest of the matrix), so one loop unit of an outer
    runner measures a different unit than its label says."""
    ns = script.parse_args(["--out", str(tmp_path), "--only", "turbo-cache-1-taef-r2,base-default-plain-r1"])
    assert [unit.name for unit in script.units_from_args(ns)] == ["base-default-plain-r1", "turbo-cache-1-taef-r2"]


def test_only_refuses_a_name_outside_the_matrix(script: ModuleType, tmp_path: Path, capsys: Any) -> None:
    """Bug: an unknown --only name is dropped silently, so a typo in an outer runner's unit runs nothing and exits 0
    as if the unit were measured."""
    with pytest.raises(SystemExit) as raised:
        script.parse_args(["--out", str(tmp_path), "--reps", "1", "--only", "turbo-default-plain-r2"])
    assert raised.value.code == 2
    assert "turbo-default-plain-r2" in capsys.readouterr().err


def test_keep_group_leaves_the_child_in_the_parent_process_group(
    script: ModuleType, tmp_path: Path, child_path: None
) -> None:
    """Bug: --keep-group still starts a new session, so an outer supervisor that caps the parent's process group
    (mlx-guard) never sees the model's memory."""
    entries = _entries("fake_command_reports_its_group")
    base = ["--models", "turbo", "--settings", "default", "--only", "turbo-default-plain-r1"]
    assert script.main(["--out", str(tmp_path / "kept"), *base, "--keep-group"], entries=entries) == 0
    assert script.main(["--out", str(tmp_path / "own"), *base], entries=entries) == 0
    kept = int((tmp_path / "kept" / "turbo-default-plain-r1" / "group.txt").read_text())
    own = int((tmp_path / "own" / "turbo-default-plain-r1" / "group.txt").read_text())
    assert kept == os.getpgrp()
    assert own != os.getpgrp()


def test_keep_group_kills_only_the_child_never_the_parent_group(tmp_path: Path) -> None:
    """Bug: in --keep-group mode the cleanup still calls killpg on the child's group, which is now the parent's own
    group, and kills the runner (and its supervisor) with it. The runner here is a helper in its own session, so the
    bug kills the helper, never pytest."""
    unit_dir = tmp_path / "turbo-default-plain-r1"
    started = time.monotonic()
    helper = _helper("_subprocess_keep_group_unit", {"BENCH_TEST_OUT": str(tmp_path)}, tmp_path)
    try:
        output, _ = helper.communicate(timeout=40)
        assert helper.returncode == 0, f"the runner did not survive the cleanup (rc {helper.returncode}): {output}"
        assert "survived: killed" in output
        # The fake sleeps 60 s: a cleanup that does not kill the child returns only when the child ends by itself.
        assert time.monotonic() - started < 30
        assert _gone(int((unit_dir / "child.pid").read_text()))
    finally:
        if helper.poll() is None:
            helper.kill()
            helper.communicate()
        # The grandchild is the outer supervisor's job in this mode; end it here so the test leaves nothing behind.
        _kill_quietly(unit_dir / "child.pid")
        _kill_quietly(unit_dir / "grandchild.pid")
