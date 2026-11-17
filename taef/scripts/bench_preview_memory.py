"""Measure how much memory and time the TAEF preview adds to a Z-Image generation at 1024x1024.

The script runs each plain mflux command and its TAEF twin under three memory settings and records, for every run,
the process's peak physical footprint, MLX's own peaks and the wall time:

- models: Z-Image Turbo (``mflux-community/z-image-turbo-mflux-q4``, 8 steps) and Z-Image Base
  (``mflux-community/z-image-base-mflux-q4 --base-model z-image``, 8 steps, guidance 4.0);
- commands: the plain mflux command and the matching ``-taef`` command;
- settings: mflux's defaults, ``--low-ram``, and ``--mlx-cache-limit-gb 1``;
- every run at 1024x1024, seed 42, prompt "a red apple on a wooden table", three reps by default.

Each run is one unit: a fresh Python process that calls the command's own ``main()`` once, in its own process group.
The units run rep by rep (every arm's first rep, then every arm's second), so slow drift in the machine spreads over
all arms. There is no warm-up run: every unit is a cold process, so its wall time includes the model load. The script
itself sets no MLX memory limit and resets no MLX counter; the only limits in play are the ones mflux sets from its
own flags. It needs no supervisor (such as mlx-guard): the watchdog below is its own safety stop.

What the record's numbers mean:

- ``footprint_lifetime_peak_bytes``: the lifetime peak physical footprint of the generating process, as macOS reports
  it (``proc_pid_rusage``, ``ri_lifetime_max_phys_footprint``), read in that process after the command returns; null
  when the call is unavailable. It covers the whole process, model load included, in every setting.
- ``mlx_peak_since_last_reset_bytes``: ``mx.get_peak_memory()`` at the end of the run. mflux resets MLX's peak counter
  after the model loads under ``--low-ram`` and ``--mlx-cache-limit-gb``, but not with its defaults, so this peak
  covers the generation only in those two settings and the whole process in the default one. Compare plain with TAEF
  within one setting, never across settings.
- ``mlx_cache_peak_bytes``: the highest MLX cache the watchdog saw, sampled every 0.05 s.
- ``mlx_cache_at_exit_bytes`` and ``wall_s`` (from just after the command's import to its end, model load included).

Each record also carries the command and its arguments, the mflux, MLX, mlx-taef and plugin versions, macOS and the
chip, the UTC start time (``started_utc``), the commit of the checkout this script runs from (``git_commit``,
``git rev-parse HEAD``) and that checkout's ``git status --porcelain`` lines (``git_dirty_files``, empty when clean;
both null when git is unavailable or the script is not in a checkout), the sha256 of the saved image's pixels
(``image_pixels_sha256``: its size and RGB bytes, so the generation time mflux writes into the file does not count),
the model's snapshot in the local Hugging Face cache (``model_revision``, the hash its ``refs/main`` names; null when
there is none), and for a TAEF unit the preview check below. Paths in a record are relative to ``--out``.

Inside each unit a watchdog thread reads MLX's active plus cache memory every 0.05 s. When the sum reaches the ceiling
(``--ceiling-gib``; by default the smaller of the machine's memory size minus 4 GiB and MLX's recommended working set,
both from MLX's ``device_info()``) or the run passes its wall backstop (``--wall-backstop-s``; 600 s for Turbo, 900 s
for Base by default), the unit writes its record to disk and exits with code 3, and the whole run stops with the
record under ``problems/``. The unit exits with code 3 even when that record cannot be written, and when the watchdog
itself fails.

A unit counts only when its command exited 0 and saved an image that can be read. A TAEF unit also needs its preview
to have run:
``image.preview.png`` in the unit's folder, written after the unit started, and no "live preview failed" in the unit's
log (mlx-taef switches the preview off on a decode error and the run goes on). Otherwise the unit is ``preview-off``.

Units never download: each one runs with ``HF_HUB_OFFLINE=1``, so a model that is not in the local Hugging Face cache
makes that unit fail. Download both models before the run. The TAEF decoder must be in mlx-taef's cache too: before
the first unit the script asks mlx-taef where its converted Z-Image decoder goes and refuses to start when the file is
not there; run any ``-taef`` command once online first.

Usage::

    python scripts/bench_preview_memory.py --out RUN_DIR --dry-run     # list the units, their commands and the ETA
    python scripts/bench_preview_memory.py --out RUN_DIR               # run (or resume) the matrix
    python scripts/bench_preview_memory.py --out RUN_DIR --models base --settings default,cache-1
    python scripts/bench_preview_memory.py --summarize RUN_DIR         # medians per arm and TAEF minus plain

A finished unit's record goes to ``RUN_DIR/records/<unit>.json``; running the same command again skips those units, so
an interrupted run resumes where it stopped. A failed, preview-off, aborted or killed unit's record goes to
``RUN_DIR/problems/<unit>.<kind>-<time>.json`` and that unit runs again next time. Resume and ``--summarize`` check
every record they read (status ok, its unit equal to its file name, its arm equal to its unit's, numbers where numbers
belong) and stop at the first one that fails, naming the file.

``--only UNIT[,UNIT...]`` runs exactly the named units (names as the dry run prints them), so an outer runner can
drive the matrix one unit per call. ``--keep-group`` keeps each unit in this process's group instead of its own, so a
supervisor that watches or caps a process group (such as mlx-guard) sees the unit's memory. What it gives up: at the
end of a unit this script then ends only the unit's own process, never the group (that is its own group, and the
supervisor's), so cleaning up anything the unit left running becomes the supervisor's job.

Exit codes:

- 0: every selected unit has a record (or the dry run or the summary printed);
- 1: at least one unit failed or ran with its preview off; the other units ran;
- 2: bad arguments, a record that fails its check, a unit, ``records/`` or ``problems/`` folder that is a symbolic
  link, or no records to summarize;
- 3: a unit's watchdog stopped it at the memory ceiling or the wall backstop (or the unit exited 3 without a record);
  the run stopped there;
- 4: a unit was killed by a signal (such as a memory kill) or hung past its backstop; the run stopped there;
- 5: the TAEF decoder is not in mlx-taef's cache; no unit ran;
- 128 + N: the script itself got signal N (SIGTERM, or SIGHUP unless it was started with SIGHUP ignored, as under
  ``nohup``) and stopped the running unit first.
"""

import argparse
import ctypes
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shlex
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType
from typing import Any

GiB = 1024**3
SCRIPT = Path(__file__).resolve()
PROMPT = "a red apple on a wooden table"
SEED = "42"
SIZE = "1024"
STEPS = "8"
POLL_S = 0.05
PARENT_GRACE_S = 120.0
CEILING_HEADROOM_BYTES = 4 * GiB
IMAGE_NAME = "image.png"
# mflux-taef's default preview path: next to the output, named <output stem>.preview.png.
PREVIEW_NAME = "image.preview.png"
# What mlx-taef logs when a decode fails and on_error="disable" switches the preview off.
PREVIEW_FAILED_TEXT = "live preview failed"
# The mlx-taef decoder both Z-Image commands use.
DECODER_KERNEL = "zimage"
FOOTPRINT = "footprint_lifetime_peak_bytes"
MLX_PEAK = "mlx_peak_since_last_reset_bytes"
CACHE_PEAK = "mlx_cache_peak_bytes"
MEASURED = ("wall_s", MLX_PEAK, CACHE_PEAK)


class BenchError(Exception):
    """A run folder or record the script refuses to use."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Model:
    repo: str
    extra_args: tuple[str, ...]
    plain_command: str
    taef_command: str
    eta_min: float
    wall_backstop_s: float


MODELS: dict[str, Model] = {
    "turbo": Model(
        repo="mflux-community/z-image-turbo-mflux-q4",
        extra_args=(),
        plain_command="mflux-generate-z-image-turbo",
        taef_command="mflux-generate-z-image-turbo-taef",
        eta_min=1.5,
        wall_backstop_s=600.0,
    ),
    "base": Model(
        repo="mflux-community/z-image-base-mflux-q4",
        extra_args=("--base-model", "z-image", "--guidance", "4.0"),
        plain_command="mflux-generate-z-image",
        taef_command="mflux-generate-z-image-taef",
        eta_min=2.9,
        wall_backstop_s=900.0,
    ),
}
SETTINGS: dict[str, tuple[str, ...]] = {
    "default": (),
    "low-ram": ("--low-ram",),
    "cache-1": ("--mlx-cache-limit-gb", "1"),
}
VARIANTS = ("plain", "taef")
ENTRIES: dict[tuple[str, str], str] = {
    ("turbo", "plain"): "mflux.models.z_image.cli.z_image_turbo_generate:main",
    ("turbo", "taef"): "mflux.extras.taef.z_image_turbo:main",
    ("base", "plain"): "mflux.models.z_image.cli.z_image_generate:main",
    ("base", "taef"): "mflux.extras.taef.z_image:main",
}


@dataclass(frozen=True, slots=True, kw_only=True)
class Unit:
    model: str
    setting: str
    variant: str
    rep: int

    @property
    def arm(self) -> str:
        return f"{self.model}-{self.setting}-{self.variant}"

    @property
    def name(self) -> str:
        return f"{self.arm}-r{self.rep}"

    def fields(self) -> dict[str, Any]:
        return {
            "unit": self.name,
            "arm": self.arm,
            "model": self.model,
            "setting": self.setting,
            "variant": self.variant,
            "rep": self.rep,
        }


def build_units(*, models: Sequence[str], settings: Sequence[str], reps: int) -> list[Unit]:
    """Every arm's rep 1, then every arm's rep 2, and so on; plain before TAEF within a setting."""
    return [
        Unit(model=model, setting=setting, variant=variant, rep=rep)
        for rep in range(1, reps + 1)
        for model in models
        for setting in settings
        for variant in VARIANTS
    ]


def command_for(unit: Unit) -> str:
    model = MODELS[unit.model]
    return model.plain_command if unit.variant == "plain" else model.taef_command


def mflux_args(unit: Unit, output: Path) -> list[str]:
    model = MODELS[unit.model]
    return [
        "--model",
        model.repo,
        *model.extra_args,
        "--prompt",
        PROMPT,
        "--steps",
        STEPS,
        "--seed",
        SEED,
        "--width",
        SIZE,
        "--height",
        SIZE,
        "--output",
        str(output),
        *SETTINGS[unit.setting],
    ]


def child_env(environ: Mapping[str, str]) -> dict[str, str]:
    return {**environ, "HF_HUB_OFFLINE": "1"}


def eta_minutes(units: Iterable[Unit]) -> float:
    return sum(MODELS[unit.model].eta_min for unit in units)


def unit_timeouts(model: str, wall_backstop_s: float | None) -> tuple[float, float]:
    """The unit's wall backstop (the model's own unless overridden) and the parent's timeout, PARENT_GRACE_S longer,
    so the child's watchdog writes its abort record before the parent gives up on it."""
    wall = MODELS[model].wall_backstop_s if wall_backstop_s is None else wall_backstop_s
    return wall, wall + PARENT_GRACE_S


def resolve_ceiling_bytes(ceiling_gib: float | None, device_info: Mapping[str, Any]) -> int:
    """The watchdog's ceiling: --ceiling-gib when given, else the smaller of the machine's memory size minus 4 GiB and
    MLX's recommended working set (MLX buffers past the working set page, and the process and the system need room
    on top of them)."""
    if ceiling_gib is not None:
        return int(ceiling_gib * GiB)
    return min(
        int(device_info["memory_size"]) - CEILING_HEADROOM_BYTES, int(device_info["max_recommended_working_set_size"])
    )


# The footprint reader: struct rusage_info_v4 from the macOS SDK's sys/resource.h.

RUSAGE_INFO_V4 = 4
_V4_FIELDS = (
    "ri_user_time ri_system_time ri_pkg_idle_wkups ri_interrupt_wkups ri_pageins ri_wired_size ri_resident_size "
    "ri_phys_footprint ri_proc_start_abstime ri_proc_exit_abstime ri_child_user_time ri_child_system_time "
    "ri_child_pkg_idle_wkups ri_child_interrupt_wkups ri_child_pageins ri_child_elapsed_abstime ri_diskio_bytesread "
    "ri_diskio_byteswritten ri_cpu_time_qos_default ri_cpu_time_qos_maintenance ri_cpu_time_qos_background "
    "ri_cpu_time_qos_utility ri_cpu_time_qos_legacy ri_cpu_time_qos_user_initiated ri_cpu_time_qos_user_interactive "
    "ri_billed_system_time ri_serviced_system_time ri_logical_writes ri_lifetime_max_phys_footprint ri_instructions "
    "ri_cycles ri_billed_energy ri_serviced_energy ri_interval_max_phys_footprint ri_runnable_time"
).split()


class RusageInfoV4(ctypes.Structure):
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16), *((name, ctypes.c_uint64) for name in _V4_FIELDS)]


def lifetime_peak_footprint(pid: int, *, libc: Any = None) -> int | None:
    """The process's lifetime peak physical footprint in bytes, or None when macOS cannot report it."""
    if libc is None:
        try:
            libc = ctypes.CDLL(None)
        except OSError:
            return None
    call = getattr(libc, "proc_pid_rusage", None)
    if call is None:
        return None
    try:
        call.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(RusageInfoV4)]
        call.restype = ctypes.c_int
    except AttributeError:
        pass  # not a C function (a stand-in); it takes the structure as it is
    info = RusageInfoV4()
    if call(pid, RUSAGE_INFO_V4, info) != 0:
        return None
    return int(info.ri_lifetime_max_phys_footprint)


# Provenance: start time and the checkout's state.


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# Variables that would point git at another repository, work tree, index or object store than the script's own
# checkout, or change how it finds that checkout.
GIT_LOCATION_VARS = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_COMMON_DIR",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_NAMESPACE",
    "GIT_CEILING_DIRECTORIES",
)


def _git(git: str, folder: Path, *args: str) -> str | None:
    env = {key: value for key, value in os.environ.items() if key not in GIT_LOCATION_VARS}
    try:
        result = subprocess.run(
            [git, "-C", str(folder), *args],
            capture_output=True,
            text=True,
            check=False,
            env={**env, "GIT_OPTIONAL_LOCKS": "0"},  # status must not write the index
        )
    except OSError:
        return None
    return result.stdout if result.returncode == 0 else None


def git_state(folder: Path, *, git: str = "git") -> dict[str, Any]:
    """The commit checked out at `folder` and its `git status --porcelain` lines (an empty list when clean); both
    None when git is unavailable or `folder` is not in a checkout."""
    head = _git(git, folder, "rev-parse", "HEAD")
    status = _git(git, folder, "status", "--porcelain")
    if head is None or status is None:
        return {"git_commit": None, "git_dirty_files": None}
    return {"git_commit": head.strip(), "git_dirty_files": [line for line in status.splitlines() if line]}


# The child: one command, once, under the watchdog.


def _sysctl(key: str) -> str | None:
    try:
        result = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, check=False)
    except OSError:
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None


def versions() -> dict[str, str | None]:
    found: dict[str, str | None] = {"python": platform.python_version()}
    for name in ("mflux", "mlx", "mlx-taef", "mflux-taef"):
        try:
            found[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            found[name] = None
    return found


def hardware() -> dict[str, Any]:
    memory = _sysctl("hw.memsize")
    return {
        "chip": _sysctl("machdep.cpu.brand_string"),
        "memory_bytes": int(memory) if memory and memory.isdigit() else None,
        "macos": platform.mac_ver()[0] or None,
    }


def _fsync_dir(folder: Path) -> None:
    fd = os.open(folder, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    """Write to a new temporary file beside `path` (a fresh name, never an existing file or link), fsync it, then
    rename, so a reader never sees half a record and the record survives a crash right after. The record is readable
    by everyone (0644), not mkstemp's 0600."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        os.fchmod(f.fileno(), 0o644)
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


class Watchdog:
    """Polls MLX active + cache memory and the elapsed wall time; calls `on_trip` once when either passes its bound.
    Keeps the highest cache it saw in `cache_peak`."""

    def __init__(
        self,
        *,
        ceiling_bytes: int,
        wall_backstop_s: float,
        read: Callable[[], tuple[int, int]],
        on_trip: Callable[[dict[str, Any]], None],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.ceiling, self.wall, self.read, self.on_trip, self.clock = (
            ceiling_bytes,
            wall_backstop_s,
            read,
            on_trip,
            clock,
        )
        self.cache_peak = 0
        self.started = clock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="memory-watchdog", daemon=True)

    def check_once(self) -> bool:
        active, cache = self.read()
        self.cache_peak = max(self.cache_peak, cache)
        elapsed = self.clock() - self.started
        info = {"active_bytes": active, "cache_bytes": cache, "ceiling_bytes": self.ceiling, "elapsed_s": elapsed}
        if active + cache >= self.ceiling:
            self.on_trip({"reason": "memory ceiling", **info})
            return True
        if elapsed >= self.wall:
            self.on_trip({"reason": "wall backstop", "wall_backstop_s": self.wall, **info})
            return True
        return False

    def _loop(self) -> None:
        """Fails closed: an exception from a check (a failed read, a trip that raised) ends the process with exit 3,
        because a run without its watchdog has no memory brake."""
        while not self._stop.wait(POLL_S):
            try:
                if self.check_once():
                    return
            except BaseException as exc:  # noqa: BLE001 -- fail closed: any failure ends the unit with exit 3
                try:
                    print(f"[watchdog] failed, stopping the unit: {exc!r}", file=sys.stderr, flush=True)
                finally:
                    os._exit(3)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)


class RecordFinisher:
    """Writes the child's record once. The first caller (the main thread at the command's end, or a watchdog trip)
    wins; a later caller waits for that write to finish and writes nothing, so a record is never "ok" with an
    "abort" key."""

    def __init__(
        self,
        record: dict[str, Any],
        path: Path,
        *,
        measure: Callable[[], Mapping[str, Any]],
        write: Callable[[Path, Mapping[str, Any]], None] = write_json,
    ) -> None:
        self.record, self.path, self.measure, self.write = record, path, measure, write
        self._lock = threading.Lock()
        self._done = False

    def finish(self, status: str, exit_code: int, **extra: Any) -> bool:
        """True when this call wrote the record; False when an earlier call already had."""
        with self._lock:
            if self._done:
                return False
            self._done = True
            self.record.update(status=status, exit_code=exit_code, **self.measure(), **extra)
            self.write(self.path, self.record)
            return True


def _resolve(entry: str) -> Callable[[], Any]:
    module, _, name = entry.partition(":")
    return getattr(importlib.import_module(module), name)


def _exit_code(code: object) -> int:
    if code is None:
        return 0
    return code if isinstance(code, int) else 1


def child_main(argv: Sequence[str]) -> int:
    if "--" not in argv:
        raise SystemExit("usage: bench_preview_memory.py _child OPTIONS -- MFLUX_ARGS")
    cut = list(argv).index("--")
    parser = argparse.ArgumentParser(prog="bench_preview_memory.py _child")
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--entry", required=True)
    parser.add_argument("--ceiling-gib", type=float, default=None)
    parser.add_argument("--wall-backstop-s", type=_positive_seconds, required=True)
    ns = parser.parse_args(argv[:cut])
    args = list(argv[cut + 1 :])
    os.environ["HF_HUB_OFFLINE"] = "1"  # before anything imports huggingface_hub
    started_utc = utc_now()
    checkout = git_state(SCRIPT.parent)

    import mlx.core as mx

    info = mx.device_info()
    ceiling = resolve_ceiling_bytes(ns.ceiling_gib, info)  # before the command loads anything
    bound = resolve_ceiling_bytes(None, info)
    if ceiling > bound:
        print(
            f"bench_preview_memory.py _child: error: --ceiling-gib {ns.ceiling_gib} is above this machine's bound of "
            f"{bound / GiB:.2f} GiB (memory size minus 4 GiB, or MLX's recommended working set when smaller)",
            file=sys.stderr,
        )
        return 2
    record: dict[str, Any] = {
        "unit": ns.unit,
        "started_utc": started_utc,
        **checkout,
        "command": ns.command,
        "argv": args,
        "entry": ns.entry,
        "versions": versions(),
        "hardware": hardware(),
        "ceiling_bytes": ceiling,
        "pid": os.getpid(),
    }

    started = time.monotonic()

    def measure() -> dict[str, Any]:
        return {
            "wall_s": time.monotonic() - started,
            MLX_PEAK: mx.get_peak_memory(),
            CACHE_PEAK: watchdog.cache_peak,
            "mlx_cache_at_exit_bytes": mx.get_cache_memory(),
            FOOTPRINT: lifetime_peak_footprint(os.getpid()),
        }

    finisher = RecordFinisher(record, ns.record, measure=measure)

    def trip(info: dict[str, Any]) -> None:
        """Writes the abort record (on disk, fsync'd) and exits 3; exits 3 also when the write fails. A stderr that
        cannot be written does not cost the record."""
        try:
            try:
                print(f"[watchdog] {json.dumps(info)}", file=sys.stderr, flush=True)
            except BaseException:  # noqa: BLE001 -- the message is lost; the record below still says why
                pass
            finisher.finish("aborted", 3, abort=info)
        except BaseException as exc:  # noqa: BLE001 -- say why, then exit 3 in the finally
            print(f"[watchdog] could not write the abort record: {exc!r}", file=sys.stderr, flush=True)
        finally:
            os._exit(3)

    watchdog = Watchdog(
        ceiling_bytes=ceiling,
        wall_backstop_s=ns.wall_backstop_s,
        read=lambda: (mx.get_active_memory(), mx.get_cache_memory()),
        on_trip=trip,
    )
    watchdog.start()  # before the command imports or loads anything
    command = _resolve(ns.entry)
    sys.argv = [ns.command, *args]
    started = time.monotonic()
    code = 1
    try:
        command()
        code = 0
    except SystemExit as exc:
        code = _exit_code(exc.code)
    finally:
        watchdog.stop()
        finisher.finish("ok" if code == 0 else "failed", code)
    return code


def child_decoder_path() -> int:
    """Prints where mlx-taef keeps the converted Z-Image decoder: the path its download.get_or_convert returns
    without downloading, built from its public CACHE_ROOT and the kernel's WeightSource.cache_key. Runs in its own
    process, so the orchestrator never imports mlx."""
    from mlx_taef.download import CACHE_ROOT
    from mlx_taef.kernels import get_kernel

    key = get_kernel(DECODER_KERNEL).source.cache_key(role="decoder")
    print(CACHE_ROOT / "converted" / f"{key}.mlx.safetensors")
    return 0


# The orchestrator.


def decoder_problem(*, python: str = sys.executable) -> str | None:
    """None when mlx-taef's converted Z-Image decoder is on disk; otherwise what is wrong. Units run offline, so a
    missing decoder would fail every TAEF unit after its model load."""
    result = subprocess.run(
        [python, str(SCRIPT), "_decoder-path"],
        capture_output=True,
        text=True,
        check=False,
        env=child_env(os.environ),
    )
    if result.returncode != 0:
        last = (result.stderr.strip().splitlines() or ["no output"])[-1]
        return f"could not ask mlx-taef where its Z-Image decoder goes ({last})"
    path = Path(result.stdout.strip())
    if path.is_file():
        return None
    return (
        f"the TAEF decoder is not in mlx-taef's cache: {path}. Units run offline, so run any -taef command once "
        "online first."
    )


def _default_wait(proc: subprocess.Popen[bytes], timeout_s: float) -> None:
    """Wait until the child has exited without reaping it, so its pid (the group id) stays taken until the caller has
    killed the group and reaped the child. Python before 3.13 has no os.waitid on macOS; there the child is reaped
    here, and macOS hands out process ids in sequence, so the group id is not given to a new process in that gap."""
    waitid = getattr(os, "waitid", None)
    if waitid is None:
        proc.wait(timeout=timeout_s)
        return
    deadline = time.monotonic() + timeout_s
    flags = os.WEXITED | os.WNOWAIT | os.WNOHANG
    while waitid(os.P_PID, proc.pid, flags) is None:
        if time.monotonic() >= deadline:
            raise subprocess.TimeoutExpired(proc.args, timeout_s)
        time.sleep(POLL_S)


def _kill_group(proc: subprocess.Popen[bytes], *, keep_group: bool = False) -> None:
    """End the unit: its whole process group, or with keep_group (the child shares this process's group) the child
    alone, so this process and an outer supervisor in the same group are never signalled. Then reap the child."""
    if keep_group:
        if proc.poll() is None:
            proc.kill()
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    proc.wait()


def _prepare_unit_dir(path: Path) -> None:
    """A fresh, empty unit folder. An earlier attempt's folder is renamed aside, never deleted; a symbolic link at
    the folder's name is refused."""
    if path.is_symlink():
        raise BenchError(f"{path} is a symbolic link; refusing to write a unit there")
    if path.exists() and any(path.iterdir()):
        path.rename(path.with_name(f"{path.name}.prev-{time.time_ns()}"))
    path.mkdir(parents=True, exist_ok=True)


def _open_log(path: Path) -> int:
    """A new log file at `path`: never an existing file, never through a symbolic link."""
    return os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)


def image_pixels_sha256(path: Path) -> str | None:
    """The sha256 of the image's size (as "<width>x<height>") followed by its RGB pixel bytes, or None when no image
    can be read from `path`. The file's own bytes would differ between runs with the same pixels: mflux writes the
    generation time into every image's metadata."""
    from PIL import Image  # Pillow comes with mflux; imported here so the orchestrator's start stays light

    try:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
    except (OSError, ValueError, Image.DecompressionBombError):
        return None
    digest = hashlib.sha256(f"{rgb.width}x{rgb.height}".encode())
    digest.update(rgb.tobytes())
    return digest.hexdigest()


_VARIABLE = re.compile(r"\$(\w+|\{[^}]*\})", re.ASCII)


def _expandvars(path: str, environ: Mapping[str, str]) -> str:
    """os.path.expandvars with `environ` in place of os.environ: $NAME and ${NAME} become the value, an unknown name
    stays as written."""

    def value(match: re.Match[str]) -> str:
        name = match[1][1:-1] if match[1].startswith("{") else match[1]
        return environ.get(name, match[0])

    return _VARIABLE.sub(value, path)


def _hub_cache(environ: Mapping[str, str]) -> Path:
    """The folder huggingface_hub reads its model cache from, by its own rules (huggingface_hub.constants):
    HF_HUB_CACHE, else HUGGINGFACE_HUB_CACHE, else HF_HOME/hub, else $XDG_CACHE_HOME/huggingface/hub, else
    ~/.cache/huggingface/hub. As there, a variable that is set counts even when empty, and "~" then $NAME are
    expanded, the variables from `environ`."""

    def expand(path: str) -> str:
        return _expandvars(os.path.expanduser(path), environ)

    xdg = environ.get("XDG_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".cache"))
    home = expand(environ.get("HF_HOME", os.path.join(xdg, "huggingface")))
    return Path(expand(environ.get("HF_HUB_CACHE", environ.get("HUGGINGFACE_HUB_CACHE", os.path.join(home, "hub")))))


_REVISION = re.compile(r"[0-9a-f]{40}")


def model_revision(repo: str, environ: Mapping[str, str], *, cwd: Path | None = None) -> str | None:
    """The snapshot hash the local cache's refs/main names for `repo`: a plain file read, never a Hub call. None when
    there is no such file or it holds no commit hash. A relative cache folder is read from `cwd`, the folder the
    command ran in (by default this process's current folder)."""
    hub = _hub_cache(environ) if cwd is None else cwd / _hub_cache(environ)
    path = hub / f"models--{repo.replace('/', '--')}" / "refs" / "main"
    try:
        text = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return None
    return text if _REVISION.fullmatch(text) else None


def _refuse_symlinked_folders(out_dir: Path) -> None:
    """records/ and problems/ must be real folders (or absent), never links that send records elsewhere."""
    for name in ("records", "problems"):
        if (out_dir / name).is_symlink():
            raise BenchError(f"{out_dir / name} is a symbolic link; refusing to write records there")


def preview_check(unit_dir: Path, unit_name: str, started_ns: int) -> dict[str, Any]:
    """Whether a TAEF unit's preview ran: its file exists, was written after the unit started, and the log does not
    say mlx-taef switched it off."""
    preview = unit_dir / PREVIEW_NAME
    exists = preview.is_file() and not preview.is_symlink()
    fresh = exists and preview.stat().st_mtime_ns > started_ns
    log = unit_dir / "log.txt"
    failed = log.is_file() and PREVIEW_FAILED_TEXT in log.read_text(encoding="utf-8", errors="replace")
    if not exists:
        reason = "no preview file"
    elif not fresh:
        reason = "preview file older than the unit"
    elif failed:
        reason = "log says live preview failed"
    else:
        reason = None
    return {
        "file": f"{unit_name}/{PREVIEW_NAME}",
        "exists": exists,
        "written_after_start": fresh,
        "log_says_failed": failed,
        "ok": reason is None,
        "reason": reason,
    }


def unit_outcome(
    *,
    rc: int,
    timed_out: bool,
    child_status: str | None,
    image_exists: bool,
    image_readable: bool,
    preview: Mapping[str, Any] | None,
) -> tuple[str, str | None]:
    """The unit's kind (ok, failed, preview-off, aborted, killed) and the reason, when there is one to give."""
    if timed_out:
        return "killed", "parent timeout"
    if rc < 0:
        return "killed", f"signal {signal.Signals(-rc).name}"
    if rc == 3 and child_status == "aborted":
        return "aborted", None
    if rc == 3 and child_status is None:  # the command's own exit 3 always gets a "failed" record from the child
        return "aborted", (
            "the watchdog stopped the unit (a trip whose record could not be written, or the watchdog itself failed)"
        )
    if rc == 3 and child_status == "ok":
        return "failed", "the watchdog tripped after the record was written"
    if rc != 0 or child_status != "ok":
        return "failed", None
    if not image_exists:
        return "failed", "no output image"
    if not image_readable:
        return "failed", "output image unreadable"
    if preview is not None and not preview["ok"]:
        return "preview-off", preview["reason"]
    return "ok", None


def run_unit(
    unit: Unit,
    out_dir: Path,
    *,
    entry: str,
    ceiling_gib: float | None,
    wall_backstop_s: float,
    parent_timeout_s: float,
    wait: Callable[[subprocess.Popen[bytes], float], object] = _default_wait,
    python: str = sys.executable,
    keep_group: bool = False,
) -> str:
    """Run one unit in a fresh process group (or, with keep_group, in this one) and write its record. Returns ok,
    failed, preview-off, aborted or killed. The child runs in `out_dir`, so the paths it is given, and records, are
    relative to it."""
    _refuse_symlinked_folders(out_dir)
    unit_dir = out_dir / unit.name
    _prepare_unit_dir(unit_dir)
    folder = Path(unit.name)
    argv = [
        python,
        str(SCRIPT),
        "_child",
        "--record",
        str(folder / "child-record.json"),
        "--unit",
        unit.name,
        "--command",
        command_for(unit),
        "--entry",
        entry,
        "--wall-backstop-s",
        str(wall_backstop_s),
        *([] if ceiling_gib is None else ["--ceiling-gib", str(ceiling_gib)]),
        "--",
        *mflux_args(unit, folder / IMAGE_NAME),
    ]
    timed_out = False
    started_ns = time.time_ns()
    with os.fdopen(_open_log(unit_dir / "log.txt"), "wb") as log:
        proc = subprocess.Popen(
            argv,
            cwd=out_dir,
            env=child_env(os.environ),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=not keep_group,
        )
        try:
            wait(proc, parent_timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            _kill_group(proc, keep_group=keep_group)  # without keep_group, also what the child left behind
    rc = proc.returncode
    child_record = unit_dir / "child-record.json"
    try:
        loaded = json.loads(child_record.read_text()) if child_record.exists() else {}
    except (OSError, ValueError):  # unreadable or not JSON: no child record
        loaded = {}
    # Anything but a JSON object is not the child's record either; the unit then gets its problem record below.
    record: dict[str, Any] = loaded if isinstance(loaded, dict) else {}
    image = unit_dir / IMAGE_NAME
    image_exists = image.is_file() and not image.is_symlink()
    digest = image_pixels_sha256(image) if image_exists else None
    preview = preview_check(unit_dir, unit.name, started_ns) if unit.variant == "taef" else None
    record.update(
        unit.fields(),
        exit_code=rc,
        log=f"{unit.name}/log.txt",
        image_pixels_sha256=digest,
        model_revision=model_revision(MODELS[unit.model].repo, os.environ, cwd=out_dir),
        preview_check=preview,
    )
    kind, reason = unit_outcome(
        rc=rc,
        timed_out=timed_out,
        child_status=record.get("status"),
        image_exists=image_exists,
        image_readable=digest is not None,
        preview=preview,
    )
    if kind == "ok":
        write_json(out_dir / "records" / f"{unit.name}.json", record)
        return "ok"
    record["status"] = kind
    if reason is not None:
        record["reason"] = reason
    write_json(out_dir / "problems" / f"{unit.name}.{kind}-{time.time_ns()}.json", record)
    return kind


# Records: the checks every reader applies.

_UNIT_NAME = re.compile(r"(?P<arm>.+)-r[0-9]+")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _is_number(value: object) -> bool:
    """An int or a finite float; never a bool, NaN or an infinity (JSON readers accept those), nor an int too large
    for a float."""
    try:
        return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:
        return False


# Each known arm and the (model, setting, variant) it is made of.
ARM_PARTS = {f"{m}-{s}-{v}": (m, s, v) for m in MODELS for s in SETTINGS for v in VARIANTS}


def check_record(path: Path, rec: object) -> dict[str, Any]:
    """`rec` as read from `path`, or BenchError naming the file and the first field that fails."""

    def bad(why: str) -> BenchError:
        return BenchError(f"{path}: {why}")

    if not isinstance(rec, dict):
        raise bad("not a JSON object")
    if rec.get("status") != "ok":
        raise bad(f"status is {rec.get('status')!r}, not 'ok'")
    if rec.get("unit") != path.stem:
        raise bad(f"unit is {rec.get('unit')!r}, not the file's name {path.stem!r}")
    match = _UNIT_NAME.fullmatch(path.stem)
    if match is None or rec.get("arm") != match["arm"]:
        raise bad(f"arm is {rec.get('arm')!r}, not the unit's")
    parts = ARM_PARTS.get(match["arm"])
    for key, expected in zip(("model", "setting", "variant"), parts or (None, None, None), strict=True):
        if expected is None or rec.get(key) != expected:
            raise bad(f"{key} is {rec.get(key)!r}, not the arm's ({match['arm']})")
    for key in MEASURED:
        if not _is_number(rec.get(key)):
            raise bad(f"{key} is {rec.get(key)!r}, not a number")
    if rec.get(FOOTPRINT) is not None and not _is_number(rec.get(FOOTPRINT)):
        raise bad(f"{FOOTPRINT} is {rec.get(FOOTPRINT)!r}, not a number or null")
    digest = rec.get("image_pixels_sha256")
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise bad(f"image_pixels_sha256 is {digest!r}, not a sha256")
    return rec


def read_record(path: Path) -> dict[str, Any]:
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError) as exc:
        raise BenchError(f"{path}: {exc}") from None
    return check_record(path, rec)


def is_recorded(out_dir: Path, unit: Unit) -> bool:
    """True when the unit has a record that passes its check; BenchError when it has one that does not."""
    path = out_dir / "records" / f"{unit.name}.json"
    if not path.exists() and not path.is_symlink():
        return False
    read_record(path)
    return True


def load_records(out_dir: Path) -> list[dict[str, Any]]:
    folder = out_dir / "records"
    paths = sorted(folder.glob("*.json")) if folder.is_dir() else []
    if not paths:
        raise BenchError(f"no records in {folder}")
    return [read_record(path) for path in paths]


def run_matrix(
    units: Sequence[Unit], out_dir: Path, *, launch: Callable[[Unit], str], say: Callable[[str], Any]
) -> int:
    failed = False
    for index, unit in enumerate(units, 1):
        if is_recorded(out_dir, unit):
            say(f"[skip] {unit.name}: already recorded")
            continue
        say(f"[unit {index}/{len(units)}] {unit.name}")
        kind = launch(unit)
        say(f"[{kind}] {unit.name}")
        if kind == "aborted":
            say(f"[stop] {unit.name} hit its watchdog; see {out_dir / 'problems'}")
            return 3
        if kind == "killed":
            say(f"[stop] {unit.name} was killed; see {out_dir / 'problems'}")
            return 4
        failed = failed or kind in ("failed", "preview-off")
    say("[done] with failed units; they run again next time" if failed else "[done]")
    return 1 if failed else 0


# The summary.

SUMMARY_NOTES = [
    "Footprint: the lifetime peak physical footprint of each generating process (proc_pid_rusage), model load "
    "included.",
    "MLX peak since last reset: mflux resets MLX's peak counter after the model loads under --low-ram and "
    "--mlx-cache-limit-gb, so there it covers the generation only; with mflux's defaults it covers the whole process. "
    "Compare plain with TAEF within one setting, not across settings.",
    f"MLX cache peak: the highest MLX cache the watchdog saw, sampled every {POLL_S} s.",
    "Wall: one cold process per run, model load included (no warm-up).",
]
DELTA_KEYS = ("footprint", "mlx_peak_since_reset", "mlx_cache_peak", "wall_s")


def _median(values: Iterable[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return statistics.median(present) if present else None


def summarize(records: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per arm: the number of reps, how many of them have a footprint, and the median footprint peak, MLX peak since
    its last reset, MLX cache peak and wall time."""
    by_arm: dict[str, list[Mapping[str, Any]]] = {}
    for rec in records:
        by_arm.setdefault(rec["arm"], []).append(rec)
    return {
        arm: {
            "n": len(recs),
            "n_footprint": sum(1 for r in recs if r.get(FOOTPRINT) is not None),
            "footprint": _median(r.get(FOOTPRINT) for r in recs),
            "mlx_peak_since_reset": _median(r.get(MLX_PEAK) for r in recs),
            "mlx_cache_peak": _median(r.get(CACHE_PEAK) for r in recs),
            "wall_s": _median(r.get("wall_s") for r in recs),
        }
        for arm, recs in by_arm.items()
    }


def taef_minus_plain(arms: Mapping[str, Mapping[str, Any]]) -> dict[tuple[str, str], dict[str, float | None]]:
    deltas: dict[tuple[str, str], dict[str, float | None]] = {}
    for model in MODELS:
        for setting in SETTINGS:
            plain, taef = arms.get(f"{model}-{setting}-plain"), arms.get(f"{model}-{setting}-taef")
            if plain is None or taef is None:
                continue
            deltas[(model, setting)] = {
                key: None if taef[key] is None or plain[key] is None else taef[key] - plain[key] for key in DELTA_KEYS
            }
    return deltas


def _distinct(values: Iterable[Any]) -> str:
    seen = sorted({str(v) for v in values})
    return " / ".join(seen) if seen else "unknown"


def _gib(value: float | None, *, sign: bool = False) -> str:
    return "n/a" if value is None else f"{value / GiB:{'+' if sign else ''}.2f} GiB"


def _secs(value: float | None, *, sign: bool = False) -> str:
    return "n/a" if value is None else f"{value:{'+' if sign else ''}.1f} s"


def _provenance(records: Sequence[Mapping[str, Any]]) -> list[str]:
    commits = _distinct(r["git_commit"] for r in records if r.get("git_commit") is not None)
    dirty = _distinct(
        "; ".join(r["git_dirty_files"]) or "none" for r in records if isinstance(r.get("git_dirty_files"), list)
    )
    started = sorted(str(r["started_utc"]) for r in records if r.get("started_utc"))
    revisions = "; ".join(
        f"{model} {_distinct(r['model_revision'] for r in recs if r.get('model_revision') is not None)}"
        for model in MODELS
        if (recs := [r for r in records if r.get("model") == model])
    )
    lines = [
        f"Commit: {commits}",
        f"Model revision: {revisions}",
        f"Dirty files: {dirty}",
        f"Started (UTC): {started[0]} to {started[-1]}" if started else "Started (UTC): unknown",
    ]
    missing = sum(1 for r in records if r.get("git_commit") is None or r.get("git_dirty_files") is None)
    if missing:
        lines.append(
            f"Warning: git_commit and git_dirty_files are null in {missing} of {len(records)} records "
            "(git unavailable, or the script was not in a checkout)."
        )
    return lines


def _image_lines(records: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = []
    for model in MODELS:
        for setting in SETTINGS:
            recs = [r for r in records if r.get("model") == model and r.get("setting") == setting]
            if not recs:
                continue
            digests = {r["image_pixels_sha256"] for r in recs}
            plain = sum(1 for r in recs if r.get("variant") == "plain")
            taef = sum(1 for r in recs if r.get("variant") == "taef")
            shared = "one image digest" if len(digests) == 1 else f"{len(digests)} image digests"
            lines.append(f"  {model} {setting}: {shared} across {plain} plain and {taef} TAEF images")
    return lines


def format_summary(records: Sequence[Mapping[str, Any]]) -> str:
    hw = [r.get("hardware", {}) for r in records]
    vs = [r.get("versions", {}) for r in records]
    memory = _distinct(f"{h['memory_bytes'] / GiB:.0f} GiB" for h in hw if h.get("memory_bytes"))
    lines = [
        f"Machine: {_distinct(h.get('chip') for h in hw)}, {memory}, macOS {_distinct(h.get('macos') for h in hw)}",
        "Versions: "
        + ", ".join(
            f"{name} {_distinct(v.get(name) for v in vs)}" for name in ("mflux", "mlx", "mlx-taef", "mflux-taef")
        ),
        *_provenance(records),
        f"Records: {len(records)}",
        "",
        *SUMMARY_NOTES,
        "",
        "Medians per arm:",
    ]
    arms = summarize(records)
    order = [f"{m}-{s}-{v}" for m in MODELS for s in SETTINGS for v in VARIANTS]
    for arm in sorted(arms, key=lambda a: order.index(a) if a in order else len(order)):
        a = arms[arm]
        lines.append(
            f"  {arm}: n={a['n']} ({a['n_footprint']} with a footprint), footprint {_gib(a['footprint'])}, "
            f"MLX peak since last reset {_gib(a['mlx_peak_since_reset'])}, "
            f"MLX cache peak {_gib(a['mlx_cache_peak'])}, wall {_secs(a['wall_s'])}"
        )
    lines += ["", "TAEF minus plain (medians):"]
    for (model, setting), d in taef_minus_plain(arms).items():
        plain, taef = arms[f"{model}-{setting}-plain"], arms[f"{model}-{setting}-taef"]
        lines.append(
            f"  {model} {setting} (n plain/TAEF {plain['n']}/{taef['n']}, "
            f"with a footprint {plain['n_footprint']}/{taef['n_footprint']}): "
            f"footprint {_gib(d['footprint'], sign=True)}, "
            f"MLX peak since last reset {_gib(d['mlx_peak_since_reset'], sign=True)}, "
            f"MLX cache peak {_gib(d['mlx_cache_peak'], sign=True)}, wall {_secs(d['wall_s'], sign=True)}"
        )
    lines += ["", "Output images (sha256 of the pixels) per model and setting:", *_image_lines(records)]
    return "\n".join(lines)


# The command line.


def _choices(allowed: Sequence[str]) -> Callable[[str], list[str]]:
    def parse(text: str) -> list[str]:
        names = [name.strip() for name in text.split(",") if name.strip()]
        unknown = [name for name in names if name not in allowed]
        if not names or unknown:
            raise argparse.ArgumentTypeError(f"choose from {','.join(allowed)} (got {text!r})")
        return names

    return parse


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be 1 or more")
    return value


def _ceiling_gib(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError(f"must be a finite number of GiB above 0 (got {text!r})")
    return value


def _positive_seconds(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError(f"must be a finite number of seconds above 0 (got {text!r})")
    return value


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="bench_preview_memory.py",
        description="Measure the memory and wall time of the TAEF preview commands against the plain mflux commands "
        "at 1024x1024, one fresh process per run.",
    )
    parser.add_argument("--out", type=Path, help="run folder: unit folders, records/ and problems/ go here")
    parser.add_argument(
        "--models", type=_choices(list(MODELS)), default=list(MODELS), help="turbo,base (default: both)"
    )
    parser.add_argument(
        "--settings",
        type=_choices(list(SETTINGS)),
        default=list(SETTINGS),
        help="default,low-ram,cache-1 (default: all)",
    )
    parser.add_argument("--reps", type=_positive_int, default=3, help="reps per arm (default: 3)")
    parser.add_argument(
        "--ceiling-gib",
        type=_ceiling_gib,
        default=None,
        help="stop a unit when MLX active + cache memory reaches this many GiB (default: the smaller of the machine's "
        "memory size minus 4 GiB and MLX's recommended working set)",
    )
    parser.add_argument(
        "--wall-backstop-s",
        type=_positive_seconds,
        default=None,
        help="stop a unit after this many seconds (default: 600 Turbo, 900 Base)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the units, their commands and the ETA; run nothing"
    )
    parser.add_argument("--summarize", type=Path, metavar="RUN_DIR", help="print medians per arm and TAEF minus plain")
    parser.add_argument(
        "--only",
        metavar="UNIT[,UNIT...]",
        help="run exactly these units of the selected matrix, in matrix order (names as the dry run prints them)",
    )
    parser.add_argument(
        "--keep-group",
        action="store_true",
        help="keep each unit in this process's group, so an outer supervisor sees and caps it; the supervisor then "
        "owns cleaning up whatever a unit leaves behind",
    )
    ns = parser.parse_args(argv)
    if ns.summarize is None and ns.out is None:
        parser.error("--out is required unless --summarize is given")
    if ns.out is not None:
        ns.out = ns.out.absolute()  # each unit runs inside the run folder, so its paths must not be relative
    if ns.only is not None:
        ns.only = [name.strip() for name in ns.only.split(",") if name.strip()]
        known = {unit.name for unit in build_units(models=ns.models, settings=ns.settings, reps=ns.reps)}
        unknown = [name for name in ns.only if name not in known]
        if not ns.only or unknown:
            parser.error(f"--only names no unit of the selected matrix: {','.join(unknown) or '(empty)'}")
    return ns


def units_from_args(ns: argparse.Namespace) -> list[Unit]:
    units = build_units(models=ns.models, settings=ns.settings, reps=ns.reps)
    return units if ns.only is None else [unit for unit in units if unit.name in ns.only]


def _print_plan(units: Sequence[Unit], out_dir: Path) -> None:
    todo = [u for u in units if not is_recorded(out_dir, u)]
    for unit in units:
        state = "recorded" if unit not in todo else f"{MODELS[unit.model].eta_min:.1f} min"
        args = mflux_args(unit, Path(unit.name) / IMAGE_NAME)  # as run_unit passes it: relative to the run folder
        print(f"{unit.name}  [{state}]  {shlex.join([command_for(unit), *args])}")
    per_unit = ", ".join(f"{name} {MODELS[name].eta_min} min" for name in dict.fromkeys(u.model for u in units))
    print(
        f"ETA {eta_minutes(todo):.1f} min for {len(todo)} units ({per_unit} per unit; {len(units) - len(todo)} recorded)"
    )


def _raise_exit(signum: int, frame: FrameType | None) -> None:
    raise SystemExit(128 + signum)


def _error(message: str) -> None:
    print(f"bench_preview_memory.py: error: {message}", file=sys.stderr)


def main(argv: Sequence[str] | None = None, *, entries: Mapping[tuple[str, str], str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["_child"]:
        return child_main(argv[1:])
    if argv[:1] == ["_decoder-path"]:
        return child_decoder_path()
    ns = parse_args(argv)
    try:
        if ns.summarize is not None:
            print(format_summary(load_records(ns.summarize)))
            return 0
        units = units_from_args(ns)
        if ns.dry_run:
            _print_plan(units, ns.out)
            return 0
        if any(unit.variant == "taef" and not is_recorded(ns.out, unit) for unit in units):
            problem = decoder_problem()
            if problem is not None:
                _error(problem)
                return 5
        return _run(ns, units, ENTRIES if entries is None else entries)
    except BenchError as exc:
        _error(str(exc))
        return 2


def _run(ns: argparse.Namespace, units: Sequence[Unit], entries: Mapping[tuple[str, str], str]) -> int:
    def launch(unit: Unit) -> str:
        wall, parent_timeout = unit_timeouts(unit.model, ns.wall_backstop_s)
        return run_unit(
            unit,
            ns.out,
            entry=entries[(unit.model, unit.variant)],
            ceiling_gib=ns.ceiling_gib,
            wall_backstop_s=wall,
            parent_timeout_s=parent_timeout,
            keep_group=ns.keep_group,
        )

    # SIGTERM and SIGHUP unwind through run_unit, which kills the running unit's process group on the way out. A
    # SIGHUP that is ignored at start (nohup) stays ignored, so closing the terminal does not end the run.
    signals = [signal.SIGTERM]
    if signal.getsignal(signal.SIGHUP) is not signal.SIG_IGN:
        signals.append(signal.SIGHUP)
    previous = {sig: signal.signal(sig, _raise_exit) for sig in signals}
    try:
        ns.out.mkdir(parents=True, exist_ok=True)
        return run_matrix(units, ns.out, launch=launch, say=lambda line: print(line, flush=True))
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    sys.exit(main())
