"""Check every plugin folder against this repository's rules.

Run from the repository root: uv run --python 3.13 --no-project scripts/check_plugins.py
Exit 0 when every rule holds. Otherwise exit 1 and print one line per problem.
Standard library only (Python 3.11 or newer, for tomllib)."""

from __future__ import annotations

import fnmatch
import keyword
import os
import re
import sys
import tomllib
from pathlib import Path

TEMPLATE_FOLDER = "template"
TEMPLATE_NAME = "myplugin"
PRIVATE_CLASSIFIER = "Private :: Do Not Upload"
FOLDER_PATTERN = re.compile(r"^[a-z][a-z0-9]*$")
REQUIREMENT_PATTERN = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*(.*)$")
OPERATOR_PATTERN = re.compile(r"\s*(===|==|!=|<=|>=|~=|<|>)")
# Each of these sets a floor; ~=, == and === also cap, which the cap rule reports on its own.
FLOOR_OPERATORS = (">=", ">", "~=", "==", "===")
CAPPING_OPERATORS = ("<", "<=", "==", "===", "~=")
RUFF_IN_LINT = re.compile(r"ruff@([0-9][0-9A-Za-z.]*)")
RUFF_PIN = re.compile(r"^\s*ruff\s*===?\s*([^\s;,]+)\s*(?:;.*)?$")
URL_INTO_TREE = re.compile(r"github\.com/mflux-community/mflux-plugins/(?:tree|blob)/main/([^/#?]*)")
# Text a renamed copy of the template must not keep in its description: a <...> placeholder, or the template's note.
# A placeholder starts and ends without a space, so "steps < 5 and guidance > 1" is not one.
TEMPLATE_TEXT = re.compile(r"<(?!\s)[^<>]+(?<!\s)>|never publish", re.IGNORECASE)
WORKFLOW_CALL = re.compile(r"^\s*workflow_call:", re.MULTILINE)
# The value of a "uses:" that names an action in another repository (owner/repo@ref), which is not a folder here.
REMOTE_ACTION = re.compile(r"^(\s*(?:-\s+)?uses:\s*)[\"']?(?!\.)[^\s\"'#]+[\"']?", re.MULTILINE)
# Before a folder name: not a letter, digit, _, ., - or /, so "cache" is not found in "teacache" or in
# "src/mflux/extras/cache/".
BEFORE_NAME = r"(?<![\w./-])"
SKIPPED_PARTS = {"__pycache__", "dist", "build"}
REQUIRED_FILES = ("README.md", "AGENTS.md", "uv.lock")
# pyproject.toml values the checker reads, and the TOML type each must have: (dotted key, table or array, items are
# strings). dependency-groups.dev may hold {include-group = ...} tables, so its items are not checked.
SHAPES = (
    ("project", dict, False),
    ("project.dependencies", list, True),
    ("project.scripts", dict, True),
    ("project.license-files", list, True),
    ("project.classifiers", list, True),
    ("project.urls", dict, True),
    ("tool", dict, False),
    ("tool.uv", dict, False),
    ("tool.uv.build-backend", dict, False),
    ("dependency-groups", dict, False),
    ("dependency-groups.dev", list, False),
)


def read(path: Path) -> str:
    """A repository file's text, or "" when it is missing. A byte that is not UTF-8 is replaced, not fatal."""
    return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""


def plugin_folders(root: Path) -> list[Path]:
    """Every top-level folder with a pyproject.toml, hidden folders excluded: the folders
    lint.yml's license loop visits."""
    return sorted(path.parent for path in root.glob("*/pyproject.toml") if not path.parent.name.startswith("."))


def package_name(folder: Path) -> str:
    """The name the plugin uses everywhere (module child, distribution and command suffix): the folder name, or
    myplugin for the template."""
    return TEMPLATE_NAME if folder.name == TEMPLATE_FOLDER else folder.name


def normalized(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def split_requirement(text: str) -> tuple[str, list[str], bool]:
    """(normalized name, version operators, is a direct reference) of a PEP 508 string; markers are ignored."""
    match = REQUIREMENT_PATTERN.match(text.split(";", 1)[0])
    if match is None:
        return "", [], False
    name, rest = match.group(1), match.group(2).strip()
    if rest.startswith("@"):
        return normalized(name), [], True
    operators = []
    for part in rest.strip("() ").split(","):
        found = OPERATOR_PATTERN.match(part)
        if found:
            operators.append(found.group(1))
    return normalized(name), operators, False


def working_directory(folder: str) -> re.Pattern[str]:
    """A working-directory line naming folder, bare or quoted, with an optional trailing comment; not a commented-out
    one."""
    return re.compile(rf"^\s*(?:-\s+)?working-directory:\s*([\"']?){re.escape(folder)}\1\s*(#.*)?$", re.MULTILINE)


def wrong_shapes(data: dict) -> list[str]:
    """One line for each pyproject.toml value in SHAPES that has the wrong TOML type."""
    found = []
    for key, kind, strings in SHAPES:
        value: object = data
        for part in key.split("."):
            value = value.get(part) if isinstance(value, dict) else None
        if value is None:
            continue
        items = list(value.values()) if isinstance(value, dict) else value
        if not isinstance(value, kind) or (strings and not all(isinstance(item, str) for item in items)):
            shape = "a table" if kind is dict else "an array"
            found.append(f"{key} must be {shape}{' of strings' if strings else ''}")
    return found


def refers_to_folder(text: str, folder: str) -> bool:
    """True when a workflow names folder: a path under it, it as the working directory, its workflow name, or its
    concurrency group."""
    name = re.escape(folder)
    text = REMOTE_ACTION.sub(r"\1", text)
    patterns = (
        re.compile(rf"{BEFORE_NAME}{name}/"),
        working_directory(folder),
        re.compile(rf"^\s*name:\s*([\"']?){name}\1\s*(#.*)?$", re.MULTILINE),
        re.compile(rf"{BEFORE_NAME}{name}-ci-"),
    )
    return any(pattern.search(text) for pattern in patterns)


def files_mentioning(folder: Path, token: str) -> list[str]:
    """Files under folder whose path or text contains token. Hidden folders (.venv, .venv-py310, caches),
    __pycache__, dist and build are skipped, and so are symlinks and the text of a binary file (a NUL byte in its
    first 8 KB)."""
    found = []
    for current, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in SKIPPED_PARTS]
        for file in files:
            path = Path(current) / file
            if path.is_symlink() or not path.is_file():
                continue
            relative = path.relative_to(folder).as_posix()
            data = path.read_bytes()
            text = "" if b"\0" in data[:8192] else data.decode("utf-8", errors="ignore")
            if token in relative or token in text:
                found.append(relative)
    return sorted(found)


def check_folder(root: Path, folder: Path) -> list[str]:
    problems: list[str] = []

    def problem(text: str) -> None:
        problems.append(f"{folder.name}: {text}")

    if not FOLDER_PATTERN.match(folder.name):
        problem(
            "folder name must be lowercase letters and digits only, starting with a letter "
            "(PEP 8 discourages underscores in package names)"
        )
        return problems
    if keyword.iskeyword(folder.name):
        problem(f"folder name must not be a Python keyword, or mflux.extras.{folder.name} can't be imported")
        return problems
    try:
        data = tomllib.loads((folder / "pyproject.toml").read_bytes().decode("utf-8"))
    except UnicodeDecodeError as exc:
        problem(f"pyproject.toml is not valid UTF-8: {exc}")
        return problems
    except tomllib.TOMLDecodeError as exc:
        problem(f"pyproject.toml is not valid TOML: {exc}")
        return problems
    shapes = wrong_shapes(data)
    if shapes:
        for line in shapes:
            problem(f"pyproject.toml: {line}")
        return problems
    name = package_name(folder)
    project = data.get("project", {})
    uv = data.get("tool", {}).get("uv", {})

    module = uv.get("build-backend", {}).get("module-name")
    if module != f"mflux.extras.{name}":
        problem(f'[tool.uv.build-backend] module-name must be "mflux.extras.{name}", got {module!r}')
    src = folder / "src" / "mflux"
    for parent in (src / "__init__.py", src / "extras" / "__init__.py"):
        if parent.exists():
            problem(
                f"remove {parent.relative_to(folder).as_posix()}: it shadows mflux core "
                "or hides the other mflux.extras packages"
            )
    if not (src / "extras" / name / "__init__.py").is_file():
        problem(f"src/mflux/extras/{name}/__init__.py is missing")

    if project.get("name") != f"mflux-{name}":
        problem(f'project.name must be "mflux-{name}", got {project.get("name")!r}')
    scripts = project.get("scripts", {})
    if not scripts:
        problem("[project.scripts] declares no command")
    for command, target in scripts.items():
        if not (command.startswith("mflux-") and command.endswith(f"-{name}")):
            problem(f'command {command!r} must start with "mflux-" and end with "-{name}"')
        if not target.startswith(f"mflux.extras.{name}."):
            problem(f"command {command!r} must point into mflux.extras.{name}, got {target!r}")

    if project.get("license") != "Apache-2.0":
        problem('license must be "Apache-2.0", the license of this repository')
    license_files = project.get("license-files", [])

    def listed(file: str) -> bool:
        """license-files holds PEP 639 globs; a plain name is the glob that matches only itself."""
        return any(fnmatch.fnmatchcase(file, pattern) for pattern in license_files)

    if not listed("LICENSE") or not (folder / "LICENSE").is_file():
        problem("the folder must carry LICENSE and list it in license-files")
    has_notice = (folder / "NOTICE").is_file()
    if has_notice and not listed("NOTICE"):
        problem("NOTICE is not in license-files, so the wheel would ship without it")
    if "NOTICE" in license_files and not has_notice:
        problem("license-files lists NOTICE, but the folder has no NOTICE file")
    for file in REQUIRED_FILES:
        if not (folder / file).is_file():
            problem(f"add {file} to the folder; every plugin folder carries README.md, AGENTS.md and uv.lock")

    if "sources" in uv:
        problem("remove [tool.uv.sources]: every dependency must install from PyPI")
    has_mflux = False
    for requirement in project.get("dependencies", []):
        dependency, operators, direct = split_requirement(requirement)
        if direct:
            problem(f"dependency {requirement!r} must come from PyPI, not a URL or a path")
        if dependency != "mflux":
            continue
        # Each mflux requirement (one per Python version, say) needs its own floor and may not cap.
        has_mflux = True
        if not any(operator in FLOOR_OPERATORS for operator in operators):
            problem(f'the mflux requirement {requirement!r} needs a lower bound, for example "mflux>=0.22"')
        capped = [operator for operator in operators if operator in CAPPING_OPERATORS]
        if capped:
            problem(f"the mflux requirement {requirement!r} must have no upper bound or pin, found {' '.join(capped)}")
    if not has_mflux:
        problem("dependencies must include mflux")

    lint_versions = set(RUFF_IN_LINT.findall(read(root / ".github" / "workflows" / "lint.yml")))
    dev = data.get("dependency-groups", {}).get("dev", [])
    pins = [found.group(1) for item in dev if isinstance(item, str) and (found := RUFF_PIN.match(item))]
    if not pins:
        problem('the dev group must pin ruff exactly ("ruff==X.Y.Z"), the version lint.yml uses')
    for pin in dict.fromkeys(pins):
        if lint_versions != {pin}:
            problem(f"ruff=={pin} must equal the ruff version in .github/workflows/lint.yml {sorted(lint_versions)}")

    entries = {f"/{folder.name}/", f"/{folder.name}/**", f"{folder.name}/", f"/{folder.name}"}
    lines = [line.split() for line in read(root / ".github" / "CODEOWNERS").splitlines()]
    # GitHub applies the last matching line, so that one must name an owner.
    matching = [fields for fields in lines if fields[:1] and fields[0] in entries]
    if not matching or len(matching[-1]) < 2:
        problem(f"add a line '/{folder.name}/ @<maintainer>' to .github/CODEOWNERS")

    workflows = root / ".github" / "workflows"
    others = [path.name for path in plugin_folders(root) if path.name != folder.name]
    ci = read(workflows / f"{folder.name}.yml")
    if not ci:
        problem(f"add .github/workflows/{folder.name}.yml to run this folder's tests")
    else:
        escaped = re.escape(folder.name)
        if not (working_directory(folder.name).search(ci) and re.search(rf"{BEFORE_NAME}{escaped}/uv\.lock", ci)):
            problem(
                f'{folder.name}.yml must run in this folder: it needs "working-directory: {folder.name}" '
                f'and "{folder.name}/uv.lock"'
            )
        if not re.search(rf"^name:\s*([\"']?){escaped}\1\s*(#.*)?$", ci, re.MULTILINE):
            problem(f'{folder.name}.yml must start with "name: {folder.name}", so its runs show under this plugin')
        if not re.search(rf"^\s*group:\s*[\"']?{escaped}-ci-", ci, re.MULTILINE):
            problem(
                f'{folder.name}.yml must use a concurrency group starting "{folder.name}-ci-", '
                "so its runs never cancel another plugin's"
            )
        if not WORKFLOW_CALL.search(ci):
            problem(
                f'{folder.name}.yml needs a "workflow_call:" trigger, so a release workflow can run its tests '
                "before it publishes"
            )
    for other in others:
        if refers_to_folder(ci, other):
            problem(f"{folder.name}.yml refers to {other}/, another plugin's folder")

    classifiers = project.get("classifiers", [])
    if folder.name == TEMPLATE_FOLDER:
        if PRIVATE_CLASSIFIER not in classifiers:
            problem(f'keep the classifier "{PRIVATE_CLASSIFIER}" so the template can never be uploaded to PyPI')
        if (workflows / f"{TEMPLATE_FOLDER}-release.yml").exists():
            problem("the template must have no release workflow")
    else:
        release = read(workflows / f"{folder.name}-release.yml")
        if not release:
            problem(f"add .github/workflows/{folder.name}-release.yml to publish from {folder.name}-v* tags")
        else:
            escaped = re.escape(folder.name)
            environment = re.search(rf"{BEFORE_NAME}pypi-{escaped}(?![\w.-])", release)
            if environment is None or not re.search(rf"{BEFORE_NAME}{escaped}-v", release):
                problem(
                    f"{folder.name}-release.yml must publish through the environment pypi-{folder.name} "
                    f"from {folder.name}-v* tags"
                )
            if not re.search(rf"uses:\s*[\"']?\./\.github/workflows/{escaped}\.yml\b", release):
                problem(
                    f"{folder.name}-release.yml must run the tests first, with "
                    f'"uses: ./.github/workflows/{folder.name}.yml"'
                )
            if not working_directory(folder.name).search(release):
                problem(
                    f'{folder.name}-release.yml must build in this folder: it needs "working-directory: {folder.name}"'
                )
        for other in others:
            if refers_to_folder(release, other):
                problem(f"{folder.name}-release.yml refers to {other}/, another plugin's folder")
        if f"[`{folder.name}/`]({folder.name}/)" not in read(root / "README.md"):
            problem("add a row for this plugin to the Plugins table in the root README.md")
        template_text = TEMPLATE_TEXT.search(str(project.get("description", "")))
        if template_text:
            problem(
                f"project.description still has template text ({template_text.group(0)!r}): rewrite it for this plugin"
            )
        if PRIVATE_CLASSIFIER in classifiers:
            problem(f'remove the classifier "{PRIVATE_CLASSIFIER}" copied from the template')
        leftovers = files_mentioning(folder, TEMPLATE_NAME)
        if leftovers:
            problem(f"the template name {TEMPLATE_NAME!r} is still in: {', '.join(leftovers)}")

    for label, url in project.get("urls", {}).items():
        for match in URL_INTO_TREE.finditer(url):
            if match.group(1) != folder.name:
                problem(f"[project.urls] {label} must point at this folder (/tree/main/{folder.name}), got {url!r}")
    return problems


def check_repo(root: Path) -> list[str]:
    problems: list[str] = []
    for folder in plugin_folders(root):
        problems.extend(check_folder(root, folder))
    return problems


def main(argv: list[str] | None = None) -> int:
    root = Path(argv[0]) if argv else Path.cwd()
    folders = plugin_folders(root)
    if not folders:
        print("check_plugins: no plugin folders found")
        return 1
    problems = check_repo(root)
    for line in problems:
        print(line)
    if not problems:
        print(f"check_plugins: {len(folders)} plugin folder(s) follow the rules")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
