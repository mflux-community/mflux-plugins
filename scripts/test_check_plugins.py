"""Tests for scripts/check_plugins.py. Each test builds a small repository in tmp_path that passes every rule,
breaks one rule, and checks that the checker reports exactly that problem."""

from pathlib import Path

import pytest

import check_plugins

LINT = """jobs:
  repo:
    steps:
      - run: uvx ruff@0.16.3 check .
      - run: uvx ruff@0.16.3 format --check .
"""
DEV = '["pytest>=9.1,<10.0", "ruff==0.16.3", "ty==0.0.72"]'
REPO_URL = "https://github.com/mflux-community/mflux-plugins"


def pyproject(
    folder: str,
    *,
    project_name: str | None = None,
    module: str | None = None,
    scripts: str | None = None,
    license_files: str = '["LICENSE"]',
    dependencies: str = '["mflux>=0.22"]',
    classifiers: str = "[]",
    dev: str = DEV,
    homepage: str | None = None,
    extra: str = "",
) -> str:
    name = check_plugins.TEMPLATE_NAME if folder == check_plugins.TEMPLATE_FOLDER else folder
    project_name = project_name or f"mflux-{name}"
    module = module or f"mflux.extras.{name}"
    if scripts is None:
        scripts = f'"mflux-generate-z-image-{name}" = "mflux.extras.{name}.z_image:main"'
    homepage = homepage or f"{REPO_URL}/tree/main/{folder}"
    return f"""[tool.uv.build-backend]
module-name = "{module}"

[project]
name = "{project_name}"
version = "0.1.0"
license = "Apache-2.0"
license-files = {license_files}
classifiers = {classifiers}
dependencies = {dependencies}

[project.urls]
Homepage = "{homepage}"
Repository = "{REPO_URL}"

[project.scripts]
{scripts}

[dependency-groups]
dev = {dev}
{extra}"""


def ci_workflow(folder: str) -> str:
    return (
        f"name: {folder}\n"
        f"on:\n  pull_request:\n    paths: ['{folder}/**']\n  workflow_call:\n"
        f"concurrency:\n  group: {folder}-ci-${{{{ github.ref }}}}\n"
        f"defaults:\n  run:\n    working-directory: {folder}\n"
        f"jobs:\n  test:\n    steps:\n      - with:\n          cache-dependency-glob: {folder}/uv.lock\n"
    )


def release_workflow(
    folder: str, *, tag: str | None = None, environment: str | None = None, tests: str | None = None
) -> str:
    tag = tag or f"{folder}-v*"
    environment = environment or f"pypi-{folder}"
    tests = tests or folder
    return (
        f"on:\n  push:\n    tags: ['{tag}']\n"
        f"jobs:\n  test:\n    uses: ./.github/workflows/{tests}.yml\n"
        f"  build:\n    defaults:\n      run:\n        working-directory: {folder}\n"
        f"  publish:\n    environment: {environment}\n"
    )


def make_repo(root: Path, folder: str = "demo", *, no_scripts: bool = False, **overrides: str) -> Path:
    """A repository that passes every rule: one plugin folder plus the files the checker reads. A second call
    with another folder adds a second plugin to the same repository."""
    template = folder == check_plugins.TEMPLATE_FOLDER
    name = check_plugins.TEMPLATE_NAME if template else folder
    workflows = root / ".github" / "workflows"
    workflows.mkdir(parents=True, exist_ok=True)
    (root / "LICENSE").write_text("Apache License\n", encoding="utf-8")
    (workflows / "lint.yml").write_text(LINT, encoding="utf-8")
    owners = root / ".github" / "CODEOWNERS"
    text = owners.read_text(encoding="utf-8") if owners.exists() else "* @owner\n"
    owners.write_text(text + f"/{folder}/ @owner\n", encoding="utf-8")
    readme = root / "README.md"
    rows = readme.read_text(encoding="utf-8") if readme.exists() else "| Plugin |\n|---|\n"
    if not template:
        rows += f"| [`{folder}/`]({folder}/) |\n"
        (workflows / f"{folder}-release.yml").write_text(release_workflow(folder), encoding="utf-8")
    readme.write_text(rows, encoding="utf-8")
    (workflows / f"{folder}.yml").write_text(ci_workflow(folder), encoding="utf-8")
    plugin = root / folder
    package = plugin / "src" / "mflux" / "extras" / name
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (plugin / "LICENSE").write_text("Apache License\n", encoding="utf-8")
    for file in check_plugins.REQUIRED_FILES:
        (plugin / file).write_text(f"{file}\n", encoding="utf-8")
    if template and "classifiers" not in overrides:
        overrides["classifiers"] = f'["{check_plugins.PRIVATE_CLASSIFIER}"]'
    if no_scripts:
        overrides["scripts"] = ""
    (plugin / "pyproject.toml").write_text(pyproject(folder, **overrides), encoding="utf-8")
    return plugin


def only_problem(root: Path) -> str:
    """The one problem the checker reports; fails when it reports none or more than one."""
    problems = check_plugins.check_repo(root)
    assert len(problems) == 1, problems
    return problems[0]


def test_a_repository_that_follows_every_rule_passes(tmp_path: Path) -> None:
    """Bug: the checker reports a problem on a correct plugin, so every pull request goes red."""
    make_repo(tmp_path, "demo")
    assert check_plugins.check_repo(tmp_path) == []


@pytest.mark.parametrize("folder", ["Demo-Plugin", "1demo", "_demo", "my_plugin", "Demo"])
def test_a_folder_name_that_is_not_lowercase_letters_and_digits_is_refused(tmp_path: Path, folder: str) -> None:
    """Bug: a folder named Demo-Plugin, 1demo, _demo, my_plugin or Demo is accepted. mflux.extras.Demo-Plugin and
    mflux.extras.1demo can never be imported, a leading underscore marks a private module, my_plugin needs a hyphen
    form for its distribution and command that its tag and environment names don't share, and Demo breaks the rule
    that every name is the folder name."""
    (tmp_path / folder).mkdir()
    (tmp_path / folder / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    assert only_problem(tmp_path) == (
        f"{folder}: folder name must be lowercase letters and digits only, starting with a letter "
        "(PEP 8 discourages underscores in package names)"
    )


def test_a_folder_named_like_a_python_keyword_is_refused(tmp_path: Path) -> None:
    """Bug: a folder named class passes the identifier pattern, but "import mflux.extras.class" is a syntax error."""
    (tmp_path / "class").mkdir()
    (tmp_path / "class" / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    assert (
        only_problem(tmp_path)
        == "class: folder name must not be a Python keyword, or mflux.extras.class can't be imported"
    )


def test_hidden_folders_are_not_plugins(tmp_path: Path) -> None:
    """Bug: plugin_folders globs .venv or another hidden folder that has a pyproject.toml, and checks it as a plugin."""
    plugin = make_repo(tmp_path, "demo")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "pyproject.toml").write_text("not [ toml\n", encoding="utf-8")
    assert check_plugins.plugin_folders(tmp_path) == [plugin]
    assert check_plugins.check_repo(tmp_path) == []


def test_a_pyproject_that_is_not_toml_is_one_problem_not_a_crash(tmp_path: Path) -> None:
    """Bug: tomllib.TOMLDecodeError escapes, so the lint job dies with a traceback that names no folder."""
    plugin = make_repo(tmp_path, "demo")
    (plugin / "pyproject.toml").write_text("[project\n", encoding="utf-8")
    assert only_problem(tmp_path).startswith("demo: pyproject.toml is not valid TOML: ")


def test_a_module_name_outside_the_plugin_child_is_refused(tmp_path: Path) -> None:
    """Bug: module-name = "mflux" passes, and the wheel ships mflux/__init__.py over mflux core's."""
    make_repo(tmp_path, "demo", module="mflux")
    assert only_problem(tmp_path) == (
        "demo: [tool.uv.build-backend] module-name must be \"mflux.extras.demo\", got 'mflux'"
    )


@pytest.mark.parametrize("parent", ["src/mflux/__init__.py", "src/mflux/extras/__init__.py"])
def test_a_parent_initializer_is_refused(tmp_path: Path, parent: str) -> None:
    """Bug: src/mflux/__init__.py or src/mflux/extras/__init__.py is accepted, and installing the plugin replaces
    mflux core's __init__ or hides every other mflux.extras package."""
    plugin = make_repo(tmp_path, "demo")
    (plugin / parent).write_text("", encoding="utf-8")
    assert only_problem(tmp_path) == (
        f"demo: remove {parent}: it shadows mflux core or hides the other mflux.extras packages"
    )


def test_a_missing_child_initializer_is_refused(tmp_path: Path) -> None:
    """Bug: a plugin without src/mflux/extras/<name>/__init__.py passes, and uv_build fails or ships nothing."""
    plugin = make_repo(tmp_path, "demo")
    (plugin / "src" / "mflux" / "extras" / "demo" / "__init__.py").unlink()
    assert only_problem(tmp_path) == "demo: src/mflux/extras/demo/__init__.py is missing"


def test_a_distribution_name_without_the_mflux_prefix_is_refused(tmp_path: Path) -> None:
    """Bug: an official plugin publishes as "demo" and nobody can tell it belongs to mflux-plugins."""
    make_repo(tmp_path, "demo", project_name="demo")
    assert only_problem(tmp_path) == "demo: project.name must be \"mflux-demo\", got 'demo'"


def test_a_plugin_without_a_command_is_refused(tmp_path: Path) -> None:
    """Bug: an empty [project.scripts] passes, and the published plugin installs no command at all."""
    make_repo(tmp_path, "demo", no_scripts=True)
    assert only_problem(tmp_path) == "demo: [project.scripts] declares no command"


@pytest.mark.parametrize("command", ["mflux-generate-z-image", "generate-z-image-demo"])
def test_a_command_without_the_mflux_prefix_or_the_plugin_suffix_is_refused(tmp_path: Path, command: str) -> None:
    """Bug: a plugin adds mflux-generate-z-image, which reads as (and shadows) mflux's own command, or
    generate-z-image-demo, which nobody finds next to the other mflux-* commands."""
    make_repo(tmp_path, "demo", scripts=f'"{command}" = "mflux.extras.demo.z_image:main"')
    assert only_problem(tmp_path) == f'demo: command {command!r} must start with "mflux-" and end with "-demo"'


def test_a_command_that_points_outside_the_plugin_is_refused(tmp_path: Path) -> None:
    """Bug: a command runs code from another package, so the plugin's tests never cover it."""
    make_repo(tmp_path, "demo", scripts='"mflux-generate-z-image-demo" = "other.module:main"')
    assert only_problem(tmp_path) == (
        "demo: command 'mflux-generate-z-image-demo' must point into mflux.extras.demo, got 'other.module:main'"
    )


def test_main_exits_1_and_prints_each_problem(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Bug: main() prints problems but exits 0, so the lint job stays green on a broken plugin."""
    make_repo(tmp_path, "demo", project_name="demo")
    assert check_plugins.main([str(tmp_path)]) == 1
    assert capsys.readouterr().out == "demo: project.name must be \"mflux-demo\", got 'demo'\n"


def test_main_exits_0_on_a_correct_repository(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Bug: main() exits 1 on a clean repository, so every pull request fails the required check."""
    make_repo(tmp_path, "demo")
    assert check_plugins.main([str(tmp_path)]) == 0
    assert capsys.readouterr().out == "check_plugins: 1 plugin folder(s) follow the rules\n"


def test_a_license_other_than_apache_is_refused(tmp_path: Path) -> None:
    """Bug: a plugin declares MIT while its folder carries the repository's Apache LICENSE copy."""
    plugin = make_repo(tmp_path, "demo")
    text = (plugin / "pyproject.toml").read_text(encoding="utf-8").replace('"Apache-2.0"', '"MIT"')
    (plugin / "pyproject.toml").write_text(text, encoding="utf-8")
    assert only_problem(tmp_path) == 'demo: license must be "Apache-2.0", the license of this repository'


@pytest.mark.parametrize("broken", ["file missing", "not listed"])
def test_a_license_file_that_is_missing_or_unlisted_is_refused(tmp_path: Path, broken: str) -> None:
    """Bug: the plugin folder has no LICENSE (packaging can't reach the root one) or license-files leaves it out,
    so the wheel ships without its license text."""
    plugin = make_repo(tmp_path, "demo", **({"license_files": "[]"} if broken == "not listed" else {}))
    if broken == "file missing":
        (plugin / "LICENSE").unlink()
    assert only_problem(tmp_path) == "demo: the folder must carry LICENSE and list it in license-files"


def test_a_notice_file_missing_from_license_files_is_refused(tmp_path: Path) -> None:
    """Bug: the plugin copies MIT code from mflux, has a NOTICE, but the wheel ships without it."""
    plugin = make_repo(tmp_path, "demo")
    (plugin / "NOTICE").write_text("mflux MIT notice\n", encoding="utf-8")
    assert only_problem(tmp_path) == "demo: NOTICE is not in license-files, so the wheel would ship without it"


def test_a_listed_notice_without_the_file_is_refused(tmp_path: Path) -> None:
    """Bug: license-files names NOTICE but the file is gone, and uv build fails only at release time."""
    make_repo(tmp_path, "demo", license_files='["LICENSE", "NOTICE"]')
    assert only_problem(tmp_path) == "demo: license-files lists NOTICE, but the folder has no NOTICE file"


def test_a_listed_notice_with_the_file_present_passes(tmp_path: Path) -> None:
    """Bug: the NOTICE rule fires on a plugin that lists and ships its NOTICE correctly."""
    plugin = make_repo(tmp_path, "demo", license_files='["LICENSE", "NOTICE"]')
    (plugin / "NOTICE").write_text("mflux MIT notice\n", encoding="utf-8")
    assert check_plugins.check_repo(tmp_path) == []


def test_uv_sources_are_refused(tmp_path: Path) -> None:
    """Bug: a git or path source in [tool.uv.sources] passes, and the published wheel can't install from PyPI."""
    make_repo(tmp_path, "demo", extra='\n[tool.uv.sources]\nmflux = { git = "https://github.com/x/y" }\n')
    assert only_problem(tmp_path) == "demo: remove [tool.uv.sources]: every dependency must install from PyPI"


def test_a_direct_reference_dependency_is_refused(tmp_path: Path) -> None:
    """Bug: "lib @ git+https://..." in dependencies passes, and PyPI rejects or can't resolve the release."""
    make_repo(tmp_path, "demo", dependencies='["mflux>=0.22", "lib @ git+https://github.com/x/lib"]')
    assert only_problem(tmp_path) == (
        "demo: dependency 'lib @ git+https://github.com/x/lib' must come from PyPI, not a URL or a path"
    )


@pytest.mark.parametrize(
    ("dependencies", "expected"),
    [
        ('["mflux>=0.22,<0.23"]', "'mflux>=0.22,<0.23' must have no upper bound or pin, found <"),
        ('["mflux~=0.22"]', "'mflux~=0.22' must have no upper bound or pin, found ~="),
        ('["mflux==0.22.0"]', "'mflux==0.22.0' must have no upper bound or pin, found =="),
        (
            "[\"mflux[x]>=0.22,<1; python_version>='3.10'\"]",
            "\"mflux[x]>=0.22,<1; python_version>='3.10'\" must have no upper bound or pin, found <",
        ),
    ],
)
def test_an_mflux_upper_bound_or_pin_is_refused(tmp_path: Path, dependencies: str, expected: str) -> None:
    """Bug: a cap (<, ~=, ==) passes, also when the requirement has extras and markers, and installing the plugin
    quietly downgrades a newer mflux."""
    make_repo(tmp_path, "demo", dependencies=dependencies)
    assert only_problem(tmp_path) == f"demo: the mflux requirement {expected}"


@pytest.mark.parametrize("dependencies", ['["mflux"]', '["mflux!=0.22.1"]'])
def test_an_mflux_requirement_without_a_floor_is_refused(tmp_path: Path, dependencies: str) -> None:
    """Bug: a bare "mflux" passes, or "!=" counts as a floor, and the plugin installs on an mflux without the command
    steps it calls."""
    make_repo(tmp_path, "demo", dependencies=dependencies)
    requirement = dependencies[2:-2]
    assert only_problem(tmp_path) == (
        f'demo: the mflux requirement {requirement!r} needs a lower bound, for example "mflux>=0.22"'
    )


@pytest.mark.parametrize(
    ("dependencies", "expected"),
    [
        (
            '["mflux>=0.21,<0.23; python_version<\'3.11\'", "mflux>=0.22"]',
            "\"mflux>=0.21,<0.23; python_version<'3.11'\" must have no upper bound or pin, found <",
        ),
        (
            '["mflux!=0.22.1; python_version<\'3.11\'", "mflux>=0.22"]',
            '"mflux!=0.22.1; python_version<\'3.11\'" needs a lower bound, for example "mflux>=0.22"',
        ),
    ],
    ids=["cap-in-the-first", "no-floor-in-the-first"],
)
def test_every_mflux_requirement_is_checked(tmp_path: Path, dependencies: str, expected: str) -> None:
    """Bug: with one mflux requirement per Python version, only the last one is checked, so a cap or a missing floor
    in an earlier one passes and the plugin downgrades mflux on that Python."""
    make_repo(tmp_path, "demo", dependencies=dependencies)
    assert only_problem(tmp_path) == f"demo: the mflux requirement {expected}"


@pytest.mark.parametrize(
    "dependencies",
    ['["mflux>0.21"]', '["mflux>=0.22,!=0.22.1"]', "[\"mflux[x]>=0.22; python_version<'3.14'\"]"],
)
def test_an_mflux_floor_without_a_cap_passes(tmp_path: Path, dependencies: str) -> None:
    """Bug: ">" is not taken as a floor, "!=" (excluding one broken release) is taken as a cap, or the "<" inside an
    environment marker is read as a cap on mflux."""
    make_repo(tmp_path, "demo", dependencies=dependencies)
    assert check_plugins.check_repo(tmp_path) == []


@pytest.mark.parametrize("dependencies", ['["numpy>=2"]', '["mflux-teacache>=1"]'])
def test_a_missing_mflux_requirement_is_refused(tmp_path: Path, dependencies: str) -> None:
    """Bug: a plugin that imports mflux but does not depend on it installs into an environment without mflux; or a
    name that only starts with "mflux" counts as the mflux requirement."""
    make_repo(tmp_path, "demo", dependencies=dependencies)
    assert only_problem(tmp_path) == "demo: dependencies must include mflux"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("lib @ git+https://github.com/x/lib", ("lib", [], True)),
        ("MFlux[x] >= 1, < 2", ("mflux", [">=", "<"], False)),
        ("mflux_teacache[mflux]>=0.13", ("mflux-teacache", [">="], False)),
        ("mflux>=0.22; os_name == 'nt'", ("mflux", [">="], False)),
        ("mflux (>=0.22,!=0.22.1)", ("mflux", [">=", "!="], False)),
    ],
)
def test_split_requirement(text: str, expected: tuple[str, list[str], bool]) -> None:
    """Bug: the parser keeps the name's case or underscores, reads a marker's comparison as a version operator,
    misses spaces or parentheses around the specifiers, or does not flag a direct reference."""
    assert check_plugins.split_requirement(text) == expected


@pytest.mark.parametrize(
    ("dev", "expected"),
    [
        (
            '["pytest>=9.1,<10.0", "ruff==0.15.0", "ty==0.0.72"]',
            "demo: ruff==0.15.0 must equal the ruff version in .github/workflows/lint.yml ['0.16.3']",
        ),
        (
            '["pytest>=9.1,<10.0", "ty==0.0.72"]',
            'demo: the dev group must pin ruff exactly ("ruff==X.Y.Z"), the version lint.yml uses',
        ),
        (
            '["pytest>=9.1,<10.0", "ruff>=0.16", "ty==0.0.72"]',
            'demo: the dev group must pin ruff exactly ("ruff==X.Y.Z"), the version lint.yml uses',
        ),
    ],
)
def test_a_ruff_pin_that_is_missing_or_differs_from_lint_is_refused(tmp_path: Path, dev: str, expected: str) -> None:
    """Bug: a plugin pins another ruff than CI lints with, or none, so local and CI formatting disagree."""
    make_repo(tmp_path, "demo", dev=dev)
    assert only_problem(tmp_path) == expected


@pytest.mark.parametrize("owners", ["* @owner\n", "* @owner\n/demo/\n", "* @owner\n/demo/ @owner\n/demo/\n"])
def test_a_folder_without_an_owner_is_refused(tmp_path: Path, owners: str) -> None:
    """Bug: a new plugin folder has no CODEOWNERS line, or a line with no owner on it, or a later owner-less line that
    wins over the owned one (GitHub applies the last matching line), so nobody is asked to review its pull
    requests."""
    make_repo(tmp_path, "demo")
    (tmp_path / ".github" / "CODEOWNERS").write_text(owners, encoding="utf-8")
    assert only_problem(tmp_path) == "demo: add a line '/demo/ @<maintainer>' to .github/CODEOWNERS"


def test_a_folder_without_its_test_workflow_is_refused(tmp_path: Path) -> None:
    """Bug: a plugin is merged with no <folder>.yml, so its tests never run in CI."""
    make_repo(tmp_path, "demo")
    (tmp_path / ".github" / "workflows" / "demo.yml").unlink()
    assert only_problem(tmp_path) == "demo: add .github/workflows/demo.yml to run this folder's tests"


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("working-directory: demo", "working-directory: ."),
        ("demo/uv.lock", "uv.lock"),
        ("working-directory: demo", "# working-directory: demo"),
    ],
    ids=["other-directory", "other-lock", "commented-out"],
)
def test_a_test_workflow_that_runs_elsewhere_is_refused(tmp_path: Path, old: str, new: str) -> None:
    """Bug: demo.yml was copied from another plugin and still runs in that folder (or caches that folder's lock), or
    its working-directory line is commented out, so the demo tests never run while the job stays green."""
    make_repo(tmp_path, "demo")
    workflow = tmp_path / ".github" / "workflows" / "demo.yml"
    workflow.write_text(ci_workflow("demo").replace(old, new), encoding="utf-8")
    assert only_problem(tmp_path) == (
        'demo: demo.yml must run in this folder: it needs "working-directory: demo" and "demo/uv.lock"'
    )


def test_a_test_workflow_naming_another_plugin_folder_is_refused(tmp_path: Path) -> None:
    """Bug: demo.yml, copied from the template, still triggers on template/** changes, so a demo change runs no
    tests."""
    make_repo(tmp_path, "demo")
    make_repo(tmp_path, check_plugins.TEMPLATE_FOLDER)
    workflow = tmp_path / ".github" / "workflows" / "demo.yml"
    workflow.write_text(ci_workflow("demo").replace("'demo/**'", "'template/**'"), encoding="utf-8")
    assert only_problem(tmp_path) == "demo: demo.yml refers to template/, another plugin's folder"


@pytest.mark.parametrize(
    ("release", "expected"),
    [
        (
            release_workflow("demo", tag="teacache-v*"),
            "demo: demo-release.yml must publish through the environment pypi-demo from demo-v* tags",
        ),
        (
            release_workflow("demo", environment="pypi-teacache"),
            "demo: demo-release.yml must publish through the environment pypi-demo from demo-v* tags",
        ),
        (None, "demo: add .github/workflows/demo-release.yml to publish from demo-v* tags"),
    ],
    ids=["wrong-tag", "wrong-environment", "missing"],
)
def test_a_release_workflow_that_is_missing_or_wired_elsewhere_is_refused(
    tmp_path: Path, release: str | None, expected: str
) -> None:
    """Bug: demo-release.yml is missing, or was copied from teacache and still fires on teacache-v* tags or publishes
    through pypi-teacache."""
    make_repo(tmp_path, "demo")
    path = tmp_path / ".github" / "workflows" / "demo-release.yml"
    if release is None:
        path.unlink()
    else:
        path.write_text(release, encoding="utf-8")
    assert only_problem(tmp_path) == expected


def test_a_release_workflow_that_builds_another_plugin_folder_is_refused(tmp_path: Path) -> None:
    """Bug: demo-release.yml has the right tag and environment but still builds in teacache/, so a demo tag uploads
    the teacache wheel."""
    make_repo(tmp_path, "demo")
    make_repo(tmp_path, "teacache")
    path = tmp_path / ".github" / "workflows" / "demo-release.yml"
    path.write_text(release_workflow("demo") + "defaults:\n  run:\n    working-directory: teacache\n", encoding="utf-8")
    assert only_problem(tmp_path) == "demo: demo-release.yml refers to teacache/, another plugin's folder"


def test_a_folder_name_inside_a_longer_one_is_not_another_plugins_folder(tmp_path: Path) -> None:
    """Bug: the other-folder rule matches "cache/" inside "teacache/uv.lock", so plugins named cache and teacache can
    never live in one repository."""
    make_repo(tmp_path, "teacache")
    make_repo(tmp_path, "cache")
    assert check_plugins.check_repo(tmp_path) == []


def test_a_plugin_missing_from_the_readme_table_is_refused(tmp_path: Path) -> None:
    """Bug: a plugin ships without a row in the root README, so users never find it."""
    make_repo(tmp_path, "demo")
    (tmp_path / "README.md").write_text("| Plugin |\n|---|\n", encoding="utf-8")
    assert only_problem(tmp_path) == "demo: add a row for this plugin to the Plugins table in the root README.md"


def test_a_repository_with_only_the_template_passes(tmp_path: Path) -> None:
    """Bug: the template is held to the published-plugin rules (release workflow, README row) and the repo goes
    red."""
    make_repo(tmp_path, check_plugins.TEMPLATE_FOLDER)
    assert check_plugins.check_repo(tmp_path) == []


def test_a_plugin_and_the_template_pass_together(tmp_path: Path) -> None:
    """Bug: the other-folder rule also counts the folder being checked, so demo.yml "refers to" demo/ and a repository
    with a plugin and the template can never pass."""
    make_repo(tmp_path, "demo")
    make_repo(tmp_path, check_plugins.TEMPLATE_FOLDER)
    assert check_plugins.check_repo(tmp_path) == []


def test_a_template_without_the_private_classifier_is_refused(tmp_path: Path) -> None:
    """Bug: the classifier that makes PyPI reject an upload is dropped, so the template could be published."""
    make_repo(tmp_path, check_plugins.TEMPLATE_FOLDER, classifiers="[]")
    assert only_problem(tmp_path) == (
        'template: keep the classifier "Private :: Do Not Upload" so the template can never be uploaded to PyPI'
    )


def test_a_template_release_workflow_is_refused(tmp_path: Path) -> None:
    """Bug: someone adds template-release.yml, and a template tag would try to publish mflux-myplugin."""
    make_repo(tmp_path, check_plugins.TEMPLATE_FOLDER)
    (tmp_path / ".github" / "workflows" / "template-release.yml").write_text("on: push\n", encoding="utf-8")
    assert only_problem(tmp_path) == "template: the template must have no release workflow"


def test_a_plugin_that_kept_the_private_classifier_is_refused(tmp_path: Path) -> None:
    """Bug: a renamed copy keeps "Private :: Do Not Upload", and its first release fails at the PyPI upload."""
    make_repo(tmp_path, "demo", classifiers=f'["{check_plugins.PRIVATE_CLASSIFIER}"]')
    assert only_problem(tmp_path) == 'demo: remove the classifier "Private :: Do Not Upload" copied from the template'


@pytest.mark.parametrize(
    ("relative", "content"),
    [
        ("uv.lock", 'name = "mflux-myplugin"\n'),
        ("NOTICE", "mflux-myplugin\nCopyright 2026 the mflux-plugins contributors\n"),
        ("src/mflux/extras/myplugin/__init__.py", ""),
    ],
    ids=["lock-not-relocked", "notice-without-suffix", "package-folder-left-behind"],
)
def test_a_leftover_template_name_is_refused(tmp_path: Path, relative: str, content: str) -> None:
    """Bug: the rename missed a file: uv.lock (not relocked), NOTICE (no suffix, so a suffix list skips it) or the
    old package folder, and the plugin still carries mflux-myplugin."""
    plugin = make_repo(tmp_path, "demo", license_files='["LICENSE", "NOTICE"]')
    (plugin / "NOTICE").write_text("mflux-demo\n", encoding="utf-8")
    path = plugin / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    assert only_problem(tmp_path) == f"demo: the template name 'myplugin' is still in: {relative}"


def test_virtualenvs_caches_build_output_and_binaries_are_not_scanned(tmp_path: Path) -> None:
    """Bug: the leftover scan reads .venv or .venv-py310 (whose site-packages may hold the template's build), a
    cache, dist/, build/, or a binary test image whose bytes happen to spell the token, and fails a correct plugin."""
    plugin = make_repo(tmp_path, "demo")
    for relative in (
        ".venv/lib/site.py",
        ".venv-py310/lib/site.py",
        "src/mflux/extras/demo/__pycache__/z_image.cpython-313.pyc",
        "dist/README.txt",
        "build/lib/x.py",
    ):
        path = plugin / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("mflux-myplugin\n", encoding="utf-8")
    (plugin / "tests").mkdir()
    (plugin / "tests" / "fixture.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00mflux-myplugin")
    assert check_plugins.check_repo(tmp_path) == []


@pytest.mark.parametrize(
    "url", [f"{REPO_URL}/tree/main/template", f"{REPO_URL}/blob/main/teacache/README.md"], ids=["tree", "blob"]
)
def test_a_project_url_into_another_folder_is_refused(tmp_path: Path, url: str) -> None:
    """Bug: the renamed copy's Homepage still points at tree/main/template (or another plugin's file), so PyPI links
    users to the wrong folder."""
    make_repo(tmp_path, "demo", homepage=url)
    assert only_problem(tmp_path) == (
        f"demo: [project.urls] Homepage must point at this folder (/tree/main/demo), got {url!r}"
    )


def test_a_project_url_into_this_folder_passes(tmp_path: Path) -> None:
    """Bug: the URL rule compares the whole rest of the URL, so the usual #readme anchor is refused."""
    make_repo(tmp_path, "demo", homepage=f"{REPO_URL}/tree/main/demo#readme")
    assert check_plugins.check_repo(tmp_path) == []


def with_description(plugin: Path, description: str) -> None:
    text = (plugin / "pyproject.toml").read_text(encoding="utf-8")
    line = f'description = "{description}"\n'
    (plugin / "pyproject.toml").write_text(text.replace("license =", line + "license =", 1), encoding="utf-8")


@pytest.mark.parametrize(
    ("description", "matched"),
    [
        (
            "mflux-demo: <one sentence about your plugin>. This is the template; never publish it.",
            "<one sentence about your plugin>",
        ),
        ("mflux-demo: adds a demo option. Never publish it.", "Never publish"),
        ("mflux-demo: <One Sentence>.", "<One Sentence>"),
    ],
    ids=["template-text", "never-publish", "placeholder"],
)
def test_a_plugin_description_that_still_has_template_text_is_refused(
    tmp_path: Path, description: str, matched: str
) -> None:
    """Bug: a sed rename keeps the template's "<one sentence about your plugin>" placeholder or its "never publish it"
    note, and the plugin is published with a description that says it must not be."""
    with_description(make_repo(tmp_path, "demo"), description)
    assert only_problem(tmp_path) == (
        f"demo: project.description still has template text ({matched!r}): rewrite it for this plugin"
    )


def test_a_plugin_description_that_mentions_templates_passes(tmp_path: Path) -> None:
    """Bug: the description rule refuses any description with the word "template", so a plugin about prompt templates
    can never describe itself."""
    with_description(make_repo(tmp_path, "demo"), "mflux-demo: prompt templates for mflux's command line.")
    assert check_plugins.check_repo(tmp_path) == []


def test_a_symlink_is_not_scanned_for_leftovers(tmp_path: Path) -> None:
    """Bug: the leftover scan follows a symlink out of the plugin folder (to a file the plugin doesn't own) and fails a
    correct plugin on that file's text, or reads a file outside the checkout."""
    plugin = make_repo(tmp_path, "demo")
    outside = tmp_path / "outside.txt"
    outside.write_text("mflux-myplugin\n", encoding="utf-8")
    (plugin / "link.txt").symlink_to(outside)
    assert check_plugins.check_repo(tmp_path) == []


@pytest.mark.parametrize("entry", ["/demo/", "/demo/**", "demo/", "/demo"])
def test_every_codeowners_form_of_the_folder_passes(tmp_path: Path, entry: str) -> None:
    """Bug: the owner rule accepts only "/demo/", so a valid CODEOWNERS line written as /demo/**, demo/ or /demo is
    refused."""
    make_repo(tmp_path, "demo")
    (tmp_path / ".github" / "CODEOWNERS").write_text(f"* @owner\n{entry} @owner\n", encoding="utf-8")
    assert check_plugins.check_repo(tmp_path) == []


def test_a_ruff_pin_with_a_marker_is_compared_by_its_version(tmp_path: Path) -> None:
    """Bug: "ruff==0.16.3; sys_platform == 'darwin'" is compared with its marker attached, so a pin that equals
    lint.yml's ruff is refused."""
    make_repo(tmp_path, "demo", dev='["pytest>=9.1,<10.0", "ruff==0.16.3; sys_platform==\'darwin\'", "ty==0.0.72"]')
    assert check_plugins.check_repo(tmp_path) == []


@pytest.mark.parametrize(
    ("lint", "listed"),
    [
        (
            "      - run: uvx ruff@0.16.3 check .\n      - run: uvx ruff@0.17.0 format --check .\n",
            "['0.16.3', '0.17.0']",
        ),
        ("      - run: uvx ruff check .\n", "[]"),
    ],
    ids=["two-versions", "no-version"],
)
def test_lint_naming_two_ruff_versions_or_none_is_refused(tmp_path: Path, lint: str, listed: str) -> None:
    """Bug: the check only asks whether the pin is one of lint.yml's ruff versions (or skips lint.yml without one), so
    lint.yml can check with one ruff and format with another, or float to the newest ruff, and no pin can match."""
    make_repo(tmp_path, "demo")
    (tmp_path / ".github" / "workflows" / "lint.yml").write_text(
        "jobs:\n  repo:\n    steps:\n" + lint, encoding="utf-8"
    )
    assert (
        only_problem(tmp_path)
        == f"demo: ruff==0.16.3 must equal the ruff version in .github/workflows/lint.yml {listed}"
    )


def test_a_project_url_outside_this_repository_passes(tmp_path: Path) -> None:
    """Bug: the URL rule reads /tree/main/ in any URL, so a link to the wrapped library's own docs is refused."""
    plugin = make_repo(tmp_path, "demo")
    text = (plugin / "pyproject.toml").read_text(encoding="utf-8")
    library = 'Library = "https://github.com/IonDen/mlx-teacache/tree/main/docs"\n'
    (plugin / "pyproject.toml").write_text(
        text.replace("[project.scripts]", library + "\n[project.scripts]"), encoding="utf-8"
    )
    assert check_plugins.check_repo(tmp_path) == []


def test_main_checks_the_working_directory_without_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Bug: main([]) ignores the working directory, the way lint.yml calls it, and checks some other path."""
    make_repo(tmp_path, "demo", project_name="demo")
    monkeypatch.chdir(tmp_path)
    assert check_plugins.main([]) == 1
    assert capsys.readouterr().out == "demo: project.name must be \"mflux-demo\", got 'demo'\n"


def test_main_fails_when_there_is_no_plugin_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Bug: run from the wrong directory (or with every pyproject.toml gone), the checker finds no folder, reports
    nothing and exits 0, so the required job stays green while checking nothing."""
    monkeypatch.chdir(tmp_path)
    assert check_plugins.main([]) == 1
    assert capsys.readouterr().out == "check_plugins: no plugin folders found\n"


@pytest.mark.parametrize("folder", ["demo", check_plugins.TEMPLATE_FOLDER])
def test_a_test_workflow_without_workflow_call_is_refused(tmp_path: Path, folder: str) -> None:
    """Bug: <folder>.yml (or the template's, which every new plugin copies) has no workflow_call trigger, so the
    release workflow can't run the tests before it publishes."""
    make_repo(tmp_path, folder)
    workflow = tmp_path / ".github" / "workflows" / f"{folder}.yml"
    workflow.write_text(ci_workflow(folder).replace("  workflow_call:\n", ""), encoding="utf-8")
    assert only_problem(tmp_path) == (
        f'{folder}: {folder}.yml needs a "workflow_call:" trigger, so a release workflow can run its tests '
        "before it publishes"
    )


def test_a_release_workflow_that_runs_another_folders_tests_is_refused(tmp_path: Path) -> None:
    """Bug: demo-release.yml, copied from teacache-release.yml, still runs teacache.yml as its test job, so a demo
    release publishes without running the demo tests."""
    make_repo(tmp_path, "demo")
    path = tmp_path / ".github" / "workflows" / "demo-release.yml"
    path.write_text(release_workflow("demo", tests="teacache"), encoding="utf-8")
    assert only_problem(tmp_path) == (
        'demo: demo-release.yml must run the tests first, with "uses: ./.github/workflows/demo.yml"'
    )


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        (
            "name: demo\n",
            "name: teacache-copy\n",
            'demo.yml must start with "name: demo", so its runs show under this plugin',
        ),
        (
            "name: demo\n",
            "name: demo-release\n",
            'demo.yml must start with "name: demo", so its runs show under this plugin',
        ),
        (
            "group: demo-ci-",
            "group: ci-",
            'demo.yml must use a concurrency group starting "demo-ci-", so its runs never cancel another plugin\'s',
        ),
    ],
    ids=["name", "name-with-a-suffix", "concurrency"],
)
def test_a_test_workflow_with_another_name_or_concurrency_group_is_refused(
    tmp_path: Path, old: str, new: str, expected: str
) -> None:
    """Bug: demo.yml keeps a copied workflow name (its runs show under another plugin) or a shared concurrency group
    (a demo push cancels another plugin's run)."""
    make_repo(tmp_path, "demo")
    workflow = tmp_path / ".github" / "workflows" / "demo.yml"
    workflow.write_text(ci_workflow("demo").replace(old, new), encoding="utf-8")
    assert only_problem(tmp_path) == f"demo: {expected}"


@pytest.mark.parametrize(
    ("folder", "release"),
    [
        ("cache", release_workflow("cache", tag="teacache-v*")),
        ("demo", release_workflow("demo", environment="pypi-demo2")),
    ],
    ids=["tag-inside-teacache-v", "environment-pypi-demo2"],
)
def test_a_release_tag_or_environment_that_only_contains_the_folder_name_is_refused(
    tmp_path: Path, folder: str, release: str
) -> None:
    """Bug: the tag and environment rules match substrings, so cache-release.yml firing on teacache-v* tags, or
    demo-release.yml publishing through pypi-demo2, passes."""
    make_repo(tmp_path, folder)
    (tmp_path / ".github" / "workflows" / f"{folder}-release.yml").write_text(release, encoding="utf-8")
    assert only_problem(tmp_path) == (
        f"{folder}: {folder}-release.yml must publish through the environment pypi-{folder} from {folder}-v* tags"
    )


def test_a_lock_path_that_only_contains_the_folder_name_is_refused(tmp_path: Path) -> None:
    """Bug: the lock rule matches a substring, so cache.yml that caches teacache/uv.lock passes."""
    make_repo(tmp_path, "cache")
    workflow = tmp_path / ".github" / "workflows" / "cache.yml"
    workflow.write_text(ci_workflow("cache").replace("cache/uv.lock", "teacache/uv.lock"), encoding="utf-8")
    assert only_problem(tmp_path) == (
        'cache: cache.yml must run in this folder: it needs "working-directory: cache" and "cache/uv.lock"'
    )


@pytest.mark.parametrize("value", ['"demo"', "'demo'", "demo  # the plugin folder"])
def test_a_quoted_or_commented_working_directory_passes(tmp_path: Path, value: str) -> None:
    """Bug: the working-directory rule accepts only a bare value, so valid YAML ("demo", or a trailing comment) is
    refused."""
    make_repo(tmp_path, "demo")
    workflow = tmp_path / ".github" / "workflows" / "demo.yml"
    workflow.write_text(
        ci_workflow("demo").replace("working-directory: demo", f"working-directory: {value}"), encoding="utf-8"
    )
    assert check_plugins.check_repo(tmp_path) == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("paths: ['teacache/**']\n", True),
        ("    working-directory: teacache\n", True),
        ('    working-directory: "teacache"  # copied\n', True),
        ("name: teacache\n", True),
        ("  group: teacache-ci-${{ github.ref }}\n", True),
        ("name: teacache-release\n", False),
        ("# like src/mflux/extras/teacache/ in that plugin\n", False),
        ("cache-dependency-glob: mflux-teacache/uv.lock\n", False),
        ("  group: mflux-teacache-ci-x\n", False),
        ("    # working-directory: teacache\n", False),
    ],
)
def test_refers_to_folder(text: str, expected: bool) -> None:
    """Bug: the other-folder rule misses a copied workflow name, concurrency group, or a quoted or commented working
    directory, or reads a package path (src/mflux/extras/teacache/) or a longer name as the folder."""
    assert check_plugins.refers_to_folder(text, "teacache") is expected


def test_a_later_owned_line_wins_over_an_earlier_owner_less_one(tmp_path: Path) -> None:
    """Bug: the owner rule refuses a folder whose first matching CODEOWNERS line has no owner, although the last
    matching line, the one GitHub applies, names one."""
    make_repo(tmp_path, "demo")
    (tmp_path / ".github" / "CODEOWNERS").write_text("* @owner\n/demo/\n/demo/ @owner\n", encoding="utf-8")
    assert check_plugins.check_repo(tmp_path) == []


@pytest.mark.parametrize(
    ("folder", "text"),
    [
        ("actions", "      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1\n"),
        ("pypa", "      - uses: 'pypa/gh-action-pypi-publish@dc37677b2e1c63e2034f94d8a5b11f265b73ba33'\n"),
    ],
)
def test_a_remote_action_is_not_a_plugin_folder(folder: str, text: str) -> None:
    """Bug: the other-folder rule reads "uses: actions/checkout@..." as the folder actions/, so a plugin named
    actions (or pypa) fails every other plugin's workflows."""
    assert check_plugins.refers_to_folder(text, folder) is False
    assert check_plugins.refers_to_folder(f"paths: ['{folder}/**']\n", folder) is True


def test_a_remote_action_does_not_fail_a_plugin_named_like_its_owner(tmp_path: Path) -> None:
    """Bug: demo.yml checks out the code with actions/checkout, and a second plugin folder named actions makes the
    checker report that demo.yml refers to actions/."""
    make_repo(tmp_path, "demo")
    make_repo(tmp_path, "actions")
    workflow = tmp_path / ".github" / "workflows" / "demo.yml"
    workflow.write_text(ci_workflow("demo") + "      - uses: actions/checkout@3d3c42e5aac5\n", encoding="utf-8")
    assert check_plugins.check_repo(tmp_path) == []


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        (
            'dependencies = ["mflux>=0.22"]',
            'dependencies = "mflux>=0.22"',
            "project.dependencies must be an array of strings",
        ),
        (
            'license-files = ["LICENSE"]',
            'license-files = "LICENSE"',
            "project.license-files must be an array of strings",
        ),
        ("classifiers = []", "classifiers = [1]", "project.classifiers must be an array of strings"),
        ('"mflux.extras.demo.z_image:main"', "1", "project.scripts must be a table of strings"),
        (f'Repository = "{REPO_URL}"', "Repository = 1", "project.urls must be a table of strings"),
        ("dev = [", 'dev = "ruff==0.16.3"\nunused = [', "dependency-groups.dev must be an array"),
    ],
    ids=["dependencies", "license-files", "classifiers", "scripts", "urls", "dev"],
)
def test_a_pyproject_value_of_the_wrong_type_is_one_problem_not_a_crash(
    tmp_path: Path, old: str, new: str, expected: str
) -> None:
    """Bug: a string where pyproject.toml needs an array (or a number where it needs a string) crashes the checker
    with a traceback, or passes: "LICENSE" in "LICENSE" is true for a string license-files."""
    plugin = make_repo(tmp_path, "demo")
    text = (plugin / "pyproject.toml").read_text(encoding="utf-8")
    assert old in text
    (plugin / "pyproject.toml").write_text(text.replace(old, new, 1), encoding="utf-8")
    assert only_problem(tmp_path) == f"demo: pyproject.toml: {expected}"


def test_a_project_that_is_not_a_table_is_one_problem_not_a_crash(tmp_path: Path) -> None:
    """Bug: "project = 1" makes the checker call .get on an int and die with a traceback that names no folder."""
    plugin = make_repo(tmp_path, "demo")
    (plugin / "pyproject.toml").write_text("project = 1\n", encoding="utf-8")
    assert only_problem(tmp_path) == "demo: pyproject.toml: project must be a table"


def test_a_pyproject_that_is_not_utf8_is_one_problem_not_a_crash(tmp_path: Path) -> None:
    """Bug: a pyproject.toml saved in Latin-1 raises UnicodeDecodeError, so the lint job dies with a traceback that
    names no folder."""
    plugin = make_repo(tmp_path, "demo")
    (plugin / "pyproject.toml").write_bytes(b'[project]\ndescription = "caf\xe9"\n')
    assert only_problem(tmp_path).startswith("demo: pyproject.toml is not valid UTF-8: ")


def test_a_workflow_that_is_not_utf8_is_still_read(tmp_path: Path) -> None:
    """Bug: one non-UTF-8 byte in a comment of demo.yml crashes the checker for every plugin."""
    make_repo(tmp_path, "demo")
    workflow = tmp_path / ".github" / "workflows" / "demo.yml"
    workflow.write_bytes(ci_workflow("demo").encode() + b"# caf\xe9\n")
    assert check_plugins.check_repo(tmp_path) == []


@pytest.mark.parametrize("license_files", ['["LICEN[CS]E", "NOTICE"]', '["LICENSE", "NOTICE*"]', '["*"]'])
def test_license_files_given_as_globs_pass(tmp_path: Path, license_files: str) -> None:
    """Bug: license-files is compared as literal names, so the PEP 639 globs "LICEN[CS]E" or "NOTICE*" are refused
    although the wheel ships both files."""
    plugin = make_repo(tmp_path, "demo", license_files=license_files)
    (plugin / "NOTICE").write_text("mflux MIT notice\n", encoding="utf-8")
    assert check_plugins.check_repo(tmp_path) == []


def test_a_ruff_pin_written_with_three_equals_signs_passes(tmp_path: Path) -> None:
    """Bug: "ruff===0.16.3" is split on "==" and read as "=0.16.3", so an exact pin that equals lint.yml's is
    refused."""
    make_repo(tmp_path, "demo", dev='["pytest>=9.1,<10.0", "ruff===0.16.3", "ty==0.0.72"]')
    assert check_plugins.check_repo(tmp_path) == []


def test_every_ruff_pin_is_compared_with_lint(tmp_path: Path) -> None:
    """Bug: only the first ruff== pin is compared, so a second pin for another platform can name another ruff."""
    make_repo(
        tmp_path,
        "demo",
        dev="[\"ruff==0.16.3; sys_platform=='darwin'\", \"ruff==0.15.0; sys_platform!='darwin'\"]",
    )
    assert only_problem(tmp_path) == (
        "demo: ruff==0.15.0 must equal the ruff version in .github/workflows/lint.yml ['0.16.3']"
    )


def test_a_plugin_description_with_comparisons_passes(tmp_path: Path) -> None:
    """Bug: the placeholder rule reads "< 5 and >" as a <...> placeholder, so a description that compares two values
    is refused."""
    with_description(make_repo(tmp_path, "demo"), "mflux-demo: skips steps < 5 when guidance > 1.")
    assert check_plugins.check_repo(tmp_path) == []


@pytest.mark.parametrize("file", ["README.md", "AGENTS.md", "uv.lock"])
def test_a_plugin_without_its_readme_agents_file_or_lock_is_refused(tmp_path: Path, file: str) -> None:
    """Bug: a plugin folder ships without the README (PyPI shows an empty page), without the AGENTS.md that
    CONTRIBUTING.md requires, or without uv.lock, so its CI's "uv sync --locked" fails."""
    plugin = make_repo(tmp_path, "demo")
    (plugin / file).unlink()
    assert (
        only_problem(tmp_path)
        == f"demo: add {file} to the folder; every plugin folder carries README.md, AGENTS.md and uv.lock"
    )


def test_a_release_workflow_that_never_names_its_folder_as_working_directory_is_refused(tmp_path: Path) -> None:
    """Bug: demo-release.yml lost its "working-directory: demo" line, so uv build runs at the repository root, where
    there is no project, and the release fails only at tag time."""
    make_repo(tmp_path, "demo")
    path = tmp_path / ".github" / "workflows" / "demo-release.yml"
    path.write_text(release_workflow("demo").replace("working-directory: demo", "shell: bash"), encoding="utf-8")
    assert only_problem(tmp_path) == (
        'demo: demo-release.yml must build in this folder: it needs "working-directory: demo"'
    )
