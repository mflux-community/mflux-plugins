"""Warning filters more than one test module needs. Kept here so test modules never import each other."""

import pytest

# mflux 0.22's parse_args reads a -C sidecar with json.load(path.open("rt")) and never closes the file
# (mflux/cli/parser/parsers.py:448); under filterwarnings = error that ResourceWarning would fail any test
# that replays a sidecar. Scoped to the sidecar files these tests write. tests/test_refusals.py checks that
# mflux still leaves the file open; when that check goes red, remove both names and every use of them.
SIDECAR_LEFT_OPEN_MESSAGE = r"unclosed file <_io\.TextIOWrapper name='[^']*\.metadata\.json'"
SIDECAR_LEFT_OPEN_BY_MFLUX = pytest.mark.filterwarnings(f"ignore:{SIDECAR_LEFT_OPEN_MESSAGE}:ResourceWarning")
