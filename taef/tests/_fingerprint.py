"""AST fingerprints of functions that ignore comments, docstrings, formatting and the CPython
version (CI runs them on 3.11 and 3.13). Copied from mlx-teacache's tests/_mflux_surface.py."""

import ast
import hashlib
import inspect
import textwrap
from collections.abc import Callable
from typing import Any

# Fields whose presence or text depends on the interpreter, not on the code. ast.dump is not
# used: 3.13 changed its defaults so that empty lists and None fields are omitted.
_INTERPRETER_FIELDS = frozenset(
    {"lineno", "col_offset", "end_lineno", "end_col_offset", "type_comment", "type_params", "kind", "ctx"}
)
# The keyword arguments whose string value is wording a user reads in --help, not what the parser accepts.
_PROSE_KEYWORDS = frozenset({"help", "help_text", "description", "epilog"})


def _strip_docstrings(tree: ast.AST) -> ast.AST:
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list) or not body:
            continue
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
            del body[0]
    return tree


def _normalized(node: Any) -> str:
    if isinstance(node, ast.AST):
        parts = [type(node).__name__]
        for field in node._fields:
            if field in _INTERPRETER_FIELDS:
                continue
            value = getattr(node, field, None)
            if value is None or value == []:
                continue
            parts.append(f"{field}={_normalized(value)}")
        return "(" + ",".join(parts) + ")"
    if isinstance(node, list):
        return "[" + ",".join(_normalized(item) for item in node) + "]"
    return repr(node)


def _function_node(fn: Callable[..., Any]) -> ast.FunctionDef | ast.AsyncFunctionDef:
    tree = _strip_docstrings(ast.parse(textwrap.dedent(inspect.getsource(fn))))
    return next(node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)))


def ast_fingerprint(fn: Callable[..., Any]) -> str:
    """Short sha256 of a function's arguments and body; the name and decorators don't move it."""
    return _digest(_function_node(fn))


def _digest(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    dumped = _normalized(node.args) + "|" + "|".join(_normalized(statement) for statement in node.body)
    return hashlib.sha256(dumped.encode()).hexdigest()[:16]


def parser_fingerprint(fn: Callable[..., Any]) -> str:
    """ast_fingerprint of a build_parser that ignores the text of the help, help_text, description and epilog strings
    it passes, so a reworded --help line doesn't move it. The keywords themselves still count, and so does a help
    value that is not a plain string (an f-string or a name can carry an option's value)."""
    node = _function_node(fn)
    for keyword in ast.walk(node):
        if (
            isinstance(keyword, ast.keyword)
            and keyword.arg in _PROSE_KEYWORDS
            and isinstance(keyword.value, ast.Constant)
            and isinstance(keyword.value.value, str)
        ):
            keyword.value = ast.Constant(value="")
    return _digest(node)


def function_body_dumps(fn: Callable[..., Any]) -> list[str]:
    """One normalized dump per top-level statement of the function body."""
    return [_normalized(statement) for statement in _function_node(fn).body]


def statements_fingerprint_from(fn: Callable[..., Any], first_statement: str) -> str:
    """Short sha256 of a function's top-level statements from the one whose source (ast.unparse) is
    `first_statement` to the end. Locating the start by content, not index, keeps the digest still when mflux
    adds or drops a line before it."""
    body = _function_node(fn).body
    starts = [index for index, statement in enumerate(body) if ast.unparse(statement) == first_statement]
    assert starts, f"{fn.__qualname__} has no top-level statement {first_statement!r}"
    dumped = "|".join(_normalized(statement) for statement in body[starts[0] :])
    return hashlib.sha256(dumped.encode()).hexdigest()[:16]
