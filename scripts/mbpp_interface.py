from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from src.data.format_tasks import nested_value


_BUILTINS = {
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float",
    "int", "isinstance", "len", "list", "map", "max", "min", "open",
    "print", "range", "round", "set", "sorted", "str", "sum", "tuple",
    "type", "zip",
}


def _calls(tree: ast.AST) -> list[ast.Call]:
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)]


def infer_interface(test_list: list[str], *, stable_id: str) -> dict[str, Any]:
    """Infer the tested function from test calls without using the solution."""
    candidates: list[tuple[str, ast.Call]] = []
    for source in test_list:
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            return {"interface_parse_failed": True, "interface_error": f"test syntax: {exc}"}
        for call in _calls(tree):
            if isinstance(call.func, ast.Name) and call.func.id not in _BUILTINS:
                candidates.append((call.func.id, call))
    if not candidates:
        return {"interface_parse_failed": True, "interface_error": "no direct function call"}

    counts: dict[str, int] = {}
    for name, _ in candidates:
        counts[name] = counts.get(name, 0) + 1
    name = max(counts, key=lambda value: counts[value])
    selected = [call for candidate, call in candidates if candidate == name]
    signatures = []
    for call in selected:
        if any(isinstance(arg, ast.Starred) for arg in call.args):
            return {"interface_parse_failed": True, "interface_error": "starred argument"}
        parts = [f"arg{index}" for index, _ in enumerate(call.args, start=1)]
        parts.extend(keyword.arg or f"arg{len(parts) + 1}" for keyword in call.keywords)
        signatures.append(tuple(parts))
    if not signatures or any(signature != signatures[0] for signature in signatures[1:]):
        return {"interface_parse_failed": True, "interface_error": "inconsistent call signatures"}
    signature = ", ".join(signatures[0])
    return {
        "interface_parse_failed": False,
        "required_function_name": name,
        "required_signature": signature,
        "interface_source": "tests_ast_call",
        "stable_id": stable_id,
    }


def build_prompt(problem: str, interface: dict[str, Any]) -> str:
    if interface.get("interface_parse_failed"):
        raise ValueError(f"{interface.get('interface_error', 'unknown')}")
    return (
        "Problem:\n"
        f"{problem.strip()}\n\n"
        "Required interface:\n"
        "Define exactly this function:\n"
        f"{interface['required_function_name']}({interface['required_signature']})\n\n"
        "Return only valid executable Python code.\n"
        "Do not rename the required function.\n"
        "Do not include explanations, markdown, examples, tests, or interactive input.\n\n"
        "Code:\n"
    )


def write_interface_manifest(
    *,
    source_manifest: Path,
    dataset: Any,
    mapping: dict[str, str],
    tokenizer: Any,
    output: Path,
) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in source_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = []
    for source in rows:
        row = dataset[int(source["row_index"])]
        tests = [str(value) for value in nested_value(row, mapping["test_list"])]
        interface = infer_interface(tests, stable_id=source["stable_id"])
        if interface.get("interface_parse_failed"):
            raise RuntimeError(f"{source['stable_id']}: {interface['interface_error']}")
        prompt = build_prompt(str(nested_value(row, mapping["prompt"])), interface)
        token_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        result.append({
            **source,
            "prompt": prompt,
            "prompt_token_ids": token_ids,
            "required_function_name": interface["required_function_name"],
            "required_signature": interface["required_signature"],
            "interface_parse_failed": False,
            "test_list": tests,
            "test_setup_code": str(nested_value(row, mapping["test_setup_code"])),
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in result),
        encoding="utf-8",
    )
    return result
