from __future__ import annotations

import json

from src.common import load_yaml
from src.data.format_tasks import (
    canonical_content_sha256,
    canonical_stable_id,
    format_task_prompt,
    format_task_target,
)


def test_formal_external_sources_are_registered_with_pinned_identity() -> None:
    config = load_yaml("configs/data.yaml")
    expected = {
        "math_train": "EleutherAI/hendrycks_math",
        "openmathinstruct2": "nvidia/OpenMathInstruct-2",
        "musique": "voidful/MuSiQue",
        "2wikimultihopqa": "xanhho/2WikiMultihopQA",
        "xcoder_80k": "banksy235/XCoder-80K",
    }
    assert set(config["external_sources"]) == set(expected)
    for key, repo_id in expected.items():
        source = config["external_sources"][key]
        assert source["repo_id"] == repo_id
        assert len(source["revision"]) == 40
        assert source["local_path"].startswith("artifacts/data/")
        assert source["official_split"]
        assert source["field_mapping"]


def test_external_math_and_code_mappings_use_native_fields() -> None:
    math_row = {"problem": "1+1?", "solution": "2", "level": "1", "type": "Algebra"}
    math_mapping = {"question": "problem", "target": "solution"}
    assert format_task_prompt("math", math_row, math_mapping) == "Question: 1+1?\nAnswer:"
    assert format_task_target("math", math_row, math_mapping) == "2"
    assert canonical_stable_id("math", "train", math_row, math_mapping)
    assert canonical_content_sha256("math", math_row, math_mapping)

    openmath_row = {
        "problem": "Solve x=1",
        "generated_solution": "x=1",
        "expected_answer": "1",
        "problem_source": "synthetic",
    }
    openmath_mapping = {
        "question": "problem",
        "target": "generated_solution",
        "answer": "expected_answer",
    }
    assert format_task_target("math", openmath_row, openmath_mapping) == "x=1"
    assert canonical_content_sha256(
        "openmathinstruct2", openmath_row, openmath_mapping
    )

    xcoder_row = {
        "id": 7,
        "messages": [
            {"role": "user", "content": "Write a function."},
            {"role": "assistant", "content": "def f(): pass"},
        ],
    }
    xcoder_mapping = {"id": "id", "messages": "messages"}
    assert format_task_prompt("code", xcoder_row, xcoder_mapping).endswith(
        "Assistant:"
    )
    assert format_task_target("code", xcoder_row, xcoder_mapping) == "def f(): pass"
    assert canonical_content_sha256("xcoder_80k", xcoder_row, xcoder_mapping)


def test_external_multihop_context_mappings_support_native_shapes() -> None:
    musique_row = {
        "id": "m1",
        "paragraphs": [
            {"idx": 0, "title": "A", "paragraph_text": "First paragraph"}
        ],
        "question": "What?",
        "answer": "Answer",
    }
    musique_mapping = {
        "id": "id",
        "question": "question",
        "context": "paragraphs",
        "target": "answer",
    }
    assert "First paragraph" in format_task_prompt(
        "multihop", musique_row, musique_mapping
    )
    assert format_task_target("multihop", musique_row, musique_mapping) == "Answer"
    assert canonical_content_sha256("musique", musique_row, musique_mapping)

    context = json.dumps([["A", ["First sentence"]]])
    wiki_row = {
        "_id": "w1",
        "question": "What?",
        "context": context,
        "answer": "Answer",
    }
    wiki_mapping = {
        "id": "_id",
        "question": "question",
        "context": "context",
        "target": "answer",
    }
    assert "First sentence" in format_task_prompt(
        "multihop", wiki_row, wiki_mapping
    )
    assert canonical_content_sha256("2wikimultihopqa", wiki_row, wiki_mapping)
