from __future__ import annotations

from finesub.llm.exchange_log import ExchangeLogger, render_message_text


def test_render_message_text_handles_text_and_file_parts() -> None:
    content = [
        {"type": "text", "text": "处理音频"},
        {
            "type": "file",
            "file": {"file_id": "files/yui", "filename": "yui.mp3", "format": "audio/mpeg"},
        },
    ]

    text = render_message_text(content)

    assert "处理音频" in text
    assert "[附件文件: yui.mp3]" in text
    assert render_message_text("纯文本") == "纯文本"


def test_exchange_logger_writes_readable_markdown_in_order(tmp_path) -> None:
    logger = ExchangeLogger(tmp_path / "exchanges")

    first = logger.log(
        "research-round1-attempt0",
        messages=[
            {"role": "system", "content": "系统指令"},
            {"role": "user", "content": "用户输入"},
        ],
        response_text="<search_queries>\n游戏B 剧情\n</search_queries>",
        metadata={
            "thinking_level": "medium",
            "input_tokens": "10 / 0 / 10 (uncached / cached / total)",
            "output_tokens_breakdown": "3 / 0 / 3 (visible / thinking / total)",
            "attempt": 0,
            "api_attempts": [
                {
                    "provider_tier": "GEMINI_FREE",
                    "model": "gemini/gemini-3.5-flash",
                    "api_key_name": "free-main",
                    "call_number_for_api_key_and_model": 1,
                    "return_code": "200",
                    "started_at": "2026-07-09T00:00:00.000+00:00",
                    "returned_at": "2026-07-09T00:00:01.000+00:00",
                    "elapsed_sec": 1.0,
                }
            ],
        },
    )
    second = logger.log(
        "correction-0001-attempt0",
        messages=None,
        response_text="<translated>\nsub|1|1.0|0.0|a|一|high|1|\n</translated>",
    )

    assert first.name == "001-research-round1-attempt0.md"
    assert second.name == "002-correction-0001-attempt0.md"
    text = first.read_text(encoding="utf-8")
    assert "# research-round1-attempt0" in text
    assert "## API Calls" in text
    assert "| GEMINI_FREE | gemini/gemini-3.5-flash | free-main | 1 | 200 |" in text
    assert "- input_tokens: 10 / 0 / 10 (uncached / cached / total)" in text
    assert "- output_tokens_breakdown: 3 / 0 / 3 (visible / thinking / total)" in text
    assert "- provider_tier:" not in text
    assert "- model:" not in text
    assert "- api_key:" not in text
    assert "- uncached_input_tokens:" not in text
    assert "- cached_input_tokens:" not in text
    assert "- total_input_tokens:" not in text
    assert "- thinking_tokens:" not in text
    assert "- output_tokens:" not in text
    assert "- total_output_tokens:" not in text
    assert "## 请求（system）" in text
    assert "系统指令" in text
    assert "## 请求（user）" in text
    assert "用户输入" in text
    assert "## 模型响应" in text
    assert "游戏B 剧情" in text
    assert "{" not in text.split("## 模型响应")[0]  # no JSON payload in header/request

    retro = second.read_text(encoding="utf-8")
    assert "（本次运行未留存请求文本）" in retro
    assert "<translated>" in retro


def test_exchange_logger_keeps_reasoning_only_in_model_response(tmp_path) -> None:
    logger = ExchangeLogger(tmp_path / "exchanges")

    path = logger.log(
        "correction-0001-attempt0",
        messages=[{"role": "user", "content": "hi"}],
        response_text="<reasoning>\n先检查术语。\n</reasoning>\n<translated></translated>",
    )

    text = path.read_text(encoding="utf-8")
    assert "## 显式推理（reasoning）" not in text
    assert text.count("先检查术语。") == 1
    assert "## 模型响应" in text
    assert "<reasoning>" in text.split("## 模型响应", 1)[1]


def test_for_task_artifact_dir_is_optional(tmp_path) -> None:
    assert ExchangeLogger.for_task_artifact_dir(None) is None
    logger = ExchangeLogger.for_task_artifact_dir(tmp_path)
    assert logger is not None
    assert logger.root == (tmp_path / "exchanges").resolve()


def test_validation_reasons_land_in_the_exchange(tmp_path) -> None:
    """`validation_ok: False` alone does not say why a window failed.

    The reasons existed all along, but only in `correction-windows.jsonl` --
    so the file you open to see the response that failed was the one file that
    would not tell you what was wrong with it.
    """

    logger = ExchangeLogger(tmp_path)
    path = logger.log(
        "correction-0001-attempt0",
        messages=[{"role": "user", "content": "x"}],
        response_text="y",
        metadata={
            "validation_ok": False,
            "validation_errors": [
                "Source id 120 appears in more than one output row.",
                "Source id 121 appears in more than one output row.",
            ],
            "validation_warnings": ["Row 79 char_count '18.5' ...; normalized."],
        },
    )
    body = path.read_text(encoding="utf-8")

    assert "## Validation" in body
    assert "**errors (2)**" in body
    assert "- Source id 120 appears in more than one output row." in body
    assert "**warnings (1)**" in body
    # The raw lists never render as `- key: [...]` noise.
    assert "- validation_errors:" not in body
    assert "- validation_warnings:" not in body


def test_a_clean_window_gets_no_validation_section(tmp_path) -> None:
    logger = ExchangeLogger(tmp_path)
    path = logger.log(
        "correction-0002-attempt0",
        messages=[{"role": "user", "content": "x"}],
        response_text="y",
        metadata={"validation_ok": True, "validation_errors": [], "validation_warnings": []},
    )
    assert "## Validation" not in path.read_text(encoding="utf-8")


def test_long_validation_lists_are_capped_with_a_pointer(tmp_path) -> None:
    logger = ExchangeLogger(tmp_path)
    path = logger.log(
        "correction-0003-attempt0",
        messages=[{"role": "user", "content": "x"}],
        response_text="y",
        metadata={
            "validation_ok": False,
            "validation_errors": [f"error {i}" for i in range(40)],
        },
    )
    body = path.read_text(encoding="utf-8")
    assert "**errors (40)**" in body
    assert "- error 24" in body
    assert "- error 25" not in body
    assert "另有 15 条" in body
