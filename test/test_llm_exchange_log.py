from __future__ import annotations

from llm.exchange_log import ExchangeLogger, render_message_text


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
        response_text="<translated>\n1|a|一\n</translated>",
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
