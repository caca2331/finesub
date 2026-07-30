from __future__ import annotations

import json
from pathlib import Path
import threading
import time
import types

import pytest

from asr_playground import batch
from asr_playground.run_metadata import update_run_metadata
from asr_playground.batch import (
    BatchItem,
    merge_item_options,
    profile_asr_workers,
    read_manifest,
    run_batch,
)


def _item(label: str, stages: dict) -> BatchItem:
    return BatchItem(label=label, stages=stages, payload=label)


def test_items_flow_through_all_bins_and_chain_payloads(tmp_path) -> None:
    calls: list[tuple[str, str]] = []
    lock = threading.Lock()

    def stage(name: str):
        def fn(payload):
            with lock:
                calls.append((name, payload))
            return payload + f"->{name}"

        return fn

    items = [
        _item(f"i{n}", {"download": stage("download"), "asr": stage("asr"), "llm": stage("llm")})
        for n in range(5)
    ]
    results = run_batch(items, status_path=tmp_path / "status.jsonl")

    assert [r.status for r in results] == ["done"] * 5
    assert results[0].payload == "i0->download->asr->llm"
    # every item hit every bin exactly once
    for n in range(5):
        assert sum(1 for name, p in calls if p.startswith(f"i{n}")) == 3


def test_missing_stages_pass_through(tmp_path) -> None:
    seen: list[str] = []
    items = [
        _item("local", {"asr": lambda p: (seen.append(p), p)[1]}),  # no download/llm
        _item("raw-only", {"download": lambda p: p, "asr": lambda p: p}),
    ]
    results = run_batch(items)
    assert [r.status for r in results] == ["done", "done"]
    assert seen == ["local"]


def test_llm_consumes_in_item_order_despite_out_of_order_upstream() -> None:
    # Item 0's download blocks until item 2's download completes, so upstream
    # completion order is 1, 2, 0 — the llm bin must still see 0, 1, 2.
    item2_done = threading.Event()
    llm_order: list[str] = []

    def slow_download(payload):
        item2_done.wait(timeout=10)
        return payload

    def download2(payload):
        item2_done.set()
        return payload

    items = [
        _item("i0", {"download": slow_download, "llm": lambda p: (llm_order.append(p), p)[1]}),
        _item("i1", {"download": lambda p: p, "llm": lambda p: (llm_order.append(p), p)[1]}),
        _item("i2", {"download": download2, "llm": lambda p: (llm_order.append(p), p)[1]}),
    ]
    results = run_batch(items, workers={"download": 2})
    assert [r.status for r in results] == ["done"] * 3
    assert llm_order == ["i0", "i1", "i2"]


def test_failed_item_skips_downstream_and_isolates(tmp_path) -> None:
    llm_seen: list[str] = []

    def boom(payload):
        raise RuntimeError("asr exploded")

    def llm(payload):
        llm_seen.append(payload)
        return payload

    items = [
        _item("ok0", {"asr": lambda p: p, "llm": llm}),
        _item("bad", {"asr": boom, "llm": llm}),
        _item("ok2", {"asr": lambda p: p, "llm": llm}),
    ]
    status_path = tmp_path / "status.jsonl"
    results = run_batch(items, status_path=status_path)

    assert results[0].status == "done"
    assert results[1].status == "failed"
    assert results[1].failed_stage == "asr"
    assert "asr exploded" in results[1].error
    assert results[2].status == "done"
    assert sorted(llm_seen) == ["ok0", "ok2"]  # ordered gate advanced past the failure

    events = [json.loads(line) for line in status_path.read_text(encoding="utf-8").splitlines()]
    failed = [e for e in events if e["status"] == "failed"]
    assert failed and failed[0]["label"] == "bad" and failed[0]["stage"] == "asr"
    item_events = [e for e in events if e["stage"] == "item"]
    assert {e["label"]: e["status"] for e in item_events} == {
        "ok0": "done",
        "bad": "failed",
        "ok2": "done",
    }


def test_asr_queue_backpressure_bounds_download_lead() -> None:
    release_asr = threading.Event()
    downloads_done = 0
    lock = threading.Lock()

    def download(payload):
        nonlocal downloads_done
        with lock:
            downloads_done += 1
        return payload

    def blocked_asr(payload):
        release_asr.wait(timeout=10)
        return payload

    items = [_item(f"i{n}", {"download": download, "asr": blocked_asr}) for n in range(12)]
    thread_result: list = []
    runner = threading.Thread(
        target=lambda: thread_result.extend(
            run_batch(items, workers={"download": 2}, asr_queue_size=2)
        ),
        daemon=True,
    )
    runner.start()

    # Wait for the download lead to stabilise, then check it stayed bounded:
    # queue capacity (2) + 1 in the asr worker + up to 2 blocked in put().
    cap = 2 + 1 + 2
    last = -1
    for _ in range(100):
        time.sleep(0.02)
        with lock:
            current = downloads_done
        if current == last:
            break
        last = current
    assert last <= cap, f"downloads ran {last} ahead despite backpressure"

    release_asr.set()
    runner.join(timeout=10)
    assert not runner.is_alive()
    assert [r.status for r in thread_result] == ["done"] * 12


def test_stop_event_skips_not_started_items() -> None:
    stop = threading.Event()
    started = threading.Event()

    def first(payload):
        started.set()
        time.sleep(0.05)
        return payload

    def others(payload):
        return payload

    items = [_item("i0", {"asr": first})] + [
        _item(f"i{n}", {"asr": others}) for n in range(1, 6)
    ]

    def trigger():
        started.wait(timeout=10)
        stop.set()

    threading.Thread(target=trigger, daemon=True).start()
    results = run_batch(items, stop_event=stop)
    assert results[0].status == "done"  # in-flight item finishes
    assert any(r.status == "skipped" for r in results[1:])


def test_run_batch_rejects_zero_workers() -> None:
    with pytest.raises(ValueError, match="workers"):
        run_batch([], workers={"asr": 0})


# --- manifest / option merging ---------------------------------------------------
def test_read_manifest_and_merge(tmp_path) -> None:
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(
        '{"source": "https://example.com/a", "language": "ja"}\n'
        "\n"
        '{"source": "data/b.wav", "stage": "raw-srt"}\n',
        encoding="utf-8",
    )
    rows = read_manifest(manifest)
    assert len(rows) == 2

    defaults = {"stage": "final-srt", "language": None, "model": "large-v3-turbo"}
    merged = [merge_item_options(row, defaults) for row in rows]
    assert merged[0]["language"] == "ja"
    assert merged[0]["stage"] == "final-srt"
    assert merged[1]["stage"] == "raw-srt"
    assert merged[1]["model"] == "large-v3-turbo"


def test_merge_rejects_unknown_keys_and_missing_source() -> None:
    with pytest.raises(ValueError, match="unknown manifest keys"):
        merge_item_options({"source": "x", "banana": 1}, {})
    with pytest.raises(ValueError, match="missing 'source'"):
        merge_item_options({"language": "ja"}, {})


def test_profile_asr_workers_runs_one_file_at_a_time() -> None:
    # Parallelism moved inside the file, so the asr bin is 1 regardless of the
    # profile mix -- that is what bounds live per-file state in one process.
    assert profile_asr_workers([{"gpu_budget_gb": 16}, {"gpu_budget_gb": 8}]) == 1
    assert profile_asr_workers([{"gpu_budget_gb": 16}, {"gpu_budget_gb": 4}]) == 1
    assert profile_asr_workers([{"gpu_budget_gb": 16, "device": "cpu"}]) == 1
    assert profile_asr_workers([]) == 1


def test_read_manifest_rejects_bad_json(tmp_path) -> None:
    manifest = tmp_path / "m.jsonl"
    manifest.write_text('{"source": "a"}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        read_manifest(manifest)


def test_batch_passes_asr_stabilize_profile_to_pipeline(tmp_path) -> None:
    source = tmp_path / "input.wav"
    source.write_bytes(b"fake")
    calls: list[dict[str, object]] = []
    pipeline_mod = types.SimpleNamespace(
        PIPELINE_STAGE_ORDER={
            "vocal": 1,
            "aligned": 2,
            "stable": 3,
            "raw-srt": 4,
            "translated-srt": 5,
            "final-srt": 6,
        },
        asr_align=types.SimpleNamespace(DEFAULT_MODEL="model"),
        asr_stabilize=types.SimpleNamespace(DEFAULT_ASR_STABILIZE_PROFILE=0),
        run_pipeline=lambda *args, **kwargs: calls.append({"args": args, **kwargs}),
    )
    opts = {
        "source": str(source),
        "stage": "raw-srt",
        "asr_stabilize_profile": -1,
    }

    item = batch._build_item(pipeline_mod, opts)
    item.stages["asr"](item.payload)

    assert calls[0]["asr_stabilize_profile"] == -1
    assert calls[0]["stage"] == "raw-srt"
    assert Path(calls[0]["args"][0]) == source


def test_batch_forwards_first_pass_stage_timing_to_final_pass(tmp_path) -> None:
    source = tmp_path / "input.wav"
    source.write_bytes(b"fake")
    metadata_path = tmp_path / "input-metadata.json"
    calls: list[dict[str, object]] = []

    def fake_run_pipeline(*args, **kwargs):
        calls.append({"args": args, **kwargs})
        if kwargs["stage"] == "raw-srt":
            update_run_metadata(
                metadata_path,
                {
                    "timing": {
                        "stages": {
                            "asr": {"status": "executed", "elapsed_sec": 4.0}
                        }
                    }
                },
            )
        return types.SimpleNamespace(metadata_json=metadata_path)

    pipeline_mod = types.SimpleNamespace(
        PIPELINE_STAGE_ORDER={
            "vocal": 1,
            "aligned": 2,
            "stable": 3,
            "raw-srt": 4,
            "translated-srt": 5,
            "final-srt": 6,
        },
        asr_align=types.SimpleNamespace(DEFAULT_MODEL="model"),
        asr_stabilize=types.SimpleNamespace(DEFAULT_ASR_STABILIZE_PROFILE=0),
        run_pipeline=fake_run_pipeline,
    )
    item = batch._build_item(
        pipeline_mod,
        {"source": str(source), "stage": "final-srt"},
    )

    payload = item.stages["asr"](item.payload)
    item.stages["llm"](payload)

    assert calls[1]["_prior_timing"]["asr"] == {
        "status": "executed",
        "elapsed_sec": 4.0,
    }
