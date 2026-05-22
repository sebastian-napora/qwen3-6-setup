import qwen_benchmark


def test_health_url_uses_server_root():
    assert qwen_benchmark._health_url("http://localhost:11111/v1") == "http://localhost:11111/health"
    assert qwen_benchmark._health_url("http://localhost:11112") == "http://localhost:11112/health"


def test_discover_model_uses_first_model_id(monkeypatch):
    monkeypatch.setattr(
        qwen_benchmark,
        "_request_json",
        lambda method, url, **kwargs: {"data": [{"id": "qwen-test"}, {"id": "backup"}]},
    )

    assert qwen_benchmark.discover_model("http://localhost:11111/v1", timeout=30) == "qwen-test"


def test_extract_output_text_handles_text_parts():
    payload = {
        "choices": [
            {
                "message": {
                    "content": [
                        {"type": "text", "text": "hello"},
                        {"type": "image_url", "image_url": {"url": "ignored"}},
                        {"type": "text", "text": " world"},
                    ]
                }
            }
        ]
    }

    assert qwen_benchmark._extract_output_text(payload) == "hello world"


def test_summarize_runs_calculates_average_rates():
    runs = [
        qwen_benchmark.RunMetrics(
            latency_seconds=2.0,
            prompt_tokens=120,
            completion_tokens=60,
            total_tokens=180,
            output_chars=240,
        ),
        qwen_benchmark.RunMetrics(
            latency_seconds=3.0,
            prompt_tokens=150,
            completion_tokens=90,
            total_tokens=240,
            output_chars=360,
        ),
    ]

    summary = qwen_benchmark.summarize_runs(
        target="litellm",
        base_url="http://localhost:11111/v1",
        model="qwen-test",
        warmup_runs=1,
        measured_runs=runs,
    )

    assert summary.avg_latency_seconds == 2.5
    assert summary.min_latency_seconds == 2.0
    assert summary.max_latency_seconds == 3.0
    assert summary.avg_prompt_tokens == 135.0
    assert summary.avg_completion_tokens == 75.0
    assert summary.avg_total_tokens == 210.0
    assert summary.avg_completion_tokens_per_second == 30.0
    assert summary.avg_total_tokens_per_second == 85.0
