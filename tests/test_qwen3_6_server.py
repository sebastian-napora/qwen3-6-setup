import qwen3_6_server


def test_build_serve_argv_uses_default_backend_model(monkeypatch):
    monkeypatch.delenv("QWEN_BACKEND_MODEL", raising=False)

    argv = qwen3_6_server.build_serve_argv()

    assert argv[0] == qwen3_6_server.DEFAULT_BACKEND_MODEL


def test_build_serve_argv_uses_env_backend_model(monkeypatch, tmp_path):
    model_dir = tmp_path / "backend-model"
    monkeypatch.setenv("QWEN_BACKEND_MODEL", str(model_dir))

    argv = qwen3_6_server.build_serve_argv()

    assert argv[0] == str(model_dir)
