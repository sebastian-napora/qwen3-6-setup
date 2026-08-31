import os
import stat
import subprocess
from pathlib import Path


def _install_offline_launcher(tmp_path: Path) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    launcher_path = repo_root / "start_offline.sh"
    temp_launcher = tmp_path / "start_offline.sh"
    temp_launcher.write_text(launcher_path.read_text())
    temp_launcher.chmod(temp_launcher.stat().st_mode | stat.S_IXUSR)
    return temp_launcher


def test_start_offline_uses_configured_models_root(tmp_path):
    launcher_path = _install_offline_launcher(tmp_path)
    models_root = tmp_path / "offline-models"

    for name in ("backend", "embedding", "asr"):
        (models_root / name).mkdir(parents=True)

    stub_start = tmp_path / "start.sh"
    stub_start.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'backend=%s\\n' \"$QWEN_BACKEND_MODEL\"\n"
        "printf 'embedding=%s\\n' \"$QWEN_RAG_EMBED_MODEL\"\n"
        "printf 'asr=%s\\n' \"$QWEN_ASR_MODEL\"\n"
        "printf 'args=%s\\n' \"$*\"\n"
    )
    stub_start.chmod(stub_start.stat().st_mode | stat.S_IXUSR)

    env = os.environ.copy()
    env["QWEN_OFFLINE_MODELS_DIR"] = str(models_root)

    result = subprocess.run(
        ["bash", str(launcher_path), "both"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert f"backend={models_root / 'backend'}" in result.stdout
    assert f"embedding={models_root / 'embedding'}" in result.stdout
    assert f"asr={models_root / 'asr'}" in result.stdout
    assert "args=both" in result.stdout
