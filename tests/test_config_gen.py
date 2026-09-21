"""Baseline characterization tests for config_gen.py.

They pin the current ordinary-GGUF behavior of ``config_gen.generate_config``
so later changes are judged against a known starting point: both modes and
their default, preset section names, model paths and group parameters, the
``model_sizes.json`` side effect, empty-directory errors, and group-params
loading.

Split GGUF files (``-NNNNN-of-MMMMM``) are intentionally NOT covered here;
their behavior belongs to the follow-up split-awareness task.
"""

from __future__ import annotations

import configparser
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config_gen  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _write_model(models_dir: Path, name: str, size: int) -> Path:
    model = models_dir / name
    model.write_bytes(b"\0" * size)
    return model


def _write_params(tmp_path: Path, params: dict) -> Path:
    path = tmp_path / "group_params.yaml"
    path.write_text(config_gen.yaml.safe_dump(params))
    return path


def _generate(
    models_dir: Path,
    out: Path,
    params: dict | None = None,
    tmp_path: Path | None = None,
):
    """Call the public entry point with an optional explicit params file."""
    params_path = (
        _write_params(tmp_path, params) if params is not None else None
    )
    config_gen.generate_config(models_dir, out, params_path)
    return out


@pytest.fixture
def no_repo_group_params(monkeypatch):
    """Hide the repo's tracked group_params.yaml from load_group_params."""
    blocked = Path(config_gen.__file__).parent / "group_params.yaml"

    class _HiddenPath(Path):
        def exists(self, *args, **kwargs):
            if self == blocked:
                return False
            return super().exists(*args, **kwargs)

    monkeypatch.setattr(config_gen, "Path", _HiddenPath)


def _read_presets(out: Path) -> configparser.ConfigParser:
    parser = configparser.ConfigParser()
    lines = out.read_text().splitlines()
    # The bare "version = 1" header sits outside any section, so drop it
    # before feeding the body to configparser (asserted here).
    assert lines[0].strip() == "version = 1"
    parser.read_string("\n".join(lines[1:]))
    return parser


def _generate_cpu_only(models_dir: Path, out: Path, tmp_path: Path):
    _generate(
        models_dir,
        out,
        {"mode": "cpu_only", "cpu": {"n-gpu-layers": 0}},
        tmp_path,
    )


def _generate_split(models_dir: Path, out: Path, tmp_path: Path):
    _generate(
        models_dir,
        out,
        {
            "mode": "split",
            "gpu": {"n-gpu-layers": 999, "flash-attn": "on"},
            "cpu": {"n-gpu-layers": 0},
        },
        tmp_path,
    )


# ---------------------------------------------------------------------------
# mode selection
# ---------------------------------------------------------------------------

def test_default_mode_is_split_without_any_params_file(
    tmp_path: Path, capsys: pytest.CaptureFixture, no_repo_group_params
):
    models = tmp_path / "models"
    models.mkdir()
    _write_model(models, "tiny.gguf", 8)
    out = tmp_path / "out" / "presets.ini"
    _generate(models, out, None, tmp_path)

    presets = _read_presets(out)
    assert sorted(presets.sections()) == ["tiny-cpu", "tiny-gpu"]


def test_cpu_only_mode_writes_one_unsuffixed_preset(
    tmp_path: Path
):
    models = tmp_path / "models"
    models.mkdir()
    model = _write_model(models, "tiny.gguf", 4096)
    out = tmp_path / "out" / "presets.ini"
    _generate_cpu_only(models, out, tmp_path)

    presets = _read_presets(out)
    assert list(presets.sections()) == ["tiny"]
    assert presets["tiny"]["model"] == str(model.resolve())
    assert presets["tiny"]["n-gpu-layers"] == "0"


def test_split_mode_writes_gpu_and_cpu_presets(
    tmp_path: Path
):
    models = tmp_path / "models"
    models.mkdir()
    model = _write_model(models, "tiny.gguf", 4096)
    out = tmp_path / "out" / "presets.ini"
    _generate_split(models, out, tmp_path)

    presets = _read_presets(out)
    assert sorted(presets.sections()) == ["tiny-cpu", "tiny-gpu"]
    assert presets["tiny-gpu"]["model"] == str(model.resolve())
    assert presets["tiny-gpu"]["n-gpu-layers"] == "999"
    assert presets["tiny-gpu"]["flash-attn"] == "on"
    assert presets["tiny-cpu"]["model"] == str(model.resolve())
    assert presets["tiny-cpu"]["n-gpu-layers"] == "0"
    assert "flash-attn" not in presets["tiny-cpu"]


def test_presets_file_starts_with_version_header(
    tmp_path: Path
):
    models = tmp_path / "models"
    models.mkdir()
    _write_model(models, "tiny.gguf", 8)
    out = tmp_path / "out" / "presets.ini"
    _generate_cpu_only(models, out, tmp_path)

    first_line = out.read_text().splitlines()[0]
    assert first_line.strip() == "version = 1"


def test_presets_are_written_in_sorted_filename_order(
    tmp_path: Path
):
    models = tmp_path / "models"
    models.mkdir()
    _write_model(models, "zeta.gguf", 8)
    _write_model(models, "alpha.gguf", 8)
    out = tmp_path / "out" / "presets.ini"
    _generate_cpu_only(models, out, tmp_path)

    presets = _read_presets(out)
    assert list(presets.sections()) == ["alpha", "zeta"]


# ---------------------------------------------------------------------------
# model_sizes.json
# ---------------------------------------------------------------------------

def test_model_sizes_json_maps_stem_to_file_bytes(
    tmp_path: Path
):
    models = tmp_path / "models"
    models.mkdir()
    _write_model(models, "alpha.gguf", 1234)
    _write_model(models, "zeta.gguf", 5678)
    out = tmp_path / "out" / "presets.ini"
    _generate_split(models, out, tmp_path)

    sizes_path = out.parent / "model_sizes.json"
    assert sizes_path.is_file()
    sizes = json.loads(sizes_path.read_text())
    assert sizes == {"alpha": 1234, "zeta": 5678}


def test_split_mode_records_each_model_once(
    tmp_path: Path
):
    models = tmp_path / "models"
    models.mkdir()
    _write_model(models, "tiny.gguf", 2048)
    out = tmp_path / "out" / "presets.ini"
    _generate_split(models, out, tmp_path)

    sizes = json.loads((out.parent / "model_sizes.json").read_text())
    assert list(sizes) == ["tiny"]


# ---------------------------------------------------------------------------
# error handling
# ---------------------------------------------------------------------------

def test_empty_model_directory_exits_with_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
):
    models = tmp_path / "models"
    models.mkdir()
    out = tmp_path / "out" / "presets.ini"
    with pytest.raises(SystemExit) as excinfo:
        _generate_cpu_only(models, out, tmp_path)
    assert excinfo.value.code == 1
    assert "No .gguf files found" in capsys.readouterr().err
    assert not out.exists()


def test_non_gguf_files_are_ignored(
    tmp_path: Path
):
    models = tmp_path / "models"
    models.mkdir()
    (models / "readme.txt").write_text("not a model")
    out = tmp_path / "out" / "presets.ini"
    with pytest.raises(SystemExit) as excinfo:
        _generate_cpu_only(models, out, tmp_path)
    assert excinfo.value.code == 1


# ---------------------------------------------------------------------------
# group params loading
# ---------------------------------------------------------------------------

def test_explicit_params_file_overrides_defaults(tmp_path: Path):
    params_path = _write_params(
        tmp_path, {"mode": "cpu_only", "cpu": {"threads": 4}}
    )
    loaded = config_gen.load_group_params(params_path)
    assert loaded == {"mode": "cpu_only", "cpu": {"threads": 4}}


def test_missing_params_path_falls_back_to_defaults(
    no_repo_group_params,
):
    loaded = config_gen.load_group_params(None)
    assert loaded == config_gen.DEFAULT_GROUP_PARAMS
    assert loaded["gpu"] == {"n-gpu-layers": 999}
    assert loaded["cpu"] == {"n-gpu-layers": 0}


def test_missing_mode_key_defaults_to_split(
    tmp_path: Path
):
    models = tmp_path / "models"
    models.mkdir()
    _write_model(models, "tiny.gguf", 8)
    out = tmp_path / "out" / "presets.ini"
    _generate(models, out, {"cpu": {"n-gpu-layers": 0}}, tmp_path)

    presets = _read_presets(out)
    assert sorted(presets.sections()) == ["tiny-cpu", "tiny-gpu"]
