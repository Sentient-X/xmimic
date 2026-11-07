"""Runtime build custody checks; no Docker, registry, network, or paid execution."""

import importlib.util
import json
from pathlib import Path
import zipfile

import pytest

spec = importlib.util.spec_from_file_location(
    "runtime_wheels", Path(__file__).parents[1] / "tools/build_runtime_wheels.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def prepared(tmp_path, *, assets=False):
    source = tmp_path / "source"
    source.mkdir()
    (source / "program.py").write_text("value = 1\n")
    output = tmp_path / "output"
    (output / "wheels").mkdir(parents=True)
    wheel = output / "wheels" / "runtime.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for root in ("sx_training", "sx_geometry", "sx_embodiments"):
            archive.writestr(root + "/__init__.py", "")
        if assets:
            archive.writestr("sx_embodiments/_assets/private.stl", "private")
    (output / "cpu.lock").write_text("# exact dependencies\n")
    manifest = {
        "xmimic_files": {"program.py": module.sha(source / "program.py")},
        "wheels": {wheel.name: module.sha(wheel)},
        "dependency_locks": {"cpu.lock": module.sha(output / "cpu.lock")},
    }
    (output / "manifest.json").write_text(json.dumps(manifest))
    return source, output


def test_source_wheel_and_lock_are_all_checked(tmp_path):
    source, output = prepared(tmp_path)
    module.verify(output, source=source)
    (source / "program.py").write_text("value = 2\n")
    with pytest.raises(ValueError, match="source changed"):
        module.verify(output, source=source)


def test_wheel_corruption_and_asset_leaks_are_refused(tmp_path):
    source, output = prepared(tmp_path, assets=True)
    with pytest.raises(ValueError, match="governed inputs"):
        module.verify(output, source=source)
    (output / "wheels/runtime.whl").write_bytes(b"different")
    with pytest.raises(ValueError, match="wheel differs"):
        module.verify(output, source=source)


def test_dependency_lock_cannot_drift(tmp_path):
    source, output = prepared(tmp_path)
    (output / "cpu.lock").write_text("unreviewed==1\n")
    with pytest.raises(ValueError, match="dependency lock"):
        module.verify(output, source=source)


def test_explicit_build_inputs_and_untracked_python_have_source_identity(tmp_path):
    (tmp_path / "Dockerfile.cpu").write_text("FROM pinned\n")
    (tmp_path / "uv.lock").write_text("version = 1\n")
    code = tmp_path / "code"
    code.mkdir()
    (code / "new.py").write_text("pass\n")
    assets = code / "assets"
    assets.mkdir()
    (assets / "private.json").write_text("{}")
    assert set(
        module.source_files(
            tmp_path, [tmp_path / "Dockerfile.cpu", tmp_path / "uv.lock", code]
        )
    ) == {"Dockerfile.cpu", "uv.lock", "code/new.py"}
