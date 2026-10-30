import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import akiki
from akiki.__main__ import main


@pytest.mark.parametrize("engine", [None, *akiki.Engine])
def test_vad_cli_converts_engine_to_enum(engine, monkeypatch):
    import akiki.__main__ as cli

    selected = []
    monkeypatch.setattr(cli, "_run_vad", lambda args: selected.append(args.engine))
    arguments = ["akiki", "vad", "recording.wav"]
    if engine is not None:
        arguments += ["--engine", engine.value]
    monkeypatch.setattr(sys, "argv", arguments)
    main()
    assert selected == [engine if engine is not None else akiki.Engine.AUTO]


def test_vad_cli_rejects_unknown_engine(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["akiki", "vad", "recording.wav", "--engine", "unknown"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "--engine: invalid Engine value" in capsys.readouterr().err


@pytest.mark.parametrize("model_precision", [None, *akiki.ModelPrecision])
@pytest.mark.parametrize("fft_precision", [None, *akiki.FFTPrecision])
def test_vad_cli_converts_precisions_to_enums(model_precision, fft_precision, monkeypatch):
    import akiki.__main__ as cli

    selected = []
    monkeypatch.setattr(
        cli, "_run_vad", lambda args: selected.append((args.model_precision, args.fft_precision))
    )
    arguments = ["akiki", "vad", "recording.wav"]
    for option, precision in [
        ("--model-precision", model_precision),
        ("--fft-precision", fft_precision),
    ]:
        if precision is not None:
            arguments += [option, precision.value]
    monkeypatch.setattr(sys, "argv", arguments)
    main()
    assert selected == [(model_precision, fft_precision or akiki.FFTPrecision.FP32)]


@pytest.mark.parametrize(
    "option,value,enum_name",
    [
        ("--model-precision", "fp8", "ModelPrecision"),
        ("--model-precision", "int4-g32", "ModelPrecision"),
        ("--fft-precision", "fp16", "FFTPrecision"),
        ("--fft-precision", "int8", "FFTPrecision"),
    ],
)
def test_vad_cli_rejects_unsupported_precision(option, value, enum_name, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["akiki", "vad", "recording.wav", option, value])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert f"{option}: invalid {enum_name} value" in capsys.readouterr().err


@pytest.mark.parametrize(
    "arguments,expected",
    [
        (["--help"], "vad"),
        (["vad", "--help"], "usage: akiki vad"),
        (["--version"], f"akiki {akiki.__version__}"),
        (["vad", "--version"], f"akiki vad {akiki.__version__}"),
        (["minilm", "--help"], "usage: akiki minilm"),
        (["minilm", "--version"], f"akiki minilm {akiki.__version__}"),
    ],
)
def test_help_and_version_do_not_load_an_engine(arguments, expected, monkeypatch, capsys):
    def unexpected_engine_load(*args, **kwargs):
        pytest.fail("Help and version must work without an inference engine")

    monkeypatch.setattr(akiki, "VAD", unexpected_engine_load)
    monkeypatch.setattr(akiki, "MiniLM", unexpected_engine_load)
    monkeypatch.setattr(sys, "argv", ["akiki", *arguments])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 0
    output = capsys.readouterr()
    assert expected in output.out
    assert output.err == ""


@pytest.mark.parametrize(
    "arguments", [[], ["vad"], ["recording.wav"], ["minilm", "--model", "."], ["minilm", "hello"]]
)
def test_command_and_audio_are_required(arguments, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["akiki", *arguments])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert "usage: akiki" in output.err


def test_vad_engine_errors_include_the_command_prefix(monkeypatch, capsys):
    def unavailable_engine(*args, **kwargs):
        raise RuntimeError("No compatible engine")

    monkeypatch.setattr(akiki, "VAD", unavailable_engine)
    monkeypatch.setattr(sys, "argv", ["akiki", "vad", "recording.wav"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == "akiki vad: No compatible engine\n"


def test_python_module_runs_vad_and_leaves_incomplete_frame_unscored(tmp_path, require_rust):
    audio = tmp_path / "recording.wav"
    sf.write(audio, np.zeros(256 * 3 + 17, np.float32), 16000)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(akiki.__file__).resolve().parents[1])
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "akiki",
            "vad",
            str(audio),
            "--engine",
            "rust",
            "--model-precision",
            "fp32",
            "--threshold",
            "0",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    frames = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(frames) == 3
    assert [frame["start_seconds"] for frame in frames] == [0, 0.016, 0.032]
    assert [frame["end_seconds"] for frame in frames] == [0.016, 0.032, 0.048]
    assert all(0 <= frame["probability"] <= 1 and frame["speech"] for frame in frames)


def test_minilm_errors_include_the_command_prefix(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["akiki", "minilm", "--model", str(tmp_path), "hello"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert (
        output.err
        == f"akiki minilm: MiniLM checkpoint file is missing: {tmp_path / 'config.json'}\n"
    )


def run_minilm(tmp_path, *arguments):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(akiki.__file__).resolve().parents[1])
    result = subprocess.run(
        [sys.executable, "-m", "akiki", "minilm", *arguments],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    return json.loads(result.stdout)


def test_python_module_embeds_and_classifies(tmp_path, tiny_minilm, require_minilm_library):
    plain = run_minilm(tmp_path, "--model", str(tiny_minilm.plain), "w4 w5 w6")
    assert list(plain) == ["embedding"] and len(plain["embedding"]) == tiny_minilm.hidden
    result = run_minilm(
        tmp_path, "--model", str(tiny_minilm.variant), "--model-precision", "fp32", "w4 w5 w6"
    )
    assert list(result) == ["index", "label", "probabilities", "logits"]
    assert result["label"] == tiny_minilm.labels[result["index"]]
    assert len(result["probabilities"]) == len(tiny_minilm.labels)
    both = run_minilm(tmp_path, "--model", str(tiny_minilm.variant), "--embedding", "w4 w5 w6")
    assert "embedding" in both and "label" in both
