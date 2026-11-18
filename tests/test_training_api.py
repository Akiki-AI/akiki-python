"""Exercise the optional training facade without duplicating the trainer."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

import akiki


def run_python(code, directory, *paths):
    environment = os.environ.copy()
    sources = [str(Path(akiki.__file__).resolve().parents[1]), *map(str, paths)]
    if environment.get("PYTHONPATH"):
        sources.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(sources)
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=directory,
        env=environment,
        capture_output=True,
        text=True,
    )


def test_missing_training_package_leaves_inference_usable_and_explains_installation(tmp_path):
    result = run_python(
        """
import importlib.abc
import sys

class BlockTraining(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'akiki_training':
            raise ModuleNotFoundError(fullname, name=fullname)

sys.meta_path.insert(0, BlockTraining())
import akiki
assert akiki.audio.vad.VAD is not None
assert dir(akiki) == ['MiniLM', 'VAD', '__version__', 'audio', 'decision', 'vision']
namespace = {}
exec('from akiki import *', namespace)
assert namespace['audio'].vad.VAD is akiki.audio.vad.VAD
assert namespace['decision'].MiniLM is akiki.MiniLM
assert 'akiki.training' not in sys.modules
import importlib
for operation in (
    lambda: importlib.import_module('akiki.training'),
    lambda: akiki.decision.Specialist().train('missing-local-model'),
    lambda: akiki.decision.Specialist().suggest_examples(object()),
):
    try:
        operation()
    except ImportError as error:
        assert 'private akiki-training companion' in str(error), error
        assert 'authorized local checkout or wheel' in str(error), error
        assert 'runtime dependencies only' in str(error), error
    else:
        raise AssertionError('The missing private companion was ignored')
""",
        tmp_path,
    )
    assert result.returncode == 0, result.stderr


def test_training_facade_preserves_types_and_loads_no_heavy_runtime(tmp_path):
    pytest.importorskip("akiki_training.training")
    result = run_python(
        """
import importlib.abc
import sys

class BlockRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'safetensors', 'datasets'}:
            raise AssertionError(f'Unexpected runtime import: {fullname}')

sys.meta_path.insert(0, BlockRuntime())
from akiki import training
import akiki.training
from akiki.training import ClassWeighting, TrainingConfig, TrainingExample, TrainingMode
import akiki_training as backend
from akiki_training import training as legacy

assert training is akiki.training
assert dir(akiki) == ['MiniLM', 'VAD', '__version__', 'audio', 'decision', 'training', 'vision']
assert dir(training) == sorted(backend.__all__)
assert training.__all__ is backend.__all__
for name in backend.__all__:
    assert getattr(training, name) is getattr(backend, name), name
assert training.train_classifier is legacy.train_classifier
assert TrainingExample is legacy.TrainingExample
assert TrainingConfig is legacy.TrainingConfig
config = TrainingConfig(mode=legacy.TrainingMode.HEAD_ONLY, class_weighting=ClassWeighting.UNIFORM)
assert config.mode is TrainingMode.HEAD_ONLY
assert akiki.training.load_examples([{'text': 'hello', 'label': 'greeting'}]) == (
    TrainingExample('hello', 'greeting'),
)
""",
        tmp_path,
    )
    assert result.returncode == 0, result.stderr


def test_companion_public_api_changes_are_forwarded_without_copying_exports(tmp_path):
    package = tmp_path / "akiki_training"
    package.mkdir()
    (package / "__init__.py").write_text('__all__ = ["initial"]\ninitial = object()\n')
    (package / "training.py").write_text('__all__ = ["legacy"]\nlegacy = object()\n')
    result = run_python(
        """
import akiki_training as backend
from akiki import training
from akiki.training import initial, legacy

assert initial is backend.initial
assert dir(training) == ['initial']
assert legacy is backend.training.legacy

# The companion replaces its public API while the facade is already imported.
backend.added = object()
backend.__all__ = ['added']
assert dir(training) == ['added']
assert training.__all__ is backend.__all__
from akiki.training import added
assert added is backend.added
namespace = {}
exec('from akiki.training import *', namespace)
assert namespace['added'] is backend.added
assert 'initial' not in namespace
assert 'legacy' not in namespace
assert training.legacy is legacy
try:
    training.initial
except AttributeError:
    pass
else:
    raise AssertionError('A removed export remained cached in the facade')
""",
        tmp_path,
        tmp_path,
    )
    assert result.returncode == 0, result.stderr


def test_internal_import_errors_keep_their_original_cause(tmp_path):
    package = tmp_path / "akiki_training"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "training.py").write_text(
        'raise ModuleNotFoundError("Missing internal dependency", name="internal_dependency")\n'
    )
    result = run_python(
        """
try:
    import akiki.training
except ModuleNotFoundError as error:
    assert error.name == 'internal_dependency', error
else:
    raise AssertionError('The internal error was hidden')
""",
        tmp_path,
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
