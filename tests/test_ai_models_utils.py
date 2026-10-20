"""Tests for cerr.ai_models helpers that do not need an installed model."""
import pytest
import yaml

from cerr.ai_models import install_utils, run_utils


def test_list_inputs_reads_execution_arguments(tmp_path, monkeypatch, capsys):
    modelName = 'Fake_Model'
    modelDir = tmp_path / modelName
    modelDir.mkdir()
    spec = {'preprocessing': {'script': 'prep.py', 'mode': 'subprocess'},
            'execution': {'entrypoint': 'run.py',
                          'arguments': [
                              {'name': 'input_path', 'type': 'positional', 'required': True},
                              {'name': 'gpu', 'type': 'flag', 'flag_string': '--gpu',
                               'required': False}]}}
    (modelDir / 'run_spec.yaml').write_text(yaml.safe_dump(spec))
    monkeypatch.setattr(run_utils, 'validateModelNum', lambda n: modelName)

    run_utils.listInputs(1, str(tmp_path))
    out = capsys.readouterr().out
    assert 'input_path' in out and 'gpu' in out


def test_missing_installer_message_gives_git_install(monkeypatch):
    def _missing(name):
        raise ModuleNotFoundError(name)
    monkeypatch.setattr(install_utils.importlib.resources, 'files', _missing)
    with pytest.raises(RuntimeError, match='git\\+https://github.com/cerr/model_installer'):
        install_utils.getScriptPath()
