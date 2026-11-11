"""Tests for cerr.ai_models helpers that do not need an installed model."""
import json
import os
import shutil
import sys
from types import SimpleNamespace

import pytest
import yaml

from cerr.ai_models import install_utils, run_utils, model_list


@pytest.fixture
def fakeModel(monkeypatch):
    """Log Fake_Model as model 99 of the smit repository."""
    monkeypatch.setitem(model_list.MODELS, 99, {'name': 'Fake_Model', 'repo': 'smit',
                                                'entrypoint': 'run_segmentation.py'})


# Mimics SMIT's get_weights.py: --list prints "<MODEL> installed: <path>" or "<MODEL> not installed"
FAKE_GET_WEIGHTS = """import os, sys
if '--list' not in sys.argv:
    sys.exit('get_weights.py should not be called to install')
for name in ('Fake_Model', 'Other_Model'):
    p = os.path.join('weights', name + '.safetensors')
    print(f"{name:28s} {'installed: ' + p if os.path.isfile(p) else 'not installed'}")
"""


def _fakeRepo(tmp_path, models=None):
    """A minimal model repository clone at <tmp_path>/smit; its uv env python is this Python."""
    repoPath = tmp_path / 'smit'
    (repoPath / 'requirements').mkdir(parents=True)
    models = models or {'Fake_Model': {'status': 'validated', 'description': 'A fake model',
                                       'labels': {1: 'Lung_L', 2: 'Lung_R'}}}
    (repoPath / 'models.yaml').write_text(yaml.safe_dump({'schema_version': 1, 'models': models}))
    (repoPath / 'run_segmentation.py').write_text('')
    (repoPath / 'get_weights.py').write_text(FAKE_GET_WEIGHTS)
    pythonExe = install_utils.getEnvPython(repoPath)
    pythonExe.parent.mkdir(parents=True)
    os.symlink(sys.executable, pythonExe)
    return repoPath


def test_read_uv_config(tmp_path):
    """Python version and uv flags are read from uv_config.txt; a missing file gives defaults."""
    (tmp_path / 'uv_config.txt').write_text(
        'python_version=3.12\nuv_flags=--find-links https://example.org/whl --no-deps\n')
    assert install_utils.readUvConfig(tmp_path) == (
        '3.12', ['--find-links', 'https://example.org/whl', '--no-deps'])
    assert install_utils.readUvConfig(tmp_path / 'missing') == (None, [])


def test_strip_pytorch_index(tmp_path):
    """The PyTorch extra index is dropped from a requirements file; other files are left alone."""
    req = tmp_path / 'r1.txt'
    req.write_text('# torch\n--extra-index-url https://download.pytorch.org/whl/cu128\ntorch==2.8.0\n')
    (tmp_path / 'out').mkdir()
    out = install_utils._stripPytorchIndex(req, tmp_path / 'out')
    assert 'download.pytorch.org' not in out.read_text() and 'torch==2.8.0' in out.read_text()
    plain = tmp_path / 'r2.txt'
    plain.write_text('numpy\n')
    assert install_utils._stripPytorchIndex(plain, tmp_path) == plain


def test_validate_model_name(tmp_path):
    """A validated model passes; one missing from models.yaml raises."""
    repoPath = _fakeRepo(tmp_path)
    assert install_utils.validateModelName('Fake_Model', repoPath)['status'] == 'validated'
    with pytest.raises(ValueError, match='Unknown model'):
        install_utils.validateModelName('Other_Model', repoPath)


def test_model_log_keeps_installer_numbers():
    """Model numbers stay contiguous and names unique, matching the installer's numbering."""
    nums = sorted(model_list.MODELS)
    assert nums == list(range(1, len(nums) + 1))
    names = [entry['name'] for entry in model_list.MODELS.values()]
    assert len(set(names)) == len(names)
    assert model_list.MODELS[10]['name'] == 'MR_Rectum_GTV_SMIT'
    assert model_list.MODELS[13]['name'] == 'MR_BrainMets_SMITplus'


def test_resolve_model_by_number_or_name():
    """A model resolves by number, numeric string or name; unknown and unsupported ones raise."""
    rectum = (10, model_list.MODELS[10])
    assert model_list.resolveModel(10) == rectum
    assert model_list.resolveModel('10') == rectum
    assert model_list.resolveModel('MR_Rectum_GTV_SMIT') == rectum
    assert install_utils.getModel(14) == ('CT_ThoraxOARs_SMITplus', 'smit', 'run_segmentation.py')
    with pytest.raises(ValueError, match='Invalid model number'):
        model_list.resolveModel(0)
    with pytest.raises(ValueError, match='Unknown model'):
        model_list.resolveModel('No_Such_Model')
    with pytest.raises(ValueError, match='not yet supported'):
        install_utils.getModel(2)
    with pytest.raises(ValueError, match='not yet supported'):
        install_utils.getModel('MR_BrainMets_SMITplus')  # in smit, but no entrypoint


def test_list_models_shows_log_and_install_status(tmp_path, capsys, fakeModel):
    """listModels lists every logged model and reports weight and support status."""
    _fakeRepo(tmp_path)
    models = install_utils.listModels(str(tmp_path))
    assert models[1] == 'CT_cardiac_structures_deeplab' and models[99] == 'Fake_Model'
    out = capsys.readouterr().out
    assert 'weights not installed' in out and 'not supported via pyCERR' in out


def test_list_installed_weights_from_weights_folder(tmp_path):
    """Installed weights are read from weights/, with or without the model's environment."""
    repoPath = _fakeRepo(tmp_path, {'Fake_Model': {}, 'Other_Model': {}, 'Multi_Model': {}})
    shutil.rmtree(repoPath / '.venv')  # works without the model's environment
    (repoPath / 'get_weights.py').unlink()  # and without get_weights.py
    assert install_utils.listInstalledWeights(repoPath) == {'Fake_Model': None, 'Other_Model': None,
                                                            'Multi_Model': None}
    weightsDir = repoPath / 'weights'
    weightsDir.mkdir()
    (weightsDir / 'Fake_Model.safetensors').write_text('')
    (weightsDir / 'Other_Model.pt').write_text('')
    (weightsDir / 'Multi_Model').mkdir()  # empty folder: not installed
    installed = install_utils.listInstalledWeights(repoPath)
    assert installed['Fake_Model'] == weightsDir / 'Fake_Model.safetensors'
    assert installed['Other_Model'] == weightsDir / 'Other_Model.pt'
    assert installed['Multi_Model'] is None
    (weightsDir / 'Multi_Model' / 'part1.pth.gz').write_text('')  # weights made of several files
    assert install_utils.getWeightsPath(repoPath, 'Multi_Model') == weightsDir / 'Multi_Model'


def test_get_weights_skips_installed(tmp_path):
    """getWeights returns weights that are already installed instead of fetching them again."""
    repoPath = _fakeRepo(tmp_path)
    (repoPath / 'weights').mkdir()
    (repoPath / 'weights' / 'Fake_Model.safetensors').write_text('')
    # the fake get_weights.py exits with an error if called to install
    assert install_utils.getWeights(repoPath, 'Fake_Model').name == 'Fake_Model.safetensors'


def test_build_command_maps_user_inputs(tmp_path):
    """User inputs become model arguments, a session input folder and GPU environment settings."""
    repoPath = _fakeRepo(tmp_path)
    nii = tmp_path / 'pt1.nii.gz'
    nii.write_text('')
    userInputs = {'input_path': str(nii), 'output_path': str(tmp_path / 'out'),
                  'session_path': str(tmp_path / 'session'), 'gpu': 1,
                  'use_tta': True, 'no_rtstruct': False, 'sw_batch_size': 2}
    cmd, env = run_utils.buildCommand(repoPath, 'Fake_Model', userInputs, 'run_segmentation.py')
    assert cmd[2:8] == ['--model', 'Fake_Model', '--data_dir', str(tmp_path / 'session' / 'input'),
                        '--results_dir', str(tmp_path / 'out')]
    assert cmd[8:] == ['--use_tta', '--sw_batch_size', '2']
    assert (tmp_path / 'session' / 'input' / 'pt1.nii.gz').is_file()
    assert env['CUDA_VISIBLE_DEVICES'] == '1' and 'PYTHONPATH' not in env
    if sys.platform == 'darwin':
        assert env['PYTORCH_ENABLE_MPS_FALLBACK'] == '1'


def test_build_command_session_dir_only_if_accepted(tmp_path):
    """--session_dir is passed only when the entrypoint accepts it and a session path is given."""
    repoPath = _fakeRepo(tmp_path)
    userInputs = {'input_path': str(tmp_path), 'output_path': str(tmp_path / 'out'),
                  'session_path': str(tmp_path / 'session')}
    cmd, _ = run_utils.buildCommand(repoPath, 'Fake_Model', userInputs, 'run_segmentation.py')
    assert '--session_dir' not in cmd and '--session_path' not in cmd
    (repoPath / 'run_hn.py').write_text("parser.add_argument('--session_dir', default=None)\n")
    cmd, _ = run_utils.buildCommand(repoPath, 'Fake_Model', userInputs, 'run_hn.py')
    assert cmd[-2:] == ['--session_dir', str(tmp_path / 'session')]
    # without a session_path nothing is added
    del userInputs['session_path']
    cmd, _ = run_utils.buildCommand(repoPath, 'Fake_Model', userInputs, 'run_hn.py')
    assert '--session_dir' not in cmd


def test_build_command_requires_output_path(tmp_path):
    """buildCommand raises when output_path is missing."""
    repoPath = _fakeRepo(tmp_path)
    with pytest.raises(ValueError, match='output_path'):
        run_utils.buildCommand(repoPath, 'Fake_Model', {'input_path': str(tmp_path)}, 'run_segmentation.py')


def test_read_labels(tmp_path, fakeModel):
    """Labels come from the model's labels.json, falling back to models.yaml."""
    _fakeRepo(tmp_path)
    out = tmp_path / 'out'
    out.mkdir()
    # From models.yaml when the model wrote no labels.json
    assert run_utils.readLabels(str(out), 99, str(tmp_path)) == {'*_seg.*': {1: 'Lung_L', 2: 'Lung_R'}}
    (out / 'labels.json').write_text(json.dumps({'model': 'Fake_Model', 'labels': {'1': 'GTV'}}))
    assert run_utils.readLabels(str(out)) == {'*_seg.*': {1: 'GTV'}}


def test_normalise_labels_several_files():
    """Labels are keyed by output file pattern; mixed or pattern-less keys raise."""
    labels = yaml.safe_load('labels:\n'
                            '  "*_chew_AI_seg.*": {1: Left masseter, 2: Right masseter}\n'
                            '  "*_larynx_AI_seg.*": {1: Larynx}\n')['labels']
    assert run_utils.normalizeLabels(labels) == {'*_chew_AI_seg.*': {1: 'Left masseter', 2: 'Right masseter'},
                                                 '*_larynx_AI_seg.*': {1: 'Larynx'}}
    with pytest.raises(ValueError, match='not both'):
        run_utils.normalizeLabels({1: 'Larynx', '*_cm_AI_seg.*': {1: 'Constrictor'}})
    with pytest.raises(ValueError, match="one '\\*'"):
        run_utils.normalizeLabels({'pt1_cm_AI_seg.nii.gz': {1: 'Constrictor'}})
    with pytest.raises(ValueError, match='No labels'):
        run_utils.normalizeLabels({})


def test_import_seg_several_files(tmp_path, monkeypatch, fakeModel):
    """Each output pattern is imported per case, for both NIfTI and DICOM output."""
    _fakeRepo(tmp_path, {'Fake_Model': {'labels': {'*_chew_AI_seg.*': {1: 'Left masseter'},
                                                   '*_larynx_AI_seg.*': {1: 'Larynx'}}}})
    out = tmp_path / 'out'
    out.mkdir()
    for name in ('pt1_chew_AI_seg.nii.gz', 'pt1_larynx_AI_seg.nii.gz', 'pt2_chew_AI_seg.nii.gz'):
        (out / name).write_text('')
    loaded = []
    monkeypatch.setattr(run_utils.pc, 'loadNiiStructure',
                        lambda path, scanNum, planC, labels_dict: loaded.append((os.path.basename(path), labels_dict)))
    with pytest.raises(RuntimeError, match='pt1, pt2'):
        run_utils.importSeg(99, str(tmp_path), str(out), 0, None)
    run_utils.importSeg(99, str(tmp_path), str(out), 0, None, caseName='pt1')
    assert loaded == [('pt1_chew_AI_seg.nii.gz', {'Left masseter': 1}),
                      ('pt1_larynx_AI_seg.nii.gz', {'Larynx': 1})]
    with pytest.raises(FileNotFoundError, match='missing for case pt2: \\*_larynx_AI_seg'):
        run_utils.importSeg(99, str(tmp_path), str(out), 0, None, caseName='pt2')

    # DICOM input: one RTSTRUCT per pattern, named like it with .dcm
    dcmOut = tmp_path / 'dcm_out'
    dcmOut.mkdir()
    for name in ('pt1_chew_AI_seg.dcm', 'pt1_larynx_AI_seg.dcm'):
        (dcmOut / name).write_text('')
    monkeypatch.setattr(run_utils.pc, 'loadDcmDir',
                        lambda path, initplanC: (initplanC or []) + [os.path.basename(path)])
    imported = run_utils.importSeg(99, str(tmp_path), str(dcmOut), 0, None)
    assert imported == ['pt1_chew_AI_seg.dcm', 'pt1_larynx_AI_seg.dcm']
    (dcmOut / 'pt1_larynx_AI_seg.dcm').unlink()
    with pytest.raises(FileNotFoundError, match='missing for case pt1: \\*_larynx_AI_seg'):
        run_utils.importSeg(99, str(tmp_path), str(dcmOut), 0, None)


def test_find_outputs_extensions(tmp_path):
    """Outputs are matched per case across the supported file extensions."""
    for name in ('pt1_cm_AI_seg.nii.gz', 'pt2_cm_AI_seg.dcm', 'pt3_cm_AI_seg.json', 'pt4_cm_AI_seg.nii',
                 'pt1_CT_HN_SMIT_RTSTRUCT.dcm'):
        (tmp_path / name).write_text('')
    assert run_utils.findOutputs(str(tmp_path), ['*_cm_AI_seg.*']) == {
        'pt1': {'*_cm_AI_seg.*': str(tmp_path / 'pt1_cm_AI_seg.nii.gz')},
        'pt2': {'*_cm_AI_seg.*': str(tmp_path / 'pt2_cm_AI_seg.dcm')}}
    assert list(run_utils.findOutputs(str(tmp_path), ['*_cm_AI_seg.*'], caseName='pt2')) == ['pt2']
    assert list(run_utils.findOutputs(str(tmp_path), ['*_CT_HN_SMIT_RTSTRUCT.dcm'])) == ['pt1']


def test_relative_install_dir(tmp_path, monkeypatch):
    """A relative install directory is made absolute, since commands run from the repository."""
    _fakeRepo(tmp_path)
    monkeypatch.chdir(tmp_path)  # commands run with cwd=repoPath, so their paths must be absolute
    assert install_utils.getRepoPath('smit', '.') == tmp_path.resolve() / 'smit'
    cmd, _ = run_utils.buildCommand('smit', 'Fake_Model', {'input_path': '.', 'output_path': 'out'},
                                    'run_segmentation.py')
    assert os.path.isabs(cmd[0]) and os.path.isabs(cmd[1])


def test_export_seg_uses_model_file_names(tmp_path, monkeypatch):
    """exportSeg names files after the model's label patterns and writes labels.json."""
    planC = SimpleNamespace(structure=[SimpleNamespace(structureName=n)
                                       for n in ('Body', 'Left masseter', 'Larynx', 'Extra')])
    written = []
    monkeypatch.setattr(run_utils.pc, 'saveNiiStructure',
                        lambda path, labelDict, planC, strNumV: written.append((os.path.basename(path), strNumV)))
    monkeypatch.setattr(run_utils.rtstruct_iod, 'create',
                        lambda strNumV, path, planC, opts: written.append((os.path.basename(path), strNumV)))
    multi = run_utils.normalizeLabels({'*_chew_AI_seg.*': {1: 'Left masseter'}, '*_larynx_AI_seg.*': {1: 'Larynx'}})
    run_utils.exportSeg(planC, [1, 2, 3], 'pt1', False, 'HN', multi, str(tmp_path))
    assert written == [('pt1_chew_AI_seg.nii.gz', [1]), ('pt1_larynx_AI_seg.nii.gz', [2])]
    assert json.loads((tmp_path / 'labels.json').read_text())['labels']['*_larynx_AI_seg.*'] == {'1': 'Larynx'}
    written.clear()
    run_utils.exportSeg(planC, [1, 2], 'pt1', True, 'HN', multi, str(tmp_path))
    assert written == [('pt1_chew_AI_seg.dcm', [1]), ('pt1_larynx_AI_seg.dcm', [2])]
    written.clear()
    single = run_utils.normalizeLabels({1: 'Larynx'})
    run_utils.exportSeg(planC, [2], 'pt1', True, 'CT_HN_SMIT', single, str(tmp_path))
    run_utils.exportSeg(planC, [2], 'pt1', False, 'CT_HN_SMIT', single, str(tmp_path))
    assert written == [('pt1_CT_HN_SMIT_RTSTRUCT.dcm', [2]), ('pt1_seg.nii.gz', [2])]


def _runWithFakeProc(tmp_path, monkeypatch, withPost):
    """main on pt1.nii.gz with a fake prep (model output to <session_path>/output), an optional fake post,
    and the model run replaced by writing pt1_seg.nii.gz to the results dir."""
    repoPath = _fakeRepo(tmp_path)
    (repoPath / 'weights').mkdir()
    (repoPath / 'weights' / 'Fake_Model.safetensors').write_text('')
    procDir = tmp_path / 'proc'
    procDir.mkdir()
    (procDir / 'prep.py').write_text(
        "import os\n"
        "def processInputData(userInputs):\n"
        "    out = os.path.join(userInputs['session_path'], 'output')\n"
        "    return 'planC', 1, 0, {**userInputs, 'output_path': out}\n")
    (procDir / 'post.py').write_text(
        "def postProcAndImportSeg(planC, procScanNum, scanNum, userInputs, outDir):\n"
        "    return planC + ':post', [7]\n")
    monkeypatch.setattr(run_utils, 'getProcScript',
                        lambda kind, name: procDir / f'{kind}.py' if kind == 'prep' or withPost else None)

    def fakeRun(cmd, **kwargs):
        """Stand in for the model run: write pt1_seg.nii.gz to the results directory."""
        with open(os.path.join(cmd[cmd.index('--results_dir') + 1], 'pt1_seg.nii.gz'), 'w') as f:
            f.write('model output')
        return SimpleNamespace(returncode=0, stdout='', stderr='')
    monkeypatch.setattr(run_utils.subprocess, 'run', fakeRun)
    nii = tmp_path / 'pt1.nii.gz'
    nii.write_text('')
    userInputs = {'input_path': str(nii), 'output_path': str(tmp_path / 'out'),
                  'session_path': str(tmp_path / 'session')}
    return run_utils.main(99, str(tmp_path), userInputs)


def test_main_post_re_exports_to_output_path(tmp_path, monkeypatch, fakeModel):
    """With a post-processing script, structures are re-exported to output_path."""
    exported = []
    monkeypatch.setattr(run_utils, 'readLabels', lambda *args: {'*_seg.*': {1: 'GTV'}})
    monkeypatch.setattr(run_utils, 'exportSeg', lambda planC, structNumV, case, isDicom, model, labels, out:
                        exported.append((planC, structNumV, case, isDicom, out)))
    assert _runWithFakeProc(tmp_path, monkeypatch, withPost=True).returncode == 0
    assert exported == [('planC:post', [7], 'pt1', False, str(tmp_path / 'out'))]


def test_main_prep_only_copies_session_output(tmp_path, monkeypatch, fakeModel):
    """Without a post-processing script, the model's session output is copied to output_path."""
    _runWithFakeProc(tmp_path, monkeypatch, withPost=False)
    assert (tmp_path / 'out' / 'pt1_seg.nii.gz').read_text() == 'model output'
