"""Log of the AI models available through cerr.ai_models: model number -> dict(model name, repository name, entrypoint).

A model is supported via pyCERR when it has a repository (https://github.com/cerr/<repo>) and an
entrypoint (its inference script, relative to the repository root); otherwise it is listed only.
"""

MODELS = {1: {'name': 'CT_cardiac_structures_deeplab', 'repo': 'CT_cardiac_structures_deeplab', 'entrypoint': None},
          2: {'name': 'CT_LungOAR_incrMRRN', 'repo': 'CT_LungOAR_incrMRRN', 'entrypoint': None},
          3: {'name': 'CT_HeadAndNeck_OARs', 'repo': 'CT_HeadAndNeck_OARs', 'entrypoint': 'run_segmentation.py'},
          4: {'name': 'MR_Prostate_Deeplab', 'repo': 'MR_Prostate_Deeplab', 'entrypoint': None},
          5: {'name':'MRI_Pancreas_Fullshot_AnatomicCtxShape', 'repo':'MRI_Pancreas_Fullshot_AnatomicCtxShape', 'entrypoint': None},
          6: {'name':'CT_Lung_SMIT', 'repo':'smit', 'entrypoint': None},
          7: {'name':'CT_HN_SMIT', 'repo':'smit', 'entrypoint': 'run_segmentation.py'},
          8: {'name':'CT_HeartSubstruct_SMIT', 'repo':'smit', 'entrypoint': 'run_segmentation.py'},
          9: {'name':'CT_WHOLEBODY_SMITplus', 'repo':'smit', 'entrypoint': 'run_segmentation.py'},
          10: {'name':'MR_Rectum_GTV_SMIT', 'repo':'smit', 'entrypoint': 'run_segmentation.py'},
          11: {'name':'MR_HN_Nodule_SMIT', 'repo':'smit', 'entrypoint': 'run_segmentation.py'},
          12: {'name':'CT_Lung_OAR_SMITplus', 'repo':'smit', 'entrypoint': None},
          13: {'name':'MR_BrainMets_SMITplus', 'repo':'smit', 'entrypoint': None},
          14: {'name':'CT_ThoraxOARs_SMITplus', 'repo':'smit', 'entrypoint': 'run_segmentation.py'},
          15: {'name':'CT_LungGTV_SMITplus', 'repo':'smit', 'entrypoint': 'run_segmentation.py'},
          16: {'name':'CT_CADSRegions_SMITplus', 'repo':'smit', 'entrypoint': 'run_segmentation.py'}}


def resolveModel(model):
    """Model number and log entry for a model given by number or name.

    Args:
        model (int or str): model number (e.g. 14 or '14') or name (e.g. 'CT_ThoraxOARs_SMITplus').

    Returns:
        tuple: (modelNum, entry), entry being a dict with 'name', 'repo' and 'entrypoint'.
    """
    if isinstance(model, str) and model.strip().isdigit():
        model = int(model)
    if isinstance(model, int) and not isinstance(model, bool):
        if model not in MODELS:
            raise ValueError(f"Invalid model number: {model}. Run install_utils.listModels() to see available models.")
        return model, MODELS[model]
    for modelNum, entry in MODELS.items():
        if entry['name'] == model:
            return modelNum, entry
    raise ValueError(f"Unknown model: {model!r}. Run install_utils.listModels() to see available models.")
