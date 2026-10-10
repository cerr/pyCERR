"""
 These tests compare NTCPs computed using pyCERR against a manually-calculated reference.
"""
import copy
import glob
import json
import os

import numpy as np
from cerr import plan_container as pc
from cerr.dataclasses.structure import getMatchingIndex
from cerr.dvh import getDVH
from cerr.roe import dosimetric_models

# Load sample dataset
currPath = os.path.abspath(__file__)
cerrPath = os.path.join(os.path.dirname(os.path.dirname(currPath)), 'cerr')
dataDir = os.path.join(cerrPath, 'datasets', 'sample_ct', 'dosimetric_model_test_data')
modelDir = os.path.join(cerrPath, 'roe', 'model_parameters')

def load_data(niiDir):
    """ Import data to plan container"""

    scanFile = os.path.join(niiDir, 'scan.nii')
    doseFile = os.path.join(niiDir, 'dose.nii')
    maskFiles = glob.glob(os.path.join(niiDir, 'mask*.nii'))

    planC = pc.loadNiiScan(scanFile, "CT SCAN")
    scanNum = len(planC.scan) - 1

    planC = pc.loadNiiDose(doseFile, scanNum, planC)

    for maskFile in maskFiles:
        structureName = (maskFile.split('_')[-1]).split('.')[0]
        planC = pc.loadNiiStructure(maskFile, scanNum, planC)
        planC.structure[-1].structureName = structureName

    return planC

planC = load_data(dataDir)
binWidth = 0.05


def create_test_dose(model, testDose, testParams, planNum, scanNum, planC):
    testPlanC = copy.deepcopy(planC)
    modelCpy = copy.deepcopy(model)

    strList = [cerrStr.structureName for cerrStr in planC.structure]

    strName = testParams["structures"]
    strIdx = getMatchingIndex(strName, strList, 'EXACT')[0]
    currentDose = getDVH(strIdx, planNum, planC)[0]
    scaleFactor = testDose / currentDose
    dA = testPlanC.dose[planNum].doseArray
    dAtest = dA * scaleFactor
    xV, yV, zV = testPlanC.dose[planNum].getDoseXYZVals()
    testPlanC = pc.importDoseArray(dAtest, xV, yV, zV, testPlanC, scanNum)
    testDoseIdx = len(testPlanC.dose) - 1

    for param, val in testParams.items():
        if isinstance(modelCpy['parameters'][param], dict):
            nameList = list(modelCpy['parameters'][param].keys())
            for name in nameList:
                if isinstance(modelCpy['parameters'][param][name], dict):
                    if isinstance(val, dict):
                        for subName, matchVal in val.items():
                            modelCpy['parameters'][param][name][subName] = matchVal
                    else:
                        modelCpy['parameters'][param][val] = modelCpy['parameters'][param][name]
                        del modelCpy['parameters'][param][name]
                    break
                else:
                    if isinstance(val, dict):
                        for subName, matchVal in val.items():
                            modelCpy['parameters'][param][subName] = matchVal
                    else:
                        modelCpy['parameters'][param][name] = [val]
                break
        else:
            modelCpy['parameters'][param] = [val]

    return modelCpy, testPlanC


def run_config(model, testDoses, testFrx, testParams, testType, refVals):
    #testFrxType: fNUmIn, fSizeIn
    #testType: ntpc, fraction_correct

    origScanNum = 0
    origPlanIdx = 0
    tol = 4 #Compare to 4 decimal places

    # Read correction type
    testFrxType = None
    if 'correctionType' in list(model.keys()):
        testFrxType = model['correctionType'].lower()

    # Loop over tests
    for testNum in range(len(testDoses)):

        # Create test dose
        modelCpy, testPlanC = create_test_dose(model, testDoses[testNum], testParams[testNum],
                                               origPlanIdx, origScanNum, planC)
        testDoseIdx = len(testPlanC.dose)-1

        # Calc. metric
        if testType == 'ntcp':

            if testFrxType == 'frxsize':
                testVal = dosimetric_models.run(modelCpy, testDoseIdx, testPlanC, fSizeIn=testFrx[testNum], mode='test')
            elif testFrxType == 'frxnum':
                testVal = dosimetric_models.run(modelCpy, testDoseIdx, testPlanC, fNumIn=testFrx[testNum], mode='test')
            else:
                # No fractionation correction
                testVal = dosimetric_models.run(modelCpy, testDoseIdx, testPlanC, mode='test')

        elif testType == 'fraction_correct':

            if testFrxType == 'frxsize':
                testValV, __, __ = dosimetric_models.getCorrectedDVbins(modelCpy, testDoseIdx, testPlanC,
                                                                         fSizeIn=testFrx[testNum], mode='test')
            elif testFrxType == 'frxnum':
                testValV, __, __ = dosimetric_models.getCorrectedDVbins(modelCpy, testDoseIdx, testPlanC,
                                                                         fNumIn=testFrx[testNum], mode='test')
            else:
                # No fractionation correction
                testValV, __, __ = dosimetric_models.run(modelCpy, testDoseIdx, testPlanC)
            testVal = testValV[0]

        # Compare vs. reference value
        np.testing.assert_almost_equal(testVal, refVals[testNum], tol)
        print('\n\tTest {} passed.'.format(testNum+1))
        del testPlanC


def test_fc1():
    print('\nBy std. fraction no.')
    testType = 'fraction_correct'
    modelFile = os.path.join(modelDir, 'Esophagitis (Huang).json')
    with open(modelFile, 'r') as f:
        model = json.load(f)

    testDoses = [29.2318, 14.6125]
    nFrx = [15, 15]
    testParams = [{"structures": "testStr1", "concurrentChemo": {"val": 1}},
                  {"structures": "testStr1", "concurrentChemo": {"val": 1}}]

    refVals = [32.0023, 15.3618]
    run_config(model, testDoses, nFrx, testParams, testType, refVals)


def test_fc2():
    print('\nBy std. fraction size')
    testType = 'fraction_correct'
    modelFile = os.path.join(modelDir, 'Rectal bleeding (grade 2+).json')
    with open(modelFile, 'r') as f:
        model = json.load(f)

    testDoses = [35]
    fSize = [7]
    testParams = [{"structures": "testStr1"}]

    refVals = [70]
    run_config(model, testDoses, fSize, testParams, testType, refVals)


def test_appelt_pneumonitis_model():
    testType = 'ntcp'
    modelFile = os.path.join(modelDir, 'Pneumonitis (Appelt).json')
    with open(modelFile, 'r') as f:
        model = json.load(f)

    testDoses = [34.4, 37.8547, 31.9303]
    nFrx = [35, 35, 35]
    testParams = [{"structures": "testStr1", 'formerSmoker': {'val':0}, 'currentSmoker': {'val': 0}, 'over63yrs': {'val': 0},
                   'pulmonaryComorbidity': {'val': 0}, 'sequentialChemo': {'val': 0}, 'lowerMidLobe': {'val': 0}},
              {"structures": "testStr1", 'formerSmoker': {'val':0}, 'currentSmoker': {'val': 1}, 'over63yrs': {'val': 0},
               'pulmonaryComorbidity': {'val': 0}, 'sequentialChemo': {'val': 0}, 'lowerMidLobe': {'val': 0}},
              {"structures": "testStr1", 'formerSmoker': {'val':0}, 'currentSmoker': {'val': 1}, 'over63yrs': {'val': 0},
               'pulmonaryComorbidity': {'val': 1}, 'sequentialChemo': {'val': 0}, 'lowerMidLobe': {'val': 0}}
              ]

    refVals = [0.5, 0.5, 0.5]
    print('\nTesting the Appelt pneumonitis model...')
    run_config(model, testDoses, nFrx, testParams, testType, refVals)


def test_huang_model():
    testType = 'ntcp'
    modelFile = os.path.join(modelDir, 'Esophagitis (Huang).json')
    with open(modelFile, 'r') as f:
        model = json.load(f)

    testDoses = [29.2318, 14.6125]
    nFrx = [15, 15]
    testParams = [{"structures": "testStr1", "concurrentChemo": {"val": 1}},
                  {"structures": "testStr1", "concurrentChemo": {"val": 1}}]

    refVals = [0.63916, 0.36051]
    print('\nTesting the Huang eshophagitis model...')
    run_config(model, testDoses, nFrx, testParams, testType, refVals)
    

def test_wijsman_esophagitis_model():
    testType = 'ntcp'
    modelFile = os.path.join(modelDir, 'Esophagitis (Wijsman).json')
    with open(modelFile, 'r') as f:
        model = json.load(f)

    testDoses = [54.8547, 44.5641, 46.3590, 21.1928]
    fSize = [2, 2, 2, 2]
    testParams = [{"structures": "testStr1", "gender": {"val": 0}, "tumorStage": {"val": 0}, "concurrentChemo": {"val": 0}},
                  {"structures": "testStr1", "gender": {"val": 1}, "tumorStage": {"val": 0}, "concurrentChemo": {"val": 0}},
                  {"structures": "testStr1", "gender": {"val": 0}, "tumorStage": {"val": 1}, "concurrentChemo": {"val": 0}},
                  {"structures": "testStr1", "gender": {"val": 0}, "tumorStage": {"val": 1}, "concurrentChemo": {"val": 0}}]

    refVals = [0.5, 0.5, 0.5, 0.05]
    print('\nTesting the Wijsman esophagitis model...')
    run_config(model, testDoses, fSize, testParams, testType, refVals)


def test_jackson_esophagitis_logistic_model():
    # Grade 2+ esophagitis in ultra-central lung
    testType = 'ntcp'
    modelFile = os.path.join(modelDir, 'Esophagitis (Jackson_logistic).json')
    with open(modelFile, 'r') as f:
        model = json.load(f)

    testDoses = [51.3609]
    fSize = [2]
    testParams = [{"structures": "testStr1"}]

    refVals = [0.5]
    print('\nTesting the Jackson esophagitis model (logistic)...')
    run_config(model, testDoses, fSize, testParams, testType, refVals)


def test_jackson_esophagitis_cox_model():
    # Grade 2+ esophagitis in ultra-central lung
    testType = 'ntcp'
    modelFile = os.path.join(modelDir, 'Esophagitis (Jackson_cox).json')
    with open(modelFile, 'r') as f:
        model = json.load(f)

    testDoses = [49.1008]
    fSize = [2]
    testParams = [{"structures": "testStr1"}]

    refVals = [0.5]
    print('\nTesting the Jackson esophagitis model (cox)...')
    run_config(model, testDoses, fSize, testParams, testType, refVals)


def test_bronchial_stenosis_logistic_model():
    testType = 'ntcp'
    modelFile = os.path.join(modelDir, 'Bronchial stenosis (logistic).json')
    with open(modelFile, 'r') as f:
        model = json.load(f)

    testDoses = [80.2771]
    testParams = [{"structures": "testStr1"}]

    refVals = [0.5]
    print('\nTesting the bronchial stenosis model (logistic)...')
    run_config(model, testDoses, [], testParams, testType, refVals)


def test_bronchial_stenosis_cox_model():
    testType = 'ntcp'
    modelFile = os.path.join(modelDir, 'Bronchial stenosis (cox).json')
    with open(modelFile, 'r') as f:
        model = json.load(f)

    testDoses = [69.2936]
    testParams = [{"structures": "testStr1"}]
    
    refVals = [0.5]
    print('\nTesting the bronchial stenosis model (cox)...')
    run_config(model, testDoses, [], testParams, testType, refVals)


def test_rectal_bleeding_model():
    testType = 'ntcp'
    modelFile = os.path.join(modelDir, 'Rectal bleeding (grade 2+).json')
    with open(modelFile, 'r') as f:
        model = json.load(f)

    testDoses = [35]
    fSize = [7]
    testParams = [{"structures": "testStr1"}]

    refVals = [0.24503]
    print('\nTesting the rectal bleeding model...')
    run_config(model, testDoses, fSize, testParams, testType, refVals)


def test_fractionation_correction():
    print('\nTesting fractionation correction...')
    test_fc1()
    test_fc2()


def test_lung_models():
    
    # Conventional
    test_appelt_pneumonitis_model()
    test_wijsman_esophagitis_model()
    test_huang_model()
    
    #Ultracentral
    test_jackson_esophagitis_logistic_model()
    test_jackson_esophagitis_cox_model()
    test_bronchial_stenosis_logistic_model()
    test_bronchial_stenosis_cox_model()


def test_prostate_models():
    test_rectal_bleeding_model()
    

def run_tests():

    test_fractionation_correction()
    test_lung_models()
    test_prostate_models()


if __name__ == "__main__":
    run_tests()

def test_run_from_predictors_every_builtin_model():
    """runFromPredictors must work for every built-in model, including the LKB
    one (LKBFn used to leave the dose unassigned when gEUD was supplied)."""
    import json
    for name in dosimetric_models.listModels():
        with open(dosimetric_models.mapModelToFile(name)) as f:
            params = json.load(f)['parameters']
        predictors = {f'{struct} {metric}': 30.0
                      for struct, metrics in params['structures'].items() for metric in metrics}
        predictors.update({key: 0 for key, entry in params.items()
                           if key != 'structures' and isinstance(entry, dict)
                           and entry.get('val', 0) is None})
        ntcp = float(dosimetric_models.runFromPredictors(name, predictors))
        assert 0.0 <= ntcp <= 1.0, name


def test_lkb_from_supplied_geud():
    """LKB: NTCP is 0.5 at gEUD = D50 and rises with gEUD."""
    import json
    name = 'Rectal bleeding (grade 2+)'
    with open(dosimetric_models.mapModelToFile(name)) as f:
        d50 = json.load(f)['parameters']['D50']['val']
    atD50 = dosimetric_models.runFromPredictors(name, {'Rectum gEUD': d50})
    np.testing.assert_allclose(atD50, 0.5, atol=1e-12)
    assert dosimetric_models.runFromPredictors(name, {'Rectum gEUD': d50 + 10}) > atD50
    assert dosimetric_models.runFromPredictors(name, {'Rectum gEUD': d50 - 10}) < atD50
