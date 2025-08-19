import json
import numpy as np
from scipy.special import erf

from cerr.dvh import *
from cerr.dvh import getDVH, doseHist
from cerr.dataclasses.structure import getMatchingIndex
from cerr.dataclasses.dose import fractionNumCorrect, fractionSizeCorrect


def linearFn(paramDict, doseBinsV, volHistV):
    """
    Evaluate linear dosimetric model.
    Args:
          paramDict: Dictionary specifying slope and intercept.
          doseBinsV: Vector of dose bins.
          volHistV:  Vector of volumes corresponding to dose bins.
    Returns:
          NTCP using linear model.
    """


    # Get parameters
    intercept = paramDict['intercept']['val']

    structDict = paramDict['structures']
    structList = list(structDict.keys())
    sum = 0
    for structName in structList:
        dvhMetricFn = structDict[structName]['val']
        slope = structDict[structName]['weight']
        sum = sum + slope * eval(dvhMetricFn + "(doseBinsV, volHistV)")

    #Compute NTCP
    ntcp = intercept + sum

    return ntcp


def LKBFn(paramDict, doseBinsV, volHistV):
    """
    Evaluate LKB model.
    Args:
          paramDict: Dictionary specifying D50, m, and n.
          doseBinsV: Vector of dose bins.
          volHistV:  Vector of volumes corresponding to dose bins.
    Returns:
          NTCP using LKB model.
    """


    D50 = paramDict['D50']['val']
    m = paramDict['m']['val']
    n = paramDict['n']['val']

    #Calc. EUD for selected struct/dose
    equivDose = eud(doseBinsV, volHistV, 1/n)

    #Calc. NTCP
    tmpv = (equivDose - D50) / (m * D50)
    ntcp = 0.5 * (1 + erf(tmpv / np.sqrt(2)))

    return ntcp


def logitFn(paramDict, doseBinList, volHistList):
    """
    Evaluate logistic dosimetric model.
    Args:
          paramDict: Dictionary specifying predictors and coefficients.
          doseBinList: List of dose bins for input structures.
          volHistList: List of volumes corresponding to dose bins.
    Returns:
          NTCP using logistic model.
    """

    def _getParCoeff(paramDict, fieldName, doseBinList, volHistList):

        keepParDict = {}

        for genField in paramDict:
            if genField.lower() == 'structures':
                structS = paramDict[genField]
                for structName in structS:
                    strParamS = structS[structName]
                    for parName in strParamS:
                        entry = strParamS[parName]
                        if 'weight' in entry:
                            fullName = f"{structName}{parName}"
                            keepParDict[fullName] = entry
            else:
                entry = paramDict[genField]
                if 'weight' in entry:
                    keepParDict[genField] = entry

        weightList = []
        paramList = []
        numStr = 0

        for predictorName, predictorVal in keepParDict.items():
            weightList.append(predictorVal[fieldName])

            if isinstance(predictorVal['val'], (int, float)):
                paramList.append(predictorVal['val'])

            else:  # val['val'] is a string referring to a function
                if isinstance(doseBinList, list):  # multi-structure
                    doseBinsV = doseBinList[numStr]
                    volHistV = volHistList[numStr]
                    numStr += 1
                else:
                    doseBinsV = doseBinList
                    volHistV = volHistList

                if isinstance(predictorVal['val'], str):
                    if 'params' not in predictorVal:
                        paramList.append(eval(predictorVal['val'])(doseBinsV, volHistV))
                    else:
                        # Pass extra parameters (e.g., numFractions, abRatio)
                        params = predictorVal['params']
                        if 'numFractions' in paramDict:
                            params['numFractions'] = {'val': paramDict['numFractions']['val']}
                        if 'abRatio' in paramDict:
                            params['abRatio'] = {'val': paramDict['abRatio']['val']}

                        paramList.append(eval(predictorVal['val'])(doseBinsV, volHistV, params))

        return weightList, paramList

    weight, x = _getParCoeff(paramDict, 'weight', doseBinList, volHistList)
    gx = np.sum(np.array(weight) * np.array(x))
    ntcp = 1 / (1 + np.exp(-gx))
    return ntcp


def appeltLogit(paramDict, doseBinList, volHistList):
    """
    Evaluate appelt-corrected logistic model.

    Ref: Ane L. Appelt, Ivan R. Vogelius, Katherina P. Farr, Azza A. Khalil & Søren
    M. Bentzen (2014) Towards individualized dose constraints: Adjusting the QUANTEC
    radiation pneumonitis model for clinical risk factors, Acta Oncologica, 53:5, 605-612.
    Args:
          paramDict: Dictionary specifying predictors and associated coefficients.
          doseBinList: List of dose bins.
          volHistList: List of volumes corresponding to dose bins.
    Returns:
          NTCP using logistic model, following Appelt correction.
    """

    def _applyAppeltMod(D50_0, gamma50_0, OR):
        """
        Return modified D50, gamma50, accounting for odds ratios.
        """

        D50Risk = (1 - np.log(OR) / (4 * gamma50_0)) * D50_0
        gamma50Risk = gamma50_0 - np.log(OR) / 4
        return D50Risk, gamma50Risk

    def _getParCoeff(paramDict, fieldName, doseBinsV, volHistV):
        keepParS = {}

        for genField, genVal in paramDict.items():
            if genField.lower() == 'structures' and isinstance(genVal, dict):
                for structName, structVal in genVal.items():
                    for parName, parEntry in structVal.items():
                        if 'weight' in parEntry:
                            fullName = f"{structName}{parName}"
                            keepParS[fullName] = parEntry
            elif 'weight' in genVal:
                keepParS[genField] = genVal

        coeffList = []
        parList = []

        for key, parEntry in keepParS.items():
            coeffList.append(parEntry[fieldName])

            if isinstance(parEntry['val'], (int, float)):
                parList.append(parEntry['val'])
            else:
                fnName = parEntry['val']
                if 'params' not in parEntry:
                    parList.append(eval(fnName)(doseBinsV, volHistV))
                else:
                    # Copy number of fractions and abRatio
                    parParams = parEntry['params']
                    if 'numFractions' in paramDict:
                        parParams['numFractions'] = {'val': paramDict['numFractions']['val']}
                    if 'abRatio' in paramDict:
                        parParams['abRatio'] = {'val': paramDict['abRatio']['val']}
                    parList.append(eval(fnName)(doseBinsV, volHistV, parParams))

        return parList, coeffList


    # Handle single and multi-structure inputs
    doseBinsV = doseBinList[0] if isinstance(doseBinList, list) else doseBinList
    volHistV = volHistList[0] if isinstance(volHistList, list) else volHistList

    # Apply Appelt modification
    if 'appeltMod' in paramDict and paramDict['appeltMod']['val'].lower() == 'yes':
            D50_0 = paramDict['D50_0']['val']
            gamma50_0 = paramDict['gamma50_0']['val']

            orList, weightList = _getParCoeff(paramDict, 'OR', doseBinsV, volHistV)
            orMult = [o for o, w in zip(orList, weightList) if w == 1]
            OR = np.prod(orMult)

            D50, gamma50 = _applyAppeltMod(D50_0, gamma50_0, OR)
    else:
            D50 = paramDict['D50']['val']
            gamma50 = paramDict['gamma50']['val']

    md = meanDose(doseBinsV, volHistV)
    ntcp = 1.0 / (1 + np.exp(4 * gamma50 * (1 - md / D50)))

    return ntcp


def coxFn(paramDict, doseBinList, volHistList):
    """
    Evaluate Cox Proportional Hazards model.
    Args:
          paramDict: Dictionary specifying baseline hazard, predictors and coefficients.
          doseBinList: List of dose bins for input structures.
          volHistList: List of volumes corresponding to dose bins.
    Returns:
          Actuarial probability or NTCP using Cox model.
    """

    def _calcHazard(H0, betaV, dkV):
        """ Cox model hazard function """
        return H0 * np.exp(np.sum(np.multiply(betaV, dkV)))

    def _calcPa(H0, betaV, dkV):
        """ Actuarial probability of complication from Cox model """
        return 1 - np.exp(-H0 * np.exp(np.dot(betaV, dkV)))

    def _getParCoeff(paramDict, fieldName, doseBinList, volHistList):
        keepParS = {}

        for genField, genVal in paramDict.items():
            if genField.lower() == 'structures' and isinstance(genVal, dict):
                for structName, structVal in genVal.items():
                    for parName, parVal in structVal.items():
                        if 'weight' in parVal:
                            keepParS[f"{structName}{parName}"] = parVal
            elif 'weight' in genVal:
                keepParS[genField] = genVal

            
        par = []
        coeff = []
        paramList = list(keepParS.keys())
        numStr = 0

        for paramName in paramList:
            entry = keepParS[paramName]
            coeff.append(entry[fieldName])

            # Determine structure-specific dose/volume
            if isinstance(entry['val'], (int, float)):
                par.append(entry['val'])
            else:
                fnName = entry['val']
                if isinstance(doseBinList, list):
                    doseBinsV = doseBinList[numStr]
                    volHistV = volHistList[numStr]
                    numStr += 1
                else:
                    doseBinsV = doseBinList
                    volHistV = volHistList

                if 'params' not in entry:
                    par.append(eval(fnName)(doseBinsV, volHistV))
                else:
                    entryParams = entry['params']
                    if 'numFractions' in paramDict:
                        entryParams['numFractions'] = {'val': paramDict['numFractions']['val']}
                    if 'abRatio' in paramDict:
                        entryParams['abRatio'] = {'val': paramDict['abRatio']['val']}
                    par.append(eval(fnName)(doseBinsV, volHistV, entryParams))

        return coeff, par, paramList

    betaV, xV, paramList = _getParCoeff(paramDict, 'weight', doseBinList, volHistList)

    # Find baseline hazard
    matchIdx = [i for i, p in enumerate(paramList) if p.lower() == 'baselineHazard']
    if not matchIdx:
        raise ValueError("Missing required parameter: baselineHazard")

    idx = matchIdx[0]
    H0 = betaV[idx]
    betaV.pop(idx)
    xV.pop(idx)

    if 'outputType' in paramDict and paramDict['outputType']['val'].lower() == 'hazard':
        prob = _calcHazard(H0, np.array(betaV), np.array(xV))
    else:
        prob = _calcPa(H0, np.array(betaV), np.array(xV))

    return prob


def biexpFn(paramDict, doseBinList, volHistList):
    """
    Evaluate bi-exponential model.
    Args:
          paramDict: Dictionary specifying, predictors and coefficients.
          doseBinList: List of dose bins for input structures.
          volHistList: List of volumes corresponding to dose bins.
    Returns:
          NTCP using biexponential model.
    """

    def _getParCoeff(paramDict, fieldName, doseBinList, volHistList):

        keepParDict = {}

        for genField in paramDict:
            if genField.lower() == 'structures':
                structS = paramDict[genField]
                for structName in structS:
                    strParamS = structS[structName]
                    for parName in strParamS:
                        entry = strParamS[parName]
                        if 'weight' in entry:
                            fullName = f"{structName}{parName}"
                            keepParDict[fullName] = entry
            else:
                entry = paramDict[genField]
                if 'weight' in entry:
                    keepParDict[genField] = entry

        weightList = []
        paramList = []
        numStr = 0

        for predictorName, predictorVal in keepParDict.items():
            weightList.append(predictorVal[fieldName])

            if isinstance(predictorVal['val'], (int, float)):
                paramList.append(predictorVal['val'])

            else:  # val['val'] is a string referring to a function
                if isinstance(doseBinList, list):  # multi-structure
                    doseBinsV = doseBinList[numStr]
                    volHistV = volHistList[numStr]
                    numStr += 1
                else:
                    doseBinsV = doseBinList
                    volHistV = volHistList

                if isinstance(predictorVal['val'], str):
                    if 'params' not in predictorVal:
                        paramList.append(eval(predictorVal['val'])(doseBinsV, volHistV))
                    else:
                        # Pass extra parameters (e.g., numFractions, abRatio)
                        params = predictorVal['params']
                        if 'numFractions' in paramDict:
                            params['numFractions'] = {'val': paramDict['numFractions']['val']}
                        if 'abRatio' in paramDict:
                            params['abRatio'] = {'val': paramDict['abRatio']['val']}

                        paramList.append(eval(predictorVal['val'])(doseBinsV, volHistV, params))

        return weightList, paramList

    weights, predictors = _getParCoeff(paramDict, 'weight', doseBinList, volHistList)
    ntcp = 0.5 * (np.exp(-weights[0] * predictors[0]) + np.exp(-weights[1] * predictors[1]))

    return ntcp

def run(modelFile, doseNum, planC, fSizeIn=None, fNumIn=None, binWidth=0.05):
    """
    Evaluate dosimetric model including fractionation correction where applicablegit add.
    Args:
          modelFile: Path to JSON file describing model parameters.
          doseNum: Index of dose in planC
          planC: plan container object
          fSizeIn: Fraction size of input plan
          fNumIn: Fraction no. of input plan
          binWidth (float): Bin width for DVH calculation. Default:0.05
    Returns:
        Model-based NTCP.
    """

    # Read model parameters
    with open(modelFile, 'r') as f:
        model = json.load(f)

    fields = list(model.keys())
    modelFn = model['function']
    paramDict = model['parameters']

    fsizeCorr = False
    fnumCorr = False
    if 'fractionCorrect' in fields and model['fractionCorrect'].lower() == 'yes':
        if model['correctionType'].lower() == 'fsize':
            fsizeCorr = True
            stdFsize = model['stdFractionSize']
            abRatio = model['abRatio']
            inputFrxsize = fSizeIn
        elif model['correctionType'].lower() == 'fnum':
            fnumCorr = True
            stdFrxNum = model['stdNumFractions']
            inputFrxNum = fNumIn

    # Extract DVHs
    modelStructs = model['parameters']['structures']
    if isinstance(modelStructs, dict):
        structureList = list(modelStructs.keys())
    elif isinstance(modelStructs, str):
        structureList = [modelStructs]
    elif isinstance(modelStructs, list):
        structureList = modelStructs
        
    availStructList = [cerrStr.structureName for cerrStr in planC.structure]

    doseBinList = []
    volHistList = []
    for struct in structureList:
        if isinstance(model['parameters']['structures'], dict) and \
        isinstance(model['parameters']['structures'][struct], dict):
            structNumV = getMatchingIndex(struct, availStructList, matchCriteria='exact')
            dosesV, volsV, __ = getDVH(structNumV[0], doseNum, planC)
            doseBinsV, volHistV = doseHist(dosesV, volsV, binWidth)

            # Fractionation correction
            if fsizeCorr:
                corrDoseBinsV = fractionSizeCorrect(doseBinsV, stdFsize, abRatio, planC, inputFrxsize)
            elif fnumCorr:
                corrDoseBinsV = fractionNumCorrect(doseBinsV, stdFrxNum, abRatio, planC, inputFrxNum)
            else:
                corrDoseBinsV = doseBinsV

            doseBinList.append(corrDoseBinsV)
            volHistList.append(volHistV)

    ntcp = eval(modelFn)(paramDict, doseBinList, volHistList)

    return ntcp