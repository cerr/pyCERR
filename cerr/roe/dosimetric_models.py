import json
import math
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
    matchIdx = [i for i, p in enumerate(paramList) if p.lower() == 'baselinehazard']
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
          paramDict: Dictionary specifying clear
          predictors and coefficients.
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


def lungBED(paramDict, doseBinsV, volHistV):
    """
    Function to compute lung BED.
    Ref. Comparison Between Mechanistic Radiobiological Modeling vs. Fowler BED Equation in Evaluating
    Lung Cancer Radiotherapy Outcome for a Broad Range of Fractionation, J Jeong et al., AAPM 2017.

    Args:
          paramDict: Dictionary specifying D50, m, and n.
          doseBinsV: Vector of dose bins.
          volHistV:  Vector of volumes corresponding to dose bins.
    Returns:
          Lung BED.
    """
    paramList = list(paramDict.keys())
    Tk = paramDict['Tk']['val'] #Kick-off time for repopulation (days)
    Tp = paramDict['Tp']['val'] #Potential tumor doubling time(days)

    alpha = paramDict['alpha']['val']
    abRatio = float(paramDict['abRatio']['val'])
    d = float(paramDict['frxSize']['val'])
    n = float(paramDict['numFractions']['val'])

    if 'treatmentDays' in paramList:
        txDaysV = paramDict['treatmentDays']['val']
        #Check for numeric input
        if ~txDaysV.isnumeric():
            try:
                txDaysV = int(txDaysV)
            except ValueError:
                # Otherwise assume function specified
                type = paramDict['treatmentDays']['params']['scheduleType']
                txDaysV = eval(txDaysV)(n, type)
        T = txDaysV[-1]
    else:
        #Default: Compute length of treatment assuming one fraction every weekday with weekend breaks.
        T = np.floor(n / 5) * 7 + (n % 5)

    #Compute BED
    bed = n * d * (1 + d/abRatio)
    if T > Tk:
        bed = bed - math.log(2) * (T - Tk) / (alpha * Tp)

    return bed


def lungTCP(paramDict, doseBinsV, volHistV):
    """
    Function to compute lung TCP.
    Ref. Jeong J. et al. Modeling the cellular response of lung cancer to radiation therapy for a
    broad range of fractionation schedules. Clin. Cancer Res. 2017; 23: pp. 5469-5479.
    Args:
          paramDict: Dictionary specifying D50, m, and n.
          doseBinsV: Vector of dose bins.
          volHistV:  Vector of volumes corresponding to dose bins.
    Returns:
          Lung TCP.
    """

    # Predefined settings
    alphaPOri = 0.305
    aOverB = 2.8
    oerI = 1.7
    rhoT = 10 ** 6
    vTRef = 3e4
    fS = 0.01
    tC = 2
    fPProIn = 0.5
    htLoss = 2
    kM = 0.3
    htLys = 3
    oerH = 1.37
    FPCyc = np.array([0.56, 0.24, 0.2])
    alphaRatioPCyc = np.array([2, 3])

    dT = 15
    clfIn = 0.92
    gfIn = 0.25

    betaPOri = alphaPOri / aOverB

    # %% EQD2 estimation for each cohort
    vT = 3e4
    alphaP = alphaPOri
    betaP = betaPOri
    nT = rhoT * vT
    nTRef = rhoT * vTRef
    deltaT = dT / (60 * 24)  # dt in days
    tStart = 0

    # Compartment sizes
    compSize = np.zeros(3)
    compSizeRef = np.zeros(3)

    clf = clfIn
    gf = gfIn

    # Initial distribution
    fPPro = fPProIn

    compSize[0] = gf / fPPro * nT
    compSize[1] = (1 - gf * (1 / fPProIn + clf * htLoss / tC)) * nT
    compSize[2] = clf * gf * htLoss / tC * nT

    compSizeRef[0] = gf / fPPro * nTRef
    compSizeRef[1] = (1 - gf * (1 / fPProIn + clf * htLoss / tC)) * nTRef
    compSizeRef[2] = clf * gf * htLoss / tC * nTRef

    # Get model parameters
    TD50 = paramDict['TD50']['val']
    gamma50 = paramDict['gamma50']['val']
    TCPUpperBound = paramDict['TCPlimit']['val']

    # Get fractionation schedule
    fxIn = paramDict['frxSize']['val']
    nFrx = paramDict['numFractions']['val']

    # Get treatment days
    scheduleIn = paramDict['treatmentSchedule']['val']
    if 'scheduleType' in paramDict['treatmentSchedule']:
        scheduleType = paramDict['treatmentSchedule']['scheduleType']
    else:
        scheduleType = 'weekday'

    # Check for numeric input
    scheduleV = []
    if isinstance(scheduleIn, str):
        try:
            scheduleV = list(map(int, scheduleIn.split()))
        except:
            scheduleV = []
    else:
            scheduleV = scheduleIn
    if scheduleV == []:
        scheduleV = getTreatmentSchedule(nFrx, scheduleType)

    #-----------------------------
    # Loop over doses
    # ----------------------------
    eqd2 = None
    for d in [fxIn]:
        treatDay = scheduleV

        # Solve root for alphaS
        def f(alphaS):
            return (
                    FPCyc[0] * np.exp(-alphaRatioPCyc[0] * alphaS * 2 - alphaRatioPCyc[0] * (alphaS / aOverB) * 4)
                    + FPCyc[1] * np.exp(-alphaS * 2 - (alphaS / aOverB) * 4)
                    + FPCyc[2] * np.exp(-alphaRatioPCyc[1] * alphaS * 2 - alphaRatioPCyc[1] * (alphaS / aOverB) * 4)
                    - np.exp(-alphaP * 2 - (alphaP / aOverB) * 4)
            )

        alphaS = 0.3
        grid = 0.1
        preF = f(alphaS)
        while abs(f(alphaS)) >= np.finfo(float).eps:
            if preF * f(alphaS) < 0:
                grid *= 0.1
            preF = f(alphaS)
            if f(alphaS) > 0:
                alphaS += grid
            else:
                alphaS -= grid

        alphaPCyc = np.zeros(3)
        alphaPCyc[1] = alphaS
        alphaPCyc[0] = alphaPCyc[1] * alphaRatioPCyc[0]
        alphaPCyc[2] = alphaPCyc[1] * alphaRatioPCyc[1]

        # Effective alpha, beta
        SuP = (
                FPCyc[0] * np.exp(-alphaPCyc[0] * d - (alphaPCyc[0] / aOverB) * d ** 2)
                + FPCyc[1] * np.exp(-alphaPCyc[1] * d - (alphaPCyc[1] / aOverB) * d ** 2)
                + FPCyc[2] * np.exp(-alphaPCyc[2] * d - (alphaPCyc[2] / aOverB) * d ** 2)
        )
        alphaPEff = -np.log(SuP) / (d * (1 + (d / aOverB)))
        betaPEff = alphaPEff / aOverB

        SuI2gy = np.exp(-alphaP / oerI * 2 - (alphaP / aOverB) / (oerI ** 2) * 2 ** 2)
        oerIG1 = (-(alphaPCyc[0] * 2) - np.sqrt(
            (alphaPCyc[0] * 2) ** 2 - 4 * np.log(SuI2gy) * (alphaPCyc[0] / aOverB) * 2 ** 2)) / (2 * np.log(SuI2gy))

        SuH2gy = np.exp(-alphaP / oerH * 2 - (alphaP / aOverB) / (oerH ** 2) * 2 ** 2)
        oerHG1 = (-(alphaPCyc[0] * 2) - np.sqrt(
            (alphaPCyc[0] * 2) ** 2 - 4 * np.log(SuH2gy) * (alphaPCyc[0] / aOverB) * 2 ** 2)) / (2 * np.log(SuH2gy))

        alphaI = alphaPCyc[0] / oerIG1
        betaI = (alphaPCyc[0] / aOverB) / (oerIG1 ** 2)
        alphaH = alphaPCyc[0] / oerHG1
        betaH = (alphaPCyc[0] / aOverB) / (oerHG1 ** 2)

        alphaP = alphaPEff
        betaP = betaPEff

        # --- SBRT schedule simulation ---
        fPPro = fPProIn
        cellDist = np.zeros(7)
        cellDist[0] = compSize[0]
        cellDist[2] = compSize[1]
        cellDist[4] = compSize[2]

        t = 0
        j = 0
        cumCellDistSbrt = []

        while t < tStart + (max(treatDay) - 1) + deltaT / 2:
            # Change in fPPro
            fPPro = 1 - 0.5 * (cellDist[0] + cellDist[1]) / compSize[0]

            # RT fraction
            if (t > (tStart + (treatDay[j] - 1) - deltaT / 2)) and (t < (tStart + (treatDay[j] - 1) + deltaT / 2)):
                cellDist[1] += cellDist[0] * (1 - np.exp(-alphaP * d - betaP * d ** 2))
                cellDist[0] *= np.exp(-alphaP * d - betaP * d ** 2)
                cellDist[3] += cellDist[2] * (1 - np.exp(-alphaI * d - betaI * d ** 2))
                cellDist[2] *= np.exp(-alphaI * d - betaI * d ** 2)
                cellDist[5] += cellDist[4] * (1 - np.exp(-alphaH * d - betaH * d ** 2))
                cellDist[4] *= np.exp(-alphaH * d - betaH * d ** 2)
                j += 1

            # Proliferation & death
            cellDist[0] *= (2) ** (fPPro * deltaT / tC)
            hPre = cellDist[4] + cellDist[5]
            cellDist[4] *= (0.5) ** (deltaT / htLoss)
            cellDist[5] *= (0.5) ** (deltaT / htLoss)
            pDPre = cellDist[1]
            cellDist[1] *= (2) ** (fPPro * (2 * kM - 1) * deltaT / tC)

            md = pDPre - cellDist[1] + (hPre - cellDist[4] - cellDist[5])
            cellDist[6] += md
            cellDist[6] *= (0.5) ** (deltaT / htLys)

            # Recompartmentalization
            if cellDist[0] + cellDist[1] >= compSize[0]:
                pEx = (cellDist[0] + cellDist[1]) - compSize[0]
                pRatio = cellDist[0] / (cellDist[0] + cellDist[1])
                cellDist[0] = compSize[0] * pRatio
                cellDist[1] = compSize[0] * (1 - pRatio)
                cellDist[2] += pEx * pRatio
                cellDist[3] += pEx * (1 - pRatio)
            else:
                if cellDist[2] + cellDist[3] > 0:
                    if cellDist[2] + cellDist[3] > compSize[0] - (cellDist[0] + cellDist[1]):
                        pDef = compSize[0] - (cellDist[0] + cellDist[1])
                        iRatio = cellDist[2] / (cellDist[2] + cellDist[3])
                        cellDist[0] += pDef * iRatio
                        cellDist[1] += pDef * (1 - iRatio)
                        cellDist[2] -= pDef * iRatio
                        cellDist[3] -= pDef * (1 - iRatio)
                    else:
                        cellDist[0] += cellDist[2]
                        cellDist[1] += cellDist[3]
                        cellDist[2] = 0
                        cellDist[3] = 0
                        if cellDist[4] + cellDist[5] > 0:
                            if cellDist[4] + cellDist[5] > compSize[0] - (cellDist[0] + cellDist[1]):
                                pDef = compSize[0] - (cellDist[0] + cellDist[1])
                                hRatio = cellDist[4] / (cellDist[4] + cellDist[5])
                                cellDist[0] += pDef * hRatio
                                cellDist[1] += pDef * (1 - hRatio)
                                cellDist[4] -= pDef * hRatio
                                cellDist[5] -= pDef * (1 - hRatio)
                            else:
                                cellDist[0] += cellDist[4]
                                cellDist[1] += cellDist[5]
                                cellDist[4] = 0
                                cellDist[5] = 0

            if cellDist[2] + cellDist[3] >= compSize[1]:
                iEx = (cellDist[2] + cellDist[3]) - compSize[1]
                iRatio = cellDist[2] / (cellDist[2] + cellDist[3])
                cellDist[2] = compSize[1] * iRatio
                cellDist[3] = compSize[1] * (1 - iRatio)
                cellDist[4] += iEx * iRatio
                cellDist[5] += iEx * (1 - iRatio)
            else:
                if cellDist[4] + cellDist[5] > 0:
                    if cellDist[4] + cellDist[5] > compSize[1] - (cellDist[2] + cellDist[3]):
                        iDef = compSize[1] - (cellDist[2] + cellDist[3])
                        hRatio = cellDist[4] / (cellDist[4] + cellDist[5])
                        cellDist[2] += iDef * hRatio
                        cellDist[3] += iDef * (1 - hRatio)
                        cellDist[4] -= iDef * hRatio
                        cellDist[5] -= iDef * (1 - hRatio)
                    else:
                        cellDist[2] += cellDist[4]
                        cellDist[3] += cellDist[5]
                        cellDist[4] = 0
                        cellDist[5] = 0

            # step
            t += deltaT
            cumCellDistSbrt.append(cellDist.copy())
            # --- End SBRT loop ---

        sSbrt = cellDist[0] + cellDist[2] + cellDist[4]

        # --- EQD2 calculation loop ---
        dEqd2 = 2
        alphaP = alphaPOri
        betaP = betaPOri
        alphaI = alphaPOri / oerI
        betaI = betaPOri / (oerI ** 2)
        alphaH = alphaPOri / oerH
        betaH = betaPOri / (oerH ** 2)

        fPPro = fPProIn
        cellDist = np.zeros(7)
        cellDist[0] = compSizeRef[0]
        cellDist[2] = compSizeRef[1]
        cellDist[4] = compSizeRef[2]

        t = 0
        j = 0
        addTime = 0
        cumCellDist = []

        sEqd2 = 0
        eqd2 = 0
        sEqd2Pre = 0
        eqd2Pre = 0

        while (cellDist[0] + cellDist[2] + cellDist[4]) > sSbrt:
            fPPro = 1 - 0.5 * (cellDist[0] + cellDist[1]) / compSize[0]

            if (t > (tStart + j + addTime - deltaT / 2)) and (t < (tStart + j + addTime + deltaT / 2)):
                cellDist[1] += cellDist[0] * (1 - np.exp(-alphaP * dEqd2 - betaP * dEqd2 ** 2))
                cellDist[0] *= np.exp(-alphaP * dEqd2 - betaP * dEqd2 ** 2)
                cellDist[3] += cellDist[2] * (1 - np.exp(-alphaI * dEqd2 - betaI * dEqd2 ** 2))
                cellDist[2] *= np.exp(-alphaI * dEqd2 - betaI * dEqd2 ** 2)
                cellDist[5] += cellDist[4] * (1 - np.exp(-alphaH * dEqd2 - betaH * dEqd2 ** 2))
                cellDist[4] *= np.exp(-alphaH * dEqd2 - betaH * dEqd2 ** 2)

                j += 1
                if j % 5 == 0:
                    addTime += 2

            cellDist[0] *= (2) ** (fPPro * deltaT / tC)
            hPre = cellDist[4] + cellDist[5]
            cellDist[4] *= (0.5) ** (deltaT / htLoss)
            cellDist[5] *= (0.5) ** (deltaT / htLoss)
            pDPre = cellDist[1]
            cellDist[1] *= (2) ** (fPPro * (2 * kM - 1) * deltaT / tC)

            md = pDPre - cellDist[1] + (hPre - cellDist[4] - cellDist[5])
            cellDist[6] += md
            cellDist[6] *= (0.5) ** (deltaT / htLys)

            # same recompt. as above
            if cellDist[0] + cellDist[1] >= compSize[0]:
                pEx = (cellDist[0] + cellDist[1]) - compSize[0]
                pRatio = cellDist[0] / (cellDist[0] + cellDist[1])
                cellDist[0] = compSize[0] * pRatio
                cellDist[1] = compSize[0] * (1 - pRatio)
                cellDist[2] += pEx * pRatio
                cellDist[3] += pEx * (1 - pRatio)
            else:
                if cellDist[2] + cellDist[3] > 0:
                    if cellDist[2] + cellDist[3] > compSize[0] - (cellDist[0] + cellDist[1]):
                        pDef = compSize[0] - (cellDist[0] + cellDist[1])
                        iRatio = cellDist[2] / (cellDist[2] + cellDist[3])
                        cellDist[0] += pDef * iRatio
                        cellDist[1] += pDef * (1 - iRatio)
                        cellDist[2] -= pDef * iRatio
                        cellDist[3] -= pDef * (1 - iRatio)
                    else:
                        cellDist[0] += cellDist[2]
                        cellDist[1] += cellDist[3]
                        cellDist[2] = 0
                        cellDist[3] = 0
                        if cellDist[4] + cellDist[5] > 0:
                            if cellDist[4] + cellDist[5] > compSize[0] - (cellDist[0] + cellDist[1]):
                                pDef = compSize[0] - (cellDist[0] + cellDist[1])
                                hRatio = cellDist[4] / (cellDist[4] + cellDist[5])
                                cellDist[0] += pDef * hRatio
                                cellDist[1] += pDef * (1 - hRatio)
                                cellDist[4] -= pDef * hRatio
                                cellDist[5] -= pDef * (1 - hRatio)
                            else:
                                cellDist[0] += cellDist[4]
                                cellDist[1] += cellDist[5]
                                cellDist[4] = 0
                                cellDist[5] = 0

            if cellDist[2] + cellDist[3] >= compSize[1]:
                iEx = (cellDist[2] + cellDist[3]) - compSize[1]
                iRatio = cellDist[2] / (cellDist[2] + cellDist[3])
                cellDist[2] = compSize[1] * iRatio
                cellDist[3] = compSize[1] * (1 - iRatio)
                cellDist[4] += iEx * iRatio
                cellDist[5] += iEx * (1 - iRatio)


    # Compute TCP
    TCP = TCPUpperBound / (1 + (TD50 / eqd2) ** (4 * gamma50))

    return TCP


def getTreatmentSchedule(nFrx, scheduleType):
    """Return RT treatment days for a given no. of fractions and schedule type.
    Args:
          nFrx: No. of fractions
          scheduleType: Supported values: 'weekday' ,'primershot','primershotopt'.
    Returns:
          Treatment days
    """

    if scheduleType.lower() == 'weekday':
        # Fraction every weekday with weekend breaks
        nWeeks = np.floor(nFrx/5)
        treatmentDays = np.array([])
        for week in range(3):
            treatmentDays = np.append(treatmentDays, 7 * week + np.arange(1, 6))

        remDays = nFrx % 5
        treatmentDays = np.append(treatmentDays, 7 * nWeeks + np.arange(1, remDays+1))

    elif scheduleType.lower() == 'primershot':
        #One initial fraction (termed a "primer shot") followed by a
        #2-week gap for full reoxygenation, with remaining fractions
        #delivered daily

        boost = 1
        remDays = 14 + getTreatmentSchedule(nFrx-1,'weekday')
        treatmentDays = np.append(boost,remDays)

    elif scheduleType.lower() == 'primershotopt':
        frxV = [15, 10, 8, 5] #4Gyx15, 7Gyx10, 7.5Gyx8, 10Gyx5
        optList = [[1, 5, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],\
                   [1, 14, 15, 16, 17, 18, 19, 20, 21, 22],\
                   [1, 13, 14, 15, 16, 17, 18, 19],\
                   [1, 13, 14, 15, 16]]
        matchIdxV = frxV.index(nFrx)
        treatmentDays = optList[matchIdxV]
        treatmentDays = np.array(treatmentDays[0])

    else:
        raise ValueError('Invalid scheduleType ' + scheduleType)
    
    return treatmentDays


def run(modelFile, doseNum, planC, fSizeIn=None, fNumIn=None, binWidth=0.05):
    """
    Evaluate dosimetric model including fractionation correction where applicablegit add.
    Args:
          modelFile: Path to JSON file describing model parameters OR
                     Dictionary of model parameters
          doseNum: Index of dose in planC
          planC: plan container object
          fSizeIn: Fraction size of input plan
          fNumIn: Fraction no. of input plan
          binWidth (float): Bin width for DVH calculation. Default:0.05
    Returns:
        Model-based NTCP.
    """

    optFields = ['numFractions','frxSize','abRatio']
    # Read model parameters
    if isinstance(modelFile, dict):
        model = modelFile
    else:
        with open(modelFile, 'r') as f:
            model = json.load(f)

    dictFields = list(model.keys())
    modelFn = model['function']
    paramDict = model['parameters']

    #Copy optional parameters
    for field in optFields:
        if field in dictFields:
            paramDict[field] = {'val': model[field]}


    fsizeCorr = False
    fnumCorr = False
    if 'fractionCorrect' in dictFields and model['fractionCorrect'].lower() == 'yes':
        if model['correctionType'].lower() == 'frxsize':
            fsizeCorr = True
            stdFsize = model['stdFractionSize']
            inputFrxsize = fSizeIn
        elif model['correctionType'].lower() == 'frxnum':
            fnumCorr = True
            stdFrxNum = model['stdNumFractions']
            inputFrxNum = fNumIn
        abRatio = float(model['abRatio'])


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