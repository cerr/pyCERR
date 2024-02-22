import os
import random
from datetime import datetime


def createSessionDir(sessionPath, inputDicomPath):
    # Create session directory for temporary files

    if inputDicomPath[-1] == os.sep:
        _, folderNam = os.path.split(inputDicomPath[:-1])
    else:
        _, folderNam = os.path.split(inputDicomPath)

    dateTimeV = datetime.now().timetuple()[:6]
    randStr = f"{random.random() * 1000:.3f}"
    sessionDir = f"session{folderNam}{dateTimeV[3]}{dateTimeV[4]}{dateTimeV[5]}{randStr}"
    fullSessionPath = os.path.join(sessionPath, sessionDir)

    # Create sub-directories
    if not os.path.exists(sessionPath):
        os.mkdir(sessionPath)
    os.mkdir(fullSessionPath)
    cerrPath = os.path.join(fullSessionPath, "dataCERR")
    os.mkdir(cerrPath)
    modInputPath = os.path.join(fullSessionPath, "inputNii")
    os.mkdir(modInputPath)
    modOutputPath = os.path.join(fullSessionPath, "outputNii")
    os.mkdir(modOutputPath)
    AIoutputPath = os.path.join(fullSessionPath, "AIoutput")
    os.mkdir(AIoutputPath)

    return modInputPath, modOutputPath
