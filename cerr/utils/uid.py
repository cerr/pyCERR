import random
from datetime import datetime

def createUID(modality):
    modality = modality.upper()
    if modality == 'SCAN':
        modality = 'CT'
    elif modality == 'STRUCTURE':
        modality = 'RS'
    elif modality == 'DOSE':
        modality = 'RD'
    elif modality == 'DVH':
        modality = 'DH'
    elif modality == 'STRUCTURESET':
        modality = 'SA'  # structure Array
    elif modality == 'IVH':  # Intensity Volume Histogram
        modality = 'IH'
    elif modality == 'BEAMS':
        modality = 'RP'
    elif modality == 'CERR':
        modality = 'CERR'
    elif modality == 'DEFORM':
        modality = 'DIR'
    elif modality == 'ANNOTATION':
        modality = 'PR'
    elif modality == 'REGISTRATION':
        modality = 'REG'
    elif modality == 'IM':
        modality = 'IM'
    elif modality == 'BEAM':
        modality = 'BM'
    elif modality == 'TEXTURE':
        modality = 'TXTR'
    elif modality == 'FEATURESET':
        modality = 'FEAT'
    elif modality == 'SEGLABEL':
        modality = 'SEGLABEL'

    randNum = random.randint(1000, 9999)
    current_datetime = datetime.now()
    datetime_str = current_datetime.strftime("%d%m%Y.%H%M%S.%f")
    return modality + "." + datetime_str + "." + str(randNum)[:3]
