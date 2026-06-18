import pydicom
from pathlib import Path


def dump_tags(ptDirName, outFileName):
    numericVR = ['US','UL','OV','SS','SL','OW','OB','FL','FD','IS','DS']
    pathObj = Path(ptDirName)
    allDcmFiles = [str(file_path) for file_path in pathObj.rglob("*.dcm")]
    with open(outFileName, 'a') as file:
        for f in allDcmFiles:
            file.write(f+'\n')
            dataset = pydicom.dcmread(f)
            # Iterate through all elements and write them
            for dataElem in dataset:
                if dataElem.VR not in numericVR:
                    txtToWrite = f"{dataElem.name}: {dataElem.value}\n"
                    file.write(txtToWrite)


# --- Main runner ---
if __name__ == '__main__':
    #ptDirName = sys.argv[1]
    #outFileName = sys.argv[2]
    ptDirName = r'N:\data\ROBIN\MCT1\15515009\15515009\20200627-1.2.840.113854.95877871384576583446023396715056731041\CT'
    outFileName = r'N:\data\ROBIN\MCT1\dicom_tag_dumps\15515009_20200627-1.2.840.113854.95877871384576583446023396715056731041_CT.txt'
    dump_tags(ptDirName, outFileName)

