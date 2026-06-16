from cerr import plan_container as pc
import os

if __name__ == "__main__":

    # Stanford Lung dataset
    dcm_dir = r"\\vpensmph\deasylab2\Aditya\Cornell_AI_Imaging_course\lung_dicom"
    nii_dir = r"\\vpensmph\deasylab2\Aditya\Cornell_AI_Imaging_course\lung_nii"

    ptDirs = []
    for d in os.scandir(dcm_dir):
        ptDir = d.path
        _, id = os.path.split(ptDir)
        ptNiiDir = os.path.join(nii_dir, id)
        if not os.path.exists(ptNiiDir):
            os.mkdir(ptNiiDir)
        else:
            continue
        planC = pc.loadDcmDir(ptDir)
        scanNiiPath = os.path.join(ptNiiDir, 'CT.nii.gz')
        maskNiiPath = os.path.join(ptNiiDir, 'tumor.nii.gz')
        planC.scan[0].saveNii(scanNiiPath)
        planC.structure[0].saveNii(maskNiiPath, planC)
