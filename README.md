# pyCERR - A Python-based Computational Environment for Radiological Research

pyCERR provides convenient data structure for imaging metadata and their associations. Utilities are provided to to extract, transform, organize metadata and visualize results of image processing for image and dosimetry features, image processing for AI model training and inference.

## Install Miniconda and Git
It is recommended to install CERR in an isolated environment such as Anaconda or VENV from GitHub repository. Please refer to 
1. https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html for installing Miniconda and 
2. https://git-scm.com/downloads for installing Git on your system.

## Install pyCERR

1. Open Git Bash and run the following commands to clone CERR from github.
````    
cd C:\Users\username\software
git clone https://github.com/cerr/pyCERR/pyCERR.git
````    
2. Launch Miniconda terminal, create a Conda environment with Python 3.8 and install CERR. Note that CERR requires Python version >= 3.8 to use Napari Viewer.
````
conda create -y --name testcerr python=3.8
conda activate testcerr
python -m pip install --upgrade pip
pip install C:\Users\username\software\pycerr
import sys
sys.path.insert(0, r'C:\Users\username\software\pycerr')
````    
The above steps will install CERR under ...\envs\testcerr\Lib\site-packages. 

## Example scripts

Run python from the above Anaconda environment and try out the following code samples.

### import modules for planC and viewer
    import numpy as np
    from cerr import plan_container as pc
    from cerr import viewer as vwr

### Read directory contents to planC
    dcm_dir = r"\\path\to\Data\dicom\directory"
    planC = pc.load_dcm_dir(dcm_dir)


### visualize scan, dose and segmentation    
    scan_num = [0]
    dose_num = [0]
    num_structs = len(planC.structure)
    str_num_list = np.arange(num_structs)
    viewer, scan_layer, dose_layer, labels_layer = vwr.show_scan_struct_dose(scan_num, str_num_list, dose_num, planC)


### Compute DVH-based metrics
    from cerr import dvh
    structNum = 0
    doseNum = 0
    dosesV, volsV, isErr = dvh.getDVH(structNum, doseNum, planC)
    binWidth = 0.025
    doseBinsV,volHistV = dvh.doseHist(dosesV, volsV, binWidth)
    percent = 70
    dvh.MOHx(doseBinsV,volHistV,percent)
