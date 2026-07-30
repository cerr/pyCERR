"""Example: generate a consensus structure from multiple observer contours.

Usage:
    python -m cerr.scripts.structure_consensus_example <dicomDir> [strNum ...]

Loads a DICOM directory into a pyCERR plan container, compares the selected
observer structures (STAPLE, Fleiss' kappa, agreement/volume statistics) and
writes a STAPLE-majority consensus structure back into the plan container.

If no structure indices are given, every structure on the first scan is used.

The same functionality is available interactively in the pyCERR viewer under
Tools > "Structure consensus (compare/STAPLE)...".
"""

import sys

import cerr.plan_container as pc
import cerr.dataclasses.scan as scn
from cerr.contour import structure_consensus as sc


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1

    dcmDir = argv[1]
    planC = pc.loadDcmDir(dcmDir)

    if len(argv) > 2:
        structNumV = [int(a) for a in argv[2:]]
    else:
        # Default: all structures associated with the first scan.
        structNumV = [i for i, st in enumerate(planC.structure)
                      if scn.getScanNumFromUID(st.assocScanUID, planC) == 0]

    if len(structNumV) < 2:
        print("Need at least two structures on the same scan to build a "
              "consensus; found: %s" % structNumV)
        return 1

    print("Comparing structures %s: %s"
          % (structNumV, [planC.structure[s].structureName
                          for s in structNumV]))

    result = sc.compareStructures(structNumV, planC)
    print()
    print(sc.summaryText(result))

    # Add a STAPLE consensus (probability >= 0.5) and a simple majority vote.
    planC, stapleNum = sc.createConsensusStructure(
        structNumV, planC, method="staple", threshold=0.5, result=result)
    planC, majNum = sc.createConsensusStructure(
        structNumV, planC, method="majority", result=result)
    print("\nAdded consensus structures:")
    print("  %d: %s" % (stapleNum, planC.structure[stapleNum].structureName))
    print("  %d: %s" % (majNum, planC.structure[majNum].structureName))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
