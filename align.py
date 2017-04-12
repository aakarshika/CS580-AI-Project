

import argparse
import cv2
import numpy as np
import os
import random
import shutil
import sys

import openface
import openface.helper
from openface.data import iterImgs

fileDir = "../openface/util/"
modelDir = fileDir+"../models/"
dlibModelDir = modelDir+'dlib/'
openfaceModelDir = modelDir+ 'openface/'


def write(vals, fName):
    if os.path.isfile(fName):
        print("{} exists. Backing up.".format(fName))
        os.rename(fName, "{}.bak".format(fName))
    with open(fName, 'w') as f:
        for p in vals:
            f.write(",".join(str(x) for x in p))
            f.write("\n")


def alignMain():
    openface.helper.mkdirP(outputDir)
    print("Aligned images directory: "+outputDir)

    imgs = list(iterImgs(inputDir))

    print("Training images directory: "+inputDir)
    
    # print(imgs)
    
    # Shuffle so multiple versions can be run at once.
    random.shuffle(imgs)

    landmarkIndices = openface.AlignDlib.OUTER_EYES_AND_NOSE
    # landmarkIndices = openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP

    align = openface.AlignDlib(dlibFacePredictor)

    nFallbacks = 0
    for imgObject in imgs:
        print("Aligning image- "+imgObject.path)
        outDir = outputDir+ "/"+imgObject.cls
        openface.helper.mkdirP(outDir)
        outputPrefix = outDir+"/"+ imgObject.name
        imgName = outputPrefix + ".png"

        if os.path.isfile(imgName):
            # if args.verbose:
            print("      x   Already found, skipping.")
        else:
            rgb = imgObject.getRGB()
            if rgb is None:
                # if args.verbose:
                #     print("  + Unable to load.")
                outRgb = None
            else:
                outRgb = align.align(size, rgb,
                                     landmarkIndices=landmarkIndices,
                                     skipMulti=skipMulti)
                if outRgb is None:
                    print("Unable to align.")

            if outRgb is not None:
                # if args.verbose:
                #     print("  + Writing aligned file to disk.")
                outBgr = cv2.cvtColor(outRgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(imgName, outBgr)

inputDir=sys.argv[1]
skipMulti=True
dlibFacePredictor=dlibModelDir+"shape_predictor_68_face_landmarks.dat"

landmarks="outerEyesAndNose"

outputDir = sys.argv[2]

size=96

alignMain()
