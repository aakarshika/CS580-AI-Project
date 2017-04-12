
import time

import argparse
import cv2
import os
import pickle
import sys

from operator import itemgetter

import numpy as np
np.set_printoptions(precision=2)
import pandas as pd

import openface



def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from sklearn.pipeline import Pipeline
from sklearn.lda import LDA
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.mixture import GMM
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

fileDir = "../openface/"
modelDir = fileDir+'models/'
dlibModelDir = "./files/"
openfaceModelDir = modelDir+'openface/'
workDir = "./generated-embeddings/"
classifierModel=workDir+"classifier.pkl"


def getRep(imgPath, multiple=False):
    bgrImg = cv2.imread(imgPath)
    if bgrImg is None:
        raise Exception("Unable to load image: "+imgPath)

    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
    print("image loaded")

    if multiple:
        bbs = align.getAllFaceBoundingBoxes(rgbImg)
    else:
        bb1 = align.getLargestFaceBoundingBox(rgbImg)
        bbs = [bb1]
    if len(bbs) == 0 or (not multiple and bb1 is None):
        raise Exception("Unable to find a face: {}".format(imgPath))
    reps = []
    for bb in bbs:
        alignedFace = align.align(
            96,
            rgbImg,
            bb,
            landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        if alignedFace is None:
            print("Unable to align image: "+imgPath)
        rep = net.forward(alignedFace)
        reps.append((bb.center().x, rep))
    sreps = sorted(reps, key=lambda x: x[0])
    return sreps


def train():
    print("Loading embeddings")
    fname = "{}/labels.csv".format(workDir)
    labels = pd.read_csv(fname, header=None).as_matrix()[:, 1]
    labels = map(itemgetter(1),
                 map(os.path.split,
                     map(os.path.dirname, labels)))  # Get the directory.
    fname = "{}/reps.csv".format(workDir)
    embeddings = pd.read_csv(fname, header=None).as_matrix()
    le = LabelEncoder().fit(labels)
    labelsNum = le.transform(labels)
    nClasses = len(le.classes_)
    print("Training for {} classes.".format(nClasses))
    
    # classifier- linear svm
    clf = SVC(C=1, kernel='linear', probability=True)

    clf.fit(embeddings, labelsNum)

    fName = classifierModel
    
    with open(fName, 'w') as f:
        pickle.dump((le, clf), f)
    print("pickled:---------classifier.pkl saved to "+fName)
    

def infer(imgPath, multiple=False):
    with open(classifierModel, 'r') as f:
        (le, clf) = pickle.load(f)
    print("image at: "+imgPath)
    reps = getRep(imgPath, False)
    for r in reps:
        rep = r[1].reshape(1, -1)
        bbx = r[0]
    predictions = clf.predict_proba(rep).ravel()
    maxI = np.argmax(predictions)
    person = le.inverse_transform(maxI)
    confidence = predictions[maxI]
    print("-------------------------------------------------------------------------------")
    print("We predicted \""+str(person)+"\" in this image with "+str(confidence)+" confidence!!")
    print("-------------------------------------------------------------------------------")


# awesomeness

dlibFacePredictor=  dlibModelDir+"shape_predictor_68_face_landmarks.dat"
torchNetworkModel= openfaceModelDir +"nn4.small2.v1.t7"
align = openface.AlignDlib(dlibFacePredictor)
net = openface.TorchNeuralNet(torchNetworkModel, 96, False)
if sys.argv[1]=='train':
    train()
elif sys.argv[1]=='infer':
    infer(sys.argv[2])