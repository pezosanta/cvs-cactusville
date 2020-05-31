import glob
import itertools
import os
import pickle
import re

import cv2
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from detectorHOGSVM.definitions import hog, winSize, ImgObject
from detectorHOGSVM.helpers import getNoObjectImages, getPartialImages, savePipeline


def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def trainTrafficSignClassifier(videos_train, annotations):
    negativeImgs = []
    positiveImgs = []

    for video in videos_train:
        for image in video:
            img = cv2.imread(image['rgb'])
            # img = cv2.GaussianBlur(img, (5,5), 0)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
            # img = np.expand_dims(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), axis=-1)
            max_y, max_x = img.shape[:2]
            objects = annotations[image['rgb']]['objects']
            objects = [ImgObject(obj) for obj in objects]

            # Filter objects for traffic signs
            objects = [obj for obj in objects if obj.classInd == 0]

            noObjImgs = getNoObjectImages(img, objects, winSize=32, ratios=((1, 1), (0.5, 1)))
            negativeImgs.extend(noObjImgs)

            # Save object image and several shifted image
            for obj in objects:
                # image part is ~10% larger than original bounding box
                x1, y1, x2, y2 = obj.getSubImgCoords(img, scale=1)
                sub_img = img[y1:y2, x1:x2]

                partialImages = getPartialImages(sub_img)
                partialImages.extend(getPartialImages(cv2.flip(sub_img, 1)))
                positiveImgs.extend(partialImages)
                # positiveImgs.append(sub_img)
                # positiveImgs.append(cv2.flip(sub_img, 1))

    features = []
    for img in tqdm(itertools.chain(negativeImgs, positiveImgs)):
        imshape = img.shape[:2]
        img = cv2.resize(img, winSize, cv2.INTER_LANCZOS4)
        des = hog.compute(img)
        features.append(des)
    features = np.array(features).squeeze()

    trainData = np.array(features)
    labels = [-1 * np.ones(len(negativeImgs)), np.ones(len(positiveImgs))]
    labels = np.concatenate(labels, axis=0).astype(np.int32)
    print(np.unique(labels, return_counts=True))

    # Train traffic sign classifier
    ts_clf = make_pipeline(StandardScaler(),
                           SGDClassifier(random_state=1, class_weight='balanced', max_iter=3000000, n_jobs=-1))
    ts_clf.fit(trainData, labels)

    acc = ts_clf.score(trainData, labels)
    print(f"Accuracy on training data: {acc * 100:.2f}%")
    return ts_clf


def trainRedVechicleClassifier(videos_train, annotations):
    negativeImgs = []
    positiveImgs = []

    for video in videos_train:
        for image in video:
            img = cv2.imread(image['rgb'])
            # img = cv2.GaussianBlur(img, (5,5), 0)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
            # img = np.expand_dims(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), axis=-1)
            max_y, max_x = img.shape[:2]
            objects = annotations[image['rgb']]['objects']
            objects = [ImgObject(obj) for obj in objects]

            # Filter objects
            objects = [obj for obj in objects if obj.classInd == 1 and obj.subClassInd in [0, 2]]

            noObjImgs = getNoObjectImages(img, objects, winSize=32, ratios=((1, 1), (0.5, 1), (0.75, 1)),
                                          scale_factor=2)
            negativeImgs.extend(noObjImgs)

            # Save object image and several shifted image
            for obj in objects:
                # image part is ~10% larger than original bounding box
                x1, y1, x2, y2 = obj.getSubImgCoords(img, scale=1)
                sub_img = img[y1:y2, x1:x2]

                partialImages = getPartialImages(sub_img)
                partialImages.extend(getPartialImages(cv2.flip(sub_img, 1)))
                positiveImgs.extend(partialImages)
                # positiveImgs.append(sub_img)
                # positiveImgs.append(cv2.flip(sub_img, 1))

    features = []
    for img in tqdm(itertools.chain(negativeImgs, positiveImgs)):
        imshape = img.shape[:2]
        img = cv2.resize(img, winSize, cv2.INTER_LANCZOS4)
        des = hog.compute(img)
        features.append(des)
    features = np.array(features).squeeze()

    trainData = np.array(features)
    labels = [-1 * np.ones(len(negativeImgs)), np.ones(len(positiveImgs))]
    labels = np.concatenate(labels, axis=0).astype(np.int32)
    print(np.unique(labels, return_counts=True))

    # Train red vehicle classifier
    vc_clf = make_pipeline(StandardScaler(),
                           SGDClassifier(random_state=1, class_weight='balanced', max_iter=3000000, n_jobs=-1))
    vc_clf.fit(trainData, labels)

    acc = vc_clf.score(trainData, labels)
    print(f"Accuracy on training data: {acc * 100:.2f}%")
    return vc_clf


def trainLargeClassifier(videos_train, annotations):
    yellowTruckImgs = []
    cactiImgs = []

    for video in videos_train:
        for image in video:
            img = cv2.imread(image['rgb'])
            # img = cv2.GaussianBlur(img, (5,5), 0)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
            # img = np.expand_dims(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), axis=-1)
            max_y, max_x = img.shape[:2]
            objects = annotations[image['rgb']]['objects']
            objects = [ImgObject(obj) for obj in objects]

            # Filter objects
            objects = [obj for obj in objects if obj.classInd == 2 or
                       (obj.classInd == 1 and obj.subClassInd == 1)]

            # Save object image and several shifted image
            for obj in objects:
                # image part is ~10% larger than original bounding box
                x1, y1, x2, y2 = obj.getSubImgCoords(img, scale=1)
                sub_img = img[y1:y2, x1:x2]

                partialImages = getPartialImages(sub_img)
                partialImages.extend(getPartialImages(cv2.flip(sub_img, 1)))
                if obj.classInd == 2:
                    cactiImgs.extend(partialImages)
                    # cactiImgs.append(sub_img)
                    # cactiImgs.append(cv2.flip(sub_img, 1))
                else:
                    yellowTruckImgs.extend(partialImages)
                    # yellowTruckImgs.append(sub_img)
                    # yellowTruckImgs.append(cv2.flip(sub_img, 1))

    features = []
    for img in tqdm(itertools.chain(yellowTruckImgs, cactiImgs)):
        imshape = img.shape[:2]
        img = cv2.resize(img, winSize, cv2.INTER_LANCZOS4)
        des = hog.compute(img)
        features.append(des)
    features = np.array(features).squeeze()

    trainData = np.array(features)
    labels = [-1 * np.ones(len(yellowTruckImgs)), np.ones(len(cactiImgs))]
    labels = np.concatenate(labels, axis=0).astype(np.int32)
    print(np.unique(labels, return_counts=True))

    # Train red vehicle classifier
    large_clf = make_pipeline(StandardScaler(),
                              SGDClassifier(random_state=1, class_weight='balanced', max_iter=3000000, n_jobs=-1))
    large_clf.fit(trainData, labels)

    acc = large_clf.score(trainData, labels)
    print(f"Accuracy on training data: {acc * 100:.2f}%")
    return large_clf


def main(dataPath):
    videoDirs = [f.path for f in os.scandir(dataPath) if f.is_dir()]
    videos = []
    for video in videoDirs:
        depths = sorted_nicely(glob.glob(os.path.join(video, 'depth', '*.png')))
        rgbs = sorted_nicely(glob.glob(os.path.join(video, 'rgb', '*.jpg')))

        assert len(depths) == len(rgbs), "RGB and depth image length not equal!"

        videos.append([])
        for depth, rgb in zip(depths, rgbs):
            videos[-1].append({
                'depth': depth,
                'rgb': rgb
            })

    # Read annotations
    file = open(os.path.join(dataPath, 'annotations.pickle'), 'rb')
    annotations = pickle.load(file)

    videos_train = videos[:3]
    videos_test = videos[3:]
    ts_clf = trainTrafficSignClassifier(videos_train, annotations)
    vc_clf = trainRedVechicleClassifier(videos_train, annotations)
    large_clf = trainLargeClassifier(videos_train, annotations)
    savePipeline(ts_clf, fp='ts_clf.gz')
    savePipeline(vc_clf, fp='vc_clf.gz')
    savePipeline(large_clf, fp='large_clf.gz')


if __name__ == '__main__':
    main('HW')
