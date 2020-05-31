import glob
import itertools
import os
import pickle
import re

import cv2
import numpy as np
from definitions import hog, winSize, ImgObject
from helpers import getNoObjectImages, getPartialImages, saveT1Modules
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from tqdm import tqdm


def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def createHeatMap(bboxes, img):
    heatMap = np.zeros_like(img[:, :, 0], dtype=np.float32)
    for box in bboxes:
        x1, y1, x2, y2 = box
        heatMap[y1:y2, x1:x2] += 1
    heatMap = np.rint(heatMap / np.max(heatMap) * 255).astype(np.uint8)
    return heatMap


def heatMapNMS(heatMap):
    pass


def trainTrafficSignDetector(videos_train, annotations):
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
            objects = [obj for obj in objects if obj.classInd == 0]

            noObjImgs = getNoObjectImages(img, objects, winSize=32, ratios=((1, 1), (0.5, 1), (0.75, 1)))
            negativeImgs.extend(noObjImgs)

            # Save object image and several shifted image
            for obj in objects:
                # image part is ~10% larger than original bounding box
                x1, y1, x2, y2 = obj.getSubImgCoords(img, scale=1)
                sub_img = img[y1:y2, x1:x2]

                partialImages = getPartialImages(sub_img)
                partialImages.extend(getPartialImages(cv2.flip(sub_img, 1)))
                # positiveImgs.extend(partialImages)
                positiveImgs.append(sub_img)
                positiveImgs.append(cv2.flip(sub_img, 1))

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

    scaler_ts = MinMaxScaler()
    trainData = scaler_ts.fit_transform(trainData)

    # Train traffic sign classifier
    svm_ts = LinearSVC(class_weight='balanced', max_iter=30000, dual=False, C=10)
    svm_ts.fit(trainData, labels)

    preds = svm_ts.predict(trainData)
    total = len(preds)
    correct = sum(preds == labels)

    print(f"Number of train descriptors: {total}, number of correct predictions: {correct}")
    print(f"Accuracy on training data: {correct / (total + 1e-3) * 100:.2f}%")
    return scaler_ts, svm_ts


def trainVechicleCactiDetector(videos_train, annotations):
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
            objects = [obj for obj in objects if obj.classInd != 0]

            noObjImgs = getNoObjectImages(img, objects, winSize=32, ratios=((1, 1), (0.5, 1), (0.75, 1)))
            negativeImgs.extend(noObjImgs)

            # Save object image and several shifted image
            for obj in objects:
                # image part is ~10% larger than original bounding box
                x1, y1, x2, y2 = obj.getSubImgCoords(img, scale=1)
                sub_img = img[y1:y2, x1:x2]

                partialImages = getPartialImages(sub_img)
                partialImages.extend(getPartialImages(cv2.flip(sub_img, 1)))
                # positiveImgs.extend(partialImages)
                positiveImgs.append(sub_img)
                positiveImgs.append(cv2.flip(sub_img, 1))

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

    scaler_vc = MinMaxScaler()
    trainData = scaler_vc.fit_transform(trainData)

    # Train traffic sign classifier
    svm_vc = LinearSVC(class_weight='balanced', max_iter=30000, dual=False, C=10)
    svm_vc.fit(trainData, labels)

    preds = svm_vc.predict(trainData)
    total = len(preds)
    correct = sum(preds == labels)

    print(f"Number of train descriptors: {total}, number of correct predictions: {correct}")
    print(f"Accuracy on training data: {correct / (total + 1e-3) * 100:.2f}%")

    return scaler_vc, svm_vc


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
    scaler_ts, svm_ts = trainTrafficSignDetector(videos_train, annotations)
    scaler_vc, svm_vc = trainVechicleCactiDetector(videos_train, annotations)
    saveT1Modules(scaler_ts, scaler_vc, svm_ts, svm_vc)


if __name__ == '__main__':
    main('HW')
