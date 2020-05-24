import cv2
import numpy as np

from trainDetector import ImgObject, winSize, hog


def slidingWindow(img_shape, minSize=32):
    size1 = np.array((minSize, minSize), dtype=np.uint32)
    size2 = np.array((3 * minSize, 4 * minSize), dtype=np.uint32) // 4
    size3 = np.array((minSize, 2 * minSize), dtype=np.uint32)
    for searchWinSize in [size2]:  # , size2, size3]:
        while searchWinSize[0] < img.shape[0] and searchWinSize[1] < img.shape[1]:
            for x in range(0, img.shape[1] - searchWinSize[0], 8):
                for y in range(0, img.shape[0] - searchWinSize[1], 8):
                    yield [x, y, x + searchWinSize[0], y + searchWinSize[1]]

            searchWinSize *= 2


def getNoObjectImages(img, objects: ImgObject):
    binary_img = np.ones_like(img).astype(np.uint8)[:, :, 0] * 255
    for obj in objects:
        x1 = max(obj.u - obj.w // 2, 0)
        x2 = x1 + obj.w
        y1 = max(obj.v - obj.h // 2, 0)
        y2 = y1 + obj.h
        binary_img[y1:y2, x1:x2] = 0

    foundImgs = []
    foundImgCount = 0
    notFoundImg = 0
    for window in slidingWindow(img.shape, minSize=128):
        x1, y1, x2, y2 = window
        if (np.all(binary_img[y1:y2, x1:x2] == 255)):
            foundImgCount += 1
            foundImgs.append(img[y1:y2, x1:x2])
        else:
            notFoundImg += 1
    return foundImgs


def getPartialImages(img):
    # Create partial images and distorted ones
    retImgs = []
    max_y, max_x = img.shape[0:2]
    scale = 0.1

    # Center crop at decreasing scales
    for i in range(1, 3):
        cscale = scale * i
        indices = [max_y * cscale, max_y * (1 - cscale), max_x * cscale, max_x * (1 - cscale)]
        indices = [int(round(idx)) for idx in indices]
        retImgs.append(img[indices[0]:indices[1], indices[2]:indices[3]])

    # Top crop
    for i in range(1, 3):
        cscale = scale * i
        indices = [max_x * scale, max_y * (1 - cscale), max_x * scale, max_x * (1 - scale)]
        indices = [int(round(idx)) for idx in indices]
        retImgs.append(img[indices[0]:indices[1], indices[2]:indices[3]])

    # Bottom crop
    for i in range(1, 3):
        cscale = scale * i
        indices = [max_y * cscale, max_y, max_x * scale, max_x * (1 - scale)]
        indices = [int(round(idx)) for idx in indices]
        retImgs.append(img[indices[0]:indices[1], indices[2]:indices[3]])

    # Left crop
    for i in range(1, 3):
        cscale = scale * i
        indices = [max_y * scale, max_y * (1 - scale), 0, max_x * (1 - cscale)]
        indices = [int(round(idx)) for idx in indices]
        retImgs.append(img[indices[0]:indices[1], indices[2]:indices[3]])

    # Right crop
    for i in range(1, 3):
        cscale = scale * i
        indices = [max_y * scale, max_y * (1 - scale), max_x * cscale, max_x]
        indices = [int(round(idx)) for idx in indices]
        retImgs.append(img[indices[0]:indices[1], indices[2]:indices[3]])

    # for _img in retImgs:
    # cv2_imshow(_img)

    return retImgs


def getTrainingData(imgs, objects, imgsPerClass):
    pass  # TODO
    HOGfeaturesPerClass = [[] for _ in range(len(imgsPerClass))]
    for clsIdx, imgsForClass in enumerate(imgsPerClass):
        for img in imgsForClass:
            img = cv2.resize(img, winSize, cv2.INTER_LANCZOS4)
            HOGfeaturesPerClass[clsIdx].append(hog.compute(img))

    trainData = []
    for i in range(len(HOGfeaturesPerClass)):
        trainData.append(np.array(HOGfeaturesPerClass[i]).squeeze())
    labels = []
    for i, feats in enumerate(trainData):
        labels.append(np.ones(len(feats)) * i)

    trainData = np.concatenate(trainData, axis=0)
    labels = np.concatenate(labels, axis=0).astype(np.int32)
    print(np.unique(labels, return_counts=True))

    return []
