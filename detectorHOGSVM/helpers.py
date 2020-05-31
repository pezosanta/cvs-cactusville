import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC


def slidingWindow(img_shape, winSize=32, ratios=((1, 1),), stride_ratio=1.0,
                  scale_factor=2.0, maxWinSize=None):
    window_shapes = []
    for ratio in ratios:
        size = (winSize * ratio[0], winSize * ratio[1])
        size = tuple((int(round(dim_size)) for dim_size in size))
        size = np.array(size, dtype=np.uint32)
        window_shapes.append(size)

    for searchWinSize in window_shapes:
        stride = int(round(winSize * stride_ratio))
        while searchWinSize[0] <= img_shape[0] and searchWinSize[1] <= img_shape[1]:
            if maxWinSize is not None and np.max(searchWinSize) > maxWinSize:
                break

            for x in range(0, img_shape[1] - searchWinSize[0] + 1, stride):
                for y in range(0, img_shape[0] - searchWinSize[1] + 1, stride):
                    yield [x, y, x + searchWinSize[0], y + searchWinSize[1]]

            searchWinSize = (searchWinSize * scale_factor).astype(np.uint32)
            stride = int(round(stride * scale_factor))


def getNoObjectImages(img, objects, winSize=32, ratios=((1, 1),),
                      stride_ratio=1.0, scale_factor=2.0, maxWinSize=None):
    binary_img = np.ones_like(img).astype(np.uint8)[:, :, 0] * 255
    for obj in objects:
        x1, y1, x2, y2 = obj.getSubImgCoords(binary_img)
        binary_img[y1:y2, x1:x2] = 0

    foundImgs = []
    for window in slidingWindow(binary_img.shape, winSize=winSize, ratios=ratios,
                                stride_ratio=stride_ratio, scale_factor=scale_factor,
                                maxWinSize=maxWinSize):
        x1, y1, x2, y2 = window
        if np.all(binary_img[y1:y2, x1:x2] == 255):
            foundImgs.append(img[y1:y2, x1:x2])

    return foundImgs


def getPartialImages(img):
    # Create partial images and distorted ones
    retImgs = []
    max_y, max_x = img.shape[0:2]
    scale = 0.1

    # Center crop at decreasing scales
    for i in range(0, 3):
        cscale = scale * i
        indices = [max_y * cscale, max_y * (1 - cscale), max_x * cscale, max_x * (1 - cscale)]
        indices = [int(round(idx)) for idx in indices]
        retImgs.append(img[indices[0]:indices[1], indices[2]:indices[3]])

    # Top crop
    for i in range(1, 3):
        cscale = scale * i
        indices = [0, max_y * (1 - cscale), 0, max_x]
        indices = [int(round(idx)) for idx in indices]
        retImgs.append(img[indices[0]:indices[1], indices[2]:indices[3]])

    # Bottom crop
    for i in range(1, 3):
        cscale = scale * i
        indices = [max_y * cscale, max_y, 0, max_x]
        indices = [int(round(idx)) for idx in indices]
        retImgs.append(img[indices[0]:indices[1], indices[2]:indices[3]])

    # Left crop
    for i in range(1, 3):
        cscale = scale * i
        indices = [0, max_y, 0, max_x * (1 - cscale)]
        indices = [int(round(idx)) for idx in indices]
        retImgs.append(img[indices[0]:indices[1], indices[2]:indices[3]])

    # Right crop
    for i in range(1, 3):
        cscale = scale * i
        indices = [0, max_y, max_x * cscale, max_x]
        indices = [int(round(idx)) for idx in indices]
        retImgs.append(img[indices[0]:indices[1], indices[2]:indices[3]])

    return retImgs


def loadT1Modules(fp='task1.gz'):
    scaler_ts, scaler_vc, svm_ts, svm_vc = joblib.load(fp)
    assert isinstance(scaler_ts, MinMaxScaler)
    assert isinstance(scaler_vc, MinMaxScaler)
    assert isinstance(svm_ts, LinearSVC)
    assert isinstance(svm_vc, LinearSVC)

    return scaler_ts, scaler_vc, svm_ts, svm_vc


def saveT1Modules(scaler_ts, scaler_vc, svm_ts, svm_vc, fp='task1.gz'):
    joblib.dump([scaler_ts, scaler_vc, svm_ts, svm_vc], fp)


def loadPipeline(fp='task1.gz'):
    pipeline = joblib.load(fp)
    assert isinstance(pipeline, Pipeline)

    return pipeline


def savePipeline(pipeline, fp='task1.gz'):
    joblib.dump(pipeline, fp)
