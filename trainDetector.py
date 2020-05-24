import cv2
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC


class ImgObject:
    classNames = ['traffic sign', 'vehicle', 'cactus']

    def __init__(self, obj: list):  # obj = [u, v, w, h, classInd, subClassInd, x, y, z]
        assert len(obj) == 9, "Object list size is not 9!"
        self.u, self.v = obj[0], obj[1]
        self.w, self.h = obj[2], obj[3]
        self.classInd = obj[4]
        self.subClassInd = obj[5]
        self.x, self.y, self.z = obj[6:9]

    def getClassName(self):
        return self.classNames[self.classInd]

    def getSubImgCoords(self, img, scale=1):
        max_y, max_x = img.shape[:2]

        x1 = max(self.u - int(self.w * scale / 2), 0)
        x2 = min(x1 + int(self.w * scale), max_x)
        y1 = max(self.v - int(self.h * scale / 2), 0)
        y2 = min(y1 + int(self.h * scale), max_y)

        return x1, y1, x2, y2

    def makeList(self):
        obj = [0 for _ in range(9)]
        obj[0:2] = self.u, self.v
        obj[2:4] = self.w, self.h
        obj[4] = self.classInd
        obj[5] = self.subClassInd
        obj[6:9] = [self.x, self.y, self.z]
        return obj


# HOG params
winSize = (48, 64)
blockSize = (16, 16)
blockStride = (8, 8)
cellSize = (8, 8)
bins = 9

# HOG feature collect
hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, bins)


def createHeatMap(bboxes, img):
    heatMap = np.zeros_like(img, dtype=np.uint32)
    for box in bboxes:
        x1, y1, x2, y2 = box
        heatMap[y1:y2, x1:x2] += 1
    return heatMap


def heatMapNMS(heatMap):
    pass


def trainObjDetector():
    pass
    raise NotImplementedError()


def trainObjClassifier():
    pass
    raise NotImplementedError()
