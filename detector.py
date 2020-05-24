import cv2
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC

from nms import non_max_suppression_fast
from trainDetector import hog, winSize, ImgObject


def loadT1Modules(fp='task1.gz'):
    scaler, objDetector, objClassifier = joblib.load(fp)
    assert isinstance(scaler, MinMaxScaler)
    assert isinstance(objDetector, LinearSVC)
    assert isinstance(objDetector, LinearSVC)

    return scaler, objDetector, objClassifier


def saveT1Modules(scaler, detector, classifier, fp='task1.gz'):
    joblib.dump([scaler, detector, classifier], fp)


def slidingWindow(img_shape, minSize=32):  # img_shape = (height, width, channels)
    size1 = np.array((minSize, minSize), dtype=np.uint32)  # (width, height)
    size2 = np.array((3 * minSize, 4 * minSize), dtype=np.uint32) // 2
    size3 = np.array((minSize, 2 * minSize), dtype=np.uint32)
    for searchWinSize in [size2, size1, size3]:
        while searchWinSize[0] < img_shape[0] // 2 and searchWinSize[1] < img_shape[1] // 2:
            for x in range(0, img_shape[1] - searchWinSize[0], 8):
                for y in range(0, img_shape[0] - searchWinSize[1], 8):
                    yield [x, y, x + searchWinSize[0], y + searchWinSize[1]]

            searchWinSize = (searchWinSize * 1.3).astype(np.uint32)


# Classifier code
def classifyBoxes(scaler, classifier, img, bboxes):
    objects = []
    for a in bboxes:
        sub_img = img[a[1]:a[3], a[0]:a[2]]
        sub_img = cv2.resize(sub_img, winSize, cv2.INTER_LANCZOS4)
        des = hog.compute(sub_img)
        des = np.squeeze(des)
        des = scaler.transform(np.expand_dims(des, axis=0))
        des = np.squeeze(des)
        classInd = classifier.predict(np.expand_dims(des, axis=0)).item()

        u = (a[0] + a[2]) // 2
        v = (a[1] + a[3]) // 2
        w = a[2] - a[0]
        h = a[3] - a[1]
        obj = ImgObject([u, v, w, h, classInd, 0, 0, 0, 0])
        objects.append(obj)

    return objects

# Object detector code
def detectAndClassify(img, scaler, detector, classifier):
    bboxes = []
    descriptors = []
    # Multiscale descriptors extraction
    for window in slidingWindow(img.shape):
        x1, y1, x2, y2 = window
        sub_img = cv2.resize(img[y1:y2, x1:x2], winSize, cv2.INTER_LANCZOS4)
        des = hog.compute(sub_img)
        des = np.squeeze(des)
        descriptors.append(des)
        bboxes.append(window)

    # Decide whether they correspond to objects
    descriptors = np.row_stack(descriptors)
    descriptors = scaler.transform(descriptors)
    scores = detector.decision_function(descriptors)
    scores = np.squeeze(scores)
    bboxes = np.array(bboxes)

    # Keep only with scores better than 1 (positive is object found, value is confidence)
    bboxes = bboxes[scores > 2]
    scores = scores[scores > 2]
    a = sorted(list(zip(bboxes, scores)), reverse=True, key=lambda x: x[1])
    bboxes = np.array([x[0] for x in a])
    scores = np.array([x[1] for x in a])

    # Non-max supression
    bboxes = non_max_suppression_fast(bboxes, 0)

    # Classify remaining boxes
    objects = classifyBoxes(scaler, classifier, img, bboxes)

    return objects


if __name__ == '__main__':
    print("No main function!")
