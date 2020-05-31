import glob
import os
import pickle

import cv2
import numpy as np
from definitions import hog, winSize, ImgObject, subclassNames
from helpers import slidingWindow, loadT1Modules
from nms import non_max_suppression_fast
# TODO
from trainDetector import sorted_nicely

scaler_ts, scaler_vc, svm_ts, svm_vc = loadT1Modules()


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


def getTrafficSignBoxes(img):
    bboxes = []
    descriptors = []
    scores = []
    # Multiscale descriptors extraction
    for window in slidingWindow(img.shape, scale_factor=1.25, maxWinSize=80, stride_ratio=0.25):
        x1, y1, x2, y2 = window
        sub_img = cv2.resize(img[y1:y2, x1:x2], winSize, cv2.INTER_LANCZOS4)
        des = hog.compute(sub_img)
        des = np.squeeze(des)
        descriptors.append(des)
        bboxes.append(window)

    # Decide whether they correspond to objects
    descriptors = np.row_stack(descriptors)
    descriptors = scaler_ts.transform(descriptors)
    scores = svm_ts.decision_function(descriptors)
    scores = np.squeeze(scores)
    bboxes = np.array(bboxes)

    # Keep only with scores better than 1 (positive is object found, value is confidence)
    bboxes = bboxes[scores > 1]
    scores = scores[scores > 1]
    a = sorted(list(zip(bboxes, scores)), reverse=True, key=lambda x: x[1])
    bboxes = np.array([x[0] for x in a])
    scores = np.array([x[1] for x in a])

    # Non-max supression
    if False:
        heatMap = createHeatMap(bboxes, img)
        cv2_imshow(img)
        cv2_imshow(heatMap)
        plt.plot(sorted(scores))
        plt.show()
        stop
    bboxes = non_max_suppression_fast(bboxes, 0)

    return bboxes


def getVehicleCactiBoxes(img):
    bboxes = []
    descriptors = []
    scores = []
    # Multiscale descriptors extraction
    for window in slidingWindow(img.shape, scale_factor=1.25, winSize=64, stride_ratio=0.2,
                                ratios=((0.75, 1), (0.5, 1), (1, 1))):
        x1, y1, x2, y2 = window
        sub_img = cv2.resize(img[y1:y2, x1:x2], winSize, cv2.INTER_LANCZOS4)
        des = hog.compute(sub_img)
        des = np.squeeze(des)
        descriptors.append(des)
        bboxes.append(window)

    # Decide whether they correspond to objects
    descriptors = np.row_stack(descriptors)
    descriptors = scaler_vc.transform(descriptors)
    scores = svm_vc.decision_function(descriptors)
    scores = np.squeeze(scores)
    bboxes = np.array(bboxes)

    # Keep only with scores better than 1 (positive is object found, value is confidence)
    bboxes = bboxes[scores > 1]
    scores = scores[scores > 1]
    a = sorted(list(zip(bboxes, scores)), reverse=True, key=lambda x: x[1])
    bboxes = np.array([x[0] for x in a])
    scores = np.array([x[1] for x in a])

    # Non-max supression
    if False:
        heatMap = createHeatMap(bboxes, img)
        cv2_imshow(img)
        cv2_imshow(heatMap)
        plt.plot(sorted(scores))
        plt.show()
        stop
    bboxes = non_max_suppression_fast(bboxes, 0)

    return bboxes


# Object detector code
def detectAndClassify(img):
    # Find traffic signs
    bboxes1 = getTrafficSignBoxes(img)
    bboxes2 = getVehicleCactiBoxes(img)

    # Classify remaining boxes
    objects = []
    for classInd, boxes in enumerate([bboxes1, bboxes2]):
        for a in boxes:
            u = (a[0] + a[2]) // 2
            v = (a[1] + a[3]) // 2
            w = a[2] - a[0]
            h = a[3] - a[1]
            obj = ImgObject([u, v, w, h, classInd, 0, 0, 0, 0])
            objects.append(obj)

    return objects


def drawBBs(BBs, img):
    img = cv2.resize(img, (1280, 960))
    colors = [(0, 0, 255), (255, 0, 255), (0, 255, 0)]
    for BB in BBs:
        u = BB[0] * 2
        v = BB[1] * 2
        w = BB[2] * 2
        h = BB[3] * 2
        c = BB[4]
        sc = BB[5]
        x = BB[6]
        y = BB[7]
        z = BB[8]
        s = (u - w // 2, v - h // 2)
        e = (u + w // 2, v + h // 2)
        cv2.rectangle(img, s, e, colors[c], 1)
        tl = (s[0], s[1] + 15)
        bl = (s[0], e[1] - 5)
        cv2.putText(img, subclassNames[c][sc], tl, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.75, colors[c])
        coords = "(%.2f, %.2f, %.2f)" % (x, y, z)
        cv2.putText(img, coords, bl, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.65, colors[c])

    return img


def main():
    videoDirs = [f.path for f in os.scandir('HW') if f.is_dir()]
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
    file = open(os.path.join('HW', 'annotations.pickle'), 'rb')
    annotations = pickle.load(file)

    videos_train = videos[:3]
    videos_test = videos[3:]

    correct = 0
    total = 0
    totalPredicted = 0
    for video in videos_test:
        for image in video:
            img = cv2.imread(image['rgb'])
            img_orig = img.copy()
            img = cv2.GaussianBlur(img, (5, 5), 0)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
            objects = annotations[image['rgb']]['objects']
            objects = [ImgObject(obj) for obj in objects]
            total += len(objects)

            objects = detectAndClassify(img)
            objects = [obj.makeList() for obj in objects]

            img = img_orig.copy()
            img = drawBBs(objects, img)
            totalPredicted += len(objects)
            cv2.imshow('Test image', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    print(f"Number of test objects: {total}, number of predictions: {totalPredicted}")
    print(f"Found/existing ratio: {totalPredicted / (total + 1e-3) * 100:.2f}")


if __name__ == '__main__':
    main()
