import glob
import os
import pickle

import cv2
import numpy as np

from detectorHOGSVM.definitions import hog, winSize, ImgObject, subclassNames
from detectorHOGSVM.helpers import loadPipeline
from task1_bbox_prediction import predict_bboxes
from trainClassifier import sorted_nicely

ts_clf = loadPipeline(fp='ts_clf.gz')
vc_clf = loadPipeline(fp='vc_clf.gz')
large_clf = loadPipeline(fp='large_clf.gz')
subcacti_clf = loadPipeline(fp='subcacti_clf.gz')


# Classifier code
def classifyBoxes(img, tsBoxes, redBoxes, largeBoxes):
    objects = []

    for a in tsBoxes:
        classInd = 0
        sub_img = img[a[1]:a[3], a[0]:a[2]]
        sub_img = cv2.resize(sub_img, winSize, cv2.INTER_LANCZOS4)
        des = hog.compute(sub_img)
        des = np.squeeze(des)
        pred = ts_clf.predict(np.expand_dims(des, axis=0)).item()
        if pred < 0:
            continue

        u = (a[0] + a[2]) // 2
        v = (a[1] + a[3]) // 2
        w = a[2] - a[0]
        h = a[3] - a[1]
        obj = ImgObject([u, v, w, h, classInd, 0, 0, 0, 0])
        objects.append(obj)

    for a in redBoxes:
        classInd = 1
        sub_img = img[a[1]:a[3], a[0]:a[2]]
        sub_img = cv2.resize(sub_img, winSize, cv2.INTER_LANCZOS4)
        des = hog.compute(sub_img)
        des = np.squeeze(des)
        pred = vc_clf.predict(np.expand_dims(des, axis=0)).item()
        if pred < 0:
            continue

        u = (a[0] + a[2]) // 2
        v = (a[1] + a[3]) // 2
        w = a[2] - a[0]
        h = a[3] - a[1]
        obj = ImgObject([u, v, w, h, classInd, 0, 0, 0, 0])
        objects.append(obj)

    for a in largeBoxes:
        classInd, subClassInd = 2, 1

        sub_img = img[a[1]:a[3], a[0]:a[2]]
        sub_img = cv2.resize(sub_img, winSize, cv2.INTER_LANCZOS4)
        des = hog.compute(sub_img)
        des = np.squeeze(des)
        pred = large_clf.predict(np.expand_dims(des, axis=0)).item()
        if pred == 0:
            continue
        elif pred == -1:
            classInd = 1

        u = (a[0] + a[2]) // 2
        v = (a[1] + a[3]) // 2
        w = a[2] - a[0]
        h = a[3] - a[1]
        obj = ImgObject([u, v, w, h, classInd, subClassInd, 0, 0, 0])
        objects.append(obj)

    return objects


def classifyCacti(img, objects):
    for i, obj in enumerate(objects):
        if obj.classInd == 2:
            x1, y1, x2, y2 = obj.getSubImgCoords(img, scale=1)
            sub_img = img[y1:y2, x1:x2]
            sub_img = cv2.resize(sub_img, winSize, cv2.INTER_LANCZOS4)
            des = hog.compute(sub_img)
            des = np.squeeze(des)
            pred = subcacti_clf.predict(np.expand_dims(des, axis=0)).item()
            objects[i].subClassInd = int(pred)

    return objects


# Object detector code
def detectAndClassify(img):
    # Detect
    tsBoxes, redBoxes, largeBoxes = predict_bboxes(img)

    # Filter and classify
    objects = classifyBoxes(img, tsBoxes, redBoxes, largeBoxes)

    # TODO: find subclasses where possible
    objects = classifyCacti(img, objects)

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
            depth = cv2.imread(image['depth'], -1)
            img_orig = img.copy()
            objects = annotations[image['rgb']]['objects']
            objects = [ImgObject(obj) for obj in objects]
            total += len(objects)

            objects = detectAndClassify(img)

            # Convert to lists
            objects = [obj.makeList() for obj in objects]

            img = img_orig.copy()
            img = drawBBs(objects, img)
            totalPredicted += len(objects)
            cv2.imshow('Test image', img)
            k = cv2.waitKey(0)
            if k == ord('q'):
                break
    cv2.destroyAllWindows()

    print(f"Number of test objects: {total}, number of predictions: {totalPredicted}")
    print(f"Found/existing ratio: {totalPredicted / (total + 1e-3) * 100:.2f}")


if __name__ == '__main__':
    main()
