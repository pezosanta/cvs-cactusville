import glob
import os
import pickle

import cv2

from detector2 import detectAndClassify, drawBBs
from task3 import get3DPosition
from trainClassifier import sorted_nicely


# model = loadModelParams()


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

    for video in videos_test:
        for image in video:
            img = cv2.imread(image['rgb'])
            depth = cv2.imread(image['depth'], -1)
            img_orig = img.copy()
            img = cv2.GaussianBlur(img, (5, 5), 0)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

            # Task 1
            objects = detectAndClassify(img)

            # Task 2
            for i, obj in enumerate(objects):
                if obj.classInd == 0:
                    x1, y1, x2, y2 = obj.getSubImgCoords(img)
                    sub_img = img[y1:y2, x1:x2]
                    sub_img = cv2.resize(sub_img, (32, 32))
                    # pred = predictTS(sub_img, model, convertToRGB=True)
                    objects[i].subClassInd = 0  # pred

            # Task 3
            objects = [get3DPosition(depth, obj) for obj in objects]

            # Convert to lists
            objects = [obj.makeList() for obj in objects]

            img = img_orig
            img = drawBBs(objects, img)

            cv2.imshow('Test image', img)
            k = cv2.waitKey(0)
            if k == ord('q'):
                break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
