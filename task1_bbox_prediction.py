import glob
import cv2
import numpy as np
from task1_bbox_prediction_utils import predictBlueTSBoundingBox, predictRedTSBoundingBox, predictYellowTSBoundingBox, predictLargeRedBoundingBox, predictLargeBoundingBox

def predict_bboxes(image):
    img = np.copy(image)

    cannyLowerThres = 10
    cannyUpperThres = 50

    brown_lower = np.array([10,100,20], dtype=np.uint8)
    brown_upper = np.array([20,255,200], dtype=np.uint8)

    yellow_lower = np.array([20, 100, 100], dtype = np.uint8)
    yellow_upper = np.array([30, 255, 255], dtype = np.uint8)

    red_lower1 = np.array([0,70,50], dtype=np.uint8)
    red_upper1 = np.array([10, 255, 255], dtype=np.uint8)
    red_lower2 = np.array([170,70,50], dtype=np.uint8)
    red_upper2 = np.array([180, 255, 200], dtype=np.uint8)

    blue_lower = np.array([70,40,0],np.uint8) #100, 150
    blue_upper = np.array([140,255,255],np.uint8)

    blueBB = predictBlueTSBoundingBox(img, blue_lower, blue_upper, cannyLowerThres, cannyUpperThres)
    redBB = predictRedTSBoundingBox(img, blueBB, red_lower1, red_upper1, red_lower2, red_upper2, cannyLowerThres, cannyUpperThres)
    yellowBB = predictYellowTSBoundingBox(img, blueBB, redBB, yellow_lower, yellow_upper, cannyLowerThres, cannyUpperThres)#
    redLargeBB = predictLargeRedBoundingBox(img, red_lower1, red_upper1, red_lower2, red_upper2, cannyLowerThres, cannyUpperThres)#
    largeBB = predictLargeBoundingBox(img, blueBB, redBB, yellowBB, brown_lower, brown_upper, blue_lower, blue_upper, cannyLowerThres, cannyUpperThres)

    return yellowBB, redLargeBB, largeBB

def test():
    rgbImagePath1 	= glob.glob('../CVS_HW_FILES/g1/rgb/*.jpg')		# 20 db
    rgbImagePath2 	= glob.glob('../CVS_HW_FILES/g2/rgb/*.jpg')		# 16 db
    rgbImagePath3 	= glob.glob('../CVS_HW_FILES/g3/rgb/*.jpg')		# 19 db
    rgbImagePath4 	= glob.glob('../CVS_HW_FILES/g4/rgb/*.jpg')		# 15 db

    trainPath 		= rgbImagePath1 + rgbImagePath2 + rgbImagePath3	# 55 db || eltérő fényviszonyos képek indexei: 18, 34, 53
    valPath			= rgbImagePath4									# 15 db || eltérő fényviszonyos képek indexei: 12

    for i in range(len(trainPath)):
        image = cv2.imread(trainPath[i])
        yellowBB, redLargeBB, largeBB = predict_bboxes(image)
        for box in yellowBB:
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]        
            cv2.rectangle(image, (x1,y1), (x2,y2), (0,0,255), 2)
        
        cv2.imwrite('../FinalTest/{}.jpg'.format(i), image)



