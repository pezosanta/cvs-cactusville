import cv2
import numpy as np
#from task1_utils import segmentBackground, predictBoundingBox
from task1_utils import segmentBackground_v2 as segmentBackground
#from task1_utils import predictBoundingBox, predictBlueBoundingBox
from task1_utils_v2 import predictBlueTSBoundingBox, predictBrownBoundingBox, predictRedTSBoundingBox, predictYellowTSBoundingBox
import matplotlib.pyplot as plt

def generateBboxes(image):
    clasterNum      = 2
    n_init          = 50
    max_iter        = 300

    lower_factor    = 1.5
    upper_factor    = 1.5
    kernel          = 5
    iterations      = 4

    cannyLowerThres = 20
    cannyUpperThres = 300

    imgHSV              = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    combinedMask        = segmentBackground(imageHSV = imgHSV, clasterNum = clasterNum, n_init = n_init, max_iter = max_iter, lower_factor = lower_factor, upper_factor = upper_factor, kernel = kernel, iterations = iterations)
    
    combinedMask = np.uint8(combinedMask)
    
    edges = cv2.Canny(combinedMask,cannyLowerThres,cannyUpperThres)
    #edges = cv2.dilate(edges , np.ones((kernel,kernel)), iterations = 1)
    #edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, np.ones((kernel,kernel)), iterations=1)
    cv2.imshow('CANNY', edges)

    '''
    kernelOpenClose = np.ones((kernel,kernel))
    edges = cv2.Canny(combinedMask,cannyLowerThres,cannyUpperThres)
    edgesClose = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernelOpenClose, iterations = 1)
    edgesOpen = cv2.morphologyEx(edgesClose, cv2.MORPH_OPEN, kernelOpenClose, iterations = 1)

    edgesCopy = np.copy(edges)
    edgesContours, edgesHierarchy = cv2.findContours(edgesCopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    edgesOpenCopy = np.copy(edgesOpen)
    edgesOpenContours, edgesOpenHierarchy = cv2.findContours(edgesOpenCopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image, edgesContours, -1, (0, 255, 0), 3)
    cv2.imshow('CANNY', edges)
    cv2.imshow('CANNYOPEN', edgesOpen)
    cv2.imshow('CONTOUR', image)
    '''
    '''
    #combinedMask[:80][:]        = 255

    imageIdx            =(combinedMask == 255)
    maskedImg           = np.copy(image)
    maskedImg[imageIdx] = 0

    maskedImgHSV        = cv2.cvtColor(maskedImg, cv2.COLOR_BGR2HSV)

    edgesMask, edgesCloseMask, edgesImageMask, edgesCloseImageMask, origMaskImg, origMaskImgClose, bboxes = predictBoundingBox(origImg = image, imageHSV = maskedImgHSV, kernel = 5, iterations = iterations, cannyLowerThres = cannyLowerThres, cannyUpperThres = cannyLowerThres)
    '''
    return combinedMask#, origMaskImgClose, bboxes

import glob

rgbImagePath1 	= glob.glob('../CVS_HW_FILES/g1/rgb/*.jpg')		# 20 db
rgbImagePath2 	= glob.glob('../CVS_HW_FILES/g2/rgb/*.jpg')		# 16 db
rgbImagePath3 	= glob.glob('../CVS_HW_FILES/g3/rgb/*.jpg')		# 19 db
rgbImagePath4 	= glob.glob('../CVS_HW_FILES/g4/rgb/*.jpg')		# 15 db

trainPath 		= rgbImagePath1 + rgbImagePath2 + rgbImagePath3	# 55 db || eltérő fényviszonyos képek indexei: 18, 34, 53
valPath			= rgbImagePath4									# 15 db || eltérő fényviszonyos képek indexei: 12


image = cv2.imread(trainPath[0])#25   #35-nél nem kell nms                       20!!!!


clasterNum      = 2
n_init          = 50
max_iter        = 300

lower_factor    = 1.5
upper_factor    = 1.5
kernel          = 5
iterations      = 4
cannyLowerThres = 20
cannyUpperThres = 300


brown_lower = np.array([10,100,20], dtype=np.uint8)
brown_upper = np.array([20,255,200], dtype=np.uint8)
green_lower = np.array([40, 52, 72], dtype=np.uint8)
green_upper = np.array([80, 255, 255], dtype=np.uint8)
yellow_lower = np.array([20, 100, 100], dtype = np.uint8)
yellow_upper = np.array([30, 255, 255], dtype = np.uint8)
red_lower1 = np.array([0,70,50], dtype=np.uint8)
red_upper1 = np.array([10, 255, 255], dtype=np.uint8)
red_lower2 = np.array([170,70,50], dtype=np.uint8)
red_upper2 = np.array([180, 255, 200], dtype=np.uint8)
blue_lower = np.array([70,40,0],np.uint8) #100, 150
blue_upper = np.array([140,255,255],np.uint8)

#blueBB = predictBlueTSBoundingBox(image, blue_lower, blue_upper, cannyLowerThres, cannyUpperThres)
#redBB = predictRedTSBoundingBox(image, blueBB, red_lower1, red_upper1, red_lower2, red_upper2, cannyLowerThres, cannyUpperThres)
#yellowBB = predictYellowTSBoundingBox(image, redBB, yellow_lower, yellow_upper, cannyLowerThres, cannyUpperThres)



'''
#combinedMask = generateBboxes(image)
imgHSV              = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#print(imgHSV[176][300][:])
h,s,v = cv2.split(imgHSV)                

#h = cv2.equalizeHist(h)
#s = cv2.equalizeHist(s)
#v = cv2.equalizeHist(v)
imgHSV = cv2.merge([h,s,v])


blueMask = cv2.inRange(imgHSV, blue_lower, blue_upper)
#blueMask = cv2.dilate(blueMask , np.ones((5,5)), iterations = 1)
#blueMask = cv2.morphologyEx(blueMask, cv2.MORPH_CLOSE, np.ones((3,3)), iterations = 3)
#blueMask = cv2.morphologyEx(blueMask, cv2.MORPH_OPEN, np.ones((2,2)), iterations = 1)
#blueMask = cv2.dilate(blueMask , np.ones((4,4)), iterations = 3)
#blueMask = cv2.erode(blueMask , np.ones((3,3)), iterations = 1)
blueMask = np.uint8(blueMask)
blue_edges = cv2.Canny(blueMask,cannyLowerThres,cannyUpperThres)
#blue_edges_close = cv2.morphologyEx(blue_edges, cv2.MORPH_CLOSE, np.ones((5,5)), iterations = 3)
#blue_edges_open = cv2.morphologyEx(blue_edges, cv2.MORPH_OPEN, np.ones((5,5)), iterations = 1)


#blueBB = predictBlueBoundingBox(image, blue_lower, blue_upper, cannyLowerThres, cannyUpperThres)

mask, firstLine = predictBrownBoundingBox(image, brown_lower, brown_upper, cannyLowerThres, cannyUpperThres)
#combinedMask = segmentBackground(imageHSV = imgHSV, firstLine = firstLine, clasterNum = clasterNum, n_init = n_init, max_iter = max_iter, lower_factor = lower_factor, upper_factor = upper_factor, kernel = kernel, iterations = iterations)

maskIdx               = (mask == 255)
invMask               = np.ones((mask.shape[0], mask.shape[1]), dtype=np.uint8) * 255
invMask[maskIdx]      = 0
invMask               = np.clip(invMask, 0, 1)

#productMask           = invMask * combinedMask

#productMask           = np.clip(productMask, 0, 255)
'''


'''
mask1 = cv2.inRange(imgHSV, red_lower1, red_upper1)
mask2 = cv2.inRange(imgHSV, red_lower2, red_upper2)
mask3 = cv2.inRange(imgHSV, brown_lower, brown_upper)
mask4 = cv2.inRange(imgHSV, yellow_lower, yellow_upper)
mask5 = cv2.inRange(imgHSV, green_lower, green_upper)
mask = mask1 + mask2 + mask3# + mask4# + mask5

mask = np.clip(mask, 0, 255)

maskIdx               = (mask == 255)
invMask               = np.ones((mask.shape[0], mask.shape[1]), dtype=np.uint8) * 255
invMask[maskIdx]      = 0
invMask               = np.clip(invMask, 0, 1)

#productMask           = invMask * combinedMask

#productMask           = np.clip(productMask, 0, 255)


hHist = cv2.calcHist(imgHSV,[0],None,[180],[0,180]) # kép, channel, mask, bin, range
sHist = cv2.calcHist(imgHSV,[1],None,[255],[0,255])
vHist = cv2.calcHist(imgHSV,[2],None,[255],[0,255])
#print(np.amax(hist))
#print(type(hist))

hsv = cv2.merge([h,s,v])
#bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

#plt.plot(hHist)
#plt.plot(sHist)
#plt.plot(vHist)
#plt.xlim([0,500])
#plt.show()
'''


'''
for i in range(len(trainPath)):
    image = cv2.imread(trainPath[i])
    blueBB = predictBlueTSBoundingBox(image, blue_lower, blue_upper, cannyLowerThres, cannyUpperThres)
    for box in blueBB:
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]        
        cv2.rectangle(image, (x1,y1), (x2,y2), (0,0,255), 2)
    cv2.imwrite('../PredictionsBlue/{}.jpg'.format(i), image)
'''
'''
for i in range(len(trainPath)):
    image = cv2.imread(trainPath[i])
    mask, firstLine = predictBrownBoundingBox(image, brown_lower, brown_upper, cannyLowerThres, cannyUpperThres)
    
    cv2.imwrite('../PredictionsRedTest/{}.jpg'.format(i), mask)
'''
'''
for i in range(len(trainPath)):
    image = cv2.imread(trainPath[i])
    blueBB = predictBlueBoundingBox(image, blue_lower, blue_upper, cannyLowerThres, cannyUpperThres)
    img, redBB = predictRedTSBoundingBox(image, blueBB, red_lower1, red_upper1, red_lower2, red_upper2, cannyLowerThres, cannyUpperThres)
    
    cv2.imwrite('../PredictionsRedBlueTest/{}.jpg'.format(i), img)
'''

for i in range(len(trainPath)):
    image = cv2.imread(trainPath[i])
    blueBB = predictBlueTSBoundingBox(image, blue_lower, blue_upper, cannyLowerThres, cannyUpperThres)
    redBB = predictRedTSBoundingBox(image, blueBB, red_lower1, red_upper1, red_lower2, red_upper2, cannyLowerThres, cannyUpperThres)
    img, yellowBB = predictYellowTSBoundingBox(image, blueBB, redBB, yellow_lower, yellow_upper, cannyLowerThres, cannyUpperThres)
    
    cv2.imwrite('../PredictionsYellowTest/{}.jpg'.format(i), img)

#cv2.imshow('COLOR BASED MASK', mask)
#cv2.imshow('COLOR BASED INVERTED MASK', invMask*255)
#cv2.imshow('BLUE BB', blueBB)
#cv2.imshow('BLUE MASK', blueMask)
#cv2.imshow('BLUE CANNY', blue_edges)
#cv2.imshow('BLUE CANNY CLOSE', blue_edges_close)
#cv2.imshow('BLUE CANNY OPEN', blue_edges_open)
#cv2.imshow('ORIGINAL', image)
#cv2.imshow('CLUSTER BASED MASK', combinedMask)
#cv2.imshow('PRODUCT MASK', productMask)
#cv2.imshow('BBOX', origMaskImgClose)
#cv2.waitKey(0)