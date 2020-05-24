import cv2
import numpy as np
from task1_utils import segmentBackground, predictBoundingBox

def generateBboxes(image):
    clasterNum      = 3
    n_init          = 100
    max_iter        = 500

    lower_factor    = 1
    upper_factor    = 1
    kernel          = 5
    iterations      = 4

    cannyLowerThres = 20
    cannyUpperThres = 300

    imgHSV              = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    combinedMask        = segmentBackground(imageHSV = imgHSV, clasterNum = clasterNum, n_init = n_init, max_iter = max_iter, lower_factor = lower_factor, upper_factor = upper_factor, kernel = kernel, iterations = iterations)
    
    combinedMask[:80][:]        = 255

    imageIdx            =(combinedMask == 255)
    maskedImg           = np.copy(image)
    maskedImg[imageIdx] = 0

    maskedImgHSV        = cv2.cvtColor(maskedImg, cv2.COLOR_BGR2HSV)

    edgesMask, edgesCloseMask, edgesImageMask, edgesCloseImageMask, origMaskImg, origMaskImgClose, bboxes = predictBoundingBox(origImg = image, imageHSV = maskedImgHSV, kernel = 5, iterations = iterations, cannyLowerThres = cannyLowerThres, cannyUpperThres = cannyLowerThres)

    return origMaskImgClose, bboxes