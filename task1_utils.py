import cv2
import numpy as np 
from sklearn.cluster import KMeans


def segmentBackground(imageHSV, clasterNum, n_init, max_iter, lower_factor, upper_factor, kernel, iterations):
    img                         = np.copy(imageHSV)

    flattenImage                = img.reshape((img.shape[0] * img.shape[1], 3))

    clt = KMeans(n_clusters = clasterNum, n_init = n_init, max_iter = max_iter)
    clt.fit(flattenImage)
   
    clustersPixelNum            = np.bincount(clt.labels_)    
    clustersPixelNumOrder       = np.argsort(clustersPixelNum)

    maskOpenList                = []
    maskPixelNumList            = []
    for i in range(len(clustersPixelNumOrder)):                    # [legkevesebb...legtöbb]
        clusterIndex            = clustersPixelNumOrder[i]
        
        boolLabels              = (clt.labels_ == clusterIndex)
        sortedImage             = flattenImage[boolLabels]

        dominantColorArray      = np.zeros((sortedImage.shape[0], sortedImage.shape[1]))
        dominantColorArray[:]   = (clt.cluster_centers_[clusterIndex][0], clt.cluster_centers_[clusterIndex][1], clt.cluster_centers_[clusterIndex][2])

        substractedArray        = np.zeros((sortedImage.shape[0], sortedImage.shape[1]))
        substractedArray[:]     = np.clip((dominantColorArray[:] - sortedImage[:]), 0, 255)

        hueMean                 = np.mean(substractedArray[:][0])
        satMean                 = np.mean(substractedArray[:][1])
        valMean                 = np.mean(substractedArray[:][2])

        dominantColor_lower     = np.array([np.clip((clt.cluster_centers_[clusterIndex][0] - (hueMean*lower_factor)), 0, 179), np.clip((clt.cluster_centers_[clusterIndex][1] - (satMean*lower_factor)), 0, 255), np.clip((clt.cluster_centers_[clusterIndex][2] - (valMean*lower_factor)), 0, 255)], np.uint8)
        dominantColor_upper     = np.array([np.clip((clt.cluster_centers_[clusterIndex][0] + (hueMean*upper_factor)), 0, 179), np.clip((clt.cluster_centers_[clusterIndex][1] + (satMean*upper_factor)), 0, 255), np.clip((clt.cluster_centers_[clusterIndex][2] + (valMean*upper_factor)), 0, 255)], np.uint8)

        kernelOpenClose         = np.ones((kernel,kernel))
        mask                    = cv2.inRange(img, dominantColor_lower, dominantColor_upper)
        maskClose               = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernelOpenClose, iterations = iterations)
        maskOpen                = cv2.morphologyEx(maskClose, cv2.MORPH_OPEN, kernelOpenClose, iterations = iterations)
        
        maskOpenList.append(maskOpen)
        maskPixelNumList.append(np.sum(np.clip(maskOpen, 0, 1)))

    # A maskPixelNumList (amiben az egyes klaszterekhez tartozó maszkok pixeleinek száma van) csökkenő sorrendbe állítása:
    backgroundIndex             = [np.argsort(np.array(maskPixelNumList))[-1], np.argsort(np.array(maskPixelNumList))[-2], np.argsort(np.array(maskPixelNumList))[-3]]#, np.argsort(np.array(maskPixelNumList))[-4], np.argsort(np.array(maskPixelNumList))[-5]]
 
    combinedMask                = maskOpenList[backgroundIndex[0]]# + maskOpenList[backgroundIndex[1]]# + maskOpenList[backgroundIndex[3]]
    np.clip(combinedMask, 0, 255)

    return combinedMask

def segmentBackground_v2(imageHSV, firstLine, clasterNum, n_init, max_iter, lower_factor, upper_factor, kernel, iterations):
    img                         = np.copy(imageHSV[firstLine:, :, :])
    #img = np.copy(imageHSV)

    imgUpper                    = img[0:(int(imageHSV.shape[0] / 3) - firstLine)][:][:]
    imgCenter                   = img[(int(imageHSV.shape[0] / 3) - firstLine): (2 * int(imageHSV.shape[0] / 3)) - firstLine][:][:]
    imgLower                    = img[int(imageHSV.shape[0] / 2) - firstLine:][:][:]
    #imgLower                    = img
    print(imgUpper.shape, imgCenter.shape, imgLower.shape)

    flattenImageUpper           = imgUpper.reshape((imgUpper.shape[0] * imgUpper.shape[1], 3))
    flattenImageCenter          = imgCenter.reshape((imgCenter.shape[0] * imgCenter.shape[1], 3))
    flattenImageLower           = imgLower.reshape((imgLower.shape[0] * imgLower.shape[1], 3))
    #flattenImageLower           = img.reshape((img.shape[0] * img.shape[1], 3))
    
    cltUpper = KMeans(n_clusters = 4, n_init = n_init, max_iter = max_iter)
    cltUpper.fit(flattenImageUpper)

    cltCenter = KMeans(n_clusters = 4, n_init = n_init, max_iter = max_iter)
    cltCenter.fit(flattenImageCenter)

    cltLower = KMeans(n_clusters = clasterNum, n_init = n_init, max_iter = max_iter)
    cltLower.fit(flattenImageLower)

    clustersPixelNumUpper       = np.bincount(cltUpper.labels_)    
    clustersPixelNumOrderUpper  = np.argsort(clustersPixelNumUpper)

    clustersPixelNumCenter       = np.bincount(cltCenter.labels_)    
    clustersPixelNumOrderCenter  = np.argsort(clustersPixelNumCenter)

    clustersPixelNumLower       = np.bincount(cltLower.labels_)    
    clustersPixelNumOrderLower  = np.argsort(clustersPixelNumLower)

    ###############################################################################################
    print(flattenImageLower.shape)
    print(len(cltLower.labels_))

    mask = np.zeros((img.shape[0], img.shape[1]))

    maskUpper = np.zeros(((imgUpper.shape[0] * imgUpper.shape[1]), 1))
    maskCenter = np.zeros(((imgCenter.shape[0] * imgCenter.shape[1]), 1))
    maskLower = np.zeros(((imgLower.shape[0] * imgLower.shape[1]), 1))    

    imageIdxLower            =(cltLower.labels_ == clustersPixelNumOrderLower[-1])
    maskLower[imageIdxLower] = 255 

    imageIdxCenter1            =(cltCenter.labels_ == clustersPixelNumOrderCenter[-1])
    maskCenter[imageIdxCenter1] = 255
    imageIdxCenter2            =(cltCenter.labels_ == clustersPixelNumOrderCenter[-2])
    maskCenter[imageIdxCenter2] = 255
    imageIdxCenter3            =(cltCenter.labels_ == clustersPixelNumOrderCenter[-3])
    maskCenter[imageIdxCenter3] = 255  

    imageIdxUpper1            =(cltUpper.labels_ == clustersPixelNumOrderUpper[-1])
    maskUpper[imageIdxUpper1] = 255
    imageIdxUpper2            =(cltUpper.labels_ == clustersPixelNumOrderUpper[-2])
    maskUpper[imageIdxUpper2] = 255
    imageIdxUpper3            =(cltUpper.labels_ == clustersPixelNumOrderUpper[-3])
    #maskUpper[imageIdxUpper3] = 255    

    maskLower = maskLower.reshape((imgLower.shape[0], imgLower.shape[1]))
    maskCenter = maskCenter.reshape((imgCenter.shape[0], imgCenter.shape[1]))
    maskUpper = maskUpper.reshape((imgUpper.shape[0], imgUpper.shape[1]))

    kernel = np.ones((5,5),np.uint8)
    #maskLower = cv2.erode(maskLower,kernel, iterations = 2)
    #maskLower = cv2.morphologyEx(maskLower, cv2.MORPH_CLOSE, kernel, iterations=2)
    #maskLower = cv2.morphologyEx(maskLower, cv2.MORPH_OPEN, kernel, iterations=2)

    #maskUpper = cv2.erode(maskUpper,kernel, iterations = 2)
    #maskUpper = cv2.morphologyEx(maskUpper, cv2.MORPH_CLOSE, kernel, iterations=2)
    #maskUpper = cv2.morphologyEx(maskUpper, cv2.MORPH_OPEN, kernel, iterations=2)
    
    mask = np.zeros((imageHSV.shape[0], imageHSV.shape[1]), dtype = np.uint8)
    mask[firstLine:int(imageHSV.shape[0] / 3)][:] = maskUpper
    mask[int(imageHSV.shape[0] / 3) : int(imageHSV.shape[0] / 2)][:] = maskCenter[0:int(imgCenter.shape[0] / 2)][:]
    mask[int(imageHSV.shape[0] / 2):][:] = maskLower

    #mask = cv2.dilate(mask,kernel, iterations = 1)
    #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    #returnMask = np.zeros((imageHSV.shape[0], imageHSV.shape[1]), dtype = np.uint8)
    #returnMask[firstLine:, :] = mask

    return mask#returnMask
    ###############################################################################################

    maskOpenList                = []
    maskPixelNumList            = []
    for i in range(len(clustersPixelNumOrderLower)):                    # [legkevesebb...legtöbb]
        clusterIndex            = clustersPixelNumOrderLower[i]
        
        boolLabels              = (cltLower.labels_ == clusterIndex)
        sortedImage             = flattenImageLower[boolLabels]

        dominantColorArray      = np.zeros((sortedImage.shape[0], sortedImage.shape[1]))
        dominantColorArray[:]   = (cltLower.cluster_centers_[clusterIndex][0], cltLower.cluster_centers_[clusterIndex][1], cltLower.cluster_centers_[clusterIndex][2])

        substractedArray        = np.zeros((sortedImage.shape[0], sortedImage.shape[1]))
        substractedArray[:]     = np.clip((dominantColorArray[:] - sortedImage[:]), 0, 255)

        hueMean                 = np.mean(substractedArray[:][0])
        satMean                 = np.mean(substractedArray[:][1])
        valMean                 = np.mean(substractedArray[:][2])

        dominantColor_lower     = np.array([np.clip((cltLower.cluster_centers_[clusterIndex][0] - (hueMean*lower_factor)), 0, 179), np.clip((cltLower.cluster_centers_[clusterIndex][1] - (satMean*lower_factor)), 0, 255), np.clip((cltLower.cluster_centers_[clusterIndex][2] - (valMean*lower_factor)), 0, 255)], np.uint8)
        dominantColor_upper     = np.array([np.clip((cltLower.cluster_centers_[clusterIndex][0] + (hueMean*upper_factor)), 0, 179), np.clip((cltLower.cluster_centers_[clusterIndex][1] + (satMean*upper_factor)), 0, 255), np.clip((cltLower.cluster_centers_[clusterIndex][2] + (valMean*upper_factor)), 0, 255)], np.uint8)

        kernelOpenClose         = np.ones((kernel,kernel))
        mask                    = cv2.inRange(imgLower, dominantColor_lower, dominantColor_upper)
        maskClose               = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernelOpenClose, iterations = iterations)
        maskOpen                = cv2.morphologyEx(maskClose, cv2.MORPH_OPEN, kernelOpenClose, iterations = iterations)
        
        maskOpenList.append(maskOpen)
        maskPixelNumList.append(np.sum(np.clip(maskOpen, 0, 1)))

    # A maskPixelNumList (amiben az egyes klaszterekhez tartozó maszkok pixeleinek száma van) csökkenő sorrendbe állítása:
    backgroundIndex             = [np.argsort(np.array(maskPixelNumList))[-1], np.argsort(np.array(maskPixelNumList))[-2]]#, np.argsort(np.array(maskPixelNumList))[-3]]#, np.argsort(np.array(maskPixelNumList))[-4], np.argsort(np.array(maskPixelNumList))[-5]]
 
    combinedMask                = maskOpenList[backgroundIndex[0]] + maskOpenList[backgroundIndex[1]]# + maskOpenList[backgroundIndex[2]]
    np.clip(combinedMask, 0, 255)

    return combinedMask

def get_iou(bb1, bb2):
    assert bb1[0] < bb1[2]          # x1, x2
    assert bb1[1] < bb1[3]
    assert bb2[0] < bb2[2]
    assert bb2[1] < bb2[3]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes	
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")

def blue_nms(boxes, overlapThresh):

    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
  
    pick = []
    
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(area)
    
    sortedx1 = x1[idxs]
    sortedy1 = y1[idxs]
    sortedx2 = x2[idxs]
    sortedy2 = y2[idxs]
    
    #idxs = idxs[::-1]

    updated_bboxes = []
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        #updated_bboxes.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        overlapIdx = (overlap >= overlapThresh)

        sortedx1 = sortedx1[:last]
        sortedy1 = sortedy1[:last]
        sortedx2 = sortedx2[:last]
        sortedy2 = sortedy2[:last]

        overlapx1 = sortedx1[overlapIdx]
        overlapx2 = sortedx2[overlapIdx] 
        overlapy1 = sortedy1[overlapIdx] 
        overlapy2 = sortedy2[overlapIdx] 

        xxx1 = np.minimum(x1[i], overlapx1)
        yyy1 = np.minimum(y1[i], overlapy1)
        xxx2 = np.maximum(x2[i], overlapx2)
        yyy2 = np.maximum(y2[i], overlapy2)

        '''
        print('XXX1 LENGTHS: {}'.format(len(xxx1)))
        print(xxx1)
        print('YYY1 LENGTHS: {}'.format(len(yyy1)))
        print(yyy1)
        print('XXX2 LENGTHS: {}'.format(len(xxx2)))
        print(xxx2)
        print('YYY2 LENGTHS: {}'.format(len(yyy2)))
        print(yyy2)
        '''

        if len(xxx1) > 0:
            updated_bboxes.append([int(np.min(xxx1)), int(np.min(yyy1)), int(np.max(xxx2)), int(np.max(yyy2))])
        else:
            updated_bboxes.append([int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])])

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap >= overlapThresh)[0])))

    return updated_bboxes

def predictBlueBoundingBox(image, blueLowerThres, blueUpperThres, cannyLowerThres, cannyUpperThres):
    img = np.copy(image)
    imgTempBB = np.copy(image)
    imgTempNMS = np.copy(image)
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    blueMask = cv2.inRange(imgHSV, blueLowerThres, blueUpperThres)

    #blueMask = cv2.dilate(blueMask , np.ones((5,5)), iterations = 1)
    #blueMask = cv2.morphologyEx(blueMask, cv2.MORPH_CLOSE, np.ones((3,3)), iterations = 4)
    #blueMask = cv2.morphologyEx(blueMask, cv2.MORPH_OPEN, np.ones((2,2)), iterations = 1)
    blueMask = cv2.dilate(blueMask , np.ones((3,3)), iterations = 2)
    #blueMask = cv2.erode(blueMask , np.ones((3,3)), iterations = 1)

    blueMask = np.uint8(blueMask)
    
    edges = cv2.Canny(blueMask,cannyLowerThres,cannyUpperThres)
    edgesCopy = np.copy(edges)

    edgesContours, edgesHierarchy = cv2.findContours(edgesCopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(img, edgesContours, -1, (0, 255, 0), 3)

    bboxes = []

    for c in edgesContours:
        rect = cv2.boundingRect(c)
        if rect[2] < 0 or rect[3] < 0 or rect[2] > 70 or rect[3] > 70: continue
        #print(cv2.contourArea(c))
        x,y,w,h = rect

        bboxes.append([x, y, x+w, y+h])

        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        #cv2.putText(image,'Obj Detected',(x+w+10,y+h),0,0.3,(0,255,0))
        #cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
    
    binImg = np.zeros((img.shape[0], img.shape[1]), dtype = np.uint8)
    binImgNMS = np.zeros((img.shape[0], img.shape[1]), dtype = np.uint8)
    
    for box in bboxes:
        if (((box[3] - box[1]) < 22) or ((box[2] - box[0]) < 22)):
            #print((box[3] - box[1]), (box[2] - box[0]))
            binImg[box[1]:box[3], box[0]:box[2]] = 255
            #print(binImg[box[1]:box[3], box[0]:box[2]])
    print(len(bboxes))
    #########################################################################################################################################################
    bboxes2 = non_max_suppression_fast(bboxes, 0.3)
    bboxes3 = blue_nms(bboxes, 0.2)
    print('BBOXEEEEEEEEEEEEES 1')
    print(bboxes)
    print('BBOXEEEEEEEEEEEEES 3')
    print(bboxes3)
    #print('ALLAHU AKBAR')
    #print(len(bboxes), len(bboxes2))
    for box in bboxes3:
        if (((box[3] - box[1]) < 22) or ((box[2] - box[0]) < 22)):
            #print((box[3] - box[1]), (box[2] - box[0]))
            binImgNMS[box[1]:box[3], box[0]:box[2]] = 255
            #print(binImg[box[1]:box[3], box[0]:box[2]])

    #binImgNMS = cv2.dilate(binImgNMS, np.ones((3,3)), iterations = 2)

    binEdgesNMS = cv2.Canny(binImgNMS,cannyLowerThres,cannyUpperThres)
    binEdgesNMSCopy = np.copy(binEdgesNMS)

    binEdgesNMSContours, binEdgesNMSHierarchy = cv2.findContours(binEdgesNMSCopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for c in binEdgesNMSContours:
        rect = cv2.boundingRect(c)
        if rect[2] < 0 or rect[3] < 0 or rect[2] > 70 or rect[3] > 70: continue
        #print(cv2.contourArea(c))
        x,y,w,h = rect

        #bboxes.append([x, y, x+w, y+h])

        cv2.rectangle(imgTempNMS,(x,y),(x+w,y+h),(0,0,255),2)
        #cv2.putText(image,'Obj Detected',(x+w+10,y+h),0,0.3,(0,255,0))
        #cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
    
    ###########################################################################################################################################################



    binImg = cv2.dilate(binImg, np.ones((3,3)), iterations = 4)

    binEdges = cv2.Canny(binImg,cannyLowerThres,cannyUpperThres)
    binEdgesCopy = np.copy(binEdges)

    binEdgesContours, binEdgesHierarchy = cv2.findContours(binEdgesCopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    #cv2.drawContours(imgTempBB, binEdgesContours, -1, (0, 255, 0), 3)
    for c in binEdgesContours:
        rect = cv2.boundingRect(c)
        if rect[2] < 0 or rect[3] < 0 or rect[2] > 70 or rect[3] > 70: continue
        #print(cv2.contourArea(c))
        x,y,w,h = rect

        #bboxes.append([x, y, x+w, y+h])

        cv2.rectangle(imgTempBB,(x,y),(x+w,y+h),(0,0,255),2)
        #cv2.putText(image,'Obj Detected',(x+w+10,y+h),0,0.3,(0,255,0))
        #cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

    cv2.imshow('BIN BOX', imgTempBB)
    cv2.imshow('BIN NMS BOX', imgTempNMS)
    cv2.imshow('BIN', binImg)


    return img



def predictBoundingBox(origImg, imageHSV, kernel, iterations, cannyLowerThres, cannyUpperThres):
    img = np.copy(origImg)
    imgClose = np.copy(origImg)
    imgHSV = np.copy(imageHSV)

    h,s,v = cv2.split(imgHSV)
    #h = cv2.equalizeHist(h)
    #s = cv2.equalizeHist(s)
    #v = cv2.equalizeHist(v)
    imgHSVEQH = cv2.merge([h,s,v])
    

    kernelOpenClose = np.ones((kernel,kernel))

    imgBGR = cv2.cvtColor(imgHSVEQH, cv2.COLOR_HSV2BGR)
    imgGray = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2GRAY)
    #imgGray = cv2.GaussianBlur(imgGray, (3, 3), 0)
    imgGray = cv2.bilateralFilter(imgGray,9,75,75)

    edges = cv2.Canny(imgGray,cannyLowerThres,cannyUpperThres)   
    edgesClose = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernelOpenClose, iterations = iterations)
    edgesOpen = cv2.morphologyEx(edgesClose, cv2.MORPH_OPEN, kernelOpenClose, iterations = iterations)

    # cv2.finContours() módosítja a paraméterben adott képet, ezért előbb copy-zzuk
    edgesCopy = np.copy(edges)
    edgesOpenCopy = np.copy(edgesOpen)

    edgesContours, edgesHierarchy = cv2.findContours(edgesCopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    edgesCloseContours, edgesCloseHierarchy = cv2.findContours(edgesOpenCopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    edgesImage = np.copy(imgBGR)
    edgesCloseImage = np.copy(imgBGR)
    cv2.drawContours(edgesImage, edgesContours, -1, (0, 255, 0), 3)
    cv2.drawContours(edgesCloseImage, edgesCloseContours, -1, (0, 255, 0), 3)

    bboxes = []

    for c in edgesContours:
        rect = cv2.boundingRect(c)
        if rect[2] < 20 or rect[3] < 20 or rect[2] > 400 or rect[3] > 400: continue
        #print(cv2.contourArea(c))
        x,y,w,h = rect
        cv2.rectangle(edgesImage,(x,y),(x+w,y+h),(0,0,255),2)
        #cv2.putText(image,'Obj Detected',(x+w+10,y+h),0,0.3,(0,255,0))
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

    for c in edgesCloseContours:
        rect = cv2.boundingRect(c)
        if rect[2] < 20 or rect[3] < 20 or rect[2] > 400 or rect[3] > 400: continue
        #print(cv2.contourArea(c))
        x,y,w,h = rect

        bboxes.append([x, y, x+w, y+h])

        cv2.rectangle(edgesCloseImage,(x,y),(x+w,y+h),(0,0,255),2)
        #cv2.putText(image,'Obj Detected',(x+w+10,y+h),0,0.3,(0,255,0))
        cv2.rectangle(imgClose,(x,y),(x+w,y+h),(0,0,255),2)

    return edges, edgesOpen, edgesImage, edgesCloseImage, img, imgClose, bboxes

