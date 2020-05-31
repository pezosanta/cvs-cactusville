import cv2
import numpy as np 

def intersection_over_union(boxA, boxB):
	
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	
	iou = interArea / float(boxAArea + boxBArea - interArea)
	
	return iou

def join_bboxes_based_on_iou(bboxes, overlapThres):
    origBoxes = bboxes.copy()
    boxes = bboxes.copy()

    deletingIdxs = []
    joinedBBoxes = []
    remainingBBoxes = []
    for i in range(len(boxes)-1):
        boxA = boxes[i]
        for j in range(i+1,len(boxes)):
            boxB = boxes[j]
            iou = intersection_over_union(boxA, boxB)
            if iou > overlapThres:
                x1 = min(boxA[0], boxB[0])
                y1 = min(boxA[1], boxB[1])
                x2 = max(boxA[2], boxB[2])
                y2 = max(boxA[3], boxB[3])

                joinedBBoxes.append([x1, y1, x2, y2])

                deletingIdxs.append(i)
                deletingIdxs.append(j)
    
    if len(deletingIdxs) > 0:
        boxes = np.array(boxes)
    
        deletingIdxs = np.array(deletingIdxs)
        unique, counts = np.unique(deletingIdxs, return_counts = True)
        
        deletingIdxs = np.array([True]*len(boxes))       
        deletingIdxs[unique] = False       

        remainingBBoxes = boxes[deletingIdxs,:]
        remainingBBoxes = remainingBBoxes.tolist()
    else:
        remainingBBoxes = []
    
    if len(deletingIdxs) > 0:
        return remainingBBoxes + joinedBBoxes
    else:
        return origBoxes

def blue_nms(boxes, overlapThresh):

    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(area)
    idxs = idxs[::-1]

    sortedx1 = x1[idxs]
    sortedy1 = y1[idxs]
    sortedx2 = x2[idxs]
    sortedy2 = y2[idxs]
    
    updated_bboxes = []
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        
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

        if len(xxx1) > 0:
            updated_bboxes.append([int(np.min(xxx1)), int(np.min(yyy1)), int(np.max(xxx2)), int(np.max(yyy2))])
        else:
            updated_bboxes.append([int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])])

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap >= overlapThresh)[0])))

    return updated_bboxes

def non_max_suppression_fast(boxes, overlapThresh):

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
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
     
        overlap = (w * h) / area[idxs[:last]]
        
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    bboxes = []
    if boxes[pick].astype("int").shape[0] > 0:
        for i in range(boxes[pick].astype("int").shape[0]):
            bboxes.append(list(boxes[pick].astype("int")[i][:]))
    
    return bboxes

def ignoreFarBBoxes(bboxes, goodBboxes):
    bboxesCopy = bboxes.copy()
    goodBboxesCopy = goodBboxes.copy()

    if len(bboxesCopy) != 0 and len(goodBboxesCopy) != 0:
        boxes = np.array(bboxesCopy)
       
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        #x2 = boxes[:,2]
        #y2 = boxes[:,3]
        
        goodboxes = np.array(goodBboxesCopy)
        goodx1 = goodboxes[:,0]
        goody1 = goodboxes[:,1]
        #goodx2 = goodboxes[:,2]
        #goody2 = goodboxes[:,3]

        remainingIdxs = []
        for i in range(len(goodx1)):
            currentgoodx1 = goodx1[i].item()
            currentgoody1 = goody1[i].item()
            for j in range(len(x1)):
                currentx1 = x1[j].item()
                currenty1 = y1[j].item()
                if (abs(currentgoodx1 - currentx1) < 75) and (abs(currentgoody1 - currenty1) < 45):
                    remainingIdxs.append(j)
        
        if len(remainingIdxs) > 0:
            remainingIdxs = np.array(remainingIdxs)
            unique, counts = np.unique(remainingIdxs, return_counts = True)
            
            remainingIdxs = np.array([False]*len(x1))       
            remainingIdxs[unique] = True       

            remainingBoxes = boxes[remainingIdxs,:]
            remainingBoxes = remainingBoxes.tolist()
        else:
            remainingBoxes = []
        
        return remainingBoxes
    else:
        return bboxesCopy

def predictBlueTSBoundingBox(image, blueLowerThres, blueUpperThres, cannyLowerThres, cannyUpperThres):
    img = np.copy(image)
    imgFinal = np.copy(image)
    #imgDilate = np.copy(image)
    
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    blueMask = cv2.inRange(imgHSV, blueLowerThres, blueUpperThres)
    blueMask = np.uint8(blueMask)

    blueMaskCopy = np.copy(blueMask)
    blueMaskCopyDilate = np.copy(blueMask)

    edges = cv2.Canny(blueMaskCopy,cannyLowerThres,cannyUpperThres)
    edgesContours, edgesHierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    blueMaskCopyDilate = cv2.dilate(blueMaskCopyDilate, np.ones((3,3)), iterations = 3)
    edgesDilate = cv2.Canny(blueMaskCopyDilate,cannyLowerThres,cannyUpperThres)
    edgesDilateContours, edgesDilateHierarchy = cv2.findContours(edgesDilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    #cv2.drawContours(img, edgesContours, -1, (0, 255, 0), 3)
    #cv2.drawContours(imgDilate, edgesContours, -1, (0, 255, 0), 3)

    possible_table_bboxes = []
    bboxes = []
    for c in edgesContours:
        rect = cv2.boundingRect(c)
        if rect[2] > 7 and rect[3] > 7 and rect[2] < 60 and rect[3] < 60:
            x,y,w,h = rect
            bboxes.append([x, y, x+w, y+h])
            #cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
    
    bboxesDilate = []
    for c in edgesDilateContours:
        rect = cv2.boundingRect(c)
        if rect[2] > 7 and rect[3] > 7 and rect[2] < 55 and rect[3] < 55:
            x,y,w,h = rect
            bboxesDilate.append([x, y, x+w, y+h])
            #cv2.rectangle(imgDilate,(x,y),(x+w,y+h),(0,0,255),2)
    
    binImg = np.zeros((img.shape[0], img.shape[1]), dtype = np.uint8)
    binImgDilate = np.zeros((img.shape[0], img.shape[1]), dtype = np.uint8)
    
    bboxes = blue_nms(bboxes, 0.2)
    for box in bboxes:
        if (((box[3] - box[1]) < 18) and ((box[2] - box[0]) < 18)):
            binImg[box[1]:box[3], box[0]:box[2]] = 255
        else:
            if ((((box[3] - box[1]) > 18) and ((box[2] - box[0]) > 18)) and (((box[3] - box[1]) < 55) and ((box[2] - box[0]) < 55))):
                possible_table_bboxes.append([box[0], box[1], box[2], box[3]])
    
    bboxesDilate = blue_nms(bboxesDilate, 0.2)
    for box in bboxesDilate:
        if (((box[3] - box[1]) < 18) and ((box[2] - box[0]) < 18)):
            binImgDilate[box[1]:box[3], box[0]:box[2]] = 255
        else:
            if ((((box[3] - box[1]) > 18) and ((box[2] - box[0]) > 18)) and (((box[3] - box[1]) < 50) and ((box[2] - box[0]) < 50))):
                possible_table_bboxes.append([box[0], box[1], box[2], box[3]])
               
    possible_table_bboxes = non_max_suppression_fast(possible_table_bboxes, 0.3)

    binPossibleImg = np.zeros((img.shape[0], img.shape[1]), dtype = np.uint8)
    for box in possible_table_bboxes:
        #if (((box[3] - box[1]) < 18) and ((box[2] - box[0]) < 18)):
        binPossibleImg[box[1]:box[3], box[0]:box[2]] = 255    
    
    #for box in possible_table_bboxes:
    #    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]        
    #    cv2.rectangle(imgFinal, (x1,y1), (x2,y2), (0,0,255), 2)

    #cv2.imshow('FINAL', imgFinal)

    return possible_table_bboxes

def predictRedTSBoundingBox(image, blueBB, red_lower1, red_upper1, red_lower2, red_upper2, cannyLowerThres, cannyUpperThres):
    img = np.copy(image)
    imgDilate = np.copy(image)
    
    imgFinal = np.copy(image)

    red_lower1 = np.array([0,70,50], dtype=np.uint8)
    red_upper1 = np.array([10, 255, 255], dtype=np.uint8)
    red_lower2 = np.array([170,70,50], dtype=np.uint8)
    red_upper2 = np.array([180, 255, 200], dtype=np.uint8)
    
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    redMask1 = cv2.inRange(imgHSV, red_lower1, red_upper1)
    redMask2 = cv2.inRange(imgHSV, red_lower2, red_upper2)
    
    mask = redMask1 + redMask2
    mask = np.clip(mask, 0, 255)
    mask = np.uint8(mask)

    maskErodeDilate = np.copy(mask)
    maskDilate = np.copy(mask)

    maskErodeDilate = cv2.erode(maskErodeDilate, np.ones((4,4)), iterations = 1)
    maskErodeDilate = cv2.dilate(maskErodeDilate, np.ones((3,3)), iterations = 2)

    maskDilate = cv2.dilate(maskDilate, np.ones((3,3)), iterations = 2)

    #for box in blueBB:
    #    mask[box[1]:box[3], box[0]:box[2]] = 0

    maskErodeDilateCopy = np.copy(maskErodeDilate)
    maskDilateCopy = np.copy(maskDilate)

    edgesErodeDilate = cv2.Canny(maskErodeDilateCopy,cannyLowerThres,cannyUpperThres)
    edgesErodeDilateContours, edgesErodeDilateHierarchy = cv2.findContours(edgesErodeDilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    edgesDilate = cv2.Canny(maskDilateCopy,cannyLowerThres,cannyUpperThres)
    edgesDilateContours, edgesDilateHierarchy = cv2.findContours(edgesDilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    #cv2.drawContours(img, edgesErodeDilateContours, -1, (0, 255, 0), 3)
    #cv2.drawContours(img, edgesDilateContours, -1, (0, 255, 0), 3)

    bboxesErodeDilate = []
    for c in edgesErodeDilateContours:
        rect = cv2.boundingRect(c)
        if (rect[2] * rect[3] > 600) and (rect[2] * rect[3] < 2050):
            x,y,w,h = rect
            bboxesErodeDilate.append([x, y, x+w, y+h])
            #cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
    
    #possible_table_bboxes = blue_nms(bboxes, 0.2)
    #possible_table_bboxes = non_max_suppression_fast(possible_table_bboxes, 0.3)

    bboxesDilate = []
    for c in edgesDilateContours:
        rect = cv2.boundingRect(c)
        if (rect[2] * rect[3] > 600) and (rect[2] * rect[3] < 2050):
            x,y,w,h = rect
            bboxesDilate.append([x, y, x+w, y+h])
            #cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

    bboxes = bboxesErodeDilate + bboxesDilate
    bboxes = non_max_suppression_fast(bboxes, 0.2)

    remainingRedBoxes = ignoreFarBBoxes(bboxes, blueBB)
    
    bboxes = remainingRedBoxes + blueBB
    bboxes = non_max_suppression_fast(bboxes, 0.3)
    '''
    for box in bboxes:
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]        
        cv2.rectangle(imgFinal, (x1,y1), (x2,y2), (0,0,255), 2)

    cv2.imshow('ORIGINAL', image)
    cv2.imshow('MASK', mask)
    cv2.imshow('MASK BBOX', img)
    cv2.imshow('FINAL', imgFinal)
    cv2.waitKey(0)
    '''
    return bboxes

def predictYellowTSBoundingBox(image, blueBB, redBB, yellow_lower, yellow_upper, cannyLowerThres, cannyUpperThres):
    img = np.copy(image)

    yellow_lower = np.array([20, 100, 100], dtype = np.uint8)
    yellow_upper = np.array([30, 255, 255], dtype = np.uint8)

    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(imgHSV, yellow_lower, yellow_upper)
    mask = np.clip(mask, 0, 255)
    mask = np.uint8(mask)

    mask = cv2.erode(mask, np.ones((3,3)), iterations = 1)
    mask = cv2.dilate(mask, np.ones((5,5)), iterations = 5)

    maskCopy = np.copy(mask)

    edges = cv2.Canny(maskCopy,cannyLowerThres,cannyUpperThres)
    edgesContours, edgesHierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    #cv2.drawContours(img, edgesContours, -1, (0, 255, 0), 3)

    bboxes = []
    for c in edgesContours:
        rect = cv2.boundingRect(c)
        if (rect[2] * rect[3] > 700) and (rect[2] * rect[3] < 2050):
            x,y,w,h = rect
            bboxes.append([x, y, x+w, y+h])
            #cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

    remainingYellowBoxes = ignoreFarBBoxes(bboxes, blueBB)
    bboxes = remainingYellowBoxes + redBB

    bboxes = non_max_suppression_fast(bboxes, 0.3)
    '''
    for box in bboxes:
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]        
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 2)
    
    
    cv2.imshow('ORIGINAL', image)
    cv2.imshow('YELLOW MASK', mask)
    cv2.imshow('YELLOW BBOX', img)
    cv2.waitKey(0)
    '''
    return bboxes

def predictLargeRedBoundingBox(image, red_lower1, red_upper1, red_lower2, red_upper2, cannyLowerThres, cannyUpperThres):
    img = np.copy(image)

    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    redMask1 = cv2.inRange(imgHSV, red_lower1, red_upper1)
    redMask2 = cv2.inRange(imgHSV, red_lower2, red_upper2)
    redMask = redMask1 + redMask2
    redMask = np.clip(redMask, 0, 255)
    redMask = np.uint8(redMask)

    mask = np.copy(redMask)

    mask = cv2.erode(mask, np.ones((3,3)), iterations = 2)
    mask = cv2.dilate(mask, np.ones((5,5)), iterations = 4)

    maskCopy = np.copy(mask)

    maskCopy[0:int(maskCopy.shape[0] / 2), :] = 0

    edges = cv2.Canny(maskCopy,cannyLowerThres,cannyUpperThres)
    edgesContours, edgesHierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    #cv2.drawContours(img, edgesContours, -1, (0, 255, 0), 3)

    bboxes = []
    for c in edgesContours:
        rect = cv2.boundingRect(c)
        if rect[2] * rect[3] > 3850 and rect[2] * rect[3] < 28000:
            x,y,w,h = rect
            bboxes.append([x, y, x+w, y+h])
            #cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
    
    bboxes = join_bboxes_based_on_iou(bboxes, 0)    
    '''
    for box in bboxes:
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]        
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 2)

    cv2.imshow('ORIGINAL', image)
    cv2.imshow('MASK BBOX', img)
    cv2.waitKey(0)
    '''
    return bboxes

def predictLargeBoundingBox(image, blueBB, redBB, yellowBB, brown_lower, brown_upper, blue_lower, blue_upper, cannyLowerThres, cannyUpperThres):
    img = np.copy(image)

    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    brownMask = cv2.inRange(imgHSV, brown_lower, brown_upper)

    blueMask = cv2.inRange(imgHSV, blue_lower, blue_upper)
    blueMask = np.uint8(blueMask)
    
    mask = brownMask
    mask = np.clip(mask, 0, 255)
    mask = np.uint8(mask)

    mask = cv2.erode(mask, np.ones((3,3)), iterations = 1)
    mask = cv2.dilate(mask, np.ones((20,20)), iterations = 2)#9,9 2
    
    mask = mask - blueMask
    mask = np.clip(mask, 0, 255)
    mask = np.uint8(mask)

    for box in yellowBB:
        mask[box[1]:box[3], box[0]:box[2]] = 0
    for box in redBB:
        mask[box[1]:box[3], box[0]:box[2]] = 0
    for box in blueBB:
        mask[box[1]:box[3], box[0]:box[2]] = 0
    
    maskCopy = np.copy(mask)
    
    maskLineIdxs = (np.where(np.sum(np.clip(maskCopy[0:90, :], 0, 1), axis = 1) >= 300)) #350
    maskLastLineIdx = (maskLineIdxs[0][-1] if maskLineIdxs[0].size > 0 else 0)
    maskCopy[0:maskLastLineIdx, :] = 0

    edges = cv2.Canny(maskCopy, cannyLowerThres, cannyUpperThres)
    edgesContours, edgesHierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    #cv2.drawContours(img, edgesContours, -1, (0, 255, 0), 3)

    bboxes = []
    for c in edgesContours:
        rect = cv2.boundingRect(c)
        if rect[2] * rect[3] > 3500 and rect[2] * rect[3] < 80000 and rect[2] < 250:#3850
            x,y,w,h = rect
            bboxes.append([x, y, x+w, y+h])
            #cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        
    bboxes = join_bboxes_based_on_iou(bboxes, 0.17)    
    bboxes = join_bboxes_based_on_iou(bboxes, 0.15)
    bboxes = join_bboxes_based_on_iou(bboxes, 0.08)
    bboxes = join_bboxes_based_on_iou(bboxes, 0.4)
    bboxes = join_bboxes_based_on_iou(bboxes, 0.05)
    '''
    for box in extendedBoxes:
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]        
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 2)
    
    cv2.imshow('ORIGINAL', image)
    cv2.imshow('TRANSFORMED MASK', maskCopy)
    cv2.imshow('MASK BBOX', img)
    cv2.imshow('CANNY', edges)
    cv2.waitKey(0)
    '''
    return bboxes