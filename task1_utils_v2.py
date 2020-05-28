import cv2
import numpy as np 

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
    
    return boxes[pick].astype("int")

def predictBlueBoundingBox(image, blueLowerThres, blueUpperThres, cannyLowerThres, cannyUpperThres):
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
        if rect[2] > 7 and rect[3] > 7 and rect[2] < 60 and rect[3] < 60:
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
            if ((((box[3] - box[1]) > 18) and ((box[2] - box[0]) > 18)) and (((box[3] - box[1]) < 60) and ((box[2] - box[0]) < 60))):
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

def predictBrownBoundingBox(image, brownLowerThres, brownUpperThres, cannyLowerThres, cannyUpperThres):
    img = np.copy(image)
    imgDilate = np.copy(image)
    imgFinal = np.copy(image)

    red_lower1 = np.array([0,70,50], dtype=np.uint8)
    red_upper1 = np.array([10, 255, 255], dtype=np.uint8)
    red_lower2 = np.array([170,70,50], dtype=np.uint8)
    red_upper2 = np.array([180, 255, 255], dtype=np.uint8)

    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(imgHSV, red_lower1, red_upper1)
    mask2 = cv2.inRange(imgHSV, red_lower2, red_upper2)
    mask3 = cv2.inRange(imgHSV, brownLowerThres, brownUpperThres)

    #brownMask = cv2.inRange(imgHSV, brownLowerThres, brownUpperThres)
    brownMask = mask1 + mask2# + mask3
    brownMask = np.clip(brownMask, 0, 255)
    brownMask = np.uint8(brownMask)

    brownMaskCopy = np.copy(brownMask)
    brownMaskCopyDilate = np.copy(brownMask)

    edges = cv2.Canny(brownMaskCopy,cannyLowerThres,cannyUpperThres)
    edgesContours, edgesHierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    brownMaskCopyDilate = cv2.dilate(brownMaskCopyDilate, np.ones((3,3)), iterations = 2)
    #brownMaskCopyDilate = cv2.morphologyEx(brownMaskCopyDilate, cv2.MORPH_CLOSE, np.ones((10,10)), iterations = 3)
    #brownMaskCopyDilate = cv2.morphologyEx(brownMaskCopyDilate, cv2.MORPH_OPEN, np.ones((10,10)), iterations = 4)
    edgesDilate = cv2.Canny(brownMaskCopyDilate,cannyLowerThres,cannyUpperThres)
    edgesDilateContours, edgesDilateHierarchy = cv2.findContours(edgesDilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(img, edgesContours, -1, (0, 255, 0), 3)
    cv2.drawContours(imgDilate, edgesContours, -1, (0, 255, 0), 3)

    possible_table_bboxes = []
    bboxes = []
    for c in edgesContours:
        rect = cv2.boundingRect(c)
        if rect[2] > 50 and rect[3] > 50 and rect[2] < 300 and rect[3] < 300:
            x,y,w,h = rect
            bboxes.append([x, y, x+w, y+h])
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
    
    bboxesDilate = []
    for c in edgesDilateContours:
        rect = cv2.boundingRect(c)
        if rect[2] > 20 and rect[3] > 20 and rect[2] < 300 and rect[3] < 300:
            x,y,w,h = rect
            bboxesDilate.append([x, y, x+w, y+h])
            cv2.rectangle(imgDilate,(x,y),(x+w,y+h),(0,0,255),2)
    
    possible_table_bboxes = blue_nms(bboxesDilate, 0.2)
    possible_table_bboxes = non_max_suppression_fast(possible_table_bboxes, 0.2)
    for box in possible_table_bboxes:
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]        
        cv2.rectangle(imgFinal, (x1,y1), (x2,y2), (0,0,255), 2)

    cv2.imshow('ORIGINAL', image)
    cv2.imshow('BROWN MASK', brownMask)
    cv2.imshow('BROWN MASK DILATE', brownMaskCopyDilate)
    cv2.imshow('MASK BBOX', img)
    cv2.imshow('MASK DILATE BBOX', imgDilate)
    cv2.imshow('FINAL', imgFinal)
    cv2.waitKey(0)


