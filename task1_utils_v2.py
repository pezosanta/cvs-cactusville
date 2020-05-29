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

    bboxes = []
    if boxes[pick].astype("int").shape[0] > 0:
        for i in range(boxes[pick].astype("int").shape[0]):
            bboxes.append(list(boxes[pick].astype("int")[i][:]))
    
    return bboxes

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

def predictRedBoundingBox(image, blueBB, red_lower1, red_upper1, red_lower2, red_upper2, cannyLowerThres, cannyUpperThres):
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
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
    
    #possible_table_bboxes = blue_nms(bboxes, 0.2)
    #possible_table_bboxes = non_max_suppression_fast(possible_table_bboxes, 0.3)

    bboxesDilate = []
    for c in edgesDilateContours:
        rect = cv2.boundingRect(c)
        if (rect[2] * rect[3] > 600) and (rect[2] * rect[3] < 2050):
            x,y,w,h = rect
            bboxesDilate.append([x, y, x+w, y+h])
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)


    bboxes = bboxesErodeDilate + bboxesDilate
    bboxes = non_max_suppression_fast(bboxes, 0.2)

    bboxesCopy = bboxes.copy()

    #bboxes = bboxes + blueBB
    #bboxes = non_max_suppression_fast(bboxes, 0.3)

    if len(bboxesCopy) != 0 and len(blueBB) != 0:
        redboxes = np.array(bboxesCopy)
       
        redx1 = redboxes[:,0]
        redy1 = redboxes[:,1]
        #redx2 = redboxes[:,2]
        #redy2 = redboxes[:,3]

        
        blueboxes = np.array(blueBB)
        bluex1 = blueboxes[:,0]
        bluey1 = blueboxes[:,1]
        #bluex2 = blueboxes[:,2]
        #bluey2 = blueboxes[:,3]

        remainingIdxs = []
        for i in range(len(bluex1)):
            currentbluex1 = bluex1[i].item()
            currentbluey1 = bluey1[i].item()
            for j in range(len(redx1)):
                currentredx1 = redx1[j].item()
                currentredy1 = redy1[j].item()
                if (abs(currentbluex1 - currentredx1) < 75) and (abs(currentbluey1 - currentredy1) < 45):
                    remainingIdxs.append(j)
        
        if len(remainingIdxs) > 0:
            remainingIdxs = np.array(remainingIdxs)
            unique, counts = np.unique(remainingIdxs, return_counts=True)
            print(unique)
            remainingIdxs = np.array([False]*len(redx1))       
            remainingIdxs[unique] = True       

            remainingRedBoxes = redboxes[remainingIdxs,:]
            remainingRedBoxes = remainingRedBoxes.tolist()
        else:
            remainingRedBoxes = []
        
    bboxes = remainingRedBoxes + blueBB
    bboxes = non_max_suppression_fast(bboxes, 0.3)

    for box in bboxes:
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]        
        cv2.rectangle(imgFinal, (x1,y1), (x2,y2), (0,0,255), 2)
    
    cv2.imshow('ORIGINAL', image)
    cv2.imshow('MASK', mask)
    cv2.imshow('MASK BBOX', img)
    cv2.imshow('FINAL', imgFinal)
    #cv2.waitKey(0)

    return imgFinal, bboxes

def predictBrownBoundingBox(image, blueBB, brownLowerThres, brownUpperThres, cannyLowerThres, cannyUpperThres):
    img = np.copy(image)
    imgDilate = np.copy(image)
    imgFinal = np.copy(image)

    print('FASZ')
    print(blueBB)

    brown_lower = np.array([10,100,20], dtype=np.uint8)
    brown_upper = np.array([20,255,200], dtype=np.uint8)

    
    yellow_lower = np.array([20, 100, 100], dtype = np.uint8)
    yellow_upper = np.array([30, 255, 255], dtype = np.uint8)

    red_lower1 = np.array([0,70,50], dtype=np.uint8)
    red_upper1 = np.array([10, 255, 255], dtype=np.uint8)
    red_lower2 = np.array([170,70,50], dtype=np.uint8)
    red_upper2 = np.array([180, 255, 200], dtype=np.uint8)
    black_lower = np.array([0, 0, 0], dtype = np.uint8)
    black_upper = np.array([180, 255, 40], dtype = np.uint8)
    white_lower = np.array([0, 0, 230], dtype = np.uint8)
    white_upper = np.array([180, 255, 255], dtype = np.uint8)
    
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    redMask1 = cv2.inRange(imgHSV, red_lower1, red_upper1)
    redMask2 = cv2.inRange(imgHSV, red_lower2, red_upper2)
    
    brownMask = cv2.inRange(imgHSV, brownLowerThres, brownUpperThres)

    blackMask = cv2.inRange(imgHSV, black_lower, black_upper)
    
    whiteMask = cv2.inRange(imgHSV, white_lower, white_upper)
    whiteMask = cv2.erode(whiteMask, np.ones((3,3)), iterations = 1)

    yellowMask = cv2.inRange(imgHSV, yellow_lower, yellow_upper)

    #brownMask = cv2.inRange(imgHSV, brownLowerThres, brownUpperThres)
    mask = redMask1 + redMask2# + whiteMask# + brownMask# + blackMask# + whiteMask
    #mask = brownMask# + whiteMask
    mask = np.clip(mask, 0, 255)
    mask = np.uint8(mask)
    origmask = np.copy(mask)
    
    mask = cv2.erode(mask, np.ones((4,4)), iterations = 1)
    #mask = cv2.erode(mask, np.ones((3,3)), iterations = 1)
    #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2,2)), iterations = 1)
    #mask = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((2,2)), iterations = 2)
    #mask = cv2.morphologyEx(mask, cv2.MORPH_BLACKHAT, np.ones((2,2)), iterations = 1)
    mask = cv2.dilate(mask, np.ones((3,3)), iterations = 2)
    #mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2,2)), iterations = 1)

    for box in blueBB:
        mask[box[1]:box[3], box[0]:box[2]] = 0
    
    '''
    maskLineIdxs = (np.where(np.sum(np.clip(mask[0:80, :], 0, 1), axis = 1) >= 350))
    maskLastLineIdx = (maskLineIdxs[0][-1] if maskLineIdxs[0].size > 0 else 0)
    maskLastLineIdx = (maskLastLineIdx if (maskLastLineIdx % 2) == 0 else maskLastLineIdx - 1)
    print(maskLineIdxs)
    print(maskLastLineIdx)

    mask[0:maskLastLineIdx, 0] = 0
    '''




    maskCopy = np.copy(mask)
    maskCopyDilate = np.copy(mask)

    edges = cv2.Canny(maskCopy,cannyLowerThres,cannyUpperThres)
    edgesContours, edgesHierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    #maskCopyDilate = cv2.morphologyEx(maskCopyDilate, cv2.MORPH_CLOSE, np.ones((3,3)), iterations = 1)
    maskCopyDilate = cv2.dilate(maskCopyDilate, np.ones((3,3)), iterations = 4)
    #maskCopyDilate = cv2.morphologyEx(maskCopyDilate, cv2.MORPH_OPEN, np.ones((3,3)), iterations = 2)
    maskCopyDilate = cv2.morphologyEx(maskCopyDilate, cv2.MORPH_CLOSE, np.ones((3,3)), iterations = 3)


    maskLineIdxs = (np.where(np.sum(np.clip(maskCopyDilate[0:80, :], 0, 1), axis = 1) >= 350))
    maskLastLineIdx = (maskLineIdxs[0][-1] if maskLineIdxs[0].size > 0 else 0)
    #maskLastLineIdx = (maskLastLineIdx if (maskLastLineIdx % 2) == 0 else maskLastLineIdx - 1)
    print(maskLineIdxs)
    print(maskLastLineIdx)

    maskCopyDilate[0:maskLastLineIdx, :] = 0


    
    edgesDilate = cv2.Canny(maskCopyDilate,cannyLowerThres,cannyUpperThres)
    edgesDilateContours, edgesDilateHierarchy = cv2.findContours(edgesDilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(img, edgesContours, -1, (0, 255, 0), 3)
    cv2.drawContours(imgDilate, edgesContours, -1, (0, 255, 0), 3)

    possible_table_bboxes = []
    bboxes = []
    for c in edgesContours:
        rect = cv2.boundingRect(c)
        if rect[2] > 20 and rect[3] > 20 and rect[2] < 50 and rect[3] < 50:
            x,y,w,h = rect
            bboxes.append([x, y, x+w, y+h])
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
    
    bboxesDilate = []
    for c in edgesDilateContours:
        rect = cv2.boundingRect(c)
        if (rect[2] * rect[3] > 1000) and (rect[2] * rect[3] < 80000) and rect[2] < 300:
            x,y,w,h = rect
            bboxesDilate.append([x, y, x+w, y+h])
            cv2.rectangle(imgDilate,(x,y),(x+w,y+h),(0,0,255),2)
    
    possible_table_bboxes = blue_nms(bboxesDilate, 0.2)
    #possible_table_bboxes = non_max_suppression_fast(possible_table_bboxes, 0.3)
    for box in possible_table_bboxes:
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]        
        cv2.rectangle(imgFinal, (x1,y1), (x2,y2), (0,0,255), 2)
    
    cv2.imshow('ORIGINAL', image)
    cv2.imshow('ORIGINAL MASK', origmask)
    cv2.imshow('TRANSFORMED MASK', mask)
    #cv2.imshow('BROWN MASK DILATE', maskCopyDilate)
    cv2.imshow('MASK BBOX', img)
    #cv2.imshow('MASK DILATE BBOX', imgDilate)
    #cv2.imshow('FINAL', imgFinal)
    cv2.imshow('CANNY', edges)
    #cv2.imshow('CANNY DILATE', edgesDilate)
    cv2.waitKey(0)
    
    return img, maskLastLineIdx


