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

