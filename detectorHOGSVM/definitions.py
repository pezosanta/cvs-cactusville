import cv2

# HOG params
winSize = (48, 64)
blockSize = (16, 16)
blockStride = (8, 8)
cellSize = (8, 8)
bins = 9

# HOG feature collect
hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, bins)

classNames = ['traffic sign', 'vehicle', 'cactus']
subclassNames = [
    ['Bump', 'Bumpy road', 'Bus stop', 'Children', 'Crossing (blue)', 'Crossing (red)', 'Cyclists',
     'Danger (other)', 'Dangerous left turn', 'Dangerous right turn', 'Give way', 'Go ahead', 'Go ahead or left',
     'Go ahead or right', 'Go around either way', 'Go around left', 'Go around right', 'Intersection', 'Limit 100',
     'Limit 120', 'Limit 20', 'Limit 30', 'Limit 50', 'Limit 60', 'Limit 70', 'Limit 80', 'Limit 80 over',
     'Limit over', 'Main road', 'Main road over', 'Multiple dangerous turns', 'Narrow road (left)',
     'Narrow road (right)', 'No entry', 'No entry (both directions)', 'No entry (truck)', 'No stopping', 'No takeover',
     'No takeover (truck)', 'No takeover (truck) end', 'No takeover end', 'No waiting', 'One way road',
     'Parking', 'Road works', 'Roundabout', 'Slippery road', 'Stop', 'Traffic light', 'Train crossing',
     'Train crossing (no barrier)', 'Wild animals', 'X - Priority', 'X - Turn left', 'X - Turn right'],
    ['SUV', 'truck', 'plane'],
    ['happy', 'sad', 'angry', 'evil']
]


class ImgObject:
    classNames = ['traffic sign', 'vehicle', 'cactus']

    def __init__(self, obj: list):  # obj = [u, v, w, h, classInd, subClassInd, x, y, z]
        assert len(obj) == 9, "Object list size is not 9!"
        self.u, self.v = obj[0], obj[1]
        self.w, self.h = obj[2], obj[3]
        self.classInd = obj[4]
        self.subClassInd = obj[5]
        self.x, self.y, self.z = obj[6:9]

    def getClassName(self):
        return self.classNames[self.classInd]

    def getSubImgCoords(self, img, scale=1):
        max_y, max_x = img.shape[:2]

        x1 = max(self.u - int(self.w * scale / 2), 0)
        x2 = min(x1 + int(self.w * scale), max_x)
        y1 = max(self.v - int(self.h * scale / 2), 0)
        y2 = min(y1 + int(self.h * scale), max_y)

        return x1, y1, x2, y2

    def makeList(self):
        obj = [0 for _ in range(9)]
        obj[0:2] = self.u, self.v
        obj[2:4] = self.w, self.h
        obj[4] = self.classInd
        obj[5] = self.subClassInd
        obj[6:9] = [self.x, self.y, self.z]
        return obj
