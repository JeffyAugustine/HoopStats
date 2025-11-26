import cv2
import numpy as np
import random
from enum import Enum
from time import time

class ObjectType(Enum):
    Unknown = 0
    Player = 1
    Ball = 37

class BoundingBox:
    def __init__(self, xLeft, xRight, yTop, yBottom, timeStamp=None, params=None, toPoint='BC'):
        self.xLeft = xLeft
        self.xRight = xRight
        self.yTop = yTop
        self.yBottom = yBottom
        self.timeStamp = timeStamp
        self.toPoint = toPoint
        self.__calculateToPoint()
        self.objType = None

        self.birdEyeX = None
        self.birdEyeY = None
        self.zoneID = None
        self.params_updated = False

        if params is not None:
            self.updateParams(params)

    def __calculateToPoint(self):
        self.x = int((self.xLeft + self.xRight) / 2)
        self.y = int(self.yBottom)

    def getTrackerCoordinates(self):
        """Return coordinates in (x, y, w, h) format."""
        return (self.xLeft, self.yTop, self.xRight - self.xLeft, self.yBottom - self.yTop)

    def updateParams(self, params, force_update=False):
        if force_update or not self.params_updated:
            p = np.array([[[self.x, self.y]]], dtype='float32')
            tmp = cv2.perspectiveTransform(p, params.unwarp_M)
            self.birdEyeX, self.birdEyeY = tmp[0][0]
            if hasattr(params, 'getZoneID'):
                self.zoneID = params.getZoneID(self.birdEyeX, self.birdEyeY)
            self.params_updated = True


class Object:
    def __init__(self, object_type):
        self.type = object_type
        self.bboxes = []
        self.tracker = None
        self.color = tuple(random.randint(0, 255) for _ in range(3))
        self.notDetectedCounter = 0
        self.id = f"{id(self)}_{time()}"
        self.teamId = -1  
        self.possession = 0  

    def createTracker(self, img, bbox, add_box=True):
        self.tracker = cv2.TrackerCSRT_create()
        success = self.tracker.init(img, bbox.getTrackerCoordinates())
        if add_box:
            self.bboxes.append(bbox)
        return success

    def updateTracker(self, img, timeStamp=None):
        success, bbox = self.tracker.update(img)
        if success:
            if timeStamp is None:
                timeStamp = time()
            bbox = BoundingBox(bbox[0], bbox[0] + bbox[2], bbox[1], bbox[1] + bbox[3], timeStamp)
            self.addBoundingBox(bbox)
            return bbox
        else:
            return False

    def addBoundingBox(self, bbox):
        self.bboxes.append(bbox)

    def getTrackerCoordinates(self):
        last = self.bboxes[-1]
        return (last.xLeft, last.yTop, last.xRight - last.xLeft, last.yBottom - last.yTop)
    
    def getParams(self, asCsv=False, speedLookback=10):
        """
        Returns core object parameters used for logging.
        Supports CSV or dict output.
        """

        if len(self.bboxes) == 0:
            return "" if asCsv else {}

        last = self.bboxes[-1]

        params = {
            "x": last.x,
            "y": last.y,
            "birdEyeX": last.birdEyeX if last.birdEyeX is not None else -1,
            "birdEyeY": last.birdEyeY if last.birdEyeY is not None else -1,
            "zoneID": last.zoneID if last.zoneID is not None else -1,
            "timeStamp": last.timeStamp if last.timeStamp is not None else -1,
            "teamId": self.teamId,
            "possession": self.possession
        }

        if asCsv:
            return "{},{},{},{},{},{},{},{}".format(
                params["x"],
                params["y"],
                params["birdEyeX"],
                params["birdEyeY"],
                params["zoneID"],
                params["timeStamp"],
                params["teamId"],
                params["possession"]
            )

        return params