import numpy as np
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
from pandas import DataFrame

class Zones:
    """
    HoopStats Zone Tracking Module
    ---------------------------------
    This class detects and analyzes player/ball movement across predefined
    basketball court zones (defined via a color-coded mask image).

    Functions:
        - Load and update zone masks
        - Track objects per zone
        - Count transitions between zones
        - Generate visual tables for analysis
    """

    def __init__(self, path, params=None, queue_size=50):
        self.loadZoneMask(path, params)
        self.zoneQueue = {}
        self.zoneCount = {}
        self.zoneTransitions = np.zeros((self.nrZones, self.nrZones)).astype(int)

        for i in range(self.nrZones):
            self.zoneQueue[i + 1] = [None] * queue_size
            self.zoneCount[i + 1] = 0


    def updateZoneMask(self, params=None):
        """Updates zone mask based on given parameters or defaults"""
        if (params is not None) and hasattr(params, 'zones_mask') and params.zones_mask is not None:
            for i, color in enumerate(params.zones_mask):
                self.mask[self.mask == color] = i + 1
        else:
            # Default: assign each unique color to a new zone ID
            unique_colors = np.unique(self.mask)
            for i, color in enumerate(unique_colors[1:]):  # Skip background (0)
                self.mask[self.mask == color] = i + 1


    def loadZoneMask(self, path, params=None):
        """Loads a color-coded zone mask image"""
        self.mask = (255 * plt.imread(path)).astype(int)
        if len(self.mask.shape) == 3 and self.mask.shape[2] > 1:
            self.mask = self.mask[:, :, 0]
        self.nrZones = len(np.unique(self.mask)) - 1
        self.updateZoneMask(params)


    def getZoneOccupancy(self, zone, timeStamp=None, timeScope=None, lookback=10):
        """
        Returns the average time or frequency of object presence in a given zone.
        Replaces the 'getMeanSpeed' concept from traffic analysis.
        """
        durations = np.array([])
        for obj in self.zoneQueue[zone]:
            if obj is None:
                continue
            if len(obj.bboxes) <= lookback:
                continue
            if (timeStamp is not None) and (timeScope is not None):
                if obj.bboxes[-1].timeStamp < timeStamp - timeScope:
                    continue
            duration = obj.getPresenceDuration(lookback=lookback)
            if duration is not None:
                durations = np.append(durations, duration)
        if len(durations) > 0:
            return np.median(durations)
        return None


    def addObject(self, obj, minBoxes=10, classes=None):
        """
        Adds an object (player or ball) to its corresponding zone.
        Tracks transitions if the object moved between zones.
        """
        if len(obj.bboxes) < minBoxes:
            return -1

        if classes is not None:
            raise ('Classes Not Implemented!')

        x = int(obj.bboxes[-1].x)
        y = int(obj.bboxes[-1].y)

        if y == self.mask.shape[0]:
            y -= 1

        if (y < self.mask.shape[0]) and (x < self.mask.shape[1]):
            zone = self.mask[y, x]
        else:
            zone = 0

        if zone > 0:
            if obj not in self.zoneQueue[zone]:
                self.zoneQueue[zone].append(obj)
                self.zoneQueue[zone].pop(0)
                self.zoneCount[zone] += 1

                # Handle zone transitions
                if hasattr(obj, 'zoneNr'):
                    self.zoneTransitions[obj.zoneNr - 1, zone - 1] += 1
                obj.zoneNr = zone

        return zone


    def addObjects(self, objects, minBoxes=10, classes=None):
        """Adds multiple detected objects to their zones"""
        for obj in objects:
            self.addObject(obj, minBoxes=minBoxes, classes=classes)


    def _color_matrix(self, df, color):
        """Helper function to fill table cells with a specific color"""
        return [[color] * df.values.shape[1]] * df.values.shape[0]


    def getDataTables(self, imgsize=(270, 300), timeStamp=None, timeScope=60,
                      cell_color='#FFFF00', header_color='#00FFFF'):
        """
        Generates zone occupancy and transition tables for visualization.
        Returns an RGB image array containing both tables stacked vertically.
        """
        backend_ = mpl.get_backend()
        mpl.use("Agg")  # Disable GUI rendering

        # --- Zone Occupancy Table ---
        df = DataFrame(columns=['Detections', 'Avg Time [s]'])
        for i in range(self.nrZones):
            duration = self.getZoneOccupancy(i + 1)
            if duration is None:
                duration = '-'
            else:
                duration = round(duration, 2)
            df.loc[i] = [self.zoneCount[i + 1], duration]

        zone_labels = [f'Zone {x + 1}' for x in range(self.nrZones)]

        fig = plt.figure()
        ax = plt.subplot(111, frame_on=False)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        dpi = plt.gcf().get_dpi()
        fig.set_size_inches(imgsize[1] / float(dpi), imgsize[0] / 2 / float(dpi))

        tmp = ax.table(cellText=df.values, colLabels=df.keys(), rowLabels=zone_labels,
                       loc='center', cellLoc='center',
                       cellColours=self._color_matrix(df, cell_color),
                       colColours=self._color_matrix(df, header_color)[0],
                       rowColours=[header_color] * self.nrZones)
        fig.canvas.draw()

        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()

        # --- Zone Transition Matrix ---
        fig = plt.figure()
        ax = plt.subplot(111, frame_on=False)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        dpi = plt.gcf().get_dpi()
        fig.set_size_inches(imgsize[1] / float(dpi), imgsize[0] / 2 / float(dpi))

        tmp = ax.table(cellText=self.zoneTransitions,
                       colLabels=zone_labels, rowLabels=zone_labels,
                       loc='center', cellLoc='center',
                       cellColours=[[cell_color] * self.nrZones] * self.nrZones,
                       colColours=[header_color] * self.nrZones,
                       rowColours=[header_color] * self.nrZones)
        fig.canvas.draw()

        data2 = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data2 = data2.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # Stack both tables
        data = np.concatenate((data, data2), axis=0)

        # Remove white padding
        white_filter = np.sum(data, axis=2) == 3 * 255
        data[white_filter, :] = 0
        plt.close()
        mpl.use(backend_)  # Restore backend
        return data
